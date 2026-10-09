"""
MRI V5.5 — ROBUST AUTOMATIC QUALITY VALIDATION

Works with the existing V5.3 output.
NO TRAINING.
NO NEW SYNTHETIC GENERATION.
NO IMAGE UPLOAD REQUIRED.

It reads:
    outputs/mri_testing_v5_3_quality_control/pilot_*/synthetic/
    outputs/mri_testing_v5_3_quality_control/pilot_*/reports/v5_generation_report.json
    data/imaging/MRI/Testing/
    outputs/mri_testing_v4_latents/run_20261001_044203/latents.pt
    outputs/mri_testing_v3_reconstruction/run_20260922_044321/models/best_autoencoder.pt

It produces:
    outputs/mri_testing_v5_5_quality_validation/run_YYYYMMDD_HHMMSS/

Main validation:
    1. Real-vs-synthetic image statistics
    2. Brightness / contrast
    3. Foreground ratio
    4. Sharpness / edge energy
    5. Histogram similarity
    6. Downsampled structural similarity
    7. V3 re-encoded latent distance
    8. Real-manifold novelty reference
    9. Synthetic duplicate rate
    10. Synthetic diversity
    11. Per-class PASS / REVIEW / FAIL
    12. HTML report with embedded preview contact sheets
    13. CSV + JSON reports

IMPORTANT:
These metrics are engineering/data-quality checks only.
They do not establish medical or clinical validity.
"""

from __future__ import annotations

import base64
import html
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)

V5_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v5_3_quality_control"
)

LATENT_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v4_latents"
    / "run_20261001_044203"
    / "latents.pt"
)

V3_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v3_reconstruction"
    / "run_20260922_044321"
    / "models"
    / "best_autoencoder.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v5_5_quality_validation"
)

IMAGE_SIZE = 128

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

SEED = 5404

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# V5.3 pilot is expected to contain 32/class.
EXPECTED_SYNTHETIC_PER_CLASS = 32

# ----------------------------------------------------------------
# PASS / REVIEW thresholds.
#
# These are deliberately conservative engineering thresholds.
# They are NOT medical validity thresholds.
# ----------------------------------------------------------------

MIN_OVERALL_SCORE_PASS = 0.68
MIN_OVERALL_SCORE_REVIEW = 0.50

MIN_HISTOGRAM_SIMILARITY = 0.55
MIN_STRUCTURAL_SIMILARITY = 0.55

MAX_MEAN_Z = 3.0
MAX_STD_RATIO_ERROR = 0.55

MIN_SHARPNESS_RATIO = 0.35
MAX_SHARPNESS_RATIO = 2.75

MAX_FOREGROUND_ERROR = 0.22

MAX_DUPLICATE_RATE = 0.10

# Latent novelty:
# We compare synthetic nearest-real distance against the real
# leave-one-out nearest-neighbour baseline.
#
# No Tumor can have a very small real baseline, so we do not use
# one fixed raw distance threshold for every class.
MAX_ROBUST_NOVELTY_Z_REVIEW = 6.0
MAX_NOVELTY_EXCEEDANCE_REVIEW = 0.50

# Diversity:
# Mean pairwise cosine distance among synthetic re-encoded
# latents should not collapse excessively.
MIN_SYNTHETIC_DIVERSITY = 0.002

# Duplicate pixel MSE threshold.
DUPLICATE_MSE_THRESHOLD = 0.0008


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    arr = (
        np.asarray(image, dtype=np.float32)
        / 255.0
    )

    return torch.from_numpy(
        arr
    ).unsqueeze(0)


def tensor_to_array(
    x: torch.Tensor,
) -> np.ndarray:
    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3:
        x = x[0]

    arr = x.numpy()

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(arr, 0.0, 1.0)


def image_mean(x: torch.Tensor) -> float:
    return float(
        x.detach().float().mean().cpu()
    )


def image_std(x: torch.Tensor) -> float:
    return float(
        x.detach().float().std().cpu()
    )


def foreground_ratio(
    x: torch.Tensor,
) -> float:
    arr = tensor_to_array(x)

    p10 = np.percentile(arr, 10)
    p75 = np.percentile(arr, 75)

    threshold = (
        p10
        + 0.18 * max(p75 - p10, 1e-6)
    )

    return float(
        (arr > threshold).mean()
    )


def sharpness(
    x: torch.Tensor,
) -> float:
    tensor = x.detach().float()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=tensor.dtype,
    ).view(1, 1, 3, 3)

    response = F.conv2d(
        tensor[:, :1],
        kernel,
        padding=1,
    )

    return float(
        response.pow(2)
        .mean()
        .cpu()
    )


def mse(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return float(
        F.mse_loss(
            a.float(),
            b.float(),
        ).cpu()
    )


# ============================================================
# IMAGE HISTOGRAM / STRUCTURE
# ============================================================

def normalized_histogram(
    x: torch.Tensor,
    bins: int = 32,
) -> np.ndarray:
    arr = tensor_to_array(x).reshape(-1)

    hist, _ = np.histogram(
        arr,
        bins=bins,
        range=(0.0, 1.0),
    )

    hist = hist.astype(np.float64)

    total = hist.sum()

    if total > 0:
        hist /= total

    return hist


def histogram_intersection(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    ha = normalized_histogram(a)
    hb = normalized_histogram(b)

    return float(
        np.minimum(ha, hb).sum()
    )


def downsample_structure(
    x: torch.Tensor,
    size: int = 32,
) -> np.ndarray:
    arr = tensor_to_array(x)

    image = Image.fromarray(
        (arr * 255).astype(np.uint8),
        mode="L",
    )

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    out = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    # Normalize independently to reduce pure brightness effects.
    out = (
        out - out.mean()
    ) / max(
        out.std(),
        1e-6,
    )

    return out


def structural_similarity_proxy(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """
    Lightweight structural similarity proxy.

    It compares normalized low-resolution images using
    normalized mean absolute difference.

    This intentionally does not call it clinical SSIM.
    """
    aa = downsample_structure(a)
    bb = downsample_structure(b)

    diff = np.mean(
        np.abs(aa - bb)
    )

    score = 1.0 / (
        1.0 + diff
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ============================================================
# V3 AUTOENCODER — EXACT ARCHITECTURE USED BY V3 TRAINING
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.SiLU(),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 96, 4, 2, 1),
            nn.SiLU(),
            ResBlock(96),
            ResBlock(96),
            nn.Conv2d(96, 16, 4, 2, 1),
            nn.SiLU(),
            ResBlock(16),
            ResBlock(16),
            nn.Conv2d(16, 16, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 96, 3, padding=1),
            nn.SiLU(),
            ResBlock(96),
            ResBlock(96),
            nn.ConvTranspose2d(96, 96, 4, 2, 1),
            nn.SiLU(),
            ResBlock(96),
            ResBlock(96),
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.SiLU(),
            ResBlock(64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class MRIReconstructionAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported V3 checkpoint format.")

    # V3 training saves: {"model": model.state_dict(), ...}
    for key in ["model", "model_state_dict", "state_dict", "autoencoder_state_dict"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError("Could not find V3 model state_dict in checkpoint.")


def load_v3():
    if not V3_CHECKPOINT.exists():
        raise FileNotFoundError(f"V3 checkpoint missing:\n{V3_CHECKPOINT}")

    checkpoint = torch.load(V3_CHECKPOINT, map_location="cpu")

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    print("V3 checkpoint config:", config)

    expected = {
        "image_size": 128,
        "latent_channels": 16,
        "latent_size": 32,
        "classes": CLASSES,
    }
    if config and config != expected:
        print("WARNING: checkpoint config differs from expected V3 config.")

    model = MRIReconstructionAE()
    state = extract_state_dict(checkpoint)

    # V3 checkpoint keys are already exactly:
    # encoder.net.... and decoder.net....
    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = value

    incompatible = model.load_state_dict(cleaned, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "V3 checkpoint architecture mismatch.\n"
            f"Missing keys: {incompatible.missing_keys[:20]}\n"
            f"Unexpected keys: {incompatible.unexpected_keys[:20]}"
        )

    model.eval().to(DEVICE)

    # Verify the exact latent shape before any validation.
    with torch.no_grad():
        probe = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        z = model.encode(probe)

    expected_shape = (1, 16, 32, 32)
    if tuple(z.shape) != expected_shape:
        raise RuntimeError(
            f"V3 encoder produced {tuple(z.shape)}, expected {expected_shape}."
        )

    print("V3 checkpoint loaded successfully with exact architecture match.")
    print("V3 encoder latent shape:", tuple(z.shape[1:]))
    return model


def preprocess_for_v3(path: Path) -> torch.Tensor:
    """Exact V3 input preprocessing used during reconstruction training."""
    image = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo + 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return torch.from_numpy(arr).float().unsqueeze(0) * 2.0 - 1.0


# ============================================================
# V5.3 RUN DISCOVERY
# ============================================================

def discover_v5_run() -> Path:
    if not V5_ROOT.exists():
        raise FileNotFoundError(
            f"V5.3 root not found:\n{V5_ROOT}"
        )

    runs = [
        p
        for p in V5_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("pilot_")
    ]

    if not runs:
        raise FileNotFoundError(
            "No V5.3 pilot run was found."
        )

    runs.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return runs[0]


def discover_synthetic_images(
    v5_run: Path,
) -> Dict[str, List[Path]]:
    synthetic_root = (
        v5_run / "synthetic"
    )

    result = {}

    for class_name in CLASSES:
        class_dir = (
            synthetic_root / class_name
        )

        if not class_dir.exists():
            # V5.3 versions may use a classes/synthetic layout.
            class_dir = (
                v5_run
                / "classes"
                / class_name
                / "synthetic"
            )

        files = []

        if class_dir.exists():
            files = sorted(
                [
                    p
                    for p in class_dir.rglob("*")
                    if p.suffix.lower()
                    in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp",
                    }
                ]
            )

        result[class_name] = files

    return result


# ============================================================
# REAL IMAGE DISCOVERY
# ============================================================

def discover_real_images():
    result = {}

    for class_name in CLASSES:
        class_dir = (
            TEST_ROOT / class_name
        )

        files = sorted(
            [
                p
                for p in class_dir.rglob("*")
                if p.suffix.lower()
                in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }
            ]
        )

        result[class_name] = files

    return result


# ============================================================
# LATENT CACHE
# ============================================================

def load_latents():
    if not LATENT_CACHE.exists():
        raise FileNotFoundError(f"Latent cache missing:\n{LATENT_CACHE}")

    obj = torch.load(LATENT_CACHE, map_location="cpu")

    if not isinstance(obj, dict):
        raise RuntimeError("V4 latent cache must be a dictionary payload.")

    # IMPORTANT: never use `a or b` with tensors.
    # Tensor truth-value evaluation raises:
    # Boolean value of Tensor with more than one value is ambiguous.
    latents = obj.get("latents", None)
    if latents is None:
        latents = obj.get("latent", None)
    if latents is None:
        latents = obj.get("z", None)

    labels = obj.get("labels", None)
    if labels is None:
        labels = obj.get("class_labels", None)
    if labels is None:
        labels = obj.get("y", None)

    if latents is None:
        raise RuntimeError("Latent tensor not found in V4 cache.")
    if labels is None:
        raise RuntimeError("Latent labels not found in V4 cache.")

    latents = torch.as_tensor(latents).float()
    labels = torch.as_tensor(labels).long().view(-1)

    expected = (16, 32, 32)
    if tuple(latents.shape[1:]) != expected:
        raise RuntimeError(
            f"Unexpected cached latent shape {tuple(latents.shape[1:])}; expected {expected}."
        )

    if len(latents) != len(labels):
        raise RuntimeError(
            f"Latent/label count mismatch: {len(latents)} vs {len(labels)}"
        )

    return latents, labels

def flatten_latents(z):
    return z.reshape(
        z.shape[0],
        -1,
    ).float()


def cosine_distance_matrix(
    a,
    b,
):
    a = F.normalize(
        a.float(),
        dim=1,
    )

    b = F.normalize(
        b.float(),
        dim=1,
    )

    return 1.0 - a @ b.T


def real_latent_baseline(
    class_latents,
):
    """
    Build a DISTRIBUTION baseline instead of using only the
    median real-to-real nearest-neighbour distance.

    This fixes the V5.4 instability: if a class has an extremely
    small nearest-neighbour median (especially No Tumor), dividing
    by that single number can produce meaningless ratios such as
    510x novelty.
    """
    flat = flatten_latents(class_latents)

    distances = cosine_distance_matrix(flat, flat)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values.cpu().numpy()

    q01, q05, q25, q50, q75, q95, q99 = np.percentile(
        nearest, [1, 5, 25, 50, 75, 95, 99]
    )
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(nearest - q50)))
    robust_scale = max(1.4826 * mad, 0.25 * iqr, 0.002)

    return {
        "count": len(class_latents),
        "median": float(q50),
        "mean": float(np.mean(nearest)),
        "std": float(np.std(nearest)),
        "min": float(np.min(nearest)),
        "max": float(np.max(nearest)),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "iqr": iqr,
        "mad": mad,
        "robust_scale": robust_scale,
        "nearest_distribution": nearest.tolist(),
    }


# ============================================================
# REAL IMAGE BASELINES
# ============================================================

def image_baseline(
    paths: List[Path],
):
    means = []
    stds = []
    sharp = []
    fg = []

    for path in paths:
        x = load_image(path)

        means.append(
            image_mean(x)
        )

        stds.append(
            image_std(x)
        )

        sharp.append(
            sharpness(x)
        )

        fg.append(
            foreground_ratio(x)
        )

    return {
        "count": len(paths),
        "mean": float(np.mean(means)),
        "mean_std": float(np.std(means)),
        "std": float(np.mean(stds)),
        "std_std": float(np.std(stds)),
        "sharpness": float(np.median(sharp)),
        "sharpness_mean": float(np.mean(sharp)),
        "foreground": float(np.median(fg)),
        "foreground_mean": float(np.mean(fg)),
    }


# ============================================================
# SYNTHETIC ANALYSIS
# ============================================================

def summarize_synthetic_images(
    paths,
    real_base,
):
    means = []
    stds = []
    sharp = []
    fg = []
    hist_scores = []
    structure_scores = []

    for path in paths:
        x = load_image(path)

        means.append(
            image_mean(x)
        )

        stds.append(
            image_std(x)
        )

        sharp.append(
            sharpness(x)
        )

        fg.append(
            foreground_ratio(x)
        )

        # Compare each synthetic image against
        # the REAL CLASS MEAN IMAGE statistics later.
        # For histogram/structure we use the median
        # real image in a separate pass.
        hist_scores.append(0.0)
        structure_scores.append(0.0)

    return {
        "means": means,
        "stds": stds,
        "sharpness": sharp,
        "foreground": fg,
    }


def choose_reference_real_images(
    real_paths,
    count=16,
):
    """
    Deterministically chooses representative real images
    distributed through the class folder.
    """
    if not real_paths:
        return []

    if len(real_paths) <= count:
        return real_paths

    positions = np.linspace(
        0,
        len(real_paths) - 1,
        count,
    ).astype(int)

    return [
        real_paths[int(i)]
        for i in positions
    ]


def class_histogram_structure_scores(
    real_paths,
    synthetic_paths,
):
    real_refs = choose_reference_real_images(
        real_paths,
        16,
    )

    if not real_refs or not synthetic_paths:
        return 0.0, 0.0

    real_images = [
        load_image(p)
        for p in real_refs
    ]

    hist_scores = []
    structure_scores = []

    for syn_path in synthetic_paths:
        syn = load_image(
            syn_path
        )

        h = [
            histogram_intersection(
                syn,
                real,
            )
            for real in real_images
        ]

        s = [
            structural_similarity_proxy(
                syn,
                real,
            )
            for real in real_images
        ]

        # Use best-of-reference rather than average:
        # a synthetic MRI should resemble at least one
        # valid real image structure.
        hist_scores.append(
            max(h)
        )

        structure_scores.append(
            max(s)
        )

    return (
        float(np.mean(hist_scores)),
        float(np.mean(structure_scores)),
    )


# ============================================================
# LATENT RE-ENCODING
# ============================================================

def encode_images(model, paths):
    latents = []

    with torch.no_grad():
        for path in paths:
            x = preprocess_for_v3(path)
            z = model.encode(
                x.unsqueeze(0).to(DEVICE)
            ).cpu()
            latents.append(z[0])

    if not latents:
        return torch.empty(0, 16, 32, 32)

    return torch.stack(latents)

def synthetic_latent_metrics(
    synthetic_latents,
    real_latents,
    real_baseline,
):
    if len(synthetic_latents) == 0:
        return {
            "nearest_distances": [],
            "mean_nearest_distance": None,
            "median_nearest_distance": None,
            "synthetic_q95": None,
            "real_q95_exceedance_fraction": None,
            "robust_z": None,
            "diversity": None,
        }

    syn_flat = flatten_latents(synthetic_latents)
    real_flat = flatten_latents(real_latents)
    distances = cosine_distance_matrix(syn_flat, real_flat)
    nearest = distances.min(dim=1).values.cpu().numpy()

    real_q95 = real_baseline["q95"]
    exceedance_fraction = float(np.mean(nearest > real_q95))
    syn_median = float(np.median(nearest))
    robust_z = float(
        (syn_median - real_baseline["median"])
        / max(real_baseline["robust_scale"], 1e-6)
    )

    if len(syn_flat) > 1:
        pairwise = cosine_distance_matrix(syn_flat, syn_flat)
        mask = ~torch.eye(len(syn_flat), dtype=torch.bool)
        diversity = float(pairwise[mask].mean())
    else:
        diversity = 0.0

    return {
        "nearest_distances": [float(x) for x in nearest.tolist()],
        "mean_nearest_distance": float(np.mean(nearest)),
        "median_nearest_distance": syn_median,
        "synthetic_q25": float(np.percentile(nearest, 25)),
        "synthetic_q75": float(np.percentile(nearest, 75)),
        "synthetic_q95": float(np.percentile(nearest, 95)),
        "real_q95_exceedance_fraction": exceedance_fraction,
        "robust_z": robust_z,
        "diversity": diversity,
    }


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def duplicate_metrics(
    paths,
):
    if len(paths) < 2:
        return {
            "duplicate_pairs": 0,
            "pair_count": 0,
            "duplicate_rate": 0.0,
        }

    images = [
        load_image(p)
        for p in paths
    ]

    duplicate_pairs = 0
    pair_count = 0

    for i in range(len(images)):
        for j in range(
            i + 1,
            len(images),
        ):
            pair_count += 1

            if mse(
                images[i],
                images[j],
            ) < DUPLICATE_MSE_THRESHOLD:
                duplicate_pairs += 1

    rate = (
        duplicate_pairs
        / max(pair_count, 1)
    )

    return {
        "duplicate_pairs": duplicate_pairs,
        "pair_count": pair_count,
        "duplicate_rate": float(rate),
    }


# ============================================================
# CLASS SCORE
# ============================================================

def clamp01(x):
    return float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )


def class_quality_score(
    *,
    hist_similarity,
    structure_similarity,
    mean_z,
    std_ratio_error,
    sharpness_ratio,
    foreground_error,
    novelty_robust_z,
    novelty_exceedance,
    duplicate_rate,
    diversity,
):
    histogram_score = clamp01(
        (
            hist_similarity
            - 0.30
        )
        / 0.70
    )

    structure_score = clamp01(
        (
            structure_similarity
            - 0.35
        )
        / 0.65
    )

    mean_score = math.exp(
        -min(mean_z, 8.0) / 2.5
    )

    std_score = math.exp(
        -min(
            std_ratio_error,
            3.0,
        )
        / 1.5
    )

    stats_score = (
        mean_score
        + std_score
    ) / 2.0

    sharpness_score = math.exp(
        -abs(
            math.log(
                max(
                    sharpness_ratio,
                    1e-6,
                )
            )
        )
    )

    foreground_score = math.exp(
        -min(
            foreground_error,
            1.0,
        )
        / 0.30
    )

    if novelty_robust_z is None:
        novelty_score = 0.0
    else:
        # Robust z-score is stable even when the real baseline
        # median is extremely close to zero.
        z = max(0.0, novelty_robust_z)
        novelty_score = float(np.clip(1.0 - z / 6.0, 0.0, 1.0))

    duplicate_score = clamp01(
        1.0
        - duplicate_rate
        / max(
            MAX_DUPLICATE_RATE,
            1e-6,
        )
    )

    diversity_score = clamp01(
        diversity
        / max(
            MIN_SYNTHETIC_DIVERSITY * 4,
            1e-6,
        )
    )

    # Weighted overall score.
    score = (
        0.20 * histogram_score
        + 0.20 * structure_score
        + 0.15 * stats_score
        + 0.10 * sharpness_score
        + 0.10 * foreground_score
        + 0.10 * novelty_score
        + 0.075 * duplicate_score
        + 0.075 * diversity_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def determine_status(
    *,
    overall_score,
    hist_similarity,
    structure_similarity,
    mean_z,
    std_ratio_error,
    sharpness_ratio,
    foreground_error,
    novelty_robust_z,
    novelty_exceedance,
    duplicate_rate,
    diversity,
    synthetic_count,
):
    hard_fail_reasons = []

    if synthetic_count == 0:
        hard_fail_reasons.append(
            "no_synthetic_images"
        )

    if hist_similarity < MIN_HISTOGRAM_SIMILARITY:
        hard_fail_reasons.append(
            "low_histogram_similarity"
        )

    if (
        structure_similarity
        < MIN_STRUCTURAL_SIMILARITY
    ):
        hard_fail_reasons.append(
            "low_structural_similarity"
        )

    if mean_z > MAX_MEAN_Z:
        hard_fail_reasons.append(
            "brightness_outlier"
        )

    if std_ratio_error > MAX_STD_RATIO_ERROR:
        hard_fail_reasons.append(
            "contrast_outlier"
        )

    if not (
        MIN_SHARPNESS_RATIO
        <= sharpness_ratio
        <= MAX_SHARPNESS_RATIO
    ):
        hard_fail_reasons.append(
            "sharpness_outlier"
        )

    if foreground_error > MAX_FOREGROUND_ERROR:
        hard_fail_reasons.append(
            "foreground_outlier"
        )

    if duplicate_rate > MAX_DUPLICATE_RATE:
        hard_fail_reasons.append(
            "high_duplicate_rate"
        )

    if diversity < MIN_SYNTHETIC_DIVERSITY:
        hard_fail_reasons.append(
            "low_synthetic_diversity"
        )

    if (
        novelty_robust_z is not None
        and novelty_robust_z > MAX_ROBUST_NOVELTY_Z_REVIEW
    ):
        hard_fail_reasons.append("latent_novelty_robust_z_outlier")

    if (
        novelty_exceedance is not None
        and novelty_exceedance > MAX_NOVELTY_EXCEEDANCE_REVIEW
    ):
        hard_fail_reasons.append("too_many_synthetic_latents_beyond_real_q95")

    if hard_fail_reasons:
        return (
            "REVIEW",
            hard_fail_reasons,
        )

    if overall_score >= MIN_OVERALL_SCORE_PASS:
        return (
            "PASS",
            [],
        )

    if overall_score >= MIN_OVERALL_SCORE_REVIEW:
        return (
            "REVIEW",
            ["overall_score_below_pass"],
        )

    return (
        "FAIL",
        ["overall_score_too_low"],
    )


# ============================================================
# HTML REPORT
# ============================================================

def image_to_data_uri(
    path: Path,
) -> str:
    data = path.read_bytes()

    encoded = base64.b64encode(
        data
    ).decode("ascii")

    return (
        "data:image/png;base64,"
        + encoded
    )


def make_html_report(
    output_dir: Path,
    v5_run: Path,
    class_reports: List[Dict],
):
    rows = []

    for report in class_reports:
        status = report["status"]

        if status == "PASS":
            status_class = "pass"
        elif status == "REVIEW":
            status_class = "review"
        else:
            status_class = "fail"

        rows.append(
            f"""
            <tr>
              <td>{html.escape(report["class_name"])}</td>
              <td class="{status_class}">
                {status}
              </td>
              <td>{report["synthetic_count"]}</td>
              <td>{report["overall_score"]:.3f}</td>
              <td>{report["histogram_similarity"]:.3f}</td>
              <td>{report["structural_similarity"]:.3f}</td>
              <td>{report["mean_z"]:.3f}</td>
              <td>{report["sharpness_ratio"]:.3f}</td>
              <td>{report["duplicate_rate"]:.3%}</td>
              <td>
                {(
                    f'{report["novelty_robust_z"]:.3f}'
                    if report["novelty_robust_z"] is not None
                    else "N/A"
                )}
              </td>
              <td>{report["novelty_q95_exceedance_fraction"]:.1%}</td>
            </tr>
            """
        )

    cards = []

    for report in class_reports:
        class_name = report["class_name"]

        selected_preview = Path(
            report["selected_preview"]
        )

        real_vs_syn = Path(
            report["real_vs_synthetic_preview"]
        )

        selected_uri = (
            image_to_data_uri(
                selected_preview
            )
            if selected_preview.exists()
            else ""
        )

        real_syn_uri = (
            image_to_data_uri(
                real_vs_syn
            )
            if real_vs_syn.exists()
            else ""
        )

        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(class_name.upper())}</h2>

              <div class="status {report["status"].lower()}">
                {report["status"]}
              </div>

              <p>
                Overall score:
                <strong>
                  {report["overall_score"]:.3f}
                </strong>
              </p>

              <p>
                Reason:
                {html.escape(
                    "; ".join(
                        report["reasons"]
                    )
                    if report["reasons"]
                    else "All automated engineering checks passed."
                )}
              </p>

              <div class="metrics">
                <div>
                  Histogram<br>
                  <strong>
                    {report["histogram_similarity"]:.3f}
                  </strong>
                </div>

                <div>
                  Structure<br>
                  <strong>
                    {report["structural_similarity"]:.3f}
                  </strong>
                </div>

                <div>
                  Sharpness ratio<br>
                  <strong>
                    {report["sharpness_ratio"]:.3f}
                  </strong>
                </div>

                <div>
                  Duplicate rate<br>
                  <strong>
                    {report["duplicate_rate"]:.2%}
                  </strong>
                </div>
              </div>

              <h3>Selected synthetic preview</h3>
              {
                  (
                      f'<img src="{selected_uri}" />'
                      if selected_uri
                      else "<p>Preview unavailable.</p>"
                  )
              }

              <h3>Real vs synthetic</h3>
              {
                  (
                      f'<img src="{real_syn_uri}" />'
                      if real_syn_uri
                      else "<p>Comparison preview unavailable.</p>"
                  )
              }
            </section>
            """
        )

    page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MRI V5.5 Quality Validation</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 32px;
    background: #f5f7fa;
    color: #1f2a44;
}}
h1 {{
    margin-bottom: 6px;
}}
.subtitle {{
    color: #5d6878;
    margin-bottom: 24px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    margin-bottom: 32px;
}}
th, td {{
    border: 1px solid #dfe4ea;
    padding: 9px;
    text-align: center;
}}
th {{
    background: #eef2f7;
}}
.pass {{
    color: #176b3a;
    font-weight: bold;
}}
.review {{
    color: #9a6500;
    font-weight: bold;
}}
.fail {{
    color: #a52828;
    font-weight: bold;
}}
.card {{
    background: white;
    padding: 22px;
    margin-bottom: 28px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}
.status {{
    display: inline-block;
    padding: 7px 14px;
    border-radius: 16px;
    font-weight: bold;
    margin-bottom: 10px;
}}
.status.pass {{
    background: #e7f6ed;
    color: #176b3a;
}}
.status.review {{
    background: #fff4d8;
    color: #8a5a00;
}}
.status.fail {{
    background: #fdeaea;
    color: #a52828;
}}
.metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 18px 0;
}}
.metrics div {{
    background: #f3f6fa;
    padding: 12px;
    text-align: center;
    border-radius: 8px;
}}
.card img {{
    width: 100%;
    max-width: 1400px;
    border: 1px solid #dfe4ea;
    margin-bottom: 16px;
}}
.note {{
    background: #fff8e6;
    padding: 15px;
    border-left: 4px solid #c28b00;
}}
</style>
</head>

<body>

<h1>MRI V5.5 Automatic Quality Validation</h1>

<div class="subtitle">
Existing V5.3 output • No training • No new generation •
Automatic local validation
</div>

<div class="note">
<strong>Important:</strong>
These are engineering/data-quality metrics.
They do not establish clinical or medical validity.
</div>

<h2>Class Summary</h2>

<table>
<thead>
<tr>
<th>Class</th>
<th>Status</th>
<th>Synthetic</th>
<th>Overall</th>
<th>Histogram</th>
<th>Structure</th>
<th>Mean Z</th>
<th>Sharpness ratio</th>
<th>Duplicate rate</th>
<th>Robust novelty z</th><th>Real-q95 exceedance</th>
</tr>
</thead>

<tbody>
{''.join(rows)}
</tbody>
</table>

{''.join(cards)}

</body>
</html>
"""

    path = (
        output_dir
        / "mri_v5_5_quality_report.html"
    )

    path.write_text(
        page,
        encoding="utf-8",
    )

    return path


# ============================================================
# MAIN CLASS VALIDATION
# ============================================================

def validate_class(
    class_name,
    real_paths,
    synthetic_paths,
    real_latents,
    v3,
    output_dir,
):
    print()
    print("-" * 78)
    print(
        f"{class_name.upper()} QUALITY VALIDATION"
    )
    print("-" * 78)

    if not synthetic_paths:
        return {
            "class_name": class_name,
            "status": "FAIL",
            "reasons": [
                "no_synthetic_images"
            ],
            "synthetic_count": 0,
            "overall_score": 0.0,
        }

    real_base = image_baseline(
        real_paths
    )

    # --------------------------------------------------------
    # Synthetic image statistics
    # --------------------------------------------------------

    synthetic_stats = (
        summarize_synthetic_images(
            synthetic_paths,
            real_base,
        )
    )

    syn_means = synthetic_stats[
        "means"
    ]

    syn_stds = synthetic_stats[
        "stds"
    ]

    syn_sharp = synthetic_stats[
        "sharpness"
    ]

    syn_fg = synthetic_stats[
        "foreground"
    ]

    mean_z = abs(
        float(np.mean(syn_means))
        - real_base["mean"]
    ) / max(
        real_base["std"],
        1e-6,
    )

    std_ratio_error = abs(
        (
            float(np.mean(syn_stds))
            / max(
                real_base["std"],
                1e-6,
            )
        )
        - 1.0
    )

    sharpness_ratio = (
        float(np.mean(syn_sharp))
        / max(
            real_base["sharpness"],
            1e-6,
        )
    )

    foreground_error = abs(
        float(np.mean(syn_fg))
        - real_base["foreground"]
    )

    # --------------------------------------------------------
    # Histogram / structure
    # --------------------------------------------------------

    (
        hist_similarity,
        structure_similarity,
    ) = class_histogram_structure_scores(
        real_paths,
        synthetic_paths,
    )

    # --------------------------------------------------------
    # Latent validation
    # --------------------------------------------------------

    synthetic_latents = encode_images(
        v3,
        synthetic_paths,
    )

    real_latent_ref = real_latent_baseline(real_latents)

    latent_metrics = synthetic_latent_metrics(
        synthetic_latents,
        real_latents,
        real_latent_ref,
    )

    novelty_robust_z = latent_metrics["robust_z"]
    novelty_exceedance = latent_metrics["real_q95_exceedance_fraction"]

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicates = duplicate_metrics(
        synthetic_paths
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    overall = class_quality_score(
        hist_similarity=hist_similarity,
        structure_similarity=structure_similarity,
        mean_z=mean_z,
        std_ratio_error=std_ratio_error,
        sharpness_ratio=sharpness_ratio,
        foreground_error=foreground_error,
        novelty_robust_z=novelty_robust_z,
        novelty_exceedance=novelty_exceedance,
        duplicate_rate=duplicates[
            "duplicate_rate"
        ],
        diversity=latent_metrics[
            "diversity"
        ],
    )

    status, reasons = determine_status(
        overall_score=overall,
        hist_similarity=hist_similarity,
        structure_similarity=structure_similarity,
        mean_z=mean_z,
        std_ratio_error=std_ratio_error,
        sharpness_ratio=sharpness_ratio,
        foreground_error=foreground_error,
        novelty_robust_z=novelty_robust_z,
        novelty_exceedance=novelty_exceedance,
        duplicate_rate=duplicates[
            "duplicate_rate"
        ],
        diversity=latent_metrics[
            "diversity"
        ],
        synthetic_count=len(
            synthetic_paths
        ),
    )

    # --------------------------------------------------------
    # Copy/use V5.3 previews in V5.5 report
    # --------------------------------------------------------

    selected_preview = (
        output_dir
        / "previews"
        / f"{class_name}_selected_v5_3.png"
    )

    real_vs_synthetic_preview = (
        output_dir
        / "previews"
        / f"{class_name}_real_vs_synthetic.png"
    )

    selected_preview.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # V5.3 preview may be in its own run.
    # The main() function copies it into this V5.5 folder.
    report = {
        "class_name": class_name,
        "status": status,
        "reasons": reasons,
        "synthetic_count": len(
            synthetic_paths
        ),
        "real_count": len(
            real_paths
        ),
        "overall_score": overall,

        "real_image_baseline": real_base,

        "histogram_similarity": (
            hist_similarity
        ),

        "structural_similarity": (
            structure_similarity
        ),

        "mean_z": mean_z,

        "std_ratio_error": (
            std_ratio_error
        ),

        "synthetic_mean": float(
            np.mean(syn_means)
        ),

        "synthetic_std": float(
            np.mean(syn_stds)
        ),

        "real_mean": real_base["mean"],

        "real_std": real_base["std"],

        "synthetic_sharpness": float(
            np.mean(syn_sharp)
        ),

        "real_sharpness": real_base[
            "sharpness"
        ],

        "sharpness_ratio": (
            sharpness_ratio
        ),

        "synthetic_foreground": float(
            np.mean(syn_fg)
        ),

        "real_foreground": (
            real_base["foreground"]
        ),

        "foreground_error": (
            foreground_error
        ),

        "real_latent_nn_median": (
            real_latent_ref["median"]
        ),

        "synthetic_nearest_real_latent_mean": (
            latent_metrics[
                "mean_nearest_distance"
            ]
        ),

        "synthetic_nearest_real_latent_median": (
            latent_metrics[
                "median_nearest_distance"
            ]
        ),

        "novelty_robust_z": novelty_robust_z,

        "novelty_q95_exceedance_fraction": novelty_exceedance,

        "real_latent_q95": real_latent_ref["q95"],
        "real_latent_q99": real_latent_ref["q99"],
        "real_latent_iqr": real_latent_ref["iqr"],
        "real_latent_mad": real_latent_ref["mad"],
        "real_latent_robust_scale": real_latent_ref["robust_scale"],

        "synthetic_diversity": (
            latent_metrics[
                "diversity"
            ]
        ),

        "duplicate_pairs": duplicates[
            "duplicate_pairs"
        ],

        "duplicate_rate": duplicates[
            "duplicate_rate"
        ],

        "selected_preview": str(
            selected_preview
        ),

        "real_vs_synthetic_preview": str(
            real_vs_synthetic_preview
        ),
    }

    print(
        f"{class_name:12s} | "
        f"{status:6s} | "
        f"score={overall:.3f} | "
        f"hist={hist_similarity:.3f} | "
        f"struct={structure_similarity:.3f} | "
        f"robust_z={novelty_robust_z:.3f} | q95_exceed={novelty_exceedance:.1%}"
        if novelty_robust_z is not None
        else
        f"{class_name:12s} | "
        f"{status:6s} | "
        f"score={overall:.3f}"
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(SEED)

    started = time.time()

    print("=" * 78)
    print(
        "MRI TESTING V5.5 — ROBUST QUALITY VALIDATION"
    )
    print("=" * 78)

    print("Device :", DEVICE)

    # --------------------------------------------------------
    # Find latest V5.3 run.
    # --------------------------------------------------------

    v5_run = discover_v5_run()

    print()
    print("V5.3 run:")
    print(v5_run)

    synthetic = (
        discover_synthetic_images(
            v5_run
        )
    )

    real = (
        discover_real_images()
    )

    for class_name in CLASSES:
        print(
            f"{class_name:12s} | "
            f"real={len(real[class_name]):4d} | "
            f"synthetic={len(synthetic[class_name]):3d}"
        )

    # --------------------------------------------------------
    # Load V3.
    # --------------------------------------------------------

    print()
    print("Loading V3 encoder/decoder...")

    v3 = load_v3()

    print(
        "V3 loaded successfully."
    )

    # --------------------------------------------------------
    # Load cached real latents.
    # --------------------------------------------------------

    print()
    print("Loading latent cache...")

    latents, labels = load_latents()

    print(
        "Latent shape:",
        tuple(latents.shape),
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        OUTPUT_ROOT
        / f"validation_{timestamp}"
    )

    reports_dir = (
        output_dir
        / "reports"
    )

    previews_dir = (
        output_dir
        / "previews"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    previews_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Copy existing V5.3 previews.
    # --------------------------------------------------------

    v5_preview_root = (
        v5_run
        / "previews"
    )

    if v5_preview_root.exists():
        for class_name in CLASSES:
            for suffix in [
                "_selected_v5_3.png",
                "_real_vs_synthetic.png",
            ]:
                source = (
                    v5_preview_root
                    / f"{class_name}{suffix}"
                )

                destination = (
                    previews_dir
                    / source.name
                )

                if source.exists():
                    destination.write_bytes(
                        source.read_bytes()
                    )

    # --------------------------------------------------------
    # Validate each class.
    # --------------------------------------------------------

    class_reports = []

    for class_name in CLASSES:
        class_id = (
            CLASS_TO_ID[class_name]
        )

        real_latents = latents[
            labels == class_id
        ].float()

        report = validate_class(
            class_name=class_name,
            real_paths=real[class_name],
            synthetic_paths=synthetic[
                class_name
            ],
            real_latents=real_latents,
            v3=v3,
            output_dir=output_dir,
        )

        class_reports.append(
            report
        )

    # --------------------------------------------------------
    # Add V5.3 preview paths.
    # --------------------------------------------------------

    for report in class_reports:
        class_name = (
            report["class_name"]
        )

        report[
            "selected_preview"
        ] = str(
            previews_dir
            / f"{class_name}_selected_v5_3.png"
        )

        report[
            "real_vs_synthetic_preview"
        ] = str(
            previews_dir
            / f"{class_name}_real_vs_synthetic.png"
        )

    # --------------------------------------------------------
    # Overall decision.
    # --------------------------------------------------------

    statuses = [
        r["status"]
        for r in class_reports
    ]

    if all(
        s == "PASS"
        for s in statuses
    ):
        production_recommendation = (
            "PROCEED_TO_CONTROLLED_PRODUCTION_REVIEW"
        )
    elif any(
        s == "FAIL"
        for s in statuses
    ):
        production_recommendation = (
            "DO_NOT_PROCEED"
        )
    else:
        production_recommendation = (
            "REVIEW_BEFORE_PRODUCTION"
        )

    report = {
        "version": (
            "V5.5 robust automatic quality validation"
        ),
        "v5_3_run": str(v5_run),
        "device": str(DEVICE),
        "seed": SEED,
        "training_performed": False,
        "new_generation_performed": False,
        "testing_modified": False,
        "v3_modified": False,
        "production_recommendation": (
            production_recommendation
        ),
        "class_results": class_reports,
        "thresholds": {
            "min_overall_score_pass": (
                MIN_OVERALL_SCORE_PASS
            ),
            "min_overall_score_review": (
                MIN_OVERALL_SCORE_REVIEW
            ),
            "min_histogram_similarity": (
                MIN_HISTOGRAM_SIMILARITY
            ),
            "min_structural_similarity": (
                MIN_STRUCTURAL_SIMILARITY
            ),
            "max_mean_z": MAX_MEAN_Z,
            "max_std_ratio_error": (
                MAX_STD_RATIO_ERROR
            ),
            "min_sharpness_ratio": (
                MIN_SHARPNESS_RATIO
            ),
            "max_sharpness_ratio": (
                MAX_SHARPNESS_RATIO
            ),
            "max_foreground_error": (
                MAX_FOREGROUND_ERROR
            ),
            "max_duplicate_rate": (
                MAX_DUPLICATE_RATE
            ),
            "max_robust_novelty_z_review": MAX_ROBUST_NOVELTY_Z_REVIEW,
            "max_novelty_q95_exceedance_review": MAX_NOVELTY_EXCEEDANCE_REVIEW,
            "min_synthetic_diversity": (
                MIN_SYNTHETIC_DIVERSITY
            ),
            "duplicate_mse_threshold": (
                DUPLICATE_MSE_THRESHOLD
            ),
        },
        "elapsed_seconds": (
            time.time() - started
        ),
    }

    # --------------------------------------------------------
    # Save JSON.
    # --------------------------------------------------------

    json_path = (
        reports_dir
        / "v5_5_quality_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    csv_rows = []

    for item in class_reports:
        row = {
            k: v
            for k, v in item.items()
            if not isinstance(
                v,
                (dict, list),
            )
        }

        csv_rows.append(row)

    csv_path = (
        reports_dir
        / "v5_5_class_summary.csv"
    )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # HTML.
    # --------------------------------------------------------

    html_path = make_html_report(
        output_dir=output_dir,
        v5_run=v5_run,
        class_reports=class_reports,
    )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("V5.5 FINAL QUALITY VALIDATION")
    print("=" * 78)

    for item in class_reports:
        print(
            f"{item['class_name']:12s} | "
            f"{item['status']:6s} | "
            f"score={item['overall_score']:.3f} | "
            f"synthetic={item['synthetic_count']}"
        )

    print()
    print(
        "Recommendation:",
        production_recommendation,
    )

    print()
    print("HTML report:")
    print(html_path)

    print()
    print("JSON report:")
    print(json_path)

    print()
    print("CSV report:")
    print(csv_path)

    print()
    print(
        "No training performed."
    )

    print(
        "No new synthetic images generated."
    )

    print(
        "No Testing data modified."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()

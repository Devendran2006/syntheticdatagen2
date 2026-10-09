
"""
MRI TESTING V5.6 — CONTROLLED PRODUCTION LATENT-MANIFOLD SYNTHETIC GENERATION

Purpose
-------
Generate synthetic MRI candidates from the already-trained V3 latent
manifold instead of learning another GAN/diffusion model.

Pipeline
--------
Real Testing MRI
    -> existing V3 encoder latents (cached)
    -> same-class local-neighbour selection
    -> convex latent interpolation
    -> very small local latent perturbation
    -> existing V3 decoder
    -> image quality gate
    -> accepted synthetic MRI

IMPORTANT
---------
- Reads ONLY data/imaging/MRI/Testing and the existing V3/V4 cache.
- Does NOT retrain V3.
- Does NOT train GAN/diffusion.
- Does NOT modify Training, Testing, or previous output folders.
- Creates a controlled 100/class production-review batch (400 images total).
- DO NOT increase beyond 100/class until V5.5 validation of this batch is reviewed.

This method is not a guarantee of medical validity. It is a quality-first
synthetic augmentation experiment and needs visual/domain validation.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ROOT = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Testing"

V3_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v3_reconstruction"
    / "run_20260922_044321"
    / "models"
    / "best_autoencoder.pt"
)

CACHE_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v4_latents"
    / "run_20261001_044203"
)

LATENT_FILE = CACHE_RUN / "latents.pt"
STATS_FILE = CACHE_RUN / "latent_statistics.pt"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mri_testing_v5_6_controlled_production"
RUN = OUTPUT_ROOT / f"controlled_100_{time.strftime('%Y%m%d_%H%M%S')}"

SYNTH_ROOT = RUN / "synthetic"
PREVIEW_ROOT = RUN / "previews"
REPORT_ROOT = RUN / "reports"

for d in [SYNTH_ROOT, PREVIEW_ROOT, REPORT_ROOT]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 128

OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "mri_testing_v5_6_controlled_production")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

LATENT_CHANNELS = 16
LATENT_SIZE = 32
LATENT_DIM = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE

# ------------------------------------------------------------
# PILOT — intentionally small
# ------------------------------------------------------------
PILOT_PER_CLASS = 100  # compatibility name; V5.6 always uses 100/class

# ------------------------------------------------------------
# Production target — NOT used unless explicitly enabled
# ------------------------------------------------------------
PRODUCTION_MODE = True
PRODUCTION_PER_CLASS = 100

# ------------------------------------------------------------
# V5.6 HARD SAFETY LIMIT
# ------------------------------------------------------------
CONTROLLED_TARGET_PER_CLASS = 100

if PRODUCTION_PER_CLASS != CONTROLLED_TARGET_PER_CLASS:
    raise RuntimeError(
        "V5.6 safety check failed: PRODUCTION_PER_CLASS must be exactly 100."
    )

if not PRODUCTION_MODE:
    raise RuntimeError(
        "V5.6 must run with PRODUCTION_MODE=True."
    )

# ------------------------------------------------------------
# Manifold generation
# ------------------------------------------------------------
# Pick one seed and one of its local same-class neighbours.
NEIGHBOURS = 12

# Interpolation stays away from endpoints so output is not a copy.
ALPHA_MIN = 0.20
ALPHA_MAX = 0.80

# Small perturbation relative to each class latent standard deviation.
PERTURBATION_STD = 0.015

# Number of candidates attempted for each accepted image.
MAX_ATTEMPTS_PER_IMAGE = 12

# ------------------------------------------------------------
# Quality gates
# These are engineering filters, NOT medical-validity claims.
# ------------------------------------------------------------

# Normalized latent distance to nearest real latent.
MIN_NEAREST_LATENT_DISTANCE = 0.003
MAX_NEAREST_LATENT_DISTANCE = 1.20

# Image foreground fraction.
MIN_FOREGROUND_RATIO = 0.03
MAX_FOREGROUND_RATIO = 0.90

# Candidate-vs-real class mean/std tolerance.
# These are now used as soft scoring signals rather than hard rejection.
MAX_MEAN_Z = 4.0
MAX_STD_RATIO_ERROR = 0.80

# Candidate sharpness relative to real class distribution.
# Also used mainly for ranking so the pilot does not collapse to 0 accepted.
MIN_SHARPNESS_RATIO = 0.20
MAX_SHARPNESS_RATIO = 3.50

# Image duplicate threshold.
DUPLICATE_MSE_THRESHOLD = 0.0008

# Number of candidates to generate before selecting the best pilot images.
# This is deliberately larger than the final accepted count.
CANDIDATES_PER_CLASS = 192

# Batch decode size for CPU.
DECODE_BATCH_SIZE = 8

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("MRI TESTING V5.6 — QUALITY-CONTROLLED PREVIEW VALIDATION")
print("=" * 80)
print(f"Device : {DEVICE}")
print(f"Output : {RUN}")
print(f"Controlled target : {PILOT_PER_CLASS}/class")
print(f"Production mode : {PRODUCTION_MODE}")
print()


# ============================================================
# V3 MODEL — EXACT ARCHITECTURE USED BY THE SAVED CHECKPOINT
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch: int):
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

            nn.Conv2d(96, LATENT_CHANNELS, 4, 2, 1),
            nn.SiLU(),
            ResBlock(LATENT_CHANNELS),
            ResBlock(LATENT_CHANNELS),

            nn.Conv2d(
                LATENT_CHANNELS,
                LATENT_CHANNELS,
                3,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 96, 3, padding=1),
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


# ============================================================
# IMAGE UTILITIES
# ============================================================

EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def paths_for_class(class_name: str):
    folder = TEST_ROOT / class_name
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    arr = np.asarray(image, dtype=np.float32) / 255.0

    lo, hi = np.percentile(arr, [1, 99])

    if hi > lo + 1e-6:
        arr = np.clip(
            (arr - lo) / (hi - lo),
            0.0,
            1.0,
        )

    return arr.astype(np.float32)


def tensor_to_array01(x: torch.Tensor) -> np.ndarray:
    """Tensor [-1,1] -> numpy [0,1], shape HxW."""
    a = ((x + 1.0) / 2.0).clamp(0, 1)

    if a.ndim == 4:
        a = a[0, 0]
    elif a.ndim == 3:
        a = a[0]

    return a.detach().cpu().numpy().astype(np.float32)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 1)
    return Image.fromarray(
        np.round(arr * 255).astype(np.uint8),
        "L",
    )


def image_sharpness(arr: np.ndarray) -> float:
    """
    Simple Laplacian-variance sharpness proxy.
    Higher means more high-frequency structure.
    """
    image = array_to_pil(arr)
    lap = np.asarray(
        image.filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    )
    return float(lap.var())


def foreground_ratio(arr: np.ndarray) -> float:
    """
    Approximate non-background fraction.
    MRI backgrounds are normally near black.
    """
    threshold = max(
        0.08,
        float(np.percentile(arr, 25)),
    )
    return float((arr > threshold).mean())


# ============================================================
# LOAD V3 CHECKPOINT
# ============================================================

def load_v3():
    if not V3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"V3 checkpoint not found:\n{V3_CHECKPOINT}"
        )

    model = MRIReconstructionAE().to(DEVICE)

    checkpoint = torch.load(
        V3_CHECKPOINT,
        map_location=DEVICE,
    )

    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("V3 decoder loaded")
    print(f"  Checkpoint : {V3_CHECKPOINT}")
    print(f"  Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(
        f"  Val PSNR   : "
        f"{checkpoint.get('val_psnr', 'unknown')}"
    )
    print()

    return model, checkpoint


# ============================================================
# LOAD CACHED V3 LATENTS
# ============================================================

def load_latent_cache():
    if not LATENT_FILE.exists():
        raise FileNotFoundError(
            f"Latent cache not found:\n{LATENT_FILE}"
        )

    data = torch.load(
        LATENT_FILE,
        map_location="cpu",
    )

    # Support the formats used by the earlier cache script.
    if isinstance(data, dict):
        latents = data.get("latents")
        labels = data.get("labels")
        paths = data.get("paths")
    else:
        raise RuntimeError(
            "Unexpected latents.pt format. "
            "Expected a dictionary containing latents/labels."
        )

    if latents is None or labels is None:
        raise RuntimeError(
            "latents.pt must contain 'latents' and 'labels'."
        )

    latents = torch.as_tensor(
        latents,
        dtype=torch.float32,
    )

    labels = torch.as_tensor(
        labels,
        dtype=torch.long,
    )

    if latents.ndim != 4:
        raise RuntimeError(
            f"Expected latent tensor [N,C,H,W], got {latents.shape}"
        )

    if tuple(latents.shape[1:]) != (
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):
        raise RuntimeError(
            "Latent shape mismatch.\n"
            f"Expected {(LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)}\n"
            f"Found {tuple(latents.shape[1:])}"
        )

    if len(latents) != len(labels):
        raise RuntimeError(
            "Latent/label count mismatch."
        )

    print("Latent cache loaded")
    print(f"  File   : {LATENT_FILE}")
    print(f"  Shape  : {tuple(latents.shape)}")
    print(f"  Labels : {tuple(labels.shape)}")
    print()

    return latents, labels, paths


# ============================================================
# REAL IMAGE STATISTICS FOR QUALITY GATES
# ============================================================

def build_real_stats():
    stats = {}

    for class_name in CLASSES:
        paths = paths_for_class(class_name)

        if not paths:
            raise RuntimeError(
                f"No Testing MRI found for class: {class_name}"
            )

        means = []
        stds = []
        sharpness = []
        foreground = []

        for path in paths:
            arr = preprocess(path)

            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            sharpness.append(image_sharpness(arr))
            foreground.append(foreground_ratio(arr))

        stats[class_name] = {
            "count": len(paths),
            "mean_mean": float(np.mean(means)),
            "mean_std": float(np.std(means) + 1e-6),
            "std_mean": float(np.mean(stds)),
            "std_std": float(np.std(stds) + 1e-6),
            "sharpness_mean": float(np.mean(sharpness)),
            "sharpness_std": float(np.std(sharpness) + 1e-6),
            "foreground_mean": float(np.mean(foreground)),
            "foreground_std": float(np.std(foreground) + 1e-6),
        }

    return stats


# ============================================================
# LOCAL MANIFOLD
# ============================================================

def prepare_class_manifold(
    latents: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Build a normalized flattened latent representation per class.

    Normalization is used only for neighbour search. Original latent
    tensors are retained for interpolation/decoding.
    """
    manifolds = {}

    for class_name in CLASSES:
        cid = CLASS_TO_ID[class_name]

        class_latents = latents[labels == cid].clone()

        if len(class_latents) < NEIGHBOURS + 2:
            raise RuntimeError(
                f"{class_name} has only {len(class_latents)} latents."
            )

        flat = class_latents.reshape(
            len(class_latents),
            -1,
        )

        mean = flat.mean(dim=0, keepdim=True)
        std = flat.std(dim=0, keepdim=True).clamp_min(1e-5)

        normalized = (flat - mean) / std

        # L2 normalize each sample for cosine/local-neighbour search.
        normalized = F.normalize(
            normalized,
            p=2,
            dim=1,
        )

        # Real-vs-real nearest-neighbour baseline (leave-one-out).
        similarity_matrix = normalized @ normalized.T
        similarity_matrix.fill_diagonal_(-1.0)
        real_nn_cosine = 1.0 - similarity_matrix.max(dim=1).values
        real_nn_median = float(real_nn_cosine.median().item())

        manifolds[class_name] = {
            "latents": class_latents,
            "flat": flat,
            "normalized": normalized,
            "mean": mean,
            "std": std,
            "real_nn_cosine_distances": real_nn_cosine,
            "real_nn_cosine_median": real_nn_median,
        }

        print(
            f"{class_name:12s} | "
            f"{len(class_latents):4d} real latent samples"
        )

    print()

    return manifolds


def nearest_neighbours(
    normalized: torch.Tensor,
    index: int,
    k: int,
):
    """
    Cosine-neighbour lookup inside one class.
    """
    query = normalized[index:index + 1]

    similarity = query @ normalized.T
    similarity[0, index] = -1.0

    k = min(k, len(normalized) - 1)

    _, indices = torch.topk(
        similarity[0],
        k=k,
        largest=True,
    )

    return indices


def normalized_distance(
    candidate_flat: torch.Tensor,
    real_flat: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    """
    RMS distance in class-normalized latent space.
    """
    candidate = (candidate_flat - mean) / std
    real = (real_flat - mean) / std

    distances = torch.sqrt(
        torch.mean(
            (real - candidate.unsqueeze(0)) ** 2,
            dim=1,
        ).clamp_min(1e-12)
    )

    return distances


# ============================================================
# CANDIDATE CREATION
# ============================================================

def make_candidate(
    manifold,
    class_name: str,
):
    z_all = manifold["latents"]
    normalized = manifold["normalized"]

    seed_index = random.randrange(len(z_all))

    neighbours = nearest_neighbours(
        normalized,
        seed_index,
        NEIGHBOURS,
    )

    neighbour_index = int(
        neighbours[
            random.randrange(len(neighbours))
        ].item()
    )

    z1 = z_all[seed_index]
    z2 = z_all[neighbour_index]

    alpha = random.uniform(
        ALPHA_MIN,
        ALPHA_MAX,
    )

    # Convex interpolation between two real same-class latent points.
    z = (
        (1.0 - alpha) * z1
        + alpha * z2
    )

    # Small local perturbation.
    local_std = (
        z_all.std(dim=0).clamp_min(1e-5)
    )

    noise = torch.randn_like(z)

    z = z + (
        PERTURBATION_STD
        * local_std
        * noise
    )

    return z, seed_index, neighbour_index, alpha


# ============================================================
# QUALITY GATE
# ============================================================

def evaluate_candidate(
    image_arr: np.ndarray,
    z: torch.Tensor,
    manifold,
    class_name: str,
    real_stats,
):
    reasons = []

    # Basic finite/range check.
    if not np.isfinite(image_arr).all():
        reasons.append("non_finite")
        return False, reasons, {}

    if image_arr.min() < -1e-5 or image_arr.max() > 1.00001:
        reasons.append("out_of_range")

    # Image statistics.
    mean = float(image_arr.mean())
    std = float(image_arr.std())
    sharp = image_sharpness(image_arr)
    fg = foreground_ratio(image_arr)

    rs = real_stats[class_name]

    mean_z = abs(
        mean - rs["mean_mean"]
    ) / rs["mean_std"]

    std_ratio_error = abs(
        std / max(rs["std_mean"], 1e-6) - 1.0
    )

    sharp_ratio = (
        sharp / max(rs["sharpness_mean"], 1e-6)
    )

    # Latent distance to the nearest real sample.
    flat = z.reshape(-1)

    distances = normalized_distance(
        flat,
        manifold["flat"],
        manifold["mean"],
        manifold["std"],
    )

    nearest = float(distances.min().item())

    if nearest < MIN_NEAREST_LATENT_DISTANCE:
        reasons.append("too_close_to_real")

    if nearest > MAX_NEAREST_LATENT_DISTANCE:
        reasons.append("too_far_from_real")

    if fg < MIN_FOREGROUND_RATIO:
        reasons.append("foreground_too_small")

    if fg > MAX_FOREGROUND_RATIO:
        reasons.append("foreground_too_large")

    if mean_z > MAX_MEAN_Z:
        reasons.append("mean_outlier")

    if std_ratio_error > MAX_STD_RATIO_ERROR:
        reasons.append("std_outlier")

    if sharp_ratio < MIN_SHARPNESS_RATIO:
        reasons.append("too_blurry")

    if sharp_ratio > MAX_SHARPNESS_RATIO:
        reasons.append("too_sharp_noisy")

    metrics = {
        "mean": mean,
        "std": std,
        "sharpness": sharp,
        "foreground_ratio": fg,
        "mean_z": mean_z,
        "std_ratio_error": std_ratio_error,
        "sharpness_ratio": sharp_ratio,
        "nearest_latent_distance": nearest,
    }

    return len(reasons) == 0, reasons, metrics


# ============================================================
# CONTACT SHEET
# ============================================================

def save_contact_sheet(
    images,
    labels,
    class_name,
    filename=None,
    columns=4,
):
    """
    Robust preview writer.

    Important V5.6 change:
    - never creates an empty/zero-row PIL image
    - writes a readable "NO IMAGES" card when selection is empty
    - supports custom filenames for multiple preview types
    """
    output = (
        PREVIEW_ROOT
        / (
            filename
            if filename
            else f"{class_name}_v5_6_contact_sheet.png"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if not images:
        sheet = Image.new("RGB", (900, 220), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (30, 35),
            f"{class_name.upper()} — NO SELECTED IMAGES",
            fill="black",
        )
        draw.text(
            (30, 100),
            "Selection was empty. Inspect v5_generation_report.json.",
            fill="black",
        )
        sheet.save(output, format="PNG")
        print(f"Contact sheet : {output}")
        return output

    cols = max(1, min(columns, len(images)))
    cell_w = IMAGE_SIZE
    cell_h = IMAGE_SIZE + 22
    rows = max(1, math.ceil(len(images) / cols))

    sheet = Image.new(
        "L",
        (cols * cell_w, rows * cell_h),
        0,
    )
    draw = ImageDraw.Draw(sheet)

    for i, image in enumerate(images):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h

        sheet.paste(image, (x, y))

        label = (
            labels[i]
            if i < len(labels)
            else f"{class_name} {i+1:02d}"
        )

        draw.text(
            (x + 4, y + IMAGE_SIZE + 3),
            str(label),
            fill=255,
        )

    sheet.save(output, format="PNG")
    print(f"Contact sheet : {output}")
    return output


def save_real_vs_synthetic_preview(
    class_name,
    synthetic_images,
    real_paths,
):
    """
    V5.6 preview:
        row 1/alternating tiles = REAL
        alternating tiles       = SYNTHETIC

    The comparison uses the same class only and is intended for visual
    inspection before any 800/class production run.
    """
    if not synthetic_images:
        return save_contact_sheet(
            [],
            [],
            class_name,
            filename=f"{class_name}_real_vs_synthetic.png",
        )

    # Use at most 16 pairs so the preview remains readable.
    pair_count = min(
        16,
        len(synthetic_images),
        len(real_paths),
    )

    tiles = []
    labels = []

    for i in range(pair_count):
        real_arr = preprocess(real_paths[i])
        real_img = array_to_pil(real_arr)

        tiles.append(real_img)
        labels.append(f"REAL {i+1:02d}")

        tiles.append(synthetic_images[i])
        labels.append(f"SYN {i+1:02d}")

    return save_contact_sheet(
        tiles,
        labels,
        class_name,
        filename=f"{class_name}_real_vs_synthetic.png",
        columns=8,
    )


# ============================================================
# MAIN GENERATION
# ============================================================

@torch.no_grad()
def generate_class(
    model,
    manifold,
    class_name,
    real_stats,
    target_count,
):
    """
    V5.2:
    - Generate same-class interpolated latent candidates.
    - Decode each candidate to an image.
    - Re-encode the generated image with the V3 encoder.
    - Compare the re-encoded image latent against the same-class real manifold.
    - Rank candidates and remove synthetic near-duplicates.

    Novelty is measured as cosine distance in class-standardized latent space.
    It is compared with the real-vs-real nearest-neighbour baseline for context.
    """
    class_dir = SYNTH_ROOT / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    target_pool = max(CANDIDATES_PER_CLASS, target_count * 4)
    candidates = []
    rejected = {}
    start = time.time()

    real_norm = manifold["normalized"].to(DEVICE)
    latent_mean = manifold["mean"].to(DEVICE)
    latent_std = manifold["std"].to(DEVICE)
    real_nn_median = manifold["real_nn_cosine_median"]

    for _ in range(target_pool):
        z, seed_idx, neighbour_idx, alpha = make_candidate(
            manifold, class_name
        )

        generated = model.decode(z.unsqueeze(0).to(DEVICE))
        image_arr = tensor_to_array01(generated)

        if not np.isfinite(image_arr).all():
            rejected["non_finite_image"] = rejected.get("non_finite_image", 0) + 1
            continue
        if image_arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
            rejected["wrong_image_shape"] = rejected.get("wrong_image_shape", 0) + 1
            continue
        if image_arr.min() < -1e-5 or image_arr.max() > 1.00001:
            rejected["image_out_of_range"] = rejected.get("image_out_of_range", 0) + 1
            continue

        mean = float(image_arr.mean())
        std = float(image_arr.std())
        sharp = image_sharpness(image_arr)
        fg = foreground_ratio(image_arr)

        if not (MIN_FOREGROUND_RATIO <= fg <= MAX_FOREGROUND_RATIO):
            rejected["foreground_range"] = rejected.get("foreground_range", 0) + 1
            continue
        if std < 0.015 or std > 0.45:
            rejected["std_sanity"] = rejected.get("std_sanity", 0) + 1
            continue

        # Critical V5.2 fix: re-encode the ACTUAL decoded image.
        reencoded = model.encode(generated).detach().float().reshape(1, -1)
        if not torch.isfinite(reencoded).all():
            rejected["non_finite_reencoded_latent"] = (
                rejected.get("non_finite_reencoded_latent", 0) + 1
            )
            continue

        reencoded_norm = F.normalize(
            (reencoded - latent_mean) / latent_std,
            p=2,
            dim=1,
        )
        sims = reencoded_norm @ real_norm.T
        nearest_real_cosine = float(
            (1.0 - sims.max(dim=1).values).clamp(0, 2).item()
        )

        # Diagnostic only: distance before decoding, not used as the
        # principal novelty metric.
        predecode = z.detach().cpu().reshape(1, -1)
        predecode_norm = F.normalize(
            (predecode - manifold["mean"]) / manifold["std"],
            p=2,
            dim=1,
        )
        predecode_distance = float(
            (1.0 - (predecode_norm @ manifold["normalized"].T).max(dim=1).values)
            .clamp(0, 2).item()
        )

        rs = real_stats[class_name]
        mean_z = abs(mean - rs["mean_mean"]) / max(rs["mean_std"], 1e-6)
        std_ratio_error = abs(std / max(rs["std_mean"], 1e-6) - 1.0)
        sharp_ratio = sharp / max(rs["sharpness_mean"], 1e-6)
        relative_novelty = nearest_real_cosine / max(real_nn_median, 1e-8)

        # Score is a ranking heuristic, not a clinical validity score.
        mean_score = min(mean_z / MAX_MEAN_Z, 3.0)
        std_score = min(std_ratio_error / MAX_STD_RATIO_ERROR, 3.0)
        sharp_score = abs(math.log(max(sharp_ratio, 1e-6)))

        if relative_novelty < 0.25:
            novelty_score = 2.0 + (0.25 - relative_novelty)
        elif relative_novelty > 3.0:
            novelty_score = 2.0 + (relative_novelty - 3.0)
        else:
            novelty_score = abs(relative_novelty - 1.0) * 0.35

        score = (
            1.20 * mean_score
            + 1.20 * std_score
            + 0.65 * sharp_score
            + novelty_score
        )

        candidates.append({
            "image": image_arr,
            "seed_idx": int(seed_idx),
            "neighbour_idx": int(neighbour_idx),
            "alpha": float(alpha),
            "score": float(score),
            "mean": mean,
            "std": std,
            "sharpness": sharp,
            "foreground_ratio": fg,
            "mean_z": float(mean_z),
            "std_ratio_error": float(std_ratio_error),
            "sharpness_ratio": float(sharp_ratio),
            "predecode_nearest_cosine_distance": predecode_distance,
            "reencoded_nearest_real_cosine_distance": nearest_real_cosine,
            "real_nn_cosine_median": float(real_nn_median),
            "relative_novelty_vs_real_nn": float(relative_novelty),
        })

    candidates.sort(key=lambda item: item["score"])
    accepted_images = []
    accepted_metrics = []

    for candidate in candidates:
        if len(accepted_images) >= target_count:
            break

        arr = candidate["image"]
        if any(
            float(np.mean((arr - previous) ** 2)) < DUPLICATE_MSE_THRESHOLD
            for previous in accepted_images
        ):
            rejected["synthetic_duplicate"] = (
                rejected.get("synthetic_duplicate", 0) + 1
            )
            continue

        number = len(accepted_images) + 1
        array_to_pil(arr).save(
            class_dir / f"{class_name}_synthetic_{number:04d}.png"
        )
        accepted_images.append(arr)

        metric = {
            "index": number,
            "source_seed_index": candidate["seed_idx"],
            "source_neighbour_index": candidate["neighbour_idx"],
            "alpha": candidate["alpha"],
            "selection_score": candidate["score"],
            "mean": candidate["mean"],
            "std": candidate["std"],
            "sharpness": candidate["sharpness"],
            "foreground_ratio": candidate["foreground_ratio"],
            "mean_z": candidate["mean_z"],
            "std_ratio_error": candidate["std_ratio_error"],
            "sharpness_ratio": candidate["sharpness_ratio"],
            "predecode_nearest_cosine_distance": candidate[
                "predecode_nearest_cosine_distance"
            ],
            "reencoded_nearest_real_cosine_distance": candidate[
                "reencoded_nearest_real_cosine_distance"
            ],
            "real_nn_cosine_median": candidate["real_nn_cosine_median"],
            "relative_novelty_vs_real_nn": candidate[
                "relative_novelty_vs_real_nn"
            ],
        }
        accepted_metrics.append(metric)

    distances = [
        m["reencoded_nearest_real_cosine_distance"]
        for m in accepted_metrics
    ]
    distance_std = float(np.std(distances)) if len(distances) > 1 else 0.0

    return {
        "class": class_name,
        "requested": target_count,
        "candidate_pool": target_pool,
        "candidate_valid": len(candidates),
        "accepted": len(accepted_images),
        "attempts": target_pool,
        "selection_fraction": len(accepted_images) / max(target_pool, 1),
        "seconds": time.time() - start,
        "rejected": rejected,
        "metrics": accepted_metrics,
        "images": accepted_images,
        "real_nn_cosine_median": float(real_nn_median),
        "reencoded_distance_std": distance_std,
        "latent_metric_variation_warning": (
            len(distances) > 1 and distance_std < 1e-7
        ),
    }


# REPORT
# ============================================================

def summarize_class(result):
    metrics = result["metrics"]

    if not metrics:
        return {
            "accepted": 0,
            "mean_image_mean": None,
            "mean_image_std": None,
            "mean_nearest_latent_distance": None,
        }

    return {
        "accepted": result["accepted"],
        "attempts": result["attempts"],
        "selection_fraction": result["selection_fraction"],
        "mean_image_mean": float(
            np.mean([
                m["mean"]
                for m in metrics
            ])
        ),
        "mean_image_std": float(
            np.mean([
                m["std"]
                for m in metrics
            ])
        ),
        "mean_sharpness": float(
            np.mean([
                m["sharpness"]
                for m in metrics
            ])
        ),
        "mean_foreground_ratio": float(
            np.mean([
                m["foreground_ratio"]
                for m in metrics
            ])
        ),
        "mean_reencoded_nearest_real_cosine_distance": float(np.mean([
            m["reencoded_nearest_real_cosine_distance"] for m in metrics
        ])),
        "min_reencoded_nearest_real_cosine_distance": float(np.min([
            m["reencoded_nearest_real_cosine_distance"] for m in metrics
        ])),
        "max_reencoded_nearest_real_cosine_distance": float(np.max([
            m["reencoded_nearest_real_cosine_distance"] for m in metrics
        ])),
        "mean_relative_novelty_vs_real_nn": float(np.mean([
            m["relative_novelty_vs_real_nn"] for m in metrics
        ])),
        "real_nn_cosine_median": result["real_nn_cosine_median"],
        "reencoded_distance_std": result["reencoded_distance_std"],
        "latent_metric_variation_warning": result["latent_metric_variation_warning"],
    }


# ============================================================
# MAIN
# ============================================================


# ============================================================
# V5.6 CONTROLLED-PRODUCTION SAFETY LIMIT
# ============================================================

CONTROLLED_TARGET_PER_CLASS = 100

if PRODUCTION_PER_CLASS != CONTROLLED_TARGET_PER_CLASS:
    raise RuntimeError(
        "V5.6 safety check failed: PRODUCTION_PER_CLASS must remain exactly 100."
    )

if not PRODUCTION_MODE:
    raise RuntimeError(
        "V5.6 must run in PRODUCTION_MODE=True for the controlled 100/class test."
    )

def main():
    total_start = time.time()

    # --------------------------------------------------------
    # Verify inputs
    # --------------------------------------------------------
    required = [
        ("V3 checkpoint", V3_CHECKPOINT),
        ("latent cache", LATENT_FILE),
    ]

    for name, path in required:
        print(
            f"{'OK' if path.exists() else 'MISSING':8s}"
            f" {name}: {path}"
        )

    missing = [
        str(path)
        for _, path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input file(s) missing:\n"
            + "\n".join(missing)
        )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    model, checkpoint = load_v3()

    latents, labels, cached_paths = (
        load_latent_cache()
    )

    real_stats = build_real_stats()

    manifolds = prepare_class_manifold(
        latents,
        labels,
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    target = (
        PRODUCTION_PER_CLASS
        if PRODUCTION_MODE
        else PILOT_PER_CLASS
    )

    print("=" * 80)
    print("GENERATION")
    print("=" * 80)
    print(
        f"Target : {target} accepted images/class"
    )
    print(
        "Method : same-class interpolation + V3 decode + V3 re-encode validation"
    )
    print()

    results = {}

    for class_name in CLASSES:
        result = generate_class(
            model=model,
            manifold=manifolds[class_name],
            class_name=class_name,
            real_stats=real_stats,
            target_count=target,
        )

        results[class_name] = result

        selected_images = [
            array_to_pil(a)
            for a in result["images"]
        ]

        save_contact_sheet(
            selected_images,
            [
                f"SYN {i+1:02d}"
                for i in range(len(selected_images))
            ],
            class_name,
            filename=f"{class_name}_selected_v5_6.png",
            columns=4,
        )

        real_paths = paths_for_class(class_name)

        save_real_vs_synthetic_preview(
            class_name,
            selected_images,
            real_paths,
        )

        print(
            f"{class_name:12s} | "
            f"selected {result['accepted']}/{target} | "
            f"valid candidates {result.get('candidate_valid', 0)} | "
            f"time {result['seconds']:.1f}s"
        )

        if result["accepted"] == 0:
            print(
                f"WARNING    {class_name}: 0 images selected. "
                "Check the rejection counts in the report."
            )
        print()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    report = {
        "version": "V5.6 quality-controlled latent-manifold generation with robust previews",
        "device": str(DEVICE),
        "testing_root": str(TEST_ROOT),
        "v3_checkpoint": str(V3_CHECKPOINT),
        "latent_cache": str(LATENT_FILE),
        "v3_epoch": checkpoint.get("epoch"),
        "v3_val_psnr": checkpoint.get("val_psnr"),
        "latent_shape": list(latents.shape[1:]),
        "real_testing_count": int(len(latents)),
        "classes": CLASSES,
        "production_mode": PRODUCTION_MODE,
        "target_per_class": target,
        "method": {
            "neighbours": NEIGHBOURS,
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "perturbation_std": PERTURBATION_STD,
            "candidates_per_class": CANDIDATES_PER_CLASS,
        },
        "quality_gate": {
            "min_nearest_latent_distance": MIN_NEAREST_LATENT_DISTANCE,
            "max_nearest_latent_distance": MAX_NEAREST_LATENT_DISTANCE,
            "min_foreground_ratio": MIN_FOREGROUND_RATIO,
            "max_foreground_ratio": MAX_FOREGROUND_RATIO,
            "max_mean_z": MAX_MEAN_Z,
            "max_std_ratio_error": MAX_STD_RATIO_ERROR,
            "min_sharpness_ratio": MIN_SHARPNESS_RATIO,
            "max_sharpness_ratio": MAX_SHARPNESS_RATIO,
            "duplicate_mse_threshold": DUPLICATE_MSE_THRESHOLD,
            "novelty_metric": "cosine distance between V3 re-encoded generated image latent and same-class real latent manifold",
            "novelty_reference": "median leave-one-out real-to-real nearest-neighbour cosine distance per class",
        },
        "class_results": {
            c: {
                **summarize_class(results[c]),
                "rejected": results[c]["rejected"],
            }
            for c in CLASSES
        },
        "preview_validation": {
            "selected_preview_per_class": True,
            "real_vs_synthetic_preview_per_class": True,
            "empty_preview_guard": True,
            "maximum_visual_pairs_per_class": 16,
            "production_requires_manual_visual_review": True,
        },
        "class_status": {
            c: (
                "PASS"
                if results[c]["accepted"] >= target
                and not results[c]["latent_metric_variation_warning"]
                else "REVIEW"
            )
            for c in CLASSES
        },
        "training_performed": False,
        "v3_modified": False,
        "testing_modified": False,
        "seconds": time.time() - total_start,
    }

    report_path = (
        REPORT_ROOT
        / "v5_generation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("V5 COMPLETE")
    print("=" * 80)
    print(f"Run folder : {RUN}")
    print(f"Synthetic  : {SYNTH_ROOT}")
    print(f"Previews   : {PREVIEW_ROOT}")
    print(f"Report     : {report_path}")
    print()
    print(
        "PREVIEW FILES:"
    )
    for class_name in CLASSES:
        print(
            f"  {class_name:12s} -> "
            f"{PREVIEW_ROOT / (class_name + '_selected_v5_6.png')}"
        )
        print(
            f"  {class_name:12s} -> "
            f"{PREVIEW_ROOT / (class_name + '_real_vs_synthetic.png')}"
        )

    print()
    print(
        "IMPORTANT: V5.6 remains PILOT mode. "
        "Review the four selected previews AND the four "
        "real-vs-synthetic previews before enabling production."
    )
    print(
        "Production mode is currently:",
        PRODUCTION_MODE,
    )


if __name__ == "__main__":
    main()

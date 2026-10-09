"""
MRI TESTING — V4.1 QUALITY-FIRST CONDITIONAL LATENT DIFFUSION

This version is designed to improve the V4 pilot rather than simply
generate more images.

IMPORTANT:
- Reuses the existing V4 latent cache:
    outputs/mri_testing_v4_latents/run_20261001_044203
- Reuses the validated V3 decoder:
    outputs/mri_testing_v3_reconstruction/run_20260922_044321/models/best_autoencoder.pt
- Reads MRI Testing indirectly through the cached latents.
- Does NOT modify Testing data or the V3 checkpoint.
- Generates ONLY a 16/class quality-validation pilot.
- It WILL NOT generate 800/class automatically.

Quality-first changes:
1. Cosine noise schedule instead of the simple linear schedule.
2. v-prediction objective, which is generally more stable for diffusion.
3. Stronger conditional U-Net.
4. Classifier-free guidance (CFG).
5. Classifier-free conditioning dropout during training.
6. EMA weights.
7. Gradient clipping.
8. Validation split in latent space.
9. Generated-latent distribution checks.
10. Nearest-neighbour checks against real cached latents.
11. A quality gate that refuses to call the pilot "ready" if basic
    latent-distribution checks fail.

IMPORTANT LIMITATION:
No software-only training script can guarantee clinically valid MRI
images. Final acceptance still requires visual inspection and, for any
medical/research use, appropriate domain-expert validation.
"""

from pathlib import Path
import sys
import json
import math
import time
import random

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v4_latents"
    / "run_20261001_044203"
)

LATENT_FILE = CACHE_RUN / "latents.pt"
STATS_FILE = CACHE_RUN / "latent_statistics.pt"

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
    / "mri_testing_v4_1_quality_diffusion"
)

RUN = OUTPUT_ROOT / f"pilot_{time.strftime('%Y%m%d_%H%M%S')}"

MODEL_DIR = RUN / "models"
SYNTH_DIR = RUN / "synthetic"
PREVIEW_DIR = RUN / "previews"
REPORT_DIR = RUN / "reports"

for directory in [
    MODEL_DIR,
    SYNTH_DIR,
    PREVIEW_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 128

LATENT_CHANNELS = 16
LATENT_SIZE = 32

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

NUM_CLASSES = len(CLASSES)

# CPU-friendly, quality-oriented training.
BATCH_SIZE = 16

# More training than V4 pilot.
TRAIN_EPOCHS = 40

# Small pilot only.
PILOT_PER_CLASS = 16

# Diffusion.
DIFFUSION_STEPS = 1000
DDIM_STEPS = 75

# Stronger U-Net.
BASE_CHANNELS = 80

# Embeddings.
TIME_EMB = 192
CLASS_EMB = 192

# Classifier-free guidance.
CONDITION_DROP_PROB = 0.10
CFG_SCALE = 2.5

# EMA.
EMA_DECAY = 0.9995

# Optimizer.
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 1e-4

# Gradient clipping.
GRAD_CLIP = 1.0

# Validation.
VAL_FRACTION = 0.10

# Quality thresholds.
# These are safety/diagnostic gates, not claims of medical validity.
MAX_GLOBAL_MEAN_Z = 1.00
MAX_GLOBAL_STD_RATIO_ERROR = 0.35
MIN_NEAREST_LATENT_DISTANCE = 0.015

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TIME EMBEDDING
# ============================================================

def sinusoidal_embedding(
    timesteps,
    dim,
):

    half = dim // 2

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )

    angles = (
        timesteps.float().unsqueeze(1)
        * frequencies.unsqueeze(0)
    )

    embedding = torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
        ],
        dim=1,
    )

    if dim % 2:
        embedding = F.pad(
            embedding,
            (0, 1),
        )

    return embedding


def group_count(channels):

    for groups in [
        32,
        16,
        8,
        4,
        2,
        1,
    ]:
        if channels % groups == 0:
            return groups

    return 1


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            group_count(in_channels),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.emb = nn.Linear(
            emb_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            group_count(out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.skip = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x,
        emb,
    ):

        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.emb(
            F.silu(emb)
        ).unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


# ============================================================
# QUALITY-FIRST CONDITIONAL U-NET
# ============================================================

class QualityConditionalUNet(nn.Module):

    def __init__(
        self,
        latent_channels=16,
        num_classes=4,
        base=80,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(
                TIME_EMB,
                TIME_EMB,
            ),
            nn.SiLU(),
            nn.Linear(
                TIME_EMB,
                TIME_EMB,
            ),
        )

        # +1 is the unconditional/null class.
        self.class_embedding = nn.Embedding(
            num_classes + 1,
            CLASS_EMB,
        )

        emb_dim = (
            TIME_EMB
            + CLASS_EMB
        )

        c1 = base
        c2 = base * 2
        c3 = base * 4

        self.in_conv = nn.Conv2d(
            latent_channels,
            c1,
            3,
            padding=1,
        )

        self.block1 = ResBlock(
            c1,
            c1,
            emb_dim,
        )

        self.down1 = nn.Conv2d(
            c1,
            c2,
            4,
            stride=2,
            padding=1,
        )

        self.block2 = ResBlock(
            c2,
            c2,
            emb_dim,
        )

        self.down2 = nn.Conv2d(
            c2,
            c3,
            4,
            stride=2,
            padding=1,
        )

        self.mid1 = ResBlock(
            c3,
            c3,
            emb_dim,
        )

        self.mid2 = ResBlock(
            c3,
            c3,
            emb_dim,
        )

        self.mid3 = ResBlock(
            c3,
            c3,
            emb_dim,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            4,
            stride=2,
            padding=1,
        )

        self.up_block2 = ResBlock(
            c2 + c2,
            c2,
            emb_dim,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            4,
            stride=2,
            padding=1,
        )

        self.up_block1 = ResBlock(
            c1 + c1,
            c1,
            emb_dim,
        )

        self.out_norm = nn.GroupNorm(
            group_count(c1),
            c1,
        )

        self.out_conv = nn.Conv2d(
            c1,
            latent_channels,
            3,
            padding=1,
        )

    def forward(
        self,
        x,
        timestep,
        labels,
    ):

        time_embedding = sinusoidal_embedding(
            timestep,
            TIME_EMB,
        )

        time_embedding = self.time_mlp(
            time_embedding
        )

        class_embedding = self.class_embedding(
            labels
        )

        embedding = torch.cat(
            [
                time_embedding,
                class_embedding,
            ],
            dim=1,
        )

        x0 = self.in_conv(x)

        d1 = self.block1(
            x0,
            embedding,
        )

        d2_in = self.down1(d1)

        d2 = self.block2(
            d2_in,
            embedding,
        )

        mid = self.down2(d2)

        mid = self.mid1(
            mid,
            embedding,
        )

        mid = self.mid2(
            mid,
            embedding,
        )

        mid = self.mid3(
            mid,
            embedding,
        )

        u2 = self.up2(mid)

        u2 = torch.cat(
            [
                u2,
                d2,
            ],
            dim=1,
        )

        u2 = self.up_block2(
            u2,
            embedding,
        )

        u1 = self.up1(u2)

        u1 = torch.cat(
            [
                u1,
                d1,
            ],
            dim=1,
        )

        u1 = self.up_block1(
            u1,
            embedding,
        )

        output = self.out_norm(u1)
        output = F.silu(output)
        output = self.out_conv(output)

        return output


# ============================================================
# EMA
# ============================================================

class EMA:

    def __init__(
        self,
        model,
        decay,
    ):

        self.decay = decay

        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter
            in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(
        self,
        model,
    ):

        for name, parameter in model.named_parameters():

            if not parameter.requires_grad:
                continue

            self.shadow[name].mul_(
                self.decay
            )

            self.shadow[name].add_(
                parameter.detach(),
                alpha=1.0 - self.decay,
            )

    @torch.no_grad()
    def apply(
        self,
        model,
    ):

        for name, parameter in model.named_parameters():

            if name in self.shadow:
                parameter.data.copy_(
                    self.shadow[name]
                )


# ============================================================
# DATASET
# ============================================================

class LatentDataset(Dataset):

    def __init__(
        self,
        latents,
        labels,
    ):

        self.latents = latents.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.latents)

    def __getitem__(
        self,
        index,
    ):

        return (
            self.latents[index],
            self.labels[index],
        )


# ============================================================
# COSINE DIFFUSION SCHEDULE
# ============================================================

def cosine_alpha_bars(
    steps,
    device,
):

    s = 0.008

    x = torch.linspace(
        0,
        steps,
        steps + 1,
        device=device,
        dtype=torch.float32,
    )

    alpha_bars = torch.cos(
        (
            (x / steps + s)
            / (1.0 + s)
        )
        * math.pi
        / 2.0
    ) ** 2

    alpha_bars = (
        alpha_bars
        / alpha_bars[0]
    )

    alpha_bars = alpha_bars.clamp(
        1e-5,
        0.9999,
    )

    betas = 1.0 - (
        alpha_bars[1:]
        / alpha_bars[:-1]
    )

    betas = betas.clamp(
        1e-5,
        0.999,
    )

    alphas = 1.0 - betas

    return (
        betas,
        alphas,
        alpha_bars[1:],
    )


def extract(
    values,
    timesteps,
    shape,
):

    result = values[timesteps]

    return result.reshape(
        timesteps.shape[0],
        *([1] * (len(shape) - 1)),
    )


# ============================================================
# V-PREDICTION
# ============================================================

def make_noisy_latent(
    x0,
    timesteps,
    noise,
    alpha_bars,
):

    alpha_bar = extract(
        alpha_bars,
        timesteps,
        x0.shape,
    )

    sqrt_alpha = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus = torch.sqrt(
        1.0 - alpha_bar
    )

    noisy = (
        sqrt_alpha * x0
        + sqrt_one_minus * noise
    )

    # v = sqrt(alpha) * eps - sqrt(1-alpha) * x0
    velocity = (
        sqrt_alpha * noise
        - sqrt_one_minus * x0
    )

    return noisy, velocity


def velocity_to_x0(
    noisy,
    velocity,
    alpha_bar,
):

    sqrt_alpha = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus = torch.sqrt(
        1.0 - alpha_bar
    )

    return (
        sqrt_alpha * noisy
        - sqrt_one_minus * velocity
    )


def velocity_to_noise(
    noisy,
    velocity,
    alpha_bar,
):

    sqrt_alpha = torch.sqrt(
        alpha_bar
    )

    sqrt_one_minus = torch.sqrt(
        1.0 - alpha_bar
    )

    return (
        sqrt_one_minus * noisy
        + sqrt_alpha * velocity
    )


# ============================================================
# CONDITION DROPOUT
# ============================================================

def apply_condition_dropout(
    labels,
):

    labels = labels.clone()

    drop_mask = (
        torch.rand(
            labels.shape,
            device=labels.device,
        )
        < CONDITION_DROP_PROB
    )

    # Null class = NUM_CLASSES.
    labels[drop_mask] = NUM_CLASSES

    return labels


# ============================================================
# TRAINING / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    alpha_bars,
    optimizer=None,
    ema=None,
    train=True,
):

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    for x0, labels in loader:

        x0 = x0.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        timesteps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (
                x0.shape[0],
            ),
            device=DEVICE,
        )

        noise = torch.randn_like(
            x0
        )

        noisy, target_v = make_noisy_latent(
            x0,
            timesteps,
            noise,
            alpha_bars,
        )

        if train:
            model_labels = apply_condition_dropout(
                labels
            )
        else:
            model_labels = labels

        with torch.set_grad_enabled(train):

            predicted_v = model(
                noisy,
                timesteps,
                model_labels,
            )

            loss = F.mse_loss(
                predicted_v,
                target_v,
            )

            if train:

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

                if ema is not None:
                    ema.update(model)

        total_loss += (
            float(loss.item())
            * x0.shape[0]
        )

        count += x0.shape[0]

    return total_loss / max(count, 1)


# ============================================================
# CLASSWISE LATENT SPLIT
# ============================================================

def make_train_val_split(
    latents,
    labels,
):

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_indices = []
    val_indices = []

    for class_id in range(NUM_CLASSES):

        indices = torch.where(
            labels == class_id
        )[0]

        permutation = indices[
            torch.randperm(
                len(indices),
                generator=generator,
            )
        ]

        val_count = max(
            1,
            int(
                len(indices)
                * VAL_FRACTION
            ),
        )

        val_indices.extend(
            permutation[:val_count].tolist()
        )

        train_indices.extend(
            permutation[val_count:].tolist()
        )

    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long,
    )

    val_indices = torch.tensor(
        val_indices,
        dtype=torch.long,
    )

    return (
        latents[train_indices],
        labels[train_indices],
        latents[val_indices],
        labels[val_indices],
    )


# ============================================================
# CLASSIFIER-FREE GUIDANCE
# ============================================================

@torch.no_grad()
def guided_velocity(
    model,
    x,
    timestep,
    labels,
    guidance_scale,
):

    conditional = model(
        x,
        timestep,
        labels,
    )

    null_labels = torch.full_like(
        labels,
        NUM_CLASSES,
    )

    unconditional = model(
        x,
        timestep,
        null_labels,
    )

    return (
        unconditional
        + guidance_scale
        * (
            conditional
            - unconditional
        )
    )


# ============================================================
# DDIM SAMPLER FOR V-PREDICTION
# ============================================================

@torch.no_grad()
def ddim_sample(
    model,
    labels,
    latent_shape,
    alpha_bars,
    steps,
    guidance_scale,
):

    model.eval()

    batch_size = labels.shape[0]

    x = torch.randn(
        batch_size,
        *latent_shape,
        device=DEVICE,
    )

    timestep_schedule = torch.linspace(
        DIFFUSION_STEPS - 1,
        0,
        steps,
        device=DEVICE,
    ).long()

    for index, timestep in enumerate(
        timestep_schedule
    ):

        timestep_batch = torch.full(
            (
                batch_size,
            ),
            int(
                timestep.item()
            ),
            device=DEVICE,
            dtype=torch.long,
        )

        predicted_v = guided_velocity(
            model,
            x,
            timestep_batch,
            labels,
            guidance_scale,
        )

        alpha_bar = alpha_bars[
            timestep
        ]

        x0 = velocity_to_x0(
            x,
            predicted_v,
            alpha_bar,
        )

        x0 = x0.clamp(
            -5.0,
            5.0,
        )

        predicted_noise = velocity_to_noise(
            x,
            predicted_v,
            alpha_bar,
        )

        if index == len(
            timestep_schedule
        ) - 1:

            x = x0
            break

        previous_timestep = (
            timestep_schedule[
                index + 1
            ]
        )

        previous_alpha_bar = alpha_bars[
            previous_timestep
        ]

        x = (
            torch.sqrt(
                previous_alpha_bar
            )
            * x0
            + torch.sqrt(
                1.0
                - previous_alpha_bar
            )
            * predicted_noise
        )

    return x


# ============================================================
# LATENT QUALITY METRICS
# ============================================================

def latent_global_metrics(
    generated,
    real,
):

    generated_flat = generated.flatten(
        start_dim=1
    )

    real_flat = real.flatten(
        start_dim=1
    )

    generated_mean = float(
        generated_flat.mean()
    )

    real_mean = float(
        real_flat.mean()
    )

    generated_std = float(
        generated_flat.std()
    )

    real_std = float(
        real_flat.std()
    )

    mean_z = abs(
        generated_mean - real_mean
    ) / max(
        real_std,
        1e-6,
    )

    std_ratio_error = abs(
        generated_std
        / max(real_std, 1e-6)
        - 1.0
    )

    return {
        "generated_mean": generated_mean,
        "real_mean": real_mean,
        "generated_std": generated_std,
        "real_std": real_std,
        "mean_z": mean_z,
        "std_ratio_error": std_ratio_error,
    }


@torch.no_grad()
def nearest_latent_distance(
    generated,
    real,
    chunk_size=4,
):

    generated_flat = generated.flatten(
        start_dim=1
    )

    real_flat = real.flatten(
        start_dim=1
    )

    generated_flat = F.normalize(
        generated_flat,
        dim=1,
    )

    real_flat = F.normalize(
        real_flat,
        dim=1,
    )

    minimums = []

    for start in range(
        0,
        len(generated_flat),
        chunk_size,
    ):

        chunk = generated_flat[
            start:start + chunk_size
        ]

        similarity = (
            chunk
            @ real_flat.T
        )

        maximum_similarity = similarity.max(
            dim=1
        ).values

        distance = (
            1.0
            - maximum_similarity
        )

        minimums.append(
            distance.cpu()
        )

    values = torch.cat(
        minimums
    )

    return {
        "mean_nearest_distance": float(
            values.mean()
        ),
        "min_nearest_distance": float(
            values.min()
        ),
    }


# ============================================================
# V3 DECODER
# ============================================================

def load_v3_decoder():

    if not V3_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"V3 checkpoint not found:\n"
            f"{V3_CHECKPOINT}"
        )

    src_dir = (
        PROJECT_ROOT
        / "src"
        / "mri"
    )

    sys.path.insert(
        0,
        str(src_dir),
    )

    from mri_testing_v3_reconstruction import (
        MRIReconstructionAE,
    )

    checkpoint = torch.load(
        V3_CHECKPOINT,
        map_location=DEVICE,
    )

    model = MRIReconstructionAE().to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print()
    print(
        "V3 decoder loaded:"
    )

    print(
        f"  Epoch    : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"  Val PSNR : "
        f"{checkpoint.get('val_psnr')} dB"
    )

    return model


# ============================================================
# IMAGE CONVERSION
# ============================================================

def tensor_to_pil(
    tensor,
):

    array = (
        (
            tensor
            + 1.0
        )
        / 2.0
    ).clamp(
        0,
        1,
    ).detach().cpu().numpy()

    if array.ndim == 4:
        array = array[0, 0]

    elif array.ndim == 3:
        array = array[0]

    array = (
        array
        * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="L",
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def save_contact_sheet(
    images,
    class_name,
):

    columns = 4

    rows = math.ceil(
        len(images)
        / columns
    )

    cell_height = (
        IMAGE_SIZE
        + 24
    )

    sheet = Image.new(
        "L",
        (
            columns
            * IMAGE_SIZE,
            rows
            * cell_height,
        ),
        255,
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for index, image in enumerate(
        images
    ):

        x = (
            index
            % columns
        ) * IMAGE_SIZE

        y = (
            index
            // columns
        ) * cell_height

        sheet.paste(
            image,
            (
                x,
                y,
            ),
        )

        draw.text(
            (
                x + 4,
                y + IMAGE_SIZE + 4,
            ),
            f"{class_name} {index + 1:02d}",
            fill=0,
        )

    output = (
        PREVIEW_DIR
        / f"{class_name}_v4_1_contact_sheet.png"
    )

    sheet.save(
        output
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 80)
    print("# MRI TESTING V4.1 — QUALITY-FIRST LATENT DIFFUSION")
    print("#" * 80)

    print()
    print(
        f"Device             : {DEVICE}"
    )

    print(
        f"Latent cache       : {LATENT_FILE}"
    )

    print(
        f"V3 checkpoint      : {V3_CHECKPOINT}"
    )

    print(
        f"Output run         : {RUN}"
    )

    print(
        f"Training epochs    : {TRAIN_EPOCHS}"
    )

    print(
        f"Pilot/class        : {PILOT_PER_CLASS}"
    )

    print(
        f"CFG scale          : {CFG_SCALE}"
    )

    print(
        f"DDIM steps         : {DDIM_STEPS}"
    )

    # --------------------------------------------------------
    # Load cache.
    # --------------------------------------------------------

    if not LATENT_FILE.exists():
        raise FileNotFoundError(
            f"Latent file not found:\n{LATENT_FILE}"
        )

    if not STATS_FILE.exists():
        raise FileNotFoundError(
            f"Statistics file not found:\n{STATS_FILE}"
        )

    payload = torch.load(
        LATENT_FILE,
        map_location="cpu",
    )

    latents = payload[
        "latents"
    ].float()

    labels = payload[
        "labels"
    ].long()

    stats = torch.load(
        STATS_FILE,
        map_location="cpu",
    )

    mean = stats[
        "mean"
    ].float()

    std = stats[
        "std"
    ].float().clamp_min(
        1e-5
    )

    print()
    print(
        f"Cached latent shape : "
        f"{tuple(latents.shape)}"
    )

    print(
        f"Cached labels       : "
        f"{tuple(labels.shape)}"
    )

    if tuple(
        latents.shape[1:]
    ) != (
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):

        raise RuntimeError(
            "Unexpected cached latent shape: "
            f"{tuple(latents.shape[1:])}"
        )

    # --------------------------------------------------------
    # Normalize.
    # --------------------------------------------------------

    normalized = (
        latents
        - mean
    ) / std

    normalized = normalized.clamp(
        -5.0,
        5.0,
    )

    # --------------------------------------------------------
    # Train / validation split.
    # --------------------------------------------------------

    (
        train_latents,
        train_labels,
        val_latents,
        val_labels,
    ) = make_train_val_split(
        normalized,
        labels,
    )

    print()
    print(
        f"Train latent samples : "
        f"{len(train_latents)}"
    )

    print(
        f"Validation samples   : "
        f"{len(val_latents)}"
    )

    train_dataset = LatentDataset(
        train_latents,
        train_labels,
    )

    val_dataset = LatentDataset(
        val_latents,
        val_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Schedule.
    # --------------------------------------------------------

    (
        betas,
        alphas,
        alpha_bars,
    ) = cosine_alpha_bars(
        DIFFUSION_STEPS,
        DEVICE,
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = QualityConditionalUNet(
        latent_channels=LATENT_CHANNELS,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"Diffusion parameters : "
        f"{parameter_count / 1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    ema = EMA(
        model,
        EMA_DECAY,
    )

    history = []

    best_val = float(
        "inf"
    )

    best_state = None

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("V4.1 TRAINING")
    print("=" * 80)

    training_start = time.time()

    for epoch in range(
        1,
        TRAIN_EPOCHS + 1,
    ):

        epoch_start = time.time()

        train_loss = run_epoch(
            model,
            train_loader,
            alpha_bars,
            optimizer=optimizer,
            ema=ema,
            train=True,
        )

        # Validation uses the current model.
        val_loss = run_epoch(
            model,
            val_loader,
            alpha_bars,
            train=False,
        )

        scheduler.step()

        elapsed = (
            time.time()
            - epoch_start
        )

        learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "seconds": elapsed,
            }
        )

        if val_loss < best_val:

            best_val = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in ema.shadow.items()
            }

        print(
            f"Epoch {epoch:02d}/{TRAIN_EPOCHS} "
            f"| Train {train_loss:.6f} "
            f"| Val {val_loss:.6f} "
            f"| LR {learning_rate:.2e} "
            f"| {elapsed:.1f}s"
        )

    training_seconds = (
        time.time()
        - training_start
    )

    # --------------------------------------------------------
    # Restore best EMA state.
    # --------------------------------------------------------

    if best_state is not None:

        for name, parameter in model.named_parameters():

            if name in best_state:
                parameter.data.copy_(
                    best_state[name].to(
                        DEVICE
                    )
                )

    model.eval()

    checkpoint_path = (
        MODEL_DIR
        / "v4_1_best_ema_diffusion.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "classes": CLASSES,
            "latent_shape": [
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ],
            "epochs": TRAIN_EPOCHS,
            "best_val_loss": best_val,
            "diffusion_steps": DIFFUSION_STEPS,
            "ddim_steps": DDIM_STEPS,
            "cfg_scale": CFG_SCALE,
            "base_channels": BASE_CHANNELS,
            "history": history,
            "cache_run": str(CACHE_RUN),
            "v3_checkpoint": str(V3_CHECKPOINT),
        },
        checkpoint_path,
    )

    print()
    print(
        f"Best EMA checkpoint saved:\n"
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # Load V3 decoder once.
    # --------------------------------------------------------

    decoder = load_v3_decoder()

    # --------------------------------------------------------
    # Generate pilot.
    # --------------------------------------------------------

    generated_summary = {}

    all_generated = []

    print()
    print("=" * 80)
    print("V4.1 PILOT GENERATION")
    print("=" * 80)

    for class_name in CLASSES:

        class_id = CLASS_TO_ID[
            class_name
        ]

        labels_batch = torch.full(
            (
                PILOT_PER_CLASS,
            ),
            class_id,
            device=DEVICE,
            dtype=torch.long,
        )

        generation_start = time.time()

        generated_normalized = ddim_sample(
            model,
            labels_batch,
            (
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            alpha_bars,
            DDIM_STEPS,
            CFG_SCALE,
        )

        # Quality check in normalized latent space.
        real_class = normalized[
            labels == class_id
        ]

        latent_metrics = latent_global_metrics(
            generated_normalized.cpu(),
            real_class,
        )

        nearest_metrics = nearest_latent_distance(
            generated_normalized.cpu(),
            real_class,
        )

        # Convert normalized latent back to V3 latent scale.
        generated_latents = (
            generated_normalized
            * std.to(DEVICE)
            + mean.to(DEVICE)
        )

        # Decode.
        with torch.no_grad():

            decoded = decoder.decode(
                generated_latents
            )

        class_dir = (
            SYNTH_DIR
            / class_name
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pil_images = []

        for index in range(
            decoded.shape[0]
        ):

            image = tensor_to_pil(
                decoded[
                    index:index + 1
                ]
            )

            pil_images.append(
                image
            )

            image_path = (
                class_dir
                / (
                    f"{class_name}"
                    f"_synthetic_"
                    f"{index + 1:04d}.png"
                )
            )

            image.save(
                image_path
            )

        contact_sheet = save_contact_sheet(
            pil_images,
            class_name,
        )

        elapsed = (
            time.time()
            - generation_start
        )

        class_result = {
            "count": PILOT_PER_CLASS,
            "generation_seconds": elapsed,
            "contact_sheet": str(
                contact_sheet
            ),
            "latent_metrics": latent_metrics,
            "nearest_metrics": nearest_metrics,
        }

        generated_summary[
            class_name
        ] = class_result

        all_generated.append(
            generated_normalized.cpu()
        )

        print()
        print(
            f"{class_name:12s} | "
            f"Mean {latent_metrics['generated_mean']:.4f} "
            f"vs {latent_metrics['real_mean']:.4f} | "
            f"Std {latent_metrics['generated_std']:.4f} "
            f"vs {latent_metrics['real_std']:.4f} | "
            f"Nearest {nearest_metrics['mean_nearest_distance']:.4f}"
        )

    # --------------------------------------------------------
    # Global quality gate.
    # --------------------------------------------------------

    global_generated = torch.cat(
        all_generated,
        dim=0,
    )

    global_metrics = latent_global_metrics(
        global_generated,
        normalized,
    )

    global_nearest = nearest_latent_distance(
        global_generated,
        normalized,
    )

    gate_reasons = []

    if (
        global_metrics["mean_z"]
        > MAX_GLOBAL_MEAN_Z
    ):
        gate_reasons.append(
            "Generated latent mean is too far from the real latent mean."
        )

    if (
        global_metrics["std_ratio_error"]
        > MAX_GLOBAL_STD_RATIO_ERROR
    ):
        gate_reasons.append(
            "Generated latent spread differs too much from the real latent spread."
        )

    if (
        global_nearest["min_nearest_distance"]
        < MIN_NEAREST_LATENT_DISTANCE
    ):
        gate_reasons.append(
            "Some generated latents are extremely close to cached real latents; inspect for possible memorization."
        )

    quality_gate_passed = (
        len(gate_reasons) == 0
    )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    report = {
        "version": "V4.1 quality-first conditional latent diffusion",
        "device": str(DEVICE),
        "cache_run": str(CACHE_RUN),
        "v3_checkpoint": str(V3_CHECKPOINT),
        "classes": CLASSES,
        "cached_images": int(
            len(latents)
        ),
        "train_images": int(
            len(train_latents)
        ),
        "validation_images": int(
            len(val_latents)
        ),
        "latent_shape": [
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ],
        "training_epochs": TRAIN_EPOCHS,
        "best_validation_loss": best_val,
        "training_seconds": training_seconds,
        "diffusion_steps": DIFFUSION_STEPS,
        "ddim_steps": DDIM_STEPS,
        "cfg_scale": CFG_SCALE,
        "condition_drop_probability": CONDITION_DROP_PROB,
        "base_channels": BASE_CHANNELS,
        "history": history,
        "global_latent_metrics": global_metrics,
        "global_nearest_metrics": global_nearest,
        "class_results": generated_summary,
        "quality_gate": {
            "passed": quality_gate_passed,
            "reasons": gate_reasons,
            "max_global_mean_z": MAX_GLOBAL_MEAN_Z,
            "max_global_std_ratio_error": MAX_GLOBAL_STD_RATIO_ERROR,
            "min_nearest_latent_distance": MIN_NEAREST_LATENT_DISTANCE,
        },
        "training_data_modified": False,
        "testing_data_modified": False,
        "v3_checkpoint_modified": False,
        "v4_cache_modified": False,
        "full_800_generation": False,
    }

    report_path = (
        REPORT_DIR
        / "v4_1_quality_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------

    print()
    print("#" * 80)
    print("# V4.1 PILOT COMPLETE")
    print("#" * 80)

    print()
    print(
        f"Quality gate : "
        f"{'PASS' if quality_gate_passed else 'FAIL'}"
    )

    if gate_reasons:

        print()
        print(
            "Quality-gate warnings:"
        )

        for reason in gate_reasons:
            print(
                f"  - {reason}"
            )

    print()
    print(
        f"Synthetic output : {SYNTH_DIR}"
    )

    print(
        f"Preview output   : {PREVIEW_DIR}"
    )

    print(
        f"Model            : {checkpoint_path}"
    )

    print(
        f"Report           : {report_path}"
    )

    print()
    print(
        "DO NOT generate 800/class automatically."
    )

    print(
        "Inspect all four V4.1 contact sheets first."
    )

    print()
    print(
        "The quality gate checks latent statistics,"
        " but visual MRI quality still requires inspection."
    )


if __name__ == "__main__":
    main()

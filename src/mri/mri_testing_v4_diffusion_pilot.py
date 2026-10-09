"""
MRI TESTING — V4 STEP 2
Conditional Latent Diffusion Pilot

Purpose:
1. Load the already-created V4 latent cache (1,600 MRI latents).
2. Train a class-conditional diffusion model in latent space.
3. Generate a SMALL pilot set: 16 synthetic images per class.
4. Decode the generated latents using the already-validated V3 autoencoder.
5. Save contact sheets + metrics/report.

IMPORTANT:
- Reads ONLY data/imaging/MRI/Testing through the cached latents.
- Does NOT modify Training, Testing, or the V3 checkpoint.
- Does NOT generate 800/class yet.
- Do NOT rerun the V3 reconstruction or V4 cache step.
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

V4_CACHE_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_v4_latents"
    / "run_20261001_044203"
)

LATENT_FILE = V4_CACHE_RUN / "latents.pt"
STATS_FILE = V4_CACHE_RUN / "latent_statistics.pt"

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
    / "mri_testing_v4_diffusion"
)

RUN = OUTPUT_ROOT / f"pilot_{time.strftime('%Y%m%d_%H%M%S')}"
MODEL_DIR = RUN / "models"
SYNTH_DIR = RUN / "synthetic"
PREVIEW_DIR = RUN / "previews"
REPORT_DIR = RUN / "reports"

for d in [MODEL_DIR, SYNTH_DIR, PREVIEW_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 128

LATENT_CHANNELS = 16
LATENT_SIZE = 32

NUM_CLASSES = 4
CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# CPU-friendly pilot
BATCH_SIZE = 16
DIFFUSION_EPOCHS = 15

# Pilot only — NOT 800 yet
PILOT_PER_CLASS = 16

# Diffusion schedule
DIFFUSION_STEPS = 1000
DDIM_STEPS = 50

# Model width
BASE_CHANNELS = 64
TIME_EMB = 128
CLASS_EMB = 128

# EMA
EMA_DECAY = 0.999

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# UTILITIES
# ============================================================

def sinusoidal_embedding(t, dim):
    """
    t: [B] integer/float timestep
    returns: [B, dim]
    """
    half = dim // 2

    freq = torch.exp(
        -math.log(10000)
        * torch.arange(
            0,
            half,
            device=t.device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )

    angles = t.float().unsqueeze(1) * freq.unsqueeze(0)

    emb = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1,
    )

    if dim % 2:
        emb = F.pad(emb, (0, 1))

    return emb


def group_count(channels):
    """
    Safe GroupNorm group count.
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
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

    def forward(self, x, emb):

        h = self.norm1(x)
        h = F.silu(h)

        h = self.conv1(h)

        h = h + self.emb(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)

        h = self.conv2(h)

        return h + self.skip(x)


# ============================================================
# CONDITIONAL LATENT UNET
# ============================================================

class ConditionalLatentUNet(nn.Module):

    def __init__(
        self,
        latent_channels=16,
        num_classes=4,
        base=64,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_EMB, TIME_EMB),
            nn.SiLU(),
            nn.Linear(TIME_EMB, TIME_EMB),
        )

        self.class_embedding = nn.Embedding(
            num_classes,
            CLASS_EMB,
        )

        emb_dim = TIME_EMB + CLASS_EMB

        c1 = base
        c2 = base * 2
        c3 = base * 4

        self.in_conv = nn.Conv2d(
            latent_channels,
            c1,
            3,
            padding=1,
        )

        self.down1 = ResBlock(c1, c1, emb_dim)

        self.downsample1 = nn.Conv2d(
            c1,
            c2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down2 = ResBlock(c2, c2, emb_dim)

        self.downsample2 = nn.Conv2d(
            c2,
            c3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.mid1 = ResBlock(c3, c3, emb_dim)
        self.mid2 = ResBlock(c3, c3, emb_dim)

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=4,
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
            kernel_size=4,
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
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, timestep, labels):

        t_emb = sinusoidal_embedding(
            timestep,
            TIME_EMB,
        )

        t_emb = self.time_mlp(t_emb)

        c_emb = self.class_embedding(labels)

        emb = torch.cat(
            [t_emb, c_emb],
            dim=1,
        )

        x0 = self.in_conv(x)

        d1 = self.down1(
            x0,
            emb,
        )

        d2_in = self.downsample1(d1)

        d2 = self.down2(
            d2_in,
            emb,
        )

        mid_in = self.downsample2(d2)

        mid = self.mid1(
            mid_in,
            emb,
        )

        mid = self.mid2(
            mid,
            emb,
        )

        u2 = self.up2(mid)

        u2 = torch.cat(
            [u2, d2],
            dim=1,
        )

        u2 = self.up_block2(
            u2,
            emb,
        )

        u1 = self.up1(u2)

        u1 = torch.cat(
            [u1, d1],
            dim=1,
        )

        u1 = self.up_block1(
            u1,
            emb,
        )

        out = self.out_norm(u1)
        out = F.silu(out)
        out = self.out_conv(out)

        return out


# ============================================================
# EMA
# ============================================================

class EMA:

    def __init__(
        self,
        model,
        decay=0.999,
    ):
        self.decay = decay

        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            self.shadow[name].mul_(self.decay)
            self.shadow[name].add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

    def apply(self, model):

        for name, param in model.named_parameters():

            if name in self.shadow:
                param.data.copy_(
                    self.shadow[name].data
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

    def __getitem__(self, index):

        return (
            self.latents[index],
            self.labels[index],
        )


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

def make_beta_schedule(
    steps,
    device,
):

    beta_start = 1e-4
    beta_end = 0.02

    betas = torch.linspace(
        beta_start,
        beta_end,
        steps,
        device=device,
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0,
    )

    return betas, alphas, alpha_bars


def extract(
    values,
    timestep,
    shape,
):

    out = values[timestep]

    return out.reshape(
        timestep.shape[0],
        *([1] * (len(shape) - 1)),
    )


def q_sample(
    x0,
    timestep,
    noise,
    alpha_bars,
):

    a_bar = extract(
        alpha_bars,
        timestep,
        x0.shape,
    )

    return (
        torch.sqrt(a_bar) * x0
        + torch.sqrt(1.0 - a_bar) * noise
    )


# ============================================================
# TRAINING
# ============================================================

def train_diffusion(
    model,
    loader,
    alpha_bars,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ema = EMA(
        model,
        EMA_DECAY,
    )

    history = []

    total_steps = len(loader)

    print()
    print("=" * 80)
    print("V4 CONDITIONAL LATENT DIFFUSION TRAINING")
    print("=" * 80)

    for epoch in range(1, DIFFUSION_EPOCHS + 1):

        model.train()

        epoch_loss = 0.0
        start = time.time()

        for step, (x0, labels) in enumerate(loader, 1):

            x0 = x0.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # Normalize latents before diffusion.
            # Statistics are applied outside this function.
            timestep = torch.randint(
                0,
                DIFFUSION_STEPS,
                (
                    x0.shape[0],
                ),
                device=DEVICE,
            )

            noise = torch.randn_like(x0)

            noisy = q_sample(
                x0,
                timestep,
                noise,
                alpha_bars,
            )

            predicted_noise = model(
                noisy,
                timestep,
                labels,
            )

            loss = F.mse_loss(
                predicted_noise,
                noise,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            ema.update(model)

            epoch_loss += float(loss.item())

        epoch_loss /= max(total_steps, 1)

        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "seconds": elapsed,
            }
        )

        print(
            f"Epoch {epoch:02d}/{DIFFUSION_EPOCHS} "
            f"| Loss {epoch_loss:.6f} "
            f"| Time {elapsed:.1f}s"
        )

    return ema, history


# ============================================================
# DDIM SAMPLING
# ============================================================

@torch.no_grad()
def ddim_sample(
    model,
    labels,
    latent_shape,
    alpha_bars,
    steps=50,
):

    model.eval()

    batch = labels.shape[0]

    x = torch.randn(
        batch,
        *latent_shape,
        device=DEVICE,
    )

    timesteps = torch.linspace(
        DIFFUSION_STEPS - 1,
        0,
        steps,
        device=DEVICE,
    ).long()

    for i, t in enumerate(timesteps):

        t_batch = torch.full(
            (batch,),
            int(t.item()),
            device=DEVICE,
            dtype=torch.long,
        )

        predicted_noise = model(
            x,
            t_batch,
            labels,
        )

        alpha_bar_t = alpha_bars[t]

        sqrt_alpha_bar_t = torch.sqrt(
            alpha_bar_t
        )

        sqrt_one_minus_alpha_bar_t = torch.sqrt(
            1.0 - alpha_bar_t
        )

        predicted_x0 = (
            x
            - sqrt_one_minus_alpha_bar_t
            * predicted_noise
        ) / sqrt_alpha_bar_t

        predicted_x0 = predicted_x0.clamp(
            -4.0,
            4.0,
        )

        if i == len(timesteps) - 1:
            x = predicted_x0
            break

        t_prev = timesteps[i + 1]

        alpha_bar_prev = alpha_bars[t_prev]

        # Deterministic DDIM update.
        x = (
            torch.sqrt(alpha_bar_prev)
            * predicted_x0
            + torch.sqrt(
                1.0 - alpha_bar_prev
            )
            * predicted_noise
        )

    return x


# ============================================================
# V3 AUTOENCODER LOADING
# ============================================================

def load_v3_decoder():

    if not V3_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"V3 checkpoint not found:\n{V3_CHECKPOINT}"
        )

    src_mri = (
        PROJECT_ROOT
        / "src"
        / "mri"
    )

    sys.path.insert(
        0,
        str(src_mri),
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
        f"V3 decoder loaded | "
        f"Epoch {checkpoint.get('epoch')} | "
        f"Val PSNR {checkpoint.get('val_psnr')} dB"
    )

    return model


# ============================================================
# IMAGE CONVERSION
# ============================================================

def tensor_to_pil(x):

    x = (
        ((x + 1.0) / 2.0)
        .clamp(0, 1)
        .detach()
        .cpu()
        .numpy()
    )

    if x.ndim == 4:
        x = x[:, 0]

    if x.ndim == 3:
        x = x[0]

    x = (
        x * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        x,
        mode="L",
    )


# ============================================================
# SAVE CONTACT SHEETS
# ============================================================

def save_contact_sheet(
    images,
    class_name,
):

    if not images:
        return

    cols = 4
    rows = math.ceil(
        len(images) / cols
    )

    cell = IMAGE_SIZE + 24

    sheet = Image.new(
        "L",
        (
            cols * IMAGE_SIZE,
            rows * cell,
        ),
        255,
    )

    draw = ImageDraw.Draw(sheet)

    for i, image in enumerate(images):

        x = (
            i % cols
        ) * IMAGE_SIZE

        y = (
            i // cols
        ) * cell

        sheet.paste(
            image,
            (x, y),
        )

        draw.text(
            (
                x + 4,
                y + IMAGE_SIZE + 4,
            ),
            f"{class_name} {i + 1:02d}",
            fill=0,
        )

    path = (
        PREVIEW_DIR
        / f"{class_name}_pilot_contact_sheet.png"
    )

    sheet.save(path)

    print(
        f"Contact sheet: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 80)
    print("# MRI TESTING V4 — CONDITIONAL LATENT DIFFUSION PILOT")
    print("#" * 80)

    print()
    print(f"Device       : {DEVICE}")
    print(f"Cache        : {LATENT_FILE}")
    print(f"Stats        : {STATS_FILE}")
    print(f"V3 checkpoint: {V3_CHECKPOINT}")
    print(f"Output       : {RUN}")
    print(
        f"Pilot target : {PILOT_PER_CLASS} images/class "
        f"({PILOT_PER_CLASS * NUM_CLASSES} total)"
    )

    # --------------------------------------------------------
    # Validate cache
    # --------------------------------------------------------

    if not LATENT_FILE.exists():
        raise FileNotFoundError(
            f"Latent cache not found:\n{LATENT_FILE}"
        )

    if not STATS_FILE.exists():
        raise FileNotFoundError(
            f"Latent statistics not found:\n{STATS_FILE}"
        )

    payload = torch.load(
        LATENT_FILE,
        map_location="cpu",
    )

    latents = payload["latents"].float()
    labels = payload["labels"].long()

    stats = torch.load(
        STATS_FILE,
        map_location="cpu",
    )

    mean = stats["mean"].float()
    std = stats["std"].float()

    print()
    print(
        f"Cached latents : {tuple(latents.shape)}"
    )

    print(
        f"Labels          : {tuple(labels.shape)}"
    )

    print(
        f"Mean/std global : "
        f"{float(latents.mean()):.6f} / "
        f"{float(latents.std()):.6f}"
    )

    if tuple(
        latents.shape[1:]
    ) != (
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
    ):

        raise RuntimeError(
            "Unexpected latent shape: "
            f"{tuple(latents.shape[1:])}"
        )

    # --------------------------------------------------------
    # Normalize latents
    # --------------------------------------------------------

    mean = mean.clamp(
        min=-10.0,
        max=10.0,
    )

    std = std.clamp_min(
        1e-5
    )

    normalized_latents = (
        latents - mean
    ) / std

    # Safety clamp for unusually large latent values.
    normalized_latents = normalized_latents.clamp(
        -5.0,
        5.0,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = LatentDataset(
        normalized_latents,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=True,
    )

    # --------------------------------------------------------
    # Diffusion model
    # --------------------------------------------------------

    model = ConditionalLatentUNet(
        latent_channels=LATENT_CHANNELS,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Diffusion parameters: "
        f"{parameter_count / 1e6:.2f}M"
    )

    betas, alphas, alpha_bars = (
        make_beta_schedule(
            DIFFUSION_STEPS,
            DEVICE,
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start_train = time.time()

    ema, history = train_diffusion(
        model,
        loader,
        alpha_bars,
    )

    training_seconds = (
        time.time()
        - start_train
    )

    # --------------------------------------------------------
    # Apply EMA
    # --------------------------------------------------------

    ema.apply(model)
    model.eval()

    checkpoint_path = (
        MODEL_DIR
        / "conditional_latent_diffusion_ema.pt"
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
            "diffusion_steps": DIFFUSION_STEPS,
            "ddim_steps": DDIM_STEPS,
            "base_channels": BASE_CHANNELS,
            "epoch": DIFFUSION_EPOCHS,
            "history": history,
            "training_seconds": training_seconds,
            "cache_run": str(V4_CACHE_RUN),
            "v3_checkpoint": str(V3_CHECKPOINT),
        },
        checkpoint_path,
    )

    print()
    print(
        f"Diffusion checkpoint saved:\n"
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # Generate pilot
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATING PILOT SYNTHETIC MRI")
    print("=" * 80)

    generated_summary = {}

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

        start = time.time()

        generated_latents = ddim_sample(
            model,
            labels_batch,
            (
                LATENT_CHANNELS,
                LATENT_SIZE,
                LATENT_SIZE,
            ),
            alpha_bars,
            DDIM_STEPS,
        )

        # Undo latent normalization.
        generated_latents = (
            generated_latents
            * std.to(DEVICE)
            + mean.to(DEVICE)
        )

        # Decode using the V3 decoder.
        v3_decoder = getattr(
            main,
            "_v3_decoder",
            None,
        )

        if v3_decoder is None:
            # Loaded once on first class.
            v3_decoder = load_v3_decoder()
            main._v3_decoder = v3_decoder

        with torch.no_grad():

            images = v3_decoder.decode(
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

        for i in range(
            images.shape[0]
        ):

            pil = tensor_to_pil(
                images[i:i + 1]
            )

            pil_images.append(pil)

            filename = (
                class_dir
                / f"{class_name}_synthetic_{i + 1:04d}.png"
            )

            pil.save(
                filename
            )

        elapsed = (
            time.time()
            - start
        )

        save_contact_sheet(
            pil_images,
            class_name,
        )

        generated_summary[
            class_name
        ] = {
            "count": int(
                len(pil_images)
            ),
            "seconds": elapsed,
            "output": str(
                class_dir
            ),
        }

        print(
            f"{class_name:12s} | "
            f"{len(pil_images):3d} images | "
            f"{elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "version": "V4 conditional latent diffusion pilot",
        "device": str(DEVICE),
        "cache_run": str(V4_CACHE_RUN),
        "latent_file": str(LATENT_FILE),
        "stats_file": str(STATS_FILE),
        "v3_checkpoint": str(V3_CHECKPOINT),
        "classes": CLASSES,
        "cached_images": int(len(latents)),
        "latent_shape": [
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
        ],
        "diffusion_epochs": DIFFUSION_EPOCHS,
        "diffusion_steps": DIFFUSION_STEPS,
        "ddim_steps": DDIM_STEPS,
        "pilot_per_class": PILOT_PER_CLASS,
        "pilot_total": int(
            PILOT_PER_CLASS * NUM_CLASSES
        ),
        "training_seconds": training_seconds,
        "history": history,
        "generated": generated_summary,
        "training_data_modified": False,
        "testing_data_modified": False,
        "v3_checkpoint_modified": False,
        "old_v4_cache_modified": False,
        "full_800_generation": False,
    }

    report_path = (
        REPORT_DIR
        / "v4_diffusion_pilot_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("#" * 80)
    print("# V4 PILOT COMPLETE")
    print("#" * 80)

    print(
        f"\nSynthetic output : {SYNTH_DIR}"
    )

    print(
        f"Preview output   : {PREVIEW_DIR}"
    )

    print(
        f"Model checkpoint : {checkpoint_path}"
    )

    print(
        f"Report           : {report_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This is only a 16/class pilot."
    )
    print(
        "Do NOT generate 800/class yet."
    )
    print(
        "First inspect all four contact sheets."
    )


if __name__ == "__main__":
    main()

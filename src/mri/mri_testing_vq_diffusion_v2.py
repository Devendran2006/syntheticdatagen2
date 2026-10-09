"""
MRI TESTING V2
VQ-VAE + Conditional Latent Diffusion

INPUT (READ ONLY)
    data/imaging/MRI/Testing/

OUTPUT (NEW FOLDER ONLY)
    outputs/mri_testing_vq_diffusion_v2/run_YYYYMMDD_HHMMSS/

TARGET
    800 synthetic images per class
    4 classes = 3,200 synthetic MRI images

ARCHITECTURE
    128x128 MRI
        -> Residual VQ-VAE
        -> 8x16x16 latent
        -> Conditional latent diffusion
        -> 8x16x16 latent
        -> VQ-VAE decoder
        -> 128x128 MRI

IMPORTANT
---------
PILOT_MODE=True is intentional.
First validate reconstruction on GLIOMA before spending time on
all 4 classes. After reconstruction looks good, set PILOT_MODE=False.

No Training data is read.
No old output/checkpoint is modified.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Testing"

RUN_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_vq_diffusion_v2"
    / time.strftime("run_%Y%m%d_%H%M%S")
)

MODEL_ROOT = RUN_ROOT / "models"
RECON_ROOT = RUN_ROOT / "reconstruction"
SYNTH_ROOT = RUN_ROOT / "synthetic"
REPORT_ROOT = RUN_ROOT / "reports"

IMAGE_SIZE = 128

# Deliberately 16x16, not 8x8.
# 128 -> 64 -> 32 -> 16 (three downsamples).
LATENT_CHANNELS = 8
LATENT_SIZE = 16

NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 0

AE_EPOCHS = 15
DIFFUSION_EPOCHS = 50

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------
SYNTHETIC_PER_CLASS = 800
GEN_BATCH_SIZE = 32
DIFFUSION_STEPS = 1000
DDIM_STEPS = 50

# ------------------------------------------------------------
# VQ-VAE
# ------------------------------------------------------------
CODEBOOK_SIZE = 512
VQ_BETA = 0.25

# ------------------------------------------------------------
# Diffusion
# ------------------------------------------------------------
BASE_CHANNELS = 64
TIME_EMBED = 128
CLASS_EMBED = 64
EMA_DECAY = 0.999

# ------------------------------------------------------------
# Reconstruction loss
# ------------------------------------------------------------
L1_WEIGHT = 1.0
SSIM_WEIGHT = 0.20
EDGE_WEIGHT = 0.15
VQ_WEIGHT = 1.0

# ------------------------------------------------------------
# Safety / experiment control
# ------------------------------------------------------------
TRAIN_VQVAE = False
TRAIN_DIFFUSION = False
GENERATE_SYNTHETIC = True

# FIRST RUN: one class only.
PILOT_MODE = True
PILOT_CLASSES = ["glioma"]

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for folder in [MODEL_ROOT, RECON_ROOT, SYNTH_ROOT, REPORT_ROOT]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA
# ============================================================

VALID_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def image_paths_for_class(class_name):
    folder = TEST_ROOT / class_name
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Robust per-image intensity normalization.
    lo, hi = np.percentile(arr, [1, 99])

    if hi > lo + 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    return arr


class MRIDataset(Dataset):
    def __init__(self, classes, augment=False):
        self.items = []
        self.augment = augment

        for class_name in classes:
            for path in image_paths_for_class(class_name):
                self.items.append(
                    (path, CLASS_TO_ID[class_name])
                )

        if not self.items:
            raise RuntimeError(
                f"No MRI images found under {TEST_ROOT}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]

        arr = preprocess_image(path)

        # Mild augmentation only.
        if self.augment and random.random() < 0.5:
            arr = np.fliplr(arr).copy()

        x = torch.from_numpy(arr).float().unsqueeze(0)
        x = x * 2.0 - 1.0

        return x, label, str(path)


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1
        )

        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1
        )

    def forward(self, x):
        h = self.conv1(
            F.silu(self.norm1(x))
        )
        h = self.conv2(
            F.silu(self.norm2(h))
        )
        return x + h


# ============================================================
# VQ-VAE ENCODER
# 128 -> 64 -> 32 -> 16
# ============================================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = nn.Conv2d(
            1, 64, 3, padding=1
        )

        self.b1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
        )

        self.d1 = nn.Conv2d(
            64, 96, 4, stride=2, padding=1
        )

        self.b2 = nn.Sequential(
            ResBlock(96),
            ResBlock(96),
        )

        self.d2 = nn.Conv2d(
            96, 128, 4, stride=2, padding=1
        )

        self.b3 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        self.d3 = nn.Conv2d(
            128,
            LATENT_CHANNELS,
            4,
            stride=2,
            padding=1,
        )

        self.out = nn.Conv2d(
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            1,
        )

    def forward(self, x):
        x = F.silu(self.in_conv(x))
        x = self.b1(x)

        x = F.silu(self.d1(x))
        x = self.b2(x)

        x = F.silu(self.d2(x))
        x = self.b3(x)

        x = F.silu(self.d3(x))
        return self.out(x)


# ============================================================
# VQ-VAE DECODER
# 16 -> 32 -> 64 -> 128
# ============================================================

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = nn.Conv2d(
            LATENT_CHANNELS, 128, 3, padding=1
        )

        self.b1 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        self.u1 = nn.ConvTranspose2d(
            128, 128, 4, stride=2, padding=1
        )

        self.b2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        self.u2 = nn.ConvTranspose2d(
            128, 96, 4, stride=2, padding=1
        )

        self.b3 = nn.Sequential(
            ResBlock(96),
            ResBlock(96),
        )

        self.u3 = nn.ConvTranspose2d(
            96, 64, 4, stride=2, padding=1
        )

        self.out = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = F.silu(self.in_conv(z))
        x = self.b1(x)

        x = F.silu(self.u1(x))
        x = self.b2(x)

        x = F.silu(self.u2(x))
        x = self.b3(x)

        x = F.silu(self.u3(x))
        return self.out(x)


# ============================================================
# VECTOR QUANTIZER
# ============================================================

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        beta=VQ_BETA,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(self, z):
        b, c, h, w = z.shape

        flat = (
            z.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, c)
        )

        emb = self.embedding.weight

        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ emb.t()
            + emb.pow(2).sum(1)
        )

        indices = distances.argmin(dim=1)

        q = self.embedding(indices)
        q = q.view(b, h, w, c)
        q = q.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(
            q,
            z.detach(),
        )

        commitment_loss = F.mse_loss(
            q.detach(),
            z,
        )

        vq_loss = (
            codebook_loss
            + self.beta * commitment_loss
        )

        # Straight-through estimator.
        q_st = z + (q - z).detach()

        return (
            q_st,
            vq_loss,
            indices.view(b, h, w),
        )


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.quantizer = VectorQuantizer(
            CODEBOOK_SIZE,
            LATENT_CHANNELS,
        )

        self.decoder = Decoder()

    def encode(self, x):
        z = self.encoder(x)

        q, vq_loss, indices = (
            self.quantizer(z)
        )

        return q, vq_loss, indices

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, vq_loss, indices = self.encode(x)
        recon = self.decode(z)

        return recon, z, vq_loss, indices


# ============================================================
# IMAGE LOSSES
# ============================================================

def sobel_edges(x):
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)

    return torch.sqrt(
        gx * gx + gy * gy + 1e-6
    )


def local_ssim_loss(x, y):
    mu_x = F.avg_pool2d(
        x, 7, stride=1, padding=3
    )
    mu_y = F.avg_pool2d(
        y, 7, stride=1, padding=3
    )

    var_x = (
        F.avg_pool2d(
            x * x, 7, stride=1, padding=3
        )
        - mu_x * mu_x
    )

    var_y = (
        F.avg_pool2d(
            y * y, 7, stride=1, padding=3
        )
        - mu_y * mu_y
    )

    cov = (
        F.avg_pool2d(
            x * y, 7, stride=1, padding=3
        )
        - mu_x * mu_y
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    score = (
        (2 * mu_x * mu_y + c1)
        * (2 * cov + c2)
        /
        (
            (mu_x * mu_x + mu_y * mu_y + c1)
            * (var_x + var_y + c2)
            + 1e-6
        )
    )

    return 1.0 - score.clamp(-1, 1).mean()


# ============================================================
# DATA REPORT
# ============================================================

def print_report(classes):
    print("\n" + "=" * 78)
    print("MRI TESTING DATASET")
    print("=" * 78)

    total = 0

    for c in classes:
        n = len(image_paths_for_class(c))
        total += n
        print(f"{c:15s}: {n:5d}")

    print("-" * 78)
    print(f"{'TOTAL':15s}: {total:5d}")
    print(f"Image           : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(
        f"Latent          : "
        f"{LATENT_CHANNELS}x{LATENT_SIZE}x{LATENT_SIZE}"
    )
    print(f"Device          : {DEVICE}")


# ============================================================
# VQ-VAE TRAINING
# ============================================================

def train_vqvae(classes):
    dataset = MRIDataset(
        classes,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = VQVAE().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    use_amp = DEVICE.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    history = []

    print("\n" + "=" * 78)
    print("STAGE 1 — RESIDUAL VQ-VAE")
    print("=" * 78)

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()

        start = time.time()

        sums = {
            "loss": 0.0,
            "l1": 0.0,
            "ssim": 0.0,
            "edge": 0.0,
            "vq": 0.0,
        }

        for x, _, _ in loader:
            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                recon, _, vq_loss, _ = model(x)

                l1 = F.l1_loss(
                    recon, x
                )

                ssim = local_ssim_loss(
                    recon, x
                )

                edge = F.l1_loss(
                    sobel_edges(recon),
                    sobel_edges(x),
                )

                loss = (
                    L1_WEIGHT * l1
                    + SSIM_WEIGHT * ssim
                    + EDGE_WEIGHT * edge
                    + VQ_WEIGHT * vq_loss
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            sums["loss"] += loss.item()
            sums["l1"] += l1.item()
            sums["ssim"] += ssim.item()
            sums["edge"] += edge.item()
            sums["vq"] += vq_loss.item()

        n = len(loader)

        row = {
            "epoch": epoch,
            "loss": sums["loss"] / n,
            "l1": sums["l1"] / n,
            "ssim": sums["ssim"] / n,
            "edge": sums["edge"] / n,
            "vq": sums["vq"] / n,
            "seconds": time.time() - start,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{AE_EPOCHS} | "
            f"Loss {row['loss']:.4f} | "
            f"L1 {row['l1']:.4f} | "
            f"SSIM {row['ssim']:.4f} | "
            f"Edge {row['edge']:.4f} | "
            f"VQ {row['vq']:.4f} | "
            f"{row['seconds']:.1f}s"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "image_size": IMAGE_SIZE,
                "latent_channels": LATENT_CHANNELS,
                "latent_size": LATENT_SIZE,
                "codebook_size": CODEBOOK_SIZE,
            },
            "history": history,
        },
        MODEL_ROOT / "vqvae_final.pt",
    )

    with open(
        REPORT_ROOT / "vqvae_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    return model


# ============================================================
# RECONSTRUCTION TEST
# ============================================================

@torch.no_grad()
def reconstruction_test(
    model,
    classes,
    per_class=16,
):
    model.eval()

    print("\n" + "=" * 78)
    print("RECONSTRUCTION GATE")
    print("=" * 78)

    all_metrics = {}

    for class_name in classes:
        paths = image_paths_for_class(
            class_name
        )[:per_class]

        output_dir = (
            RECON_ROOT / class_name
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for i, path in enumerate(paths):
            arr = preprocess_image(path)

            x = (
                torch.from_numpy(arr)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            x = x * 2.0 - 1.0

            recon, _, _, _ = model(x)

            real01 = (
                (x + 1) / 2
            ).clamp(0, 1)

            recon01 = (
                (recon + 1) / 2
            ).clamp(0, 1)

            mse = F.mse_loss(
                recon01,
                real01,
            ).item()

            mae = F.l1_loss(
                recon01,
                real01,
            ).item()

            psnr = -10.0 * math.log10(
                max(mse, 1e-8)
            )

            ssim_loss = local_ssim_loss(
                recon01,
                real01,
            ).item()

            rows.append(
                {
                    "file": path.name,
                    "mae": mae,
                    "psnr": psnr,
                    "ssim_loss": ssim_loss,
                }
            )

            real_pil = Image.fromarray(
                (
                    real01[0, 0]
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            )

            recon_pil = Image.fromarray(
                (
                    recon01[0, 0]
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            )

            sheet = Image.new(
                "L",
                (
                    IMAGE_SIZE * 2,
                    IMAGE_SIZE,
                ),
            )

            sheet.paste(
                real_pil,
                (0, 0),
            )

            sheet.paste(
                recon_pil,
                (IMAGE_SIZE, 0),
            )

            sheet.save(
                output_dir
                / f"{i+1:03d}_real_vs_recon.png"
            )

        all_metrics[class_name] = rows

        if rows:
            print(
                f"{class_name:15s} | "
                f"MAE {np.mean([r['mae'] for r in rows]):.4f} | "
                f"PSNR {np.mean([r['psnr'] for r in rows]):.2f} dB | "
                f"SSIM-loss {np.mean([r['ssim_loss'] for r in rows]):.4f}"
            )

    with open(
        REPORT_ROOT
        / "reconstruction_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_metrics,
            f,
            indent=2,
        )


# ============================================================
# LATENT COLLECTION
# ============================================================

@torch.no_grad()
def collect_latents(model, classes):
    model.eval()

    dataset = MRIDataset(
        classes,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    latents = []
    labels = []

    for x, y, _ in loader:
        x = x.to(DEVICE)

        z, _, _ = model.encode(x)

        latents.append(
            z.cpu()
        )

        labels.append(y)

    latents = torch.cat(
        latents,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    mean = latents.mean(
        dim=0,
        keepdim=True,
    )

    std = latents.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-4)

    torch.save(
        {
            "mean": mean,
            "std": std,
            "shape": list(
                latents.shape[1:]
            ),
        },
        MODEL_ROOT
        / "latent_statistics.pt",
    )

    print(
        f"Latents : {tuple(latents.shape)}"
    )

    return (
        latents,
        labels,
        mean.to(DEVICE),
        std.to(DEVICE),
    )


# ============================================================
# DIFFUSION
# ============================================================

def sinusoidal_embedding(t, dim):
    half = dim // 2

    freq = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=t.device,
        ).float()
        / max(half - 1, 1)
    )

    values = (
        t.float().unsqueeze(1)
        * freq.unsqueeze(0)
    )

    emb = torch.cat(
        [
            torch.sin(values),
            torch.cos(values),
        ],
        dim=1,
    )

    if dim % 2:
        emb = F.pad(
            emb,
            (0, 1),
        )

    return emb


class FiLMResBlock(nn.Module):
    def __init__(
        self,
        channels,
        cond_dim,
    ):
        super().__init__()

        self.n1 = nn.GroupNorm(
            8,
            channels,
        )

        self.c1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.n2 = nn.GroupNorm(
            8,
            channels,
        )

        self.c2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.condition = nn.Linear(
            cond_dim,
            channels * 2,
        )

    def forward(self, x, cond):
        h = self.n1(x)

        scale, shift = (
            self.condition(cond)
            .chunk(2, dim=1)
        )

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = (
            h * (1 + scale)
            + shift
        )

        h = self.c1(
            F.silu(h)
        )

        h = self.c2(
            F.silu(
                self.n2(h)
            )
        )

        return x + h


class LatentDiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        cond_dim = (
            TIME_EMBED
            + CLASS_EMBED
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(
                TIME_EMBED,
                TIME_EMBED,
            ),
            nn.SiLU(),
            nn.Linear(
                TIME_EMBED,
                TIME_EMBED,
            ),
        )

        self.class_emb = nn.Embedding(
            NUM_CLASSES,
            CLASS_EMBED,
        )

        self.input = nn.Conv2d(
            LATENT_CHANNELS,
            BASE_CHANNELS,
            3,
            padding=1,
        )

        self.b1 = FiLMResBlock(
            BASE_CHANNELS,
            cond_dim,
        )

        self.b2 = FiLMResBlock(
            BASE_CHANNELS,
            cond_dim,
        )

        self.mid = nn.Conv2d(
            BASE_CHANNELS,
            BASE_CHANNELS * 2,
            3,
            stride=2,
            padding=1,
        )

        self.b3 = FiLMResBlock(
            BASE_CHANNELS * 2,
            cond_dim,
        )

        self.b4 = FiLMResBlock(
            BASE_CHANNELS * 2,
            cond_dim,
        )

        self.up = nn.ConvTranspose2d(
            BASE_CHANNELS * 2,
            BASE_CHANNELS,
            4,
            stride=2,
            padding=1,
        )

        self.b5 = FiLMResBlock(
            BASE_CHANNELS,
            cond_dim,
        )

        self.output = nn.Sequential(
            nn.GroupNorm(
                8,
                BASE_CHANNELS,
            ),
            nn.SiLU(),
            nn.Conv2d(
                BASE_CHANNELS,
                LATENT_CHANNELS,
                3,
                padding=1,
            ),
        )

    def forward(
        self,
        x,
        t,
        labels,
    ):
        te = self.time_mlp(
            sinusoidal_embedding(
                t,
                TIME_EMBED,
            )
        )

        ce = self.class_emb(
            labels
        )

        cond = torch.cat(
            [te, ce],
            dim=1,
        )

        h = self.input(x)
        h = self.b1(h, cond)
        h = self.b2(h, cond)

        skip = h

        h = F.silu(
            self.mid(h)
        )

        h = self.b3(h, cond)
        h = self.b4(h, cond)

        h = F.silu(
            self.up(h)
        )

        h = h + skip
        h = self.b5(h, cond)

        return self.output(h)


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

def cosine_schedule(timesteps):
    s = 0.008

    x = torch.linspace(
        0,
        timesteps,
        timesteps + 1,
    )

    alpha_bar = (
        torch.cos(
            ((x / timesteps) + s)
            / (1 + s)
            * math.pi
            * 0.5
        )
        ** 2
    )

    alpha_bar = (
        alpha_bar
        / alpha_bar[0]
    )

    betas = (
        1
        - alpha_bar[1:]
        / alpha_bar[:-1]
    )

    return betas.clamp(
        1e-5,
        0.999,
    )


BETAS = cosine_schedule(
    DIFFUSION_STEPS
).to(DEVICE)

ALPHAS = 1.0 - BETAS

ALPHA_BAR = torch.cumprod(
    ALPHAS,
    dim=0,
)


def q_sample(
    x0,
    t,
    noise,
):
    a = ALPHA_BAR[t].view(
        -1, 1, 1, 1
    )

    return (
        torch.sqrt(a) * x0
        + torch.sqrt(1 - a) * noise
    )


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(
        self,
        model,
        decay=EMA_DECAY,
    ):
        self.decay = decay

        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if (
                k in self.shadow
                and v.dtype.is_floating_point
            ):
                self.shadow[k].mul_(
                    self.decay
                ).add_(
                    v.detach(),
                    alpha=1 - self.decay,
                )

    def apply(self, model):
        state = model.state_dict()

        for k, v in self.shadow.items():
            state[k].copy_(v)

    def state_dict(self):
        return {
            k: v.cpu()
            for k, v in self.shadow.items()
        }

    def load_state_dict(self, state):
        self.shadow = {
            k: v.clone()
            for k, v in state.items()
        }


# ============================================================
# DIFFUSION TRAINING
# ============================================================

def train_diffusion(
    vqvae,
    classes,
):
    latents, labels, mean, std = (
        collect_latents(
            vqvae,
            classes,
        )
    )

    latents = (
        latents - mean.cpu()
    ) / std.cpu()

    dataset = torch.utils.data.TensorDataset(
        latents,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = LatentDiffusionUNet().to(
        DEVICE
    )

    ema = EMA(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    use_amp = DEVICE.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    history = []

    print("\n" + "=" * 78)
    print("STAGE 2 — CONDITIONAL LATENT DIFFUSION")
    print("=" * 78)

    for epoch in range(
        1,
        DIFFUSION_EPOCHS + 1,
    ):
        model.train()

        start = time.time()
        total = 0.0

        for z0, labels_batch in loader:
            z0 = z0.to(DEVICE)
            labels_batch = labels_batch.to(
                DEVICE
            )

            t = torch.randint(
                0,
                DIFFUSION_STEPS,
                (z0.shape[0],),
                device=DEVICE,
            )

            noise = torch.randn_like(z0)

            noisy = q_sample(
                z0,
                t,
                noise,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                predicted = model(
                    noisy,
                    t,
                    labels_batch,
                )

                loss = F.mse_loss(
                    predicted,
                    noise,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            total += loss.item()

        row = {
            "epoch": epoch,
            "loss": total / len(loader),
            "seconds": time.time() - start,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{DIFFUSION_EPOCHS} | "
            f"Loss {row['loss']:.5f} | "
            f"{row['seconds']:.1f}s"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": {
                "latent_channels": LATENT_CHANNELS,
                "latent_size": LATENT_SIZE,
                "num_classes": NUM_CLASSES,
                "diffusion_steps": DIFFUSION_STEPS,
            },
        },
        MODEL_ROOT
        / "diffusion_final.pt",
    )

    with open(
        REPORT_ROOT
        / "diffusion_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    return (
        model,
        ema,
        mean,
        std,
    )


# ============================================================
# DDIM
# ============================================================

@torch.no_grad()
def ddim_sample(
    model,
    labels,
    latent_mean,
    latent_std,
):
    model.eval()

    batch = labels.shape[0]

    x = torch.randn(
        batch,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=DEVICE,
    )

    times = torch.linspace(
        DIFFUSION_STEPS - 1,
        0,
        DDIM_STEPS,
        device=DEVICE,
    ).long()

    for i, tv in enumerate(times):
        t = torch.full(
            (batch,),
            int(tv.item()),
            device=DEVICE,
            dtype=torch.long,
        )

        eps = model(
            x,
            t,
            labels,
        )

        a_t = ALPHA_BAR[t].view(
            -1, 1, 1, 1
        )

        if i + 1 < len(times):
            prev_t = torch.full(
                (batch,),
                int(
                    times[i + 1].item()
                ),
                device=DEVICE,
                dtype=torch.long,
            )

            a_prev = ALPHA_BAR[
                prev_t
            ].view(
                -1, 1, 1, 1
            )
        else:
            a_prev = torch.ones_like(
                a_t
            )

        x0 = (
            x
            - torch.sqrt(1 - a_t)
            * eps
        ) / torch.sqrt(a_t)

        x = (
            torch.sqrt(a_prev)
            * x0
            + torch.sqrt(
                1 - a_prev
            )
            * eps
        )

    return (
        x * latent_std
        + latent_mean
    )


# ============================================================
# GENERATION
# ============================================================

def tensor_to_image(x):
    x = (
        (x + 1) / 2
    ).clamp(0, 1)

    arr = (
        x.detach()
        .cpu()
        .numpy()[0]
        * 255
    ).astype(np.uint8)

    return Image.fromarray(
        arr,
        mode="L",
    )

@torch.no_grad()
def generate(
    vqvae,
    diffusion,
    ema,
    latent_mean,
    latent_std,
    classes,
):
    diffusion.eval()

    # Use EMA weights for generation.
    ema.apply(diffusion)

    print("\n" + "=" * 78)
    print("STAGE 3 — 800 SYNTHETIC MRI / CLASS")
    print("=" * 78)

    summary = {}

    for class_name in classes:
        class_id = CLASS_TO_ID[
            class_name
        ]

        raw_dir = (
            SYNTH_ROOT / class_name
        )

        preview_dir = (
            raw_dir / "preview"
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        start = time.time()
        total = 0

        while total < SYNTHETIC_PER_CLASS:
            n = min(
                GEN_BATCH_SIZE,
                SYNTHETIC_PER_CLASS
                - total,
            )

            labels = torch.full(
                (n,),
                class_id,
                device=DEVICE,
                dtype=torch.long,
            )

            z = ddim_sample(
                diffusion,
                labels,
                latent_mean,
                latent_std,
            )

            images = vqvae.decode(z)

            for j in range(n):
                image = tensor_to_image(
                    images[j:j + 1]
                )

                index = total + j + 1

                image.save(
                    raw_dir
                    / f"synthetic_{index:04d}.png"
                )

                # Preview only. Raw image remains untouched.
                preview = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=1.0,
                        percent=60,
                        threshold=3,
                    )
                )

                preview.save(
                    preview_dir
                    / f"synthetic_{index:04d}.png"
                )

            total += n

            print(
                f"{class_name:15s}: "
                f"{total:4d}/{SYNTHETIC_PER_CLASS}"
            )

        summary[class_name] = {
            "real_count": len(
                image_paths_for_class(
                    class_name
                )
            ),
            "synthetic_count": total,
            "seconds": time.time() - start,
        }

    with open(
        REPORT_ROOT
        / "generation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 78)
    print("MRI TESTING V2")
    print("RESIDUAL VQ-VAE + CONDITIONAL LATENT DIFFUSION")
    print("=" * 78)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Testing root : {TEST_ROOT}")
    print(f"Output root  : {RUN_ROOT}")
    print(f"Device       : {DEVICE}")
    print(
        f"Latent       : "
        f"{LATENT_CHANNELS}x{LATENT_SIZE}x{LATENT_SIZE}"
    )
    print(
        f"Target       : "
        f"{SYNTHETIC_PER_CLASS}/class"
    )

    if not TEST_ROOT.exists():
        raise FileNotFoundError(
            f"Testing folder not found: {TEST_ROOT}"
        )

    classes = (
        PILOT_CLASSES
        if PILOT_MODE
        else CLASS_NAMES
    )

    print_report(classes)

    # --------------------------------------------------------
    # VQ-VAE
    # --------------------------------------------------------
    if TRAIN_VQVAE:
        vqvae = train_vqvae(
            classes
        )
    else:
        checkpoint = torch.load(
            MODEL_ROOT
            / "vqvae_final.pt",
            map_location=DEVICE,
        )

        vqvae = VQVAE().to(DEVICE)

        vqvae.load_state_dict(
            checkpoint["model"]
        )

    # Reconstruction gate is ALWAYS executed.
    reconstruction_test(
        vqvae,
        classes,
        per_class=16,
    )

    # --------------------------------------------------------
    # Diffusion
    # --------------------------------------------------------
    if TRAIN_DIFFUSION:
        (
            diffusion,
            ema,
            latent_mean,
            latent_std,
        ) = train_diffusion(
            vqvae,
            classes,
        )
    else:
        checkpoint = torch.load(
            MODEL_ROOT
            / "diffusion_final.pt",
            map_location=DEVICE,
        )

        diffusion = (
            LatentDiffusionUNet()
            .to(DEVICE)
        )

        diffusion.load_state_dict(
            checkpoint["model"]
        )

        ema = EMA(diffusion)

        ema.load_state_dict(
            checkpoint["ema"]
        )

        stats = torch.load(
            MODEL_ROOT
            / "latent_statistics.pt",
            map_location=DEVICE,
        )

        latent_mean = stats[
            "mean"
        ].to(DEVICE)

        latent_std = stats[
            "std"
        ].to(DEVICE)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    if GENERATE_SYNTHETIC:
        generate(
            vqvae,
            diffusion,
            ema,
            latent_mean,
            latent_std,
            classes,
        )

    config = {
        "image_size": IMAGE_SIZE,
        "latent_channels": LATENT_CHANNELS,
        "latent_size": LATENT_SIZE,
        "codebook_size": CODEBOOK_SIZE,
        "ae_epochs": AE_EPOCHS,
        "diffusion_epochs": DIFFUSION_EPOCHS,
        "synthetic_per_class": SYNTHETIC_PER_CLASS,
        "ddim_steps": DDIM_STEPS,
        "classes": classes,
        "device": str(DEVICE),
        "testing_only": True,
        "training_data_modified": False,
        "previous_outputs_modified": False,
    }

    with open(
        REPORT_ROOT
        / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("V2 COMPLETE")
    print("=" * 78)
    print(f"Run       : {RUN_ROOT}")
    print(f"Models    : {MODEL_ROOT}")
    print(f"Recon     : {RECON_ROOT}")
    print(f"Synthetic : {SYNTH_ROOT}")
    print(f"Reports   : {REPORT_ROOT}")
    print("\nTraining MRI data: NOT MODIFIED")
    print("Previous outputs: NOT MODIFIED")
    print("Testing source : NOT MODIFIED")
    print("=" * 78)


if __name__ == "__main__":
    main()

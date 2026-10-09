"""
MRI TESTING GAN V6
==================
Purpose
-------
Generate synthetic MRI images using ONLY the MRI Testing dataset.

IMPORTANT
---------
This script never writes to:
    data/imaging/MRI/Training
    data/imaging/MRI/Training_Preprocessed
    outputs/mri
    outputs/mri_testing_gan
    outputs/mri_testing_gan_v3
    outputs/mri_testing_gan_v4
    outputs/mri_testing_gan_v5

V6 focuses on the failure seen in V5:
    - very dark outputs
    - weak image structure
    - unstable adversarial training

Approach
--------
1. Read one testing class at a time.
2. Normalize MRI intensities per image using robust percentiles.
3. Train an autoencoder first.
4. Use the trained decoder as the GAN generator.
5. Use latent codes learned from real testing images instead of starting
   from an unrelated Gaussian distribution.
6. Add adversarial + feature-matching + intensity-statistics losses.
7. Keep an EMA copy of the generator.
8. Save side-by-side REAL / RECONSTRUCTION / SYNTHETIC previews.
9. Save 100 synthetic images per completed class.

NOTE
----
Because this script learns from the Testing folder, its generated data is
TEST-DERIVED synthetic data. Do not use these generated images as an
independent evaluation of the original Testing set.
"""

from __future__ import annotations

import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

IMAGE_SIZE = 128
AE_IMAGE_SIZE = 128

AE_EPOCHS = 5
GAN_EPOCHS = 8

BATCH_SIZE = 8
NUM_WORKERS = 0

LATENT_DIM = 256

SYNTHETIC_IMAGES_PER_CLASS = 100

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
LEARNING_RATE_AE = 2e-4

BETA1 = 0.5
BETA2 = 0.999

EMA_DECAY = 0.995

# Loss weights.
ADV_WEIGHT = 1.0
FEATURE_WEIGHT = 8.0
PIXEL_WEIGHT = 4.0
MEAN_WEIGHT = 2.0
STD_WEIGHT = 2.0

# Save previews every N epochs.
PREVIEW_EVERY = 2

# Minimum number of images required for a class.
MIN_IMAGES = 10

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# Output is intentionally new and isolated.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TESTING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_gan_v6"
)

PREVIEW_ROOT = OUTPUT_ROOT / "previews"
MODEL_ROOT = OUTPUT_ROOT / "models"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
SYNTHETIC_ROOT = OUTPUT_ROOT / "synthetic_samples"
HISTORY_ROOT = OUTPUT_ROOT / "training_history"
AE_ROOT = OUTPUT_ROOT / "autoencoder"
LATENT_ROOT = OUTPUT_ROOT / "latent_codes"


# ============================================================================
# RANDOM / DEVICE
# ============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PROTECTION
# ============================================================================

def print_protection_message() -> None:
    print()
    print("=" * 76)
    print("MRI TESTING GAN V6")
    print("=" * 76)
    print("Testing dataset ONLY")
    print("New output directory ONLY")
    print()
    print("PROTECTED DATA")
    print("-" * 76)
    print("Training                  : NOT MODIFIED")
    print("Training_Preprocessed     : NOT MODIFIED")
    print("outputs/mri               : NOT MODIFIED")
    print("mri_testing_gan           : NOT MODIFIED")
    print("mri_testing_gan_v3       : NOT MODIFIED")
    print("mri_testing_gan_v4       : NOT MODIFIED")
    print("mri_testing_gan_v5       : NOT MODIFIED")
    print("-" * 76)


def prepare_output_dirs() -> None:
    for path in [
        OUTPUT_ROOT,
        PREVIEW_ROOT,
        MODEL_ROOT,
        CHECKPOINT_ROOT,
        SYNTHETIC_ROOT,
        HISTORY_ROOT,
        AE_ROOT,
        LATENT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_classes() -> list[str]:
    if not TESTING_ROOT.exists():
        raise FileNotFoundError(
            "\nTesting dataset not found:\n"
            f"{TESTING_ROOT}\n\n"
            "Expected structure:\n"
            "data/imaging/MRI/Testing/glioma/*.jpg\n"
            "data/imaging/MRI/Testing/meningioma/*.jpg\n"
            "data/imaging/MRI/Testing/notumor/*.jpg\n"
            "data/imaging/MRI/Testing/pituitary/*.jpg"
        )

    classes = []

    for path in sorted(TESTING_ROOT.iterdir()):
        if not path.is_dir():
            continue

        count = len(image_files(path))

        if count >= MIN_IMAGES:
            classes.append(path.name)

    if not classes:
        raise RuntimeError(
            f"No usable MRI classes found under:\n{TESTING_ROOT}"
        )

    return classes


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    files = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def class_files(class_name: str) -> list[Path]:
    return image_files(TESTING_ROOT / class_name)


# ============================================================================
# IMAGE NORMALIZATION
# ============================================================================

def robust_normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Robust per-image normalization.

    This avoids the V5 failure mode where a small absolute intensity range
    can become a nearly black generated image.
    """
    arr = arr.astype(np.float32)

    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    values = arr[finite]

    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))

    if high - low < 1e-6:
        low = float(values.min())
        high = float(values.max())

    if high - low < 1e-6:
        return np.full_like(arr, 0.5, dtype=np.float32)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)

    # Keep background near black but prevent the entire image from
    # becoming numerically tiny.
    return np.clip(arr, 0.0, 1.0)


def load_grayscale_tensor(
    path: Path,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    image = Image.open(path).convert("L")

    # Preserve aspect ratio and pad instead of stretching.
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (size, size), color=0)

    x = (size - image.width) // 2
    y = (size - image.height) // 2

    canvas.paste(image, (x, y))

    arr = np.asarray(canvas, dtype=np.float32)

    arr = robust_normalize_array(arr)

    tensor = torch.from_numpy(arr).unsqueeze(0)

    return tensor


# ============================================================================
# DATASET
# ============================================================================

class MRIDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        size: int = IMAGE_SIZE,
    ):
        self.files = files
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]

        tensor = load_grayscale_tensor(path, self.size)

        return tensor, str(path)


# ============================================================================
# MODELS
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # Sigmoid is important here:
            # output is guaranteed to remain in [0, 1].
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


class Discriminator(nn.Module):
    """
    Patch discriminator with spectral normalization.

    Returns:
        score
        intermediate feature representation

    Feature matching is used to reduce the tendency toward blank/black
    generated images.
    """

    def sn(self, layer):
        return nn.utils.spectral_norm(layer)

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            self.sn(
                nn.Conv2d(
                    1,
                    32,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            self.sn(
                nn.Conv2d(
                    32,
                    64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block3 = nn.Sequential(
            self.sn(
                nn.Conv2d(
                    64,
                    128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block4 = nn.Sequential(
            self.sn(
                nn.Conv2d(
                    128,
                    256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block5 = nn.Sequential(
            self.sn(
                nn.Conv2d(
                    256,
                    512,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = self.sn(
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )

    def forward(self, x, return_features=False):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)

        score = self.head(f5).view(-1)

        if return_features:
            return score, [f1, f2, f3, f4, f5]

        return score


# ============================================================================
# EMA
# ============================================================================

@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = EMA_DECAY,
) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name in ema_params:
        ema_params[name].mul_(decay).add_(
            model_params[name],
            alpha=1.0 - decay,
        )

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())

    for name in ema_buffers:
        ema_buffers[name].copy_(model_buffers[name])


# ============================================================================
# PREVIEW / IMAGE SAVING
# ============================================================================

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)

    if tensor.ndim == 4:
        tensor = tensor[0]

    if tensor.ndim == 3:
        tensor = tensor[0]

    arr = (tensor.numpy() * 255.0).round().astype(np.uint8)

    return Image.fromarray(arr, mode="L")


def make_grid(
    images: list[Image.Image],
    columns: int = 4,
    cell_size: int = IMAGE_SIZE,
    border: int = 1,
) -> Image.Image:
    if not images:
        return Image.new("L", (columns * cell_size, cell_size), 0)

    rows = math.ceil(len(images) / columns)

    width = columns * cell_size
    height = rows * cell_size

    canvas = Image.new("L", (width, height), 0)

    for index, image in enumerate(images):
        image = image.convert("L").resize(
            (cell_size, cell_size),
            Image.Resampling.LANCZOS,
        )

        x = (index % columns) * cell_size
        y = (index // columns) * cell_size

        canvas.paste(image, (x, y))

    return canvas


@torch.no_grad()
def save_training_preview(
    real_images: torch.Tensor,
    recon_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_path: Path,
) -> None:
    real_list = [
        tensor_to_image(x)
        for x in real_images[:8]
    ]

    recon_list = [
        tensor_to_image(x)
        for x in recon_images[:8]
    ]

    fake_list = [
        tensor_to_image(x)
        for x in fake_images[:8]
    ]

    real_grid = make_grid(real_list, columns=4)
    recon_grid = make_grid(recon_list, columns=4)
    fake_grid = make_grid(fake_list, columns=4)

    final = Image.new(
        "L",
        (
            IMAGE_SIZE * 4,
            IMAGE_SIZE * 6,
        ),
        0,
    )

    final.paste(real_grid, (0, 0))
    final.paste(recon_grid, (0, IMAGE_SIZE * 2))
    final.paste(fake_grid, (0, IMAGE_SIZE * 4))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)


def save_single_image(
    tensor: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_image(tensor)

    # Save PNG with no JPEG compression.
    image.save(output_path, format="PNG")


# ============================================================================
# IMAGE STATISTICS
# ============================================================================

def image_stats(batch: torch.Tensor) -> dict[str, float]:
    flat = batch.detach().float().reshape(batch.size(0), -1)

    mean = flat.mean(dim=1)
    std = flat.std(dim=1)

    q05 = torch.quantile(flat, 0.05, dim=1)
    q50 = torch.quantile(flat, 0.50, dim=1)
    q95 = torch.quantile(flat, 0.95, dim=1)

    return {
        "mean": float(mean.mean().item()),
        "std": float(std.mean().item()),
        "q05": float(q05.mean().item()),
        "median": float(q50.mean().item()),
        "q95": float(q95.mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


# ============================================================================
# AUTOENCODER TRAINING
# ============================================================================

def train_autoencoder(
    class_name: str,
    dataset: MRIDataset,
    device: torch.device,
):
    print()
    print("=" * 72)
    print(f"AUTOENCODER PRETRAINING: {class_name.upper()}")
    print("=" * 72)
    print(f"Images     : {len(dataset)}")
    print(f"Resolution : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Epochs     : {AE_EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Device     : {device}")
    print("=" * 72)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = AutoEncoder(LATENT_DIM).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE_AE,
        betas=(0.5, 0.999),
    )

    # L1 is more stable for medical-image reconstruction than relying only
    # on MSE and helps preserve edges.
    criterion = nn.L1Loss()

    best_loss = float("inf")

    history = []

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        sample_count = 0

        start = time.time()

        preview_real = None
        preview_recon = None

        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            recon, _ = model(images)

            loss = criterion(recon, images)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = images.size(0)

            epoch_loss += float(loss.item()) * batch_size
            sample_count += batch_size

            if preview_real is None:
                preview_real = images.detach().clone()
                preview_recon = recon.detach().clone()

        avg_loss = epoch_loss / max(sample_count, 1)

        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "reconstruction_loss": avg_loss,
                "time_seconds": elapsed,
            }
        )

        print(
            f"AE Epoch [{epoch:02d}/{AE_EPOCHS}] | "
            f"Reconstruction Loss: {avg_loss:.5f} | "
            f"Time: {elapsed:.1f}s"
        )

        if preview_real is not None and (
            epoch == 1
            or epoch == AE_EPOCHS
            or epoch % PREVIEW_EVERY == 0
        ):
            path = (
                AE_ROOT
                / class_name
                / f"epoch_{epoch:04d}.png"
            )

            save_training_preview(
                preview_real,
                preview_recon,
                preview_recon,
                path,
            )

            print(f"AE preview saved: {path}")

        if avg_loss < best_loss:
            best_loss = avg_loss

            model_path = (
                AE_ROOT
                / f"{class_name}_autoencoder_best.pth"
            )

            torch.save(
                {
                    "class_name": class_name,
                    "image_size": IMAGE_SIZE,
                    "latent_dim": LATENT_DIM,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                },
                model_path,
            )

    final_path = (
        AE_ROOT
        / f"{class_name}_autoencoder.pth"
    )

    torch.save(
        {
            "class_name": class_name,
            "image_size": IMAGE_SIZE,
            "latent_dim": LATENT_DIM,
            "model_state_dict": model.state_dict(),
            "loss": best_loss,
        },
        final_path,
    )

    history_path = (
        HISTORY_ROOT
        / f"{class_name}_autoencoder_history.csv"
    )

    save_csv(history, history_path)

    print(f"Autoencoder saved: {final_path}")

    return model


# ============================================================================
# LATENT CODE COLLECTION
# ============================================================================

@torch.no_grad()
def collect_real_latents(
    autoencoder: AutoEncoder,
    dataset: MRIDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    autoencoder.eval()

    all_latents = []
    all_images = []

    for images, _ in loader:
        images = images.to(device)

        z = autoencoder.encoder(images)

        all_latents.append(z.detach().cpu())
        all_images.append(images.detach().cpu())

    latents = torch.cat(all_latents, dim=0)
    images = torch.cat(all_images, dim=0)

    return latents, images


def save_latents(
    class_name: str,
    latents: torch.Tensor,
) -> None:
    path = LATENT_ROOT / f"{class_name}_latents.pt"

    torch.save(
        {
            "class_name": class_name,
            "latent_dim": LATENT_DIM,
            "latents": latents,
        },
        path,
    )


# ============================================================================
# LATENT SAMPLING
# ============================================================================

def build_latent_sampler(
    latents: torch.Tensor,
):
    """
    Samples around real latent codes.

    This is deliberately different from a pure N(0,1) latent prior.
    The decoder therefore starts from a latent region actually learned
    from the MRI testing class.
    """
    mean = latents.mean(dim=0)

    std = latents.std(dim=0).clamp_min(0.02)

    # Keep the perturbation conservative to prevent meaningless images.
    noise_scale = 0.35

    def sample(batch_size: int, device: torch.device):
        indices = torch.randint(
            0,
            latents.size(0),
            (batch_size,),
        )

        base = latents[indices]

        noise = torch.randn_like(base) * std * noise_scale

        z = base + noise

        return z.to(device)

    return sample


# ============================================================================
# GAN TRAINING
# ============================================================================

def feature_matching_loss(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
) -> torch.Tensor:
    loss = torch.zeros(
        (),
        device=real_features[0].device,
    )

    for real, fake in zip(real_features, fake_features):
        real_mean = real.mean(dim=(0, 2, 3))
        fake_mean = fake.mean(dim=(0, 2, 3))

        loss = loss + F.l1_loss(
            fake_mean,
            real_mean.detach(),
        )

    return loss


def intensity_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_flat = real.flatten(1)
    fake_flat = fake.flatten(1)

    real_mean = real_flat.mean(dim=1)
    fake_mean = fake_flat.mean(dim=1)

    real_std = real_flat.std(dim=1)
    fake_std = fake_flat.std(dim=1)

    mean_loss = F.l1_loss(fake_mean, real_mean)
    std_loss = F.l1_loss(fake_std, real_std)

    return mean_loss, std_loss


def gradient_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """
    Encourages similar local edge energy without forcing exact pixels.
    """
    real_dx = real[:, :, :, 1:] - real[:, :, :, :-1]
    fake_dx = fake[:, :, :, 1:] - fake[:, :, :, :-1]

    real_dy = real[:, :, 1:, :] - real[:, :, :-1, :]
    fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]

    return (
        F.l1_loss(fake_dx.abs().mean(dim=(2, 3)), real_dx.abs().mean(dim=(2, 3)))
        + F.l1_loss(fake_dy.abs().mean(dim=(2, 3)), real_dy.abs().mean(dim=(2, 3)))
    )


def train_gan(
    class_name: str,
    autoencoder: AutoEncoder,
    dataset: MRIDataset,
    real_latents: torch.Tensor,
    device: torch.device,
):
    print()
    print("=" * 72)
    print(f"GAN TRAINING V6: {class_name.upper()}")
    print("=" * 72)
    print(f"Real Testing images : {len(dataset)}")
    print(f"Resolution           : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"GAN epochs           : {GAN_EPOCHS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print("Objective             : LSGAN + feature matching")
    print("Latent sampling       : real latent neighbourhood")
    print("Generator             : pretrained AE decoder")
    print("EMA generator         : enabled")
    print("Intensity protection  : enabled")
    print(f"Device               : {device}")
    print("=" * 72)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Generator = pretrained decoder.
    generator = copy.deepcopy(autoencoder.decoder).to(device)

    discriminator = Discriminator().to(device)

    ema_generator = copy.deepcopy(generator).to(device)

    for param in ema_generator.parameters():
        param.requires_grad_(False)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE_G,
        betas=(BETA1, BETA2),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE_D,
        betas=(BETA1, BETA2),
    )

    sampler = build_latent_sampler(real_latents)

    history = []

    best_score = float("inf")

    # Fixed latent preview for stable epoch-to-epoch comparison.
    preview_indices = torch.arange(
        min(16, real_latents.size(0))
    )

    fixed_z = real_latents[preview_indices].to(device)

    for epoch in range(1, GAN_EPOCHS + 1):
        generator.train()
        discriminator.train()

        total_g = 0.0
        total_d = 0.0
        total_adv = 0.0
        total_feature = 0.0
        total_pixel = 0.0
        total_mean = 0.0
        total_std = 0.0

        count = 0

        start = time.time()

        preview_real = None

        for real, _ in loader:
            real = real.to(device)

            batch_size = real.size(0)

            # --------------------------------------------------------------
            # DISCRIMINATOR
            # --------------------------------------------------------------

            z = sampler(batch_size, device)

            with torch.no_grad():
                fake = generator(z)

            real_score, real_features = discriminator(
                real,
                return_features=True,
            )

            fake_score, _ = discriminator(
                fake.detach(),
                return_features=True,
            )

            # LSGAN targets.
            real_target = torch.ones_like(real_score)
            fake_target = torch.zeros_like(fake_score)

            d_real = F.mse_loss(
                real_score,
                real_target,
            )

            d_fake = F.mse_loss(
                fake_score,
                fake_target,
            )

            d_loss = 0.5 * (d_real + d_fake)

            optimizer_d.zero_grad(set_to_none=True)

            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0,
            )

            optimizer_d.step()

            # --------------------------------------------------------------
            # GENERATOR
            # --------------------------------------------------------------

            z = sampler(batch_size, device)

            fake = generator(z)

            fake_score, fake_features = discriminator(
                fake,
                return_features=True,
            )

            # Generator wants discriminator output close to real target.
            adv_loss = F.mse_loss(
                fake_score,
                torch.ones_like(fake_score),
            )

            # Compare feature distributions.
            with torch.no_grad():
                _, real_features_for_g = discriminator(
                    real,
                    return_features=True,
                )

            fm_loss = feature_matching_loss(
                real_features_for_g,
                fake_features,
            )

            # Batch-level intensity matching.
            mean_loss, std_loss = intensity_loss(
                real,
                fake,
            )

            # Edge-energy matching.
            edge_loss = gradient_loss(
                real,
                fake,
            )

            g_loss = (
                ADV_WEIGHT * adv_loss
                + FEATURE_WEIGHT * fm_loss
                + PIXEL_WEIGHT * edge_loss
                + MEAN_WEIGHT * mean_loss
                + STD_WEIGHT * std_loss
            )

            optimizer_g.zero_grad(set_to_none=True)

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=5.0,
            )

            optimizer_g.step()

            update_ema(
                ema_generator,
                generator,
                EMA_DECAY,
            )

            total_g += float(g_loss.item()) * batch_size
            total_d += float(d_loss.item()) * batch_size
            total_adv += float(adv_loss.item()) * batch_size
            total_feature += float(fm_loss.item()) * batch_size
            total_pixel += float(edge_loss.item()) * batch_size
            total_mean += float(mean_loss.item()) * batch_size
            total_std += float(std_loss.item()) * batch_size

            count += batch_size

            if preview_real is None:
                preview_real = real.detach().clone()

        avg_g = total_g / max(count, 1)
        avg_d = total_d / max(count, 1)
        avg_adv = total_adv / max(count, 1)
        avg_feature = total_feature / max(count, 1)
        avg_pixel = total_pixel / max(count, 1)
        avg_mean = total_mean / max(count, 1)
        avg_std = total_std / max(count, 1)

        elapsed = time.time() - start

        # --------------------------------------------------------------
        # PREVIEW
        # --------------------------------------------------------------

        generator.eval()
        ema_generator.eval()

        with torch.no_grad():
            preview_fake = ema_generator(fixed_z)

            if preview_real is None:
                preview_real = torch.zeros(
                    min(16, fixed_z.size(0)),
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    device=device,
                )

            preview_real = preview_real[:16]

            # Reconstruction is from the AE encoder.
            preview_recon, _ = autoencoder(
                preview_real
            )

        if (
            epoch == 1
            or epoch == GAN_EPOCHS
            or epoch % PREVIEW_EVERY == 0
        ):
            preview_path = (
                PREVIEW_ROOT
                / class_name
                / f"epoch_{epoch:04d}.png"
            )

            save_training_preview(
                preview_real,
                preview_recon,
                preview_fake,
                preview_path,
            )

            print(f"Preview saved: {preview_path}")

        # --------------------------------------------------------------
        # QUALITY PROXY
        # --------------------------------------------------------------

        with torch.no_grad():
            real_stat = image_stats(preview_real)
            fake_stat = image_stats(preview_fake)

        stat_distance = (
            abs(real_stat["mean"] - fake_stat["mean"])
            + abs(real_stat["std"] - fake_stat["std"])
            + abs(real_stat["median"] - fake_stat["median"])
        )

        history.append(
            {
                "epoch": epoch,
                "generator_loss": avg_g,
                "discriminator_loss": avg_d,
                "adversarial_loss": avg_adv,
                "feature_matching_loss": avg_feature,
                "edge_loss": avg_pixel,
                "mean_loss": avg_mean,
                "std_loss": avg_std,
                "real_mean": real_stat["mean"],
                "fake_mean": fake_stat["mean"],
                "real_std": real_stat["std"],
                "fake_std": fake_stat["std"],
                "stat_distance": stat_distance,
                "time_seconds": elapsed,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{GAN_EPOCHS}] | "
            f"G: {avg_g:.4f} | "
            f"D: {avg_d:.4f} | "
            f"FM: {avg_feature:.4f} | "
            f"RealMean: {real_stat['mean']:.4f} | "
            f"FakeMean: {fake_stat['mean']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --------------------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------------------

        checkpoint_path = (
            CHECKPOINT_ROOT
            / f"{class_name}_latest.pth"
        )

        torch.save(
            {
                "class_name": class_name,
                "epoch": epoch,
                "image_size": IMAGE_SIZE,
                "latent_dim": LATENT_DIM,
                "generator": generator.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )

        # Smaller statistic distance is used only as a checkpoint proxy,
        # not as a clinical-quality score.
        if stat_distance < best_score:
            best_score = stat_distance

            best_path = (
                CHECKPOINT_ROOT
                / f"{class_name}_best.pth"
            )

            torch.save(
                {
                    "class_name": class_name,
                    "epoch": epoch,
                    "image_size": IMAGE_SIZE,
                    "latent_dim": LATENT_DIM,
                    "ema_generator": ema_generator.state_dict(),
                    "stat_distance": best_score,
                },
                best_path,
            )

    # Save final models.
    generator_path = (
        MODEL_ROOT
        / f"{class_name}_generator.pth"
    )

    ema_path = (
        MODEL_ROOT
        / f"{class_name}_generator_ema.pth"
    )

    discriminator_path = (
        MODEL_ROOT
        / f"{class_name}_discriminator.pth"
    )

    torch.save(
        {
            "class_name": class_name,
            "image_size": IMAGE_SIZE,
            "latent_dim": LATENT_DIM,
            "model_state_dict": generator.state_dict(),
        },
        generator_path,
    )

    torch.save(
        {
            "class_name": class_name,
            "image_size": IMAGE_SIZE,
            "latent_dim": LATENT_DIM,
            "model_state_dict": ema_generator.state_dict(),
        },
        ema_path,
    )

    torch.save(
        {
            "class_name": class_name,
            "image_size": IMAGE_SIZE,
            "model_state_dict": discriminator.state_dict(),
        },
        discriminator_path,
    )

    history_path = (
        HISTORY_ROOT
        / f"{class_name}_gan_history.csv"
    )

    save_csv(history, history_path)

    print()
    print(f"Generator saved    : {generator_path}")
    print(f"EMA Generator saved: {ema_path}")
    print(f"Discriminator saved: {discriminator_path}")
    print(f"History saved      : {history_path}")

    return ema_generator


# ============================================================================
# SYNTHETIC GENERATION
# ============================================================================

@torch.no_grad()
def generate_synthetic_images(
    class_name: str,
    generator: nn.Module,
    real_latents: torch.Tensor,
    device: torch.device,
    count: int = SYNTHETIC_IMAGES_PER_CLASS,
) -> Path:
    print()
    print("-" * 72)
    print(f"GENERATING {count} SYNTHETIC {class_name.upper()} IMAGES")
    print("-" * 72)

    generator.eval()

    output_dir = SYNTHETIC_ROOT / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = build_latent_sampler(real_latents)

    generated = 0

    while generated < count:
        current = min(BATCH_SIZE, count - generated)

        z = sampler(current, device)

        fake = generator(z)

        # Explicit clamp protects against numerical drift.
        fake = fake.clamp(0.0, 1.0)

        for index in range(current):
            output_path = (
                output_dir
                / f"synthetic_{generated + index + 1:04d}.png"
            )

            save_single_image(
                fake[index],
                output_path,
            )

        generated += current

        if generated % 25 == 0 or generated == count:
            print(
                f"Generated: {generated}/{count}"
            )

    print(f"Synthetic folder: {output_dir}")

    return output_dir


# ============================================================================
# CSV
# ============================================================================

def save_csv(
    rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# CLASS TRAINING
# ============================================================================

def train_one_class(
    class_name: str,
    device: torch.device,
) -> dict:
    files = class_files(class_name)

    if len(files) < MIN_IMAGES:
        raise RuntimeError(
            f"Class '{class_name}' has only {len(files)} images."
        )

    dataset = MRIDataset(
        files,
        size=IMAGE_SIZE,
    )

    print()
    print("#" * 76)
    print(f"STARTING V6: {class_name.upper()}")
    print("#" * 76)
    print(f"Testing images : {len(files)}")
    print(f"Resolution     : {IMAGE_SIZE} x {IMAGE_SIZE}")

    # --------------------------------------------------------------
    # SOURCE STATISTICS
    # --------------------------------------------------------------

    print()
    print("Calculating source image statistics...")

    source_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    source_means = []
    source_stds = []

    for images, _ in source_loader:
        source_means.append(
            images.mean(dim=(1, 2, 3))
        )
        source_stds.append(
            images.std(dim=(1, 2, 3))
        )

    source_mean = float(
        torch.cat(source_means).mean().item()
    )

    source_std = float(
        torch.cat(source_stds).mean().item()
    )

    print(f"Source mean intensity : {source_mean:.4f}")
    print(f"Source std intensity  : {source_std:.4f}")

    # --------------------------------------------------------------
    # AUTOENCODER
    # --------------------------------------------------------------

    autoencoder = train_autoencoder(
        class_name,
        dataset,
        device,
    )

    # --------------------------------------------------------------
    # LATENTS
    # --------------------------------------------------------------

    print()
    print("Collecting learned latent codes...")

    autoencoder.eval()

    latents, real_images = collect_real_latents(
        autoencoder,
        dataset,
        device,
    )

    save_latents(
        class_name,
        latents,
    )

    print(
        f"Latent codes collected: {latents.shape[0]} "
        f"x {latents.shape[1]}"
    )

    # --------------------------------------------------------------
    # GAN
    # --------------------------------------------------------------

    ema_generator = train_gan(
        class_name,
        autoencoder,
        dataset,
        latents,
        device,
    )

    # --------------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------------

    synthetic_dir = generate_synthetic_images(
        class_name,
        ema_generator,
        latents,
        device,
        SYNTHETIC_IMAGES_PER_CLASS,
    )

    # --------------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------------

    synthetic_files = image_files(synthetic_dir)

    synthetic_dataset = MRIDataset(
        synthetic_files,
        size=IMAGE_SIZE,
    )

    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    synthetic_means = []
    synthetic_stds = []

    for images, _ in synthetic_loader:
        synthetic_means.append(
            images.mean(dim=(1, 2, 3))
        )
        synthetic_stds.append(
            images.std(dim=(1, 2, 3))
        )

    synthetic_mean = float(
        torch.cat(synthetic_means).mean().item()
    )

    synthetic_std = float(
        torch.cat(synthetic_stds).mean().item()
    )

    print()
    print("-" * 76)
    print(f"{class_name.upper()} V6 COMPLETED")
    print("-" * 76)
    print(f"Real testing images : {len(files)}")
    print(f"Synthetic images    : {len(synthetic_files)}")
    print(f"Real mean           : {source_mean:.4f}")
    print(f"Synthetic mean      : {synthetic_mean:.4f}")
    print(f"Real std            : {source_std:.4f}")
    print(f"Synthetic std       : {synthetic_std:.4f}")
    print(f"Synthetic folder    : {synthetic_dir}")
    print("-" * 76)

    return {
        "class": class_name,
        "real_images": len(files),
        "synthetic_images": len(synthetic_files),
        "resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "ae_epochs": AE_EPOCHS,
        "gan_epochs": GAN_EPOCHS,
        "real_mean": source_mean,
        "synthetic_mean": synthetic_mean,
        "real_std": source_std,
        "synthetic_std": synthetic_std,
    }


# ============================================================================
# MAIN
# ============================================================================

def train_all_classes() -> None:
    seed_everything(SEED)

    device = get_device()

    prepare_output_dirs()
    print_protection_message()

    print()
    print("=" * 76)
    print("MRI TESTING DATASET - GAN V6")
    print("=" * 76)

    print()
    print("INPUT:")
    print(TESTING_ROOT)

    print()
    print("OUTPUT:")
    print(OUTPUT_ROOT)

    print()
    print("DEVICE:")
    print(device)

    print()
    print("RESOLUTION:")
    print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

    print()
    print("V6 FIXES:")
    print("  - Robust per-image intensity normalization")
    print("  - Native 128x128")
    print("  - Autoencoder pretraining")
    print("  - Real-latent neighbourhood sampling")
    print("  - LSGAN objective")
    print("  - Spectral-normalized discriminator")
    print("  - Feature matching")
    print("  - Intensity mean/std matching")
    print("  - Edge-energy matching")
    print("  - EMA generator")
    print("  - Explicit [0,1] output clamp")
    print("=" * 76)

    classes = discover_classes()

    print()
    print("Classes detected:")

    for class_name in classes:
        print(
            f"  {class_name:<15} "
            f"{len(class_files(class_name))} images"
        )

    results = []

    # Default behaviour:
    # process all classes.
    #
    # If you want a pilot run, change:
    # PILOT_CLASSES = ["glioma"]
    #
    # to process only one class.
    PILOT_CLASSES = None

    selected_classes = (
        PILOT_CLASSES
        if PILOT_CLASSES
        else classes
    )

    for class_name in selected_classes:
        try:
            result = train_one_class(
                class_name,
                device,
            )

            results.append(result)

        except KeyboardInterrupt:
            print()
            print("Training interrupted by user.")
            break

        except Exception as exc:
            print()
            print("=" * 76)
            print(
                f"ERROR WHILE PROCESSING CLASS: "
                f"{class_name.upper()}"
            )
            print("=" * 76)
            print(type(exc).__name__, ":", exc)
            print()
            print(
                "Other classes will not be trained after this error."
            )
            raise

    if results:
        summary_path = (
            OUTPUT_ROOT
            / "v6_summary.csv"
        )

        save_csv(
            results,
            summary_path,
        )

        print()
        print("=" * 76)
        print("MRI TESTING GAN V6 FINISHED")
        print("=" * 76)

        print()
        print(
            f"{'class':<15}"
            f"{'real_images':>14}"
            f"{'synthetic':>14}"
            f"{'resolution':>14}"
        )

        for row in results:
            print(
                f"{row['class']:<15}"
                f"{row['real_images']:>14}"
                f"{row['synthetic_images']:>14}"
                f"{row['resolution']:>14}"
            )

        print()
        print(f"V6 output       : {OUTPUT_ROOT}")
        print(f"Synthetic MRI   : {SYNTHETIC_ROOT}")
        print(f"Models          : {MODEL_ROOT}")
        print(f"Previews        : {PREVIEW_ROOT}")
        print(f"Summary         : {summary_path}")
        print("=" * 76)


if __name__ == "__main__":
    train_all_classes()

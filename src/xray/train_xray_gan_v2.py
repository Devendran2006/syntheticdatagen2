"""
X-Ray GAN V2 training script.

Uses:
    src/xray/xray_gan2.py
    src/xray/xray_dataset.py

Writes all V2 artifacts to:
    outputs/xray_gan_v2/

IMPORTANT:
- This script does not load V1 checkpoints.
- Only the train and validation splits are loaded; the test split is never used.
- Synthetic images are research artifacts, not for diagnosis or clinical use.
"""

from pathlib import Path
import sys
import csv
import random
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


# ---------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xray.xray_dataset import XRayDataset, IMAGE_SIZE, CLASSES
from src.xray.xray_gan2 import (
    XRayGenerator,
    XRayDiscriminator,
    LATENT_DIM,
    DEVICE,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BATCH_SIZE = 16
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0001
BETAS = (0.0, 0.9)

DEFAULT_EPOCHS = 10
NUM_WORKERS = 0
SEED = 42
PREVIEW_COUNT_PER_CLASS = 8
CHECKPOINT_INTERVAL = 1
USE_BALANCED_SAMPLER = True

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "xray_gan_v2"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
LOG_DIR = OUTPUT_ROOT / "logs"
HISTORY_FILE = LOG_DIR / "training_history.csv"

for folder in (CHECKPOINT_DIR, PREVIEW_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def make_train_loader():
    dataset = XRayDataset(split="train", image_size=IMAGE_SIZE)

    sampler = None
    shuffle = True

    # Build balanced sampling weights from the known canonical folder layout.
    # This avoids relying on private attributes inside XRayDataset.
    if USE_BALANCED_SAMPLER:
        data_root = PROJECT_ROOT / "data" / "imaging" / "xray" / "chest_xray"
        class_counts = []
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for class_name in CLASSES:
            class_dir = data_root / "train" / class_name
            count = sum(
                1 for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in supported
            )
            class_counts.append(count)

        if len(class_counts) == len(CLASSES) and all(c > 0 for c in class_counts):
            per_class_weight = {
                class_index: 1.0 / count
                for class_index, count in enumerate(class_counts)
            }

            # Dataset order is expected to follow sorted class directories,
            # matching the dataset sanity check. Verify this against labels
            # by reading each item once; this is a one-time CPU preprocessing pass.
            sample_weights = []
            for index in range(len(dataset)):
                _, label = dataset[index]
                label_value = int(label.item()) if torch.is_tensor(label) else int(label)
                sample_weights.append(per_class_weight.get(label_value, 1.0))

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            print("Balanced sampler unavailable; using shuffled training data.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return dataset, loader


def make_val_loader():
    dataset = XRayDataset(split="val", image_size=IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return dataset, loader


# ---------------------------------------------------------------------
# Fixed preview inputs (same noise/labels every epoch for fair comparison)
# ---------------------------------------------------------------------

def make_fixed_preview_inputs():
    total = len(CLASSES) * PREVIEW_COUNT_PER_CLASS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 100)

    noise = torch.randn(total, LATENT_DIM, generator=generator)
    labels = []
    for class_index in range(len(CLASSES)):
        labels.extend([class_index] * PREVIEW_COUNT_PER_CLASS)

    return noise.to(DEVICE), torch.tensor(labels, dtype=torch.long, device=DEVICE)


def save_preview(generator, epoch, fixed_noise, fixed_labels):
    from torchvision.utils import save_image

    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise, fixed_labels)
        images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)

    output_path = PREVIEW_DIR / f"epoch_{epoch:04d}.png"
    save_image(images.cpu(), output_path, nrow=PREVIEW_COUNT_PER_CLASS, padding=3)

    if was_training:
        generator.train()
    return output_path


def save_real_preview(train_loader):
    from torchvision.utils import save_image

    output_path = PREVIEW_DIR / "real_training_sample.png"
    if output_path.exists():
        return output_path

    images, _ = next(iter(train_loader))
    images = ((images[:16] + 1.0) / 2.0).clamp(0.0, 1.0)
    save_image(images.cpu(), output_path, nrow=8, padding=3)
    return output_path


# ---------------------------------------------------------------------
# Simple validation diagnostics (not a clinical quality metric)
# ---------------------------------------------------------------------

def validate_generator(generator, fixed_noise, fixed_labels):
    was_training = generator.training
    generator.eval()

    with torch.no_grad():
        images = generator(fixed_noise, fixed_labels)
        mean_value = images.mean().item()
        std_value = images.std().item()
        min_value = images.min().item()
        max_value = images.max().item()

        # Per-image standard deviation is a coarse indicator only; it does not
        # establish visual quality or diversity.
        per_image_std = images.flatten(1).std(dim=1)
        mean_image_std = per_image_std.mean().item()

    if was_training:
        generator.train()

    return {
        "val_mean": mean_value,
        "val_std": std_value,
        "val_min": min_value,
        "val_max": max_value,
        "mean_per_image_std": mean_image_std,
    }


# ---------------------------------------------------------------------
# History / checkpoints
# ---------------------------------------------------------------------

FIELDNAMES = [
    "epoch",
    "generator_loss",
    "discriminator_loss",
    "real_loss",
    "fake_loss",
    "val_mean",
    "val_std",
    "val_min",
    "val_max",
    "mean_per_image_std",
]


def write_history(history):
    with HISTORY_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    epoch, generator, discriminator, optimizer_g, optimizer_d,
    history, fixed_noise, fixed_labels
):
    checkpoint = {
        "model_version": "xray_gan_v2",
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "history": history,
        "fixed_noise": fixed_noise.detach().cpu(),
        "fixed_labels": fixed_labels.detach().cpu(),
        "config": {
            "image_size": IMAGE_SIZE,
            "latent_dim": LATENT_DIM,
            "classes": list(CLASSES),
            "batch_size": BATCH_SIZE,
            "lr_g": LEARNING_RATE_G,
            "lr_d": LEARNING_RATE_D,
        },
    }

    latest_path = CHECKPOINT_DIR / "latest.pt"
    epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt"

    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, epoch_path)

    print(f"Checkpoint saved : {latest_path}")
    print(f"Epoch checkpoint : {epoch_path}")


def load_checkpoint(path, generator, discriminator, optimizer_g, optimizer_d):
    # weights_only=False is explicit because this checkpoint contains training
    # metadata as well as model tensors. Only load checkpoints you created.
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(path, map_location=DEVICE)

    if checkpoint.get("model_version") != "xray_gan_v2":
        raise RuntimeError(
            "This is not an X-Ray GAN V2 checkpoint. "
            "Do not load a V1 checkpoint into V2."
        )

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    history = checkpoint.get("history", [])
    start_epoch = int(checkpoint["epoch"]) + 1
    print(f"Resumed V2 checkpoint from epoch {checkpoint['epoch']}.")
    print(f"Next epoch: {start_epoch}")
    return start_epoch, history


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(epochs, resume=False):
    if epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    set_seed()
    print("=" * 70)
    print("X-RAY CONDITIONAL GAN V2 TRAINING")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes       : {CLASSES}")
    print(f"Latent dim    : {LATENT_DIM}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Target epochs : {epochs}")
    print(f"Balanced sampler: {USE_BALANCED_SAMPLER}")
    print(f"Output root   : {OUTPUT_ROOT}")
    print("=" * 70)

    train_dataset, train_loader = make_train_loader()
    val_dataset, val_loader = make_val_loader()

    if len(train_loader) == 0:
        raise RuntimeError(
            "Training loader has zero batches. Check dataset size and BATCH_SIZE."
        )

    print(f"Training images   : {len(train_dataset)}")
    print(f"Training batches  : {len(train_loader)}")
    print(f"Validation images : {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")
    print("Test split        : NOT LOADED")

    save_real_preview(train_loader)

    generator = XRayGenerator().to(DEVICE)
    discriminator = XRayDiscriminator().to(DEVICE)

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE_G, betas=BETAS
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE_D, betas=BETAS
    )
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise, fixed_labels = make_fixed_preview_inputs()
    start_epoch = 1
    history = []

    latest_path = CHECKPOINT_DIR / "latest.pt"
    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"No V2 checkpoint found at {latest_path}. "
                "Run without --resume for a fresh V2 experiment."
            )
        start_epoch, history = load_checkpoint(
            latest_path, generator, discriminator, optimizer_g, optimizer_d
        )

    if start_epoch > epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"target is {epochs}. Nothing to train."
        )
        return

    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        discriminator.train()

        g_losses, d_losses, real_losses, fake_losses = [], [], [], []
        print(f"\n{'=' * 70}\nEPOCH {epoch}/{epochs}\n{'=' * 70}")

        for batch_index, (real_images, labels) in enumerate(train_loader, start=1):
            real_images = real_images.to(DEVICE)
            labels = labels.to(DEVICE).long()
            batch_size = real_images.size(0)

            # Train discriminator
            optimizer_d.zero_grad(set_to_none=True)

            real_logits = discriminator(real_images, labels)
            real_targets = torch.full_like(real_logits, 0.9)  # mild label smoothing
            real_loss = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = generator(noise, labels)

            fake_logits = discriminator(fake_images.detach(), labels)
            fake_targets = torch.zeros_like(fake_logits)
            fake_loss = criterion(fake_logits, fake_targets)

            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            generated = generator(noise, labels)
            generated_logits = discriminator(generated, labels)
            g_targets = torch.ones_like(generated_logits)
            g_loss = criterion(generated_logits, g_targets)

            g_loss.backward()
            optimizer_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

            if batch_index % 50 == 0 or batch_index == len(train_loader):
                print(
                    f"Batch {batch_index:>3}/{len(train_loader)} | "
                    f"G={g_loss.item():.4f} | D={d_loss.item():.4f}"
                )

        diagnostics = validate_generator(generator, fixed_noise, fixed_labels)

        row = {
            "epoch": epoch,
            "generator_loss": float(np.mean(g_losses)),
            "discriminator_loss": float(np.mean(d_losses)),
            "real_loss": float(np.mean(real_losses)),
            "fake_loss": float(np.mean(fake_losses)),
            **diagnostics,
        }
        history.append(row)
        write_history(history)

        print("\nEpoch summary")
        print(f"Generator loss       : {row['generator_loss']:.6f}")
        print(f"Discriminator loss   : {row['discriminator_loss']:.6f}")
        print(f"Real loss             : {row['real_loss']:.6f}")
        print(f"Fake loss             : {row['fake_loss']:.6f}")
        print(f"Generated pixel range : {row['val_min']:.4f} to {row['val_max']:.4f}")
        print(f"Mean image std        : {row['mean_per_image_std']:.6f}")

        preview_path = save_preview(
            generator, epoch, fixed_noise, fixed_labels
        )
        print(f"Preview saved         : {preview_path}")

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                epoch, generator, discriminator, optimizer_g, optimizer_d,
                history, fixed_noise, fixed_labels
            )

    print("\n" + "=" * 70)
    print("X-RAY GAN V2 TRAINING FINISHED")
    print(f"History     : {HISTORY_FILE}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"Previews    : {PREVIEW_DIR}")
    print("=" * 70)


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

def smoke_test():
    set_seed()
    print("=" * 70)
    print("X-RAY GAN V2 TRAINING SMOKE TEST")
    print("=" * 70)

    train_dataset = XRayDataset(split="train", image_size=IMAGE_SIZE)
    val_dataset = XRayDataset(split="val", image_size=IMAGE_SIZE)

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    real_images, labels = next(iter(loader))
    real_images = real_images.to(DEVICE)
    labels = labels.to(DEVICE).long()

    generator = XRayGenerator().to(DEVICE)
    discriminator = XRayDiscriminator().to(DEVICE)

    noise = torch.randn(real_images.size(0), LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise, labels)
        scores = discriminator(fake_images, labels)

    print(f"Device              : {DEVICE}")
    print(f"Train dataset size  : {len(train_dataset)}")
    print(f"Validation size     : {len(val_dataset)}")
    print(f"Real batch shape    : {tuple(real_images.shape)}")
    print(f"Generated shape     : {tuple(fake_images.shape)}")
    print(f"Generated range     : {fake_images.min().item():.4f} to {fake_images.max().item():.4f}")
    print(f"Discriminator shape : {tuple(scores.shape)}")
    print("No training performed.")
    print("No checkpoints changed.")
    print("Test split not loaded.")
    print("=" * 70)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train X-Ray GAN V2")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        train(epochs=args.epochs, resume=args.resume)

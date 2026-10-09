from pathlib import Path
import sys
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.xray.xray_dataset import XRayDataset, IMAGE_SIZE, CLASSES
from src.xray.xray_gan import (
    XRayGenerator,
    XRayDiscriminator,
    LATENT_DIM,
    DEVICE,
)

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)

DEFAULT_EPOCHS = 50
NUM_WORKERS = 0

PREVIEW_COUNT_PER_CLASS = 8
SEED = 42

# Save checkpoint after EVERY epoch
CHECKPOINT_INTERVAL = 1

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "xray_gan_training"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
LOG_DIR = OUTPUT_ROOT / "logs"

HISTORY_FILE = LOG_DIR / "training_history.csv"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders():

    train_dataset = XRayDataset(
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = XRayDataset(
        split="val",
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# PREVIEW SAVING
# ============================================================

def save_preview(generator, epoch):

    generator.eval()

    total = len(CLASSES) * PREVIEW_COUNT_PER_CLASS

    noise = torch.randn(
        total,
        LATENT_DIM,
        device=DEVICE,
    )

    labels = []

    for class_index in range(len(CLASSES)):
        labels.extend(
            [class_index] * PREVIEW_COUNT_PER_CLASS
        )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        fake_images = generator(
            noise,
            labels,
        )

    # Convert [-1, 1] -> [0, 1]
    fake_images = (fake_images + 1.0) / 2.0
    fake_images = fake_images.clamp(0.0, 1.0)

    # Save using torchvision
    from torchvision.utils import save_image

    output_file = PREVIEW_DIR / f"epoch_{epoch:04d}.png"

    save_image(
        fake_images,
        output_file,
        nrow=PREVIEW_COUNT_PER_CLASS,
    )

    generator.train()

    return output_file


# ============================================================
# SAVE REAL TRAINING SAMPLE
# ============================================================

def save_real_training_sample(train_loader):

    output_file = PREVIEW_DIR / "real_training_sample.png"

    if output_file.exists():
        return

    from torchvision.utils import save_image

    images, labels = next(iter(train_loader))

    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    save_image(
        images[:16],
        output_file,
        nrow=8,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_generator(generator):

    generator.eval()

    total = len(CLASSES)

    noise = torch.randn(
        total,
        LATENT_DIM,
        device=DEVICE,
    )

    labels = torch.arange(
        total,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        fake_images = generator(
            noise,
            labels,
        )

    mean_value = fake_images.mean().item()
    std_value = fake_images.std().item()

    generator.train()

    return mean_value, std_value


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    epoch,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    history,
):

    checkpoint = {
        "epoch": epoch,

        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "generator_optimizer_state_dict":
            generator_optimizer.state_dict(),

        "discriminator_optimizer_state_dict":
            discriminator_optimizer.state_dict(),

        "history": history,
    }

    latest_file = CHECKPOINT_DIR / "latest.pt"

    epoch_file = CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt"

    torch.save(
        checkpoint,
        latest_file,
    )

    torch.save(
        checkpoint,
        epoch_file,
    )

    print(f"Checkpoint saved : {latest_file}")
    print(f"Epoch checkpoint : {epoch_file}")


# ============================================================
# CHECKPOINT LOAD
# ============================================================

def load_checkpoint(
    checkpoint_file,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"]
    )

    generator_optimizer.load_state_dict(
        checkpoint["generator_optimizer_state_dict"]
    )

    discriminator_optimizer.load_state_dict(
        checkpoint["discriminator_optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    history = checkpoint.get(
        "history",
        [],
    )

    print()
    print("=" * 70)
    print("CHECKPOINT RESUMED")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_file}")
    print(f"Previous epoch : {checkpoint['epoch']}")
    print(f"Next epoch : {start_epoch}")
    print("=" * 70)

    return start_epoch, history


# ============================================================
# CSV LOGGING
# ============================================================

def write_history(history):

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "generator_loss",
                "discriminator_loss",
                "real_loss",
                "fake_loss",
                "val_mean",
                "val_std",
            ],
        )

        writer.writeheader()

        writer.writerows(history)


# ============================================================
# TRAINING
# ============================================================

def train(epochs, resume=False):

    set_seed()

    print("=" * 70)
    print("X-RAY CONDITIONAL GAN TRAINING")
    print("=" * 70)

    print(f"Device        : {DEVICE}")
    print(f"Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes       : {CLASSES}")
    print(f"Latent dim    : {LATENT_DIM}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Target epochs : {epochs}")
    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_loader, val_loader = create_dataloaders()

    print()
    print(f"Training batches   : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")

    save_real_training_sample(train_loader)

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    generator = XRayGenerator().to(DEVICE)

    discriminator = XRayDiscriminator().to(DEVICE)

    # --------------------------------------------------------
    # OPTIMIZERS
    # --------------------------------------------------------

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = torch.nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 1
    history = []

    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"

    if resume and latest_checkpoint.exists():

        start_epoch, history = load_checkpoint(
            latest_checkpoint,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
        )

    elif resume:

        print()
        print("No checkpoint found.")
        print("Starting training from epoch 1.")

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(start_epoch, epochs + 1):

        generator.train()
        discriminator.train()

        generator_losses = []
        discriminator_losses = []
        real_losses = []
        fake_losses = []

        print()
        print("=" * 70)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 70)

        for real_images, labels in train_loader:

            real_images = real_images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            batch_size = real_images.size(0)

            # =================================================
            # TRAIN DISCRIMINATOR
            # =================================================

            discriminator_optimizer.zero_grad()

            real_output = discriminator(
                real_images,
                labels,
            )

            real_targets = torch.ones_like(
                real_output
            )

            real_loss = criterion(
                real_output,
                real_targets,
            )

            noise = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE,
            )

            fake_images = generator(
                noise,
                labels,
            )

            fake_output = discriminator(
                fake_images.detach(),
                labels,
            )

            fake_targets = torch.zeros_like(
                fake_output
            )

            fake_loss = criterion(
                fake_output,
                fake_targets,
            )

            discriminator_loss = (
                real_loss + fake_loss
            ) / 2.0

            discriminator_loss.backward()

            discriminator_optimizer.step()

            # =================================================
            # TRAIN GENERATOR
            # =================================================

            generator_optimizer.zero_grad()

            noise = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE,
            )

            fake_images = generator(
                noise,
                labels,
            )

            fake_output = discriminator(
                fake_images,
                labels,
            )

            generator_targets = torch.ones_like(
                fake_output
            )

            generator_loss = criterion(
                fake_output,
                generator_targets,
            )

            generator_loss.backward()

            generator_optimizer.step()

            # =================================================
            # RECORD LOSSES
            # =================================================

            generator_losses.append(
                generator_loss.item()
            )

            discriminator_losses.append(
                discriminator_loss.item()
            )

            real_losses.append(
                real_loss.item()
            )

            fake_losses.append(
                fake_loss.item()
            )

        # ----------------------------------------------------
        # EPOCH METRICS
        # ----------------------------------------------------

        generator_loss_value = float(
            np.mean(generator_losses)
        )

        discriminator_loss_value = float(
            np.mean(discriminator_losses)
        )

        real_loss_value = float(
            np.mean(real_losses)
        )

        fake_loss_value = float(
            np.mean(fake_losses)
        )

        val_mean, val_std = validate_generator(
            generator
        )

        row = {
            "epoch": epoch,
            "generator_loss": generator_loss_value,
            "discriminator_loss": discriminator_loss_value,
            "real_loss": real_loss_value,
            "fake_loss": fake_loss_value,
            "val_mean": val_mean,
            "val_std": val_std,
        }

        history.append(row)

        write_history(history)

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(f"Generator loss     : {generator_loss_value:.6f}")
        print(f"Discriminator loss : {discriminator_loss_value:.6f}")
        print(f"Real loss          : {real_loss_value:.6f}")
        print(f"Fake loss          : {fake_loss_value:.6f}")
        print(f"Validation mean    : {val_mean:.6f}")
        print(f"Validation std     : {val_std:.6f}")

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        preview_file = save_preview(
            generator,
            epoch,
        )

        print(f"Preview saved      : {preview_file}")

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if epoch % CHECKPOINT_INTERVAL == 0:

            save_checkpoint(
                epoch,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                history,
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"History     : {HISTORY_FILE}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"Previews    : {PREVIEW_DIR}")
    print("=" * 70)


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test():

    set_seed()

    print("=" * 70)
    print("X-RAY GAN TRAINING SMOKE TEST")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Image size : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes    : {CLASSES}")
    print("=" * 70)

    train_loader, val_loader = create_dataloaders()

    real_images, labels = next(
        iter(train_loader)
    )

    print()
    print("Training batch")
    print(f"Images shape : {tuple(real_images.shape)}")
    print(f"Labels shape : {tuple(labels.shape)}")
    print(
        f"Pixel range  : "
        f"{real_images.min().item():.4f} "
        f"to "
        f"{real_images.max().item():.4f}"
    )

    generator = XRayGenerator().to(DEVICE)

    discriminator = XRayDiscriminator().to(DEVICE)

    real_images = real_images.to(DEVICE)
    labels = labels.to(DEVICE)

    noise = torch.randn(
        real_images.size(0),
        LATENT_DIM,
        device=DEVICE,
    )

    fake_images = generator(
        noise,
        labels,
    )

    discriminator_output = discriminator(
        fake_images,
        labels,
    )

    print()
    print("Generator")
    print(f"Output shape : {tuple(fake_images.shape)}")
    print(
        f"Output range : "
        f"{fake_images.min().item():.6f} "
        f"to "
        f"{fake_images.max().item():.6f}"
    )

    print()
    print("Discriminator")
    print(
        f"Output shape : "
        f"{tuple(discriminator_output.shape)}"
    )

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("No training was performed.")
    print("No checkpoint was modified.")
    print("No test images were loaded.")


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Train conditional X-ray GAN"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run model/data smoke test only",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/xray_gan_training/checkpoints/latest.pt",
    )

    args = parser.parse_args()

    if args.smoke_test:

        smoke_test()

    else:

        train(
            epochs=args.epochs,
            resume=args.resume,
        )
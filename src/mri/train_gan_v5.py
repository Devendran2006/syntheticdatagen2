"""
MRI GAN V5
-----------
Goal:
- Reduce V4's strong intra-class mode collapse.
- Preserve the good image-distribution/sharpness behavior from V4.
- Improve class conditioning and latent sensitivity.
- Use discriminator-side augmentation for robustness.
- Save frequent checkpoints/previews so training can be stopped early.

IMPORTANT:
- GAN training uses ONLY MRI Training_Preprocessed.
- Real MRI Testing is NEVER used for GAN training.
- This is an engineering synthetic-data experiment; it does not establish
  medical/clinical validity.

Run from the project root:
    python src/mri/train_gan_v5.py
"""

import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 128
CHANNELS = 1
NUM_CLASSES = 4
LATENT_DIM = 128

BATCH_SIZE = 16

TOTAL_EPOCHS = 12

# Save checkpoints often so we do not waste CPU time.
SAVE_EVERY = 1
PREVIEW_EVERY = 1

# Learning rates
G_LR = 0.00015
D_LR = 0.00020

BETAS = (0.0, 0.99)

# Regularization / auxiliary objectives
MODE_SEEKING_WEIGHT = 2.5
MODE_SEEKING_EPS = 1e-5

FEATURE_MATCH_WEIGHT = 1.0

# EMA helps produce smoother evaluation samples.
EMA_DECAY = 0.995

# Discriminator augmentation probability.
AUGMENT_PROB = 0.55

SEED = 42

# CPU settings
CPU_THREADS = min(4, os.cpu_count() or 4)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Training_Preprocessed"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mri" / "v5"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PREVIEW_DIR = OUTPUT_ROOT / "generated_samples"
SAMPLE_DIR = OUTPUT_ROOT / "samples"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

if DEVICE.type == "cpu":
    torch.set_num_threads(CPU_THREADS)


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_DIR}\n"
        "Make sure prepare_mri.py has already been run."
    )

dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform,
)

if len(dataset.classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, found {len(dataset.classes)}: "
        f"{dataset.classes}"
    )

class_names = dataset.classes

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)


# ============================================================
# GENERATOR
# ============================================================

class NoiseInjection(nn.Module):
    """
    Per-feature-channel learnable noise strength.

    The noise is deliberately small. Its purpose is to prevent every
    latent vector from collapsing to the same texture/template.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(
            x.size(0),
            1,
            x.size(2),
            x.size(3),
            device=x.device,
        )
        return x + self.weight * noise


class GeneratorV5(nn.Module):
    def __init__(
        self,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        image_channels=CHANNELS,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim,
        )

        # Small mapping network improves the separation of latent directions.
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.block1 = self._up_block(512, 256)
        self.block2 = self._up_block(256, 128)
        self.block3 = self._up_block(128, 64)
        self.block4 = self._up_block(64, 32)
        self.block5 = self._up_block(32, 16)

        self.final = nn.Sequential(
            nn.Conv2d(16, image_channels, 3, 1, 1),
            nn.Tanh(),
        )

        self._initialize()

    @staticmethod
    def _up_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            NoiseInjection(out_channels),
        )

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, z, labels):
        label_vector = self.label_embedding(labels)

        z_mapped = self.mapping(z)

        x = torch.cat([z_mapped, label_vector], dim=1)

        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.block1(x)   # 8
        x = self.block2(x)   # 16
        x = self.block3(x)   # 32
        x = self.block4(x)   # 64
        x = self.block5(x)   # 128

        return self.final(x)


# ============================================================
# DISCRIMINATOR
# ============================================================

class DiscriminatorV5(nn.Module):
    """
    Conditional projection discriminator.

    Returns:
        logits
        feature representation

    The feature representation is used for a lightweight feature-matching
    objective to reduce unstable generator behavior.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        image_channels=CHANNELS,
    ):
        super().__init__()

        self.features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(image_channels, 64, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(64, 128, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(128, 256, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(256, 512, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(512, 512, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(512, 512, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feature_dim = 512

        self.fc = spectral_norm(
            nn.Linear(512, 1)
        )

        self.embedding = nn.Embedding(
            num_classes,
            512,
        )

    def forward(self, x, labels):
        x = self.features(x)

        # Global average pooling.
        features = x.mean(dim=(2, 3))

        logits = self.fc(features)

        class_vector = self.embedding(labels)

        projection = (features * class_vector).sum(
            dim=1,
            keepdim=True,
        )

        logits = logits + projection

        return logits.squeeze(1), features


# ============================================================
# DISCRIMINATOR-SIDE AUGMENTATION
# ============================================================

def discriminator_augment(x, probability=AUGMENT_PROB):
    """
    Apply the same type of mild augmentation to both real and fake images
    before discriminator evaluation.

    We intentionally avoid aggressive transformations because these are
    medical images and the purpose is GAN stabilization, not synthetic
    medical manipulation.
    """

    if probability <= 0:
        return x

    batch = x.size(0)
    device = x.device

    apply_mask = (
        torch.rand(batch, device=device) < probability
    )

    if not apply_mask.any():
        return x

    out = x.clone()

    selected = torch.where(apply_mask)[0]

    selected_x = out[selected]

    # Small translation.
    max_shift = 4

    shifts_y = torch.randint(
        -max_shift,
        max_shift + 1,
        (len(selected),),
        device=device,
    )

    shifts_x = torch.randint(
        -max_shift,
        max_shift + 1,
        (len(selected),),
        device=device,
    )

    translated = torch.empty_like(selected_x)

    for i in range(len(selected)):
        translated[i] = torch.roll(
            selected_x[i],
            shifts=(
                int(shifts_y[i].item()),
                int(shifts_x[i].item()),
            ),
            dims=(1, 2),
        )

    # Mild contrast variation.
    contrast = (
        0.90
        + 0.20 * torch.rand(
            len(selected),
            1,
            1,
            1,
            device=device,
        )
    )

    translated = translated * contrast

    # Mild brightness variation.
    brightness = (
        -0.05
        + 0.10 * torch.rand(
            len(selected),
            1,
            1,
            1,
            device=device,
        )
    )

    translated = translated + brightness

    out[selected] = translated.clamp(-1.0, 1.0)

    return out


# ============================================================
# EMA
# ============================================================

def update_ema(ema_model, model, decay=EMA_DECAY):
    with torch.no_grad():
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


# ============================================================
# MODE SEEKING LOSS
# ============================================================

def mode_seeking_loss(
    generator,
    z1,
    z2,
    labels,
):
    """
    Encourage different latent vectors to produce different images.

    This directly targets V4's visual failure:
    many latent vectors were producing nearly identical templates.
    """

    img1 = generator(z1, labels)
    img2 = generator(z2, labels)

    image_difference = (
        img1 - img2
    ).abs().mean(
        dim=(1, 2, 3)
    )

    latent_difference = (
        z1 - z2
    ).abs().mean(
        dim=1
    )

    # We want image difference to remain meaningful relative to
    # latent difference.
    ratio = image_difference / (
        latent_difference + MODE_SEEKING_EPS
    )

    loss = (
        1.0 / (ratio + MODE_SEEKING_EPS)
    ).mean()

    return loss, img1, img2, ratio.detach().mean().item()


# ============================================================
# UTILS
# ============================================================

def save_class_previews(
    generator,
    epoch,
    device,
    num_images=16,
):
    generator.eval()

    with torch.no_grad():
        for class_index, class_name in enumerate(class_names):
            labels = torch.full(
                (num_images,),
                class_index,
                dtype=torch.long,
                device=device,
            )

            # Fixed but diverse latent vectors for epoch-to-epoch
            # visual comparison.
            generator_seed = (
                SEED
                + epoch * 100
                + class_index
            )

            gen = torch.Generator(device=device)
            gen.manual_seed(generator_seed)

            z = torch.randn(
                num_images,
                LATENT_DIM,
                generator=gen,
                device=device,
            )

            fake = generator(z, labels)

            path = (
                PREVIEW_DIR
                / f"epoch_{epoch:03d}_{class_name}.png"
            )

            save_image(
                fake,
                str(path),
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )

    generator.train()


def save_all_class_grid(
    generator,
    epoch,
    device,
    num_images=8,
):
    generator.eval()

    all_images = []

    with torch.no_grad():
        for class_index in range(NUM_CLASSES):
            labels = torch.full(
                (num_images,),
                class_index,
                dtype=torch.long,
                device=device,
            )

            gen = torch.Generator(device=device)
            gen.manual_seed(
                SEED + epoch * 1000 + class_index
            )

            z = torch.randn(
                num_images,
                LATENT_DIM,
                generator=gen,
                device=device,
            )

            fake = generator(z, labels)
            all_images.append(fake.cpu())

    grid = make_grid(
        torch.cat(all_images, dim=0),
        nrow=num_images,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )

    path = PREVIEW_DIR / f"epoch_{epoch:03d}_all_classes.png"

    save_image(grid, str(path))

    generator.train()


def latest_checkpoint():
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("mri_gan_v5_epoch_*.pth")
    )

    if not checkpoints:
        return None

    return checkpoints[-1]


def save_checkpoint(
    epoch,
    generator,
    discriminator,
    ema_generator,
    optimizer_g,
    optimizer_d,
    history,
):
    path = (
        CHECKPOINT_DIR
        / f"mri_gan_v5_epoch_{epoch:03d}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": {
                "image_size": IMAGE_SIZE,
                "channels": CHANNELS,
                "num_classes": NUM_CLASSES,
                "latent_dim": LATENT_DIM,
                "batch_size": BATCH_SIZE,
                "g_lr": G_LR,
                "d_lr": D_LR,
                "mode_seeking_weight": MODE_SEEKING_WEIGHT,
                "feature_match_weight": FEATURE_MATCH_WEIGHT,
                "ema_decay": EMA_DECAY,
                "seed": SEED,
            },
            "class_names": class_names,
        },
        str(path),
    )

    return path


def load_checkpoint(
    path,
    generator,
    discriminator,
    ema_generator,
    optimizer_g,
    optimizer_d,
):
    checkpoint = torch.load(
        str(path),
        map_location=DEVICE,
    )

    generator.load_state_dict(
        checkpoint["generator"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator"]
    )

    if "ema_generator" in checkpoint:
        ema_generator.load_state_dict(
            checkpoint["ema_generator"]
        )
    else:
        ema_generator.load_state_dict(
            generator.state_dict()
        )

    optimizer_g.load_state_dict(
        checkpoint["optimizer_g"]
    )

    optimizer_d.load_state_dict(
        checkpoint["optimizer_d"]
    )

    return (
        int(checkpoint["epoch"]),
        checkpoint.get("history", []),
    )


# ============================================================
# TRAINING
# ============================================================

def train():
    print("=" * 72)
    print("MRI GAN V5")
    print("=" * 72)

    print(f"Device          : {DEVICE}")
    print(f"Image size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Latent dim      : {LATENT_DIM}")
    print(f"Classes         : {class_names}")
    print(f"Total epochs    : {TOTAL_EPOCHS}")
    print(f"G learning rate : {G_LR}")
    print(f"D learning rate : {D_LR}")
    print(f"Mode-seeking wt : {MODE_SEEKING_WEIGHT}")
    print(f"Feature-match wt: {FEATURE_MATCH_WEIGHT}")
    print(f"Augmentation p  : {AUGMENT_PROB}")
    print(f"EMA decay       : {EMA_DECAY}")
    print(f"CPU threads     : {CPU_THREADS}")
    print()
    print(f"Training data   : {TRAIN_DIR}")
    print(f"Training images : {len(dataset)}")

    for index, class_name in enumerate(class_names):
        count = sum(
            1
            for _, label in dataset.samples
            if label == index
        )
        print(
            f"  {class_name:<12}: {count}"
        )

    print()
    print("IMPORTANT: Real Testing data is NOT used.")
    print("=" * 72)

    generator = GeneratorV5().to(DEVICE)
    discriminator = DiscriminatorV5().to(DEVICE)

    ema_generator = deepcopy(generator).to(DEVICE)
    ema_generator.eval()

    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=G_LR,
        betas=BETAS,
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=D_LR,
        betas=BETAS,
    )

    start_epoch = 1
    history = []

    # --------------------------------------------------------
    # Resume ONLY V5 checkpoints.
    # V4 is intentionally NOT loaded because the architecture
    # and training objective have changed.
    # --------------------------------------------------------

    checkpoint_path = latest_checkpoint()

    if checkpoint_path is not None:
        print()
        print(
            f"Resuming V5 from: {checkpoint_path}"
        )

        loaded_epoch, history = load_checkpoint(
            checkpoint_path,
            generator,
            discriminator,
            ema_generator,
            optimizer_g,
            optimizer_d,
        )

        start_epoch = loaded_epoch + 1

        print(
            f"Resuming at epoch {start_epoch}"
        )
    else:
        print()
        print(
            "No V5 checkpoint found."
        )
        print(
            "Starting V5 training from scratch."
        )

    if start_epoch > TOTAL_EPOCHS:
        print()
        print(
            "Training is already complete."
        )
        return

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        epoch_start = time.time()

        generator.train()
        discriminator.train()

        g_running = 0.0
        d_running = 0.0
        mode_running = 0.0
        feature_running = 0.0
        ratio_running = 0.0

        batches = 0

        for real_images, real_labels in loader:
            real_images = real_images.to(
                DEVICE,
                non_blocking=False,
            )

            real_labels = real_labels.to(
                DEVICE,
                non_blocking=False,
            )

            batch_size = real_images.size(0)

            # =================================================
            # 1. DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad(
                set_to_none=True
            )

            z = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE,
            )

            fake_labels = torch.randint(
                0,
                NUM_CLASSES,
                (batch_size,),
                device=DEVICE,
            )

            fake_images = generator(
                z,
                fake_labels,
            ).detach()

            real_aug = discriminator_augment(
                real_images
            )

            fake_aug = discriminator_augment(
                fake_images
            )

            real_logits, _ = discriminator(
                real_aug,
                real_labels,
            )

            fake_logits, _ = discriminator(
                fake_aug,
                fake_labels,
            )

            # Hinge discriminator loss.
            d_real_loss = F.relu(
                1.0 - real_logits
            ).mean()

            d_fake_loss = F.relu(
                1.0 + fake_logits
            ).mean()

            d_loss = (
                d_real_loss
                + d_fake_loss
            )

            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0,
            )

            optimizer_d.step()

            # =================================================
            # 2. GENERATOR
            # =================================================

            optimizer_g.zero_grad(
                set_to_none=True
            )

            z1 = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE,
            )

            z2 = torch.randn(
                batch_size,
                LATENT_DIM,
                device=DEVICE,
            )

            labels = torch.randint(
                0,
                NUM_CLASSES,
                (batch_size,),
                device=DEVICE,
            )

            fake1 = generator(
                z1,
                labels,
            )

            fake2 = generator(
                z2,
                labels,
            )

            fake_aug = discriminator_augment(
                fake1
            )

            fake_logits, fake_features = discriminator(
                fake_aug,
                labels,
            )

            # Generator wants discriminator to classify
            # generated images as real.
            adversarial_loss = -fake_logits.mean()

            # -------------------------------------------------
            # Feature matching
            # -------------------------------------------------

            real_reference = discriminator_augment(
                real_images
            )

            with torch.no_grad():
                _, real_features = discriminator(
                    real_reference,
                    real_labels,
                )

            feature_loss = F.l1_loss(
                fake_features.mean(dim=0),
                real_features.mean(dim=0),
            )

            # -------------------------------------------------
            # Mode-seeking
            # -------------------------------------------------

            image_difference = (
                fake1 - fake2
            ).abs().mean(
                dim=(1, 2, 3)
            )

            latent_difference = (
                z1 - z2
            ).abs().mean(
                dim=1
            )

            ratio = (
                image_difference
                / (
                    latent_difference
                    + MODE_SEEKING_EPS
                )
            )

            mode_loss = (
                1.0
                / (
                    ratio
                    + MODE_SEEKING_EPS
                )
            ).mean()

            # -------------------------------------------------
            # Total generator objective
            # -------------------------------------------------

            g_loss = (
                adversarial_loss
                + FEATURE_MATCH_WEIGHT * feature_loss
                + MODE_SEEKING_WEIGHT * mode_loss
            )

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=5.0,
            )

            optimizer_g.step()

            # EMA update after generator update.
            update_ema(
                ema_generator,
                generator,
                EMA_DECAY,
            )

            # -------------------------------------------------
            # Running metrics
            # -------------------------------------------------

            g_running += float(
                g_loss.detach().item()
            )

            d_running += float(
                d_loss.detach().item()
            )

            mode_running += float(
                mode_loss.detach().item()
            )

            feature_running += float(
                feature_loss.detach().item()
            )

            ratio_running += float(
                ratio.detach().mean().item()
            )

            batches += 1

        # =====================================================
        # Epoch summary
        # =====================================================

        g_avg = g_running / max(batches, 1)
        d_avg = d_running / max(batches, 1)
        mode_avg = mode_running / max(batches, 1)
        feature_avg = feature_running / max(batches, 1)
        ratio_avg = ratio_running / max(batches, 1)

        elapsed_minutes = (
            time.time() - epoch_start
        ) / 60.0

        epoch_record = {
            "epoch": epoch,
            "g_loss": g_avg,
            "d_loss": d_avg,
            "mode_loss": mode_avg,
            "feature_loss": feature_avg,
            "mode_ratio": ratio_avg,
            "minutes": elapsed_minutes,
        }

        history.append(epoch_record)

        print()
        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"G {g_avg:.4f} | "
            f"D {d_avg:.4f} | "
            f"Mode {mode_avg:.5f} | "
            f"FM {feature_avg:.5f} | "
            f"Ratio {ratio_avg:.4f} | "
            f"{elapsed_minutes:.2f} min"
        )

        # =====================================================
        # Preview
        # =====================================================

        if epoch % PREVIEW_EVERY == 0:
            save_class_previews(
                ema_generator,
                epoch,
                DEVICE,
                num_images=16,
            )

            save_all_class_grid(
                ema_generator,
                epoch,
                DEVICE,
                num_images=8,
            )

            print(
                f"Preview saved: {PREVIEW_DIR}"
            )

        # =====================================================
        # Checkpoint
        # =====================================================

        if epoch % SAVE_EVERY == 0:
            checkpoint = save_checkpoint(
                epoch,
                generator,
                discriminator,
                ema_generator,
                optimizer_g,
                optimizer_d,
                history,
            )

            print(
                f"Checkpoint saved: {checkpoint}"
            )

        # =====================================================
        # EARLY OBSERVATION RULE
        # =====================================================

        print(
            "Next action: inspect this epoch's preview before "
            "letting the training run much further."
        )

    # =========================================================
    # Final checkpoint alias
    # =========================================================

    final_path = (
        CHECKPOINT_DIR
        / "mri_gan_v5_final.pth"
    )

    torch.save(
        {
            "epoch": TOTAL_EPOCHS,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": {
                "image_size": IMAGE_SIZE,
                "channels": CHANNELS,
                "num_classes": NUM_CLASSES,
                "latent_dim": LATENT_DIM,
                "batch_size": BATCH_SIZE,
                "g_lr": G_LR,
                "d_lr": D_LR,
                "mode_seeking_weight": MODE_SEEKING_WEIGHT,
                "feature_match_weight": FEATURE_MATCH_WEIGHT,
                "ema_decay": EMA_DECAY,
                "seed": SEED,
            },
            "class_names": class_names,
        },
        str(final_path),
    )

    print()
    print("=" * 72)
    print("V5 TRAINING COMPLETE")
    print("=" * 72)
    print(f"Final checkpoint : {final_path}")
    print(f"Preview folder   : {PREVIEW_DIR}")
    print()
    print(
        "IMPORTANT: Do not automatically assume the final epoch is "
        "the best epoch."
    )
    print(
        "Inspect the epoch previews and run the existing GAN evaluator "
        "on the strongest checkpoint."
    )


if __name__ == "__main__":
    train()

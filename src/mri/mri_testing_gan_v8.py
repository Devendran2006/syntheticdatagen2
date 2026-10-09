# ============================================================
# MRI TESTING GAN V8
# ============================================================
# Purpose:
#   Generate synthetic MRI images from ONLY the Testing dataset.
#
# IMPORTANT:
#   - Does NOT read Training for model training
#   - Does NOT modify Training
#   - Does NOT modify Training_Preprocessed
#   - Does NOT modify previous GAN output folders
#   - Creates a NEW timestamped V8 run folder
#
# Project:
#   E:\A1593 DA python live\synthetic data
#
# Input:
#   data/imaging/MRI/Testing
#
# Output:
#   outputs/mri_testing_gan_v8/run_YYYYMMDD_HHMMSS/
#
# Pilot:
#   By default only GLIOMA is trained first.
#   After checking quality, change PILOT_CLASSES = None
#   to train all detected classes.
# ============================================================

import os
import csv
import json
import math
import time
import random
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

CHANNELS = 1

LATENT_DIM = 128
CLASS_EMBED_DIM = 32

BATCH_SIZE = 8

EPOCHS = 15

D_STEPS = 1

LR_G = 2e-4
LR_D = 4e-4

BETAS = (0.0, 0.9)

EMA_DECAY = 0.995

NUM_SYNTHETIC = 100

SEED = 42

# ------------------------------------------------------------
# PILOT MODE
# ------------------------------------------------------------
# First run:
#     ["glioma"]
#
# After verifying quality:
#     None
#
# None = automatically train every class in Testing.
# ------------------------------------------------------------

PILOT_CLASSES = ["glioma"]

# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

FEATURE_MATCH_WEIGHT = 5.0
STATISTICS_WEIGHT = 1.0
EDGE_WEIGHT = 1.0
DIVERSITY_WEIGHT = 0.05

# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------

PREVIEW_COUNT = 16

SAVE_PREVIEW_EVERY = 1

# ------------------------------------------------------------
# Quality filtering
# ------------------------------------------------------------

QUALITY_STD_LIMIT = 3.0

MIN_DIVERSITY = 0.002

# ------------------------------------------------------------
# CPU configuration
# ------------------------------------------------------------

NUM_WORKERS = 0

# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MRI_ROOT = PROJECT_ROOT / "data" / "imaging" / "MRI"

TESTING_PATH = MRI_ROOT / "Testing"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mri_testing_gan_v8"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR = OUTPUT_ROOT / f"run_{RUN_ID}"

SYNTHETIC_ROOT = RUN_DIR / "synthetic_samples"

MODEL_ROOT = RUN_DIR / "models"

PREVIEW_ROOT = RUN_DIR / "previews"

REPORT_ROOT = RUN_DIR / "reports"

CHECKPOINT_ROOT = RUN_DIR / "checkpoints"


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 4. RANDOM SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# 5. DIRECTORY SETUP
# ============================================================

def create_directories():

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    SYNTHETIC_ROOT.mkdir(parents=True, exist_ok=True)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. PRINT CONFIGURATION
# ============================================================

def print_configuration():

    print("\n" + "=" * 70)
    print("MRI TESTING GAN V8")
    print("=" * 70)

    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Testing dataset   : {TESTING_PATH}")
    print(f"Output root       : {RUN_DIR}")
    print(f"Device            : {DEVICE}")
    print(f"Image size        : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Latent dimension  : {LATENT_DIM}")
    print(f"EMA decay         : {EMA_DECAY}")
    print(f"Synthetic/class   : {NUM_SYNTHETIC}")
    print(f"Pilot classes     : {PILOT_CLASSES}")

    print("=" * 70 + "\n")


# ============================================================
# 7. IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 8. FIND CLASSES
# ============================================================

def discover_classes():

    if not TESTING_PATH.exists():

        raise FileNotFoundError(
            f"\nTesting dataset not found:\n{TESTING_PATH}\n"
        )

    classes = []

    for item in sorted(TESTING_PATH.iterdir()):

        if item.is_dir():

            files = [
                f
                for f in item.rglob("*")
                if f.is_file()
                and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if files:
                classes.append(item.name)

    return classes


# ============================================================
# 9. FIND IMAGES
# ============================================================

def get_class_images(class_name):

    class_dir = TESTING_PATH / class_name

    if not class_dir.exists():

        return []

    files = [
        f
        for f in class_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(files)


# ============================================================
# 10. ROBUST MRI NORMALIZATION
# ============================================================

def normalize_mri_image(image):

    image = image.convert("L")

    arr = np.asarray(image).astype(np.float32)

    if arr.size == 0:

        raise ValueError("Empty image encountered.")

    low = np.percentile(arr, 1.0)

    high = np.percentile(arr, 99.0)

    if high <= low:

        low = arr.min()
        high = arr.max()

    if high <= low:

        arr = np.zeros_like(arr)

    else:

        arr = (arr - low) / (high - low)

        arr = np.clip(arr, 0.0, 1.0)

    return arr


# ============================================================
# 11. DATASET
# ============================================================

class MRITestingDataset(Dataset):

    def __init__(self, image_paths, class_index):

        self.image_paths = image_paths

        self.class_index = class_index

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        path = self.image_paths[index]

        try:

            image = Image.open(path)

            image = image.convert("L")

            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.LANCZOS,
            )

            arr = normalize_mri_image(image)

            tensor = torch.from_numpy(arr).float()

            tensor = tensor.unsqueeze(0)

            # [0,1] -> [-1,1]

            tensor = tensor * 2.0 - 1.0

            return tensor, self.class_index

        except Exception as exc:

            print(f"Warning: could not read {path}: {exc}")

            return (
                torch.zeros(
                    CHANNELS,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                self.class_index,
            )


# ============================================================
# 12. RESIDUAL GENERATOR BLOCK
# ============================================================

class GeneratorResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.norm1 = nn.BatchNorm2d(channels)

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm2 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):

        residual = x

        out = self.norm1(x)

        out = F.silu(out)

        out = self.conv1(out)

        out = self.norm2(out)

        out = F.silu(out)

        out = self.conv2(out)

        return out + residual


# ============================================================
# 13. UPSAMPLE BLOCK
# ============================================================

class GeneratorUpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm = nn.BatchNorm2d(out_channels)

        self.residual = GeneratorResidualBlock(
            out_channels
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

        x = self.conv(x)

        x = self.norm(x)

        x = F.silu(x)

        x = self.residual(x)

        return x


# ============================================================
# 14. CONDITIONAL GENERATOR
# ============================================================

class ConditionalGenerator(nn.Module):

    def __init__(
        self,
        num_classes,
        latent_dim=LATENT_DIM,
        embed_dim=CLASS_EMBED_DIM,
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.class_embedding = nn.Embedding(
            num_classes,
            embed_dim,
        )

        self.fc = nn.Linear(
            latent_dim + embed_dim,
            512 * 8 * 8,
        )

        self.block8 = nn.Sequential(
            GeneratorResidualBlock(512),
            GeneratorResidualBlock(512),
        )

        self.up16 = GeneratorUpBlock(
            512,
            256,
        )

        self.up32 = GeneratorUpBlock(
            256,
            128,
        )

        self.up64 = GeneratorUpBlock(
            128,
            64,
        )

        self.up128 = GeneratorUpBlock(
            64,
            32,
        )

        self.final_norm = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(
            32,
            CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z, labels):

        class_vector = self.class_embedding(labels)

        x = torch.cat(
            [z, class_vector],
            dim=1,
        )

        x = self.fc(x)

        x = x.view(
            x.size(0),
            512,
            8,
            8,
        )

        x = self.block8(x)

        x = self.up16(x)

        x = self.up32(x)

        x = self.up64(x)

        x = self.up128(x)

        x = self.final_norm(x)

        x = F.silu(x)

        x = self.final_conv(x)

        return torch.tanh(x)


# ============================================================
# 15. DISCRIMINATOR BLOCK
# ============================================================

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        self.conv2 = spectral_norm(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x):

        x = self.conv(x)

        x = F.leaky_relu(
            x,
            0.2,
            inplace=True,
        )

        x = self.conv2(x)

        x = F.leaky_relu(
            x,
            0.2,
            inplace=True,
        )

        return x


# ============================================================
# 16. GLOBAL CONDITIONAL DISCRIMINATOR
# ============================================================

class GlobalDiscriminator(nn.Module):

    def __init__(
        self,
        num_classes,
        embed_dim=CLASS_EMBED_DIM,
    ):

        super().__init__()

        self.block1 = DiscriminatorBlock(
            CHANNELS,
            64,
        )

        self.block2 = DiscriminatorBlock(
            64,
            128,
        )

        self.block3 = DiscriminatorBlock(
            128,
            256,
        )

        self.block4 = DiscriminatorBlock(
            256,
            512,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = spectral_norm(
            nn.Linear(
                512,
                1,
            )
        )

        self.class_embedding = nn.Embedding(
            num_classes,
            512,
        )

    def forward(self, x, labels):

        features = []

        x = self.block1(x)
        features.append(x)

        x = self.block2(x)
        features.append(x)

        x = self.block3(x)
        features.append(x)

        x = self.block4(x)
        features.append(x)

        pooled = self.pool(x)

        pooled = pooled.flatten(1)

        unconditional = self.fc(pooled)

        class_vector = self.class_embedding(labels)

        projection = (
            pooled * class_vector
        ).sum(
            dim=1,
            keepdim=True,
        )

        logits = unconditional + projection

        return logits, features


# ============================================================
# 17. PATCH DISCRIMINATOR
# ============================================================

class PatchDiscriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.block1 = DiscriminatorBlock(
            CHANNELS,
            64,
        )

        self.block2 = DiscriminatorBlock(
            64,
            128,
        )

        self.block3 = DiscriminatorBlock(
            128,
            256,
        )

        self.block4 = DiscriminatorBlock(
            256,
            512,
        )

        self.final = spectral_norm(
            nn.Conv2d(
                512,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x):

        features = []

        x = self.block1(x)
        features.append(x)

        x = self.block2(x)
        features.append(x)

        x = self.block3(x)
        features.append(x)

        x = self.block4(x)
        features.append(x)

        logits = self.final(x)

        return logits, features


# ============================================================
# 18. SOBEL EDGE DETECTOR
# ============================================================

def sobel_edges(x):

    device = x.device

    gx = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    gy = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(
        x,
        gx,
        padding=1,
    )

    gy = F.conv2d(
        x,
        gy,
        padding=1,
    )

    magnitude = torch.sqrt(
        gx * gx + gy * gy + 1e-8
    )

    return magnitude


# ============================================================
# 19. IMAGE STATISTICS
# ============================================================

def image_statistics(x):

    mean = x.mean(
        dim=(1, 2, 3)
    )

    std = x.std(
        dim=(1, 2, 3)
    )

    edges = sobel_edges(x)

    edge_mean = edges.mean(
        dim=(1, 2, 3)
    )

    return mean, std, edge_mean


# ============================================================
# 20. FEATURE MATCHING LOSS
# ============================================================

def feature_matching_loss(
    real_features,
    fake_features,
):

    total = torch.tensor(
        0.0,
        device=DEVICE,
    )

    count = 0

    for real, fake in zip(
        real_features,
        fake_features,
    ):

        real_mean = real.mean(
            dim=(0, 2, 3)
        )

        fake_mean = fake.mean(
            dim=(0, 2, 3)
        )

        total = total + F.l1_loss(
            fake_mean,
            real_mean.detach(),
        )

        count += 1

    if count == 0:

        return total

    return total / count


# ============================================================
# 21. HINGE DISCRIMINATOR LOSS
# ============================================================

def hinge_discriminator_loss(
    real_logits,
    fake_logits,
):

    real_loss = F.relu(
        1.0 - real_logits
    ).mean()

    fake_loss = F.relu(
        1.0 + fake_logits
    ).mean()

    return real_loss + fake_loss


# ============================================================
# 22. SAVE TENSOR IMAGE
# ============================================================

def tensor_to_image(tensor):

    tensor = tensor.detach().cpu()

    tensor = tensor.clamp(
        -1.0,
        1.0,
    )

    tensor = (
        tensor + 1.0
    ) / 2.0

    array = (
        tensor.squeeze()
        .numpy()
        * 255.0
    )

    array = np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="L",
    )


# ============================================================
# 23. DISPLAY ENHANCEMENT
# ============================================================

def enhance_for_display(image):

    image = image.convert("L")

    arr = np.asarray(
        image
    ).astype(np.float32)

    low = np.percentile(
        arr,
        1,
    )

    high = np.percentile(
        arr,
        99,
    )

    if high > low:

        arr = (
            arr - low
        ) / (
            high - low
        )

        arr = np.clip(
            arr,
            0,
            1,
        )

    arr = (
        arr * 255
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        arr,
        mode="L",
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    # Very mild sharpening only for preview.
    # Raw synthetic data is NOT sharpened.
    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=0.6,
            percent=80,
            threshold=3,
        )
    )

    return image


# ============================================================
# 24. CONTACT SHEET
# ============================================================

def save_contact_sheet(
    images,
    output_path,
    columns=4,
    cell_size=IMAGE_SIZE,
):

    if not images:

        return

    rows = math.ceil(
        len(images) / columns
    )

    sheet = Image.new(
        "L",
        (
            columns * cell_size,
            rows * cell_size,
        ),
        color=0,
    )

    for index, image in enumerate(images):

        image = image.resize(
            (
                cell_size,
                cell_size,
            ),
            Image.Resampling.LANCZOS,
        )

        x = (
            index % columns
        ) * cell_size

        y = (
            index // columns
        ) * cell_size

        sheet.paste(
            image,
            (
                x,
                y,
            ),
        )

    sheet.save(
        output_path
    )


# ============================================================
# 25. SAVE REAL/Fake PREVIEW
# ============================================================

def save_training_preview(
    generator,
    fixed_noise,
    fixed_labels,
    epoch,
    class_name,
):

    generator.eval()

    with torch.no_grad():

        fake = generator(
            fixed_noise,
            fixed_labels,
        )

    generator.train()

    images = []

    for image in fake:

        pil_image = tensor_to_image(
            image
        )

        pil_image = enhance_for_display(
            pil_image
        )

        images.append(
            pil_image
        )

    class_preview_dir = (
        PREVIEW_ROOT / class_name
    )

    class_preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        class_preview_dir
        / f"epoch_{epoch:03d}.png"
    )

    save_contact_sheet(
        images,
        path,
        columns=4,
    )

    return path


# ============================================================
# 26. SAVE REAL DATA PREVIEW
# ============================================================

def save_real_preview(
    dataset,
    class_name,
):

    images = []

    count = min(
        PREVIEW_COUNT,
        len(dataset),
    )

    for index in range(count):

        tensor, _ = dataset[index]

        image = tensor_to_image(
            tensor
        )

        image = enhance_for_display(
            image
        )

        images.append(
            image
        )

    class_preview_dir = (
        PREVIEW_ROOT
        / class_name
    )

    class_preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        class_preview_dir
        / "real_testing_samples.png"
    )

    save_contact_sheet(
        images,
        path,
        columns=4,
    )


# ============================================================
# 27. UPDATE EMA
# ============================================================

@torch.no_grad()
def update_ema(
    ema_model,
    model,
    decay,
):

    ema_params = dict(
        ema_model.named_parameters()
    )

    model_params = dict(
        model.named_parameters()
    )

    for name in ema_params:

        ema_params[name].mul_(decay)

        ema_params[name].add_(
            model_params[name],
            alpha=1.0 - decay,
        )

    ema_buffers = dict(
        ema_model.named_buffers()
    )

    model_buffers = dict(
        model.named_buffers()
    )

    for name in ema_buffers:

        ema_buffers[name].copy_(
            model_buffers[name]
        )


# ============================================================
# 28. SOURCE STATISTICS
# ============================================================

def calculate_source_statistics(
    dataset
):

    means = []

    stds = []

    edges = []

    for index in range(
        len(dataset)
    ):

        image, _ = dataset[index]

        image = image.unsqueeze(0)

        mean, std, edge = image_statistics(
            image
        )

        means.append(
            float(mean.item())
        )

        stds.append(
            float(std.item())
        )

        edges.append(
            float(edge.item())
        )

    statistics = {

        "mean": {
            "mean": float(
                np.mean(means)
            ),
            "std": float(
                np.std(means) + 1e-6
            ),
        },

        "std": {
            "mean": float(
                np.mean(stds)
            ),
            "std": float(
                np.std(stds) + 1e-6
            ),
        },

        "edge": {
            "mean": float(
                np.mean(edges)
            ),
            "std": float(
                np.std(edges) + 1e-6
            ),
        },
    }

    return statistics


# ============================================================
# 29. QUALITY CHECK
# ============================================================

def image_quality_score(
    image,
    source_stats,
):

    tensor = image.unsqueeze(0)

    mean, std, edge = image_statistics(
        tensor
    )

    mean_value = float(
        mean.item()
    )

    std_value = float(
        std.item()
    )

    edge_value = float(
        edge.item()
    )

    mean_target = source_stats[
        "mean"
    ]["mean"]

    mean_std = source_stats[
        "mean"
    ]["std"]

    std_target = source_stats[
        "std"
    ]["mean"]

    std_std = source_stats[
        "std"
    ]["std"]

    edge_target = source_stats[
        "edge"
    ]["mean"]

    edge_std = source_stats[
        "edge"
    ]["std"]

    z_mean = abs(
        mean_value - mean_target
    ) / max(
        mean_std,
        1e-6,
    )

    z_std = abs(
        std_value - std_target
    ) / max(
        std_std,
        1e-6,
    )

    z_edge = abs(
        edge_value - edge_target
    ) / max(
        edge_std,
        1e-6,
    )

    score = (
        z_mean
        + z_std
        + z_edge
    )

    accepted = (
        z_mean <= QUALITY_STD_LIMIT
        and z_std <= QUALITY_STD_LIMIT
        and z_edge <= QUALITY_STD_LIMIT
    )

    return {
        "score": float(score),
        "accepted": bool(accepted),
        "mean": mean_value,
        "std": std_value,
        "edge": edge_value,
    }


# ============================================================
# 30. DIVERSITY CHECK
# ============================================================

def is_diverse_enough(
    image,
    accepted_images,
    threshold=MIN_DIVERSITY,
):

    if not accepted_images:

        return True

    image_flat = image.flatten()

    # Compare against at most the latest
    # 20 accepted images for speed.
    comparison_images = accepted_images[-20:]

    distances = []

    for previous in comparison_images:

        previous_flat = previous.flatten()

        distance = torch.mean(
            torch.abs(
                image_flat
                - previous_flat
            )
        ).item()

        distances.append(
            distance
        )

    if not distances:

        return True

    return (
        np.mean(distances)
        >= threshold
    )


# ============================================================
# 31. SAVE CSV
# ============================================================

def save_csv(
    path,
    rows,
    fieldnames,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
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


# ============================================================
# 32. TRAIN ONE CLASS
# ============================================================

def train_class(
    class_name,
    class_index,
    class_names,
):

    print("\n" + "=" * 70)

    print(
        f"TRAINING CLASS: {class_name.upper()}"
    )

    print("=" * 70)

    image_paths = get_class_images(
        class_name
    )

    print(
        f"Testing images: {len(image_paths)}"
    )

    if len(image_paths) == 0:

        print(
            f"No images found for {class_name}"
        )

        return None

    dataset = MRITestingDataset(
        image_paths,
        class_index,
    )

    save_real_preview(
        dataset,
        class_name,
    )

    print(
        "Calculating source image statistics..."
    )

    source_stats = calculate_source_statistics(
        dataset
    )

    print(
        json.dumps(
            source_stats,
            indent=2,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(class_names)

    generator = ConditionalGenerator(
        num_classes=num_classes,
    ).to(DEVICE)

    generator_ema = deepcopy(
        generator
    ).to(DEVICE)

    discriminator_global = GlobalDiscriminator(
        num_classes=num_classes,
    ).to(DEVICE)

    discriminator_patch = PatchDiscriminator().to(
        DEVICE
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LR_G,
        betas=BETAS,
    )

    optimizer_d = torch.optim.Adam(
        list(
            discriminator_global.parameters()
        )
        + list(
            discriminator_patch.parameters()
        ),
        lr=LR_D,
        betas=BETAS,
    )

    fixed_noise = torch.randn(
        PREVIEW_COUNT,
        LATENT_DIM,
        device=DEVICE,
    )

    fixed_labels = torch.full(
        (
            PREVIEW_COUNT,
        ),
        class_index,
        dtype=torch.long,
        device=DEVICE,
    )

    history = []

    best_g_loss = float("inf")

    class_checkpoint_dir = (
        CHECKPOINT_ROOT
        / class_name
    )

    class_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        generator.train()

        discriminator_global.train()

        discriminator_patch.train()

        epoch_d = []

        epoch_g = []

        epoch_fm = []

        epoch_edge = []

        epoch_stat = []

        for real_images, labels in loader:

            real_images = real_images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            # ==================================================
            # DISCRIMINATOR
            # ==================================================

            for _ in range(D_STEPS):

                noise = torch.randn(
                    real_images.size(0),
                    LATENT_DIM,
                    device=DEVICE,
                )

                with torch.no_grad():

                    fake_images = generator(
                        noise,
                        labels,
                    )

                # Small instance noise early in training.
                noise_scale = max(
                    0.0,
                    0.04
                    * (
                        1.0
                        - (
                            epoch
                            / max(
                                EPOCHS,
                                1,
                            )
                        )
                    ),
                )

                real_input = (
                    real_images
                    + torch.randn_like(
                        real_images
                    )
                    * noise_scale
                )

                fake_input = (
                    fake_images
                    + torch.randn_like(
                        fake_images
                    )
                    * noise_scale
                )

                real_global, _ = (
                    discriminator_global(
                        real_input,
                        labels,
                    )
                )

                fake_global, _ = (
                    discriminator_global(
                        fake_input,
                        labels,
                    )
                )

                real_patch, _ = (
                    discriminator_patch(
                        real_input
                    )
                )

                fake_patch, _ = (
                    discriminator_patch(
                        fake_input
                    )
                )

                d_global_loss = (
                    hinge_discriminator_loss(
                        real_global,
                        fake_global,
                    )
                )

                d_patch_loss = (
                    hinge_discriminator_loss(
                        real_patch,
                        fake_patch,
                    )
                )

                d_loss = (
                    d_global_loss
                    + d_patch_loss
                ) * 0.5

                optimizer_d.zero_grad(
                    set_to_none=True
                )

                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(
                        discriminator_global.parameters()
                    )
                    + list(
                        discriminator_patch.parameters()
                    ),
                    max_norm=5.0,
                )

                optimizer_d.step()

            # ==================================================
            # GENERATOR
            # ==================================================

            noise = torch.randn(
                real_images.size(0),
                LATENT_DIM,
                device=DEVICE,
            )

            fake_images = generator(
                noise,
                labels,
            )

            fake_global, fake_global_features = (
                discriminator_global(
                    fake_images,
                    labels,
                )
            )

            with torch.no_grad():

                real_global, real_global_features = (
                    discriminator_global(
                        real_images,
                        labels,
                    )
                )

            fake_patch, fake_patch_features = (
                discriminator_patch(
                    fake_images
                )
            )

            with torch.no_grad():

                real_patch, real_patch_features = (
                    discriminator_patch(
                        real_images
                    )
                )

            # --------------------------------------------------
            # Adversarial loss
            # --------------------------------------------------

            g_global_adv = (
                -fake_global.mean()
            )

            g_patch_adv = (
                -fake_patch.mean()
            )

            g_adv = (
                g_global_adv
                + g_patch_adv
            ) * 0.5

            # --------------------------------------------------
            # Feature matching
            # --------------------------------------------------

            fm_global = feature_matching_loss(
                real_global_features,
                fake_global_features,
            )

            fm_patch = feature_matching_loss(
                real_patch_features,
                fake_patch_features,
            )

            fm_loss = (
                fm_global
                + fm_patch
            ) * 0.5

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            real_mean, real_std, real_edge = (
                image_statistics(
                    real_images
                )
            )

            fake_mean, fake_std, fake_edge = (
                image_statistics(
                    fake_images
                )
            )

            stat_loss = (
                F.l1_loss(
                    fake_mean.mean(),
                    real_mean.mean().detach(),
                )
                +
                F.l1_loss(
                    fake_std.mean(),
                    real_std.mean().detach(),
                )
            )

            # --------------------------------------------------
            # Edge energy loss
            # --------------------------------------------------

            edge_loss = F.l1_loss(
                fake_edge.mean(),
                real_edge.mean().detach(),
            )

            # --------------------------------------------------
            # Diversity regularization
            # --------------------------------------------------

            if fake_images.size(0) > 1:

                shuffled = fake_images[
                    torch.randperm(
                        fake_images.size(0),
                        device=DEVICE,
                    )
                ]

                pair_distance = torch.mean(
                    torch.abs(
                        fake_images
                        - shuffled
                    )
                )

                diversity_loss = (
                    1.0
                    / (
                        pair_distance
                        + 1e-4
                    )
                )

            else:

                diversity_loss = torch.tensor(
                    0.0,
                    device=DEVICE,
                )

            # --------------------------------------------------
            # Total generator loss
            # --------------------------------------------------

            g_loss = (
                g_adv
                + FEATURE_MATCH_WEIGHT
                * fm_loss
                + STATISTICS_WEIGHT
                * stat_loss
                + EDGE_WEIGHT
                * edge_loss
                + DIVERSITY_WEIGHT
                * diversity_loss
            )

            optimizer_g.zero_grad(
                set_to_none=True
            )

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=5.0,
            )

            optimizer_g.step()

            update_ema(
                generator_ema,
                generator,
                EMA_DECAY,
            )

            epoch_d.append(
                float(d_loss.item())
            )

            epoch_g.append(
                float(g_loss.item())
            )

            epoch_fm.append(
                float(fm_loss.item())
            )

            epoch_edge.append(
                float(edge_loss.item())
            )

            epoch_stat.append(
                float(stat_loss.item())
            )

        elapsed = (
            time.time()
            - epoch_start
        )

        avg_d = float(
            np.mean(epoch_d)
        )

        avg_g = float(
            np.mean(epoch_g)
        )

        avg_fm = float(
            np.mean(epoch_fm)
        )

        avg_edge = float(
            np.mean(epoch_edge)
        )

        avg_stat = float(
            np.mean(epoch_stat)
        )

        row = {

            "epoch": epoch,

            "d_loss": avg_d,

            "g_loss": avg_g,

            "feature_matching": avg_fm,

            "edge_loss": avg_edge,

            "statistics_loss": avg_stat,

            "seconds": elapsed,
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"D {avg_d:.4f} | "
            f"G {avg_g:.4f} | "
            f"FM {avg_fm:.4f} | "
            f"Edge {avg_edge:.4f} | "
            f"Stat {avg_stat:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ======================================================
        # SAVE PREVIEW
        # ======================================================

        if (
            epoch % SAVE_PREVIEW_EVERY == 0
        ):

            preview_path = (
                save_training_preview(
                    generator_ema,
                    fixed_noise,
                    fixed_labels,
                    epoch,
                    class_name,
                )
            )

            print(
                f"Preview saved: {preview_path}"
            )

        # ======================================================
        # SAVE BEST GENERATOR
        # ======================================================

        if avg_g < best_g_loss:

            best_g_loss = avg_g

            best_path = (
                class_checkpoint_dir
                / "best_generator_ema.pt"
            )

            torch.save(
                generator_ema.state_dict(),
                best_path,
            )

        # ======================================================
        # SAVE LATEST CHECKPOINT
        # ======================================================

        checkpoint = {

            "epoch": epoch,

            "generator": generator.state_dict(),

            "generator_ema":
                generator_ema.state_dict(),

            "global_discriminator":
                discriminator_global.state_dict(),

            "patch_discriminator":
                discriminator_patch.state_dict(),

            "optimizer_g":
                optimizer_g.state_dict(),

            "optimizer_d":
                optimizer_d.state_dict(),

            "class_name":
                class_name,

            "class_index":
                class_index,

            "image_size":
                IMAGE_SIZE,

            "latent_dim":
                LATENT_DIM,
        }

        torch.save(
            checkpoint,
            class_checkpoint_dir
            / "latest_checkpoint.pt",
        )

    # ==========================================================
    # SAVE TRAINING HISTORY
    # ==========================================================

    history_path = (
        REPORT_ROOT
        / f"{class_name}_training_history.csv"
    )

    save_csv(
        history_path,
        history,
        [
            "epoch",
            "d_loss",
            "g_loss",
            "feature_matching",
            "edge_loss",
            "statistics_loss",
            "seconds",
        ],
    )

    # ==========================================================
    # SAVE SOURCE STATS
    # ==========================================================

    stats_path = (
        REPORT_ROOT
        / f"{class_name}_source_statistics.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            source_stats,
            file,
            indent=2,
        )

    # ==========================================================
    # SAVE FINAL EMA MODEL
    # ==========================================================

    model_path = (
        MODEL_ROOT
        / f"{class_name}_generator_ema.pt"
    )

    torch.save(
        generator_ema.state_dict(),
        model_path,
    )

    print(
        f"\nModel saved: {model_path}"
    )

    return {

        "class_name": class_name,

        "images": len(image_paths),

        "source_statistics":
            source_stats,

        "history": history,

        "model_path":
            str(model_path),
    }


# ============================================================
# 33. GENERATE SYNTHETIC IMAGES
# ============================================================

def generate_synthetic_images(
    class_name,
    class_index,
    class_names,
):

    print("\n" + "-" * 70)

    print(
        f"GENERATING SYNTHETIC: {class_name.upper()}"
    )

    print("-" * 70)

    class_dir = (
        SYNTHETIC_ROOT
        / class_name
    )

    class_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load source dataset statistics
    # --------------------------------------------------------

    dataset = MRITestingDataset(
        get_class_images(class_name),
        class_index,
    )

    source_stats = calculate_source_statistics(
        dataset
    )

    # --------------------------------------------------------
    # Build generator
    # --------------------------------------------------------

    generator = ConditionalGenerator(
        num_classes=len(class_names),
    ).to(DEVICE)

    model_path = (
        MODEL_ROOT
        / f"{class_name}_generator_ema.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Generator model not found:\n{model_path}"
        )

    generator.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )
    )

    generator.eval()

    accepted_images = []

    metadata = []

    generated_count = 0

    attempt = 0

    max_attempts = (
        NUM_SYNTHETIC * 10
    )

    while (
        generated_count < NUM_SYNTHETIC
        and attempt < max_attempts
    ):

        attempt += 1

        current_batch = min(
            BATCH_SIZE,
            NUM_SYNTHETIC
            - generated_count,
        )

        noise = torch.randn(
            current_batch,
            LATENT_DIM,
            device=DEVICE,
        )

        labels = torch.full(
            (
                current_batch,
            ),
            class_index,
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.no_grad():

            generated = generator(
                noise,
                labels,
            )

        for image in generated:

            if generated_count >= NUM_SYNTHETIC:

                break

            quality = image_quality_score(
                image,
                source_stats,
            )

            diversity = is_diverse_enough(
                image,
                accepted_images,
            )

            accepted = (
                quality["accepted"]
                and diversity
            )

            # ------------------------------------------------
            # Fallback:
            # If quality filtering is too strict near the end,
            # allow statistically reasonable images so the
            # requested count is reached.
            # ------------------------------------------------

            if (
                not accepted
                and attempt
                > max_attempts * 0.75
            ):

                accepted = (
                    quality["score"]
                    < QUALITY_STD_LIMIT * 3
                )

            if not accepted:

                continue

            image_copy = (
                image.detach()
                .cpu()
                .clone()
            )

            accepted_images.append(
                image_copy
            )

            raw_image = tensor_to_image(
                image_copy
            )

            file_name = (
                f"{class_name}_synthetic_"
                f"{generated_count + 1:04d}.png"
            )

            raw_path = (
                class_dir
                / file_name
            )

            raw_image.save(
                raw_path
            )

            # Display-only version.
            display_image = (
                enhance_for_display(
                    raw_image
                )
            )

            display_dir = (
                class_dir
                / "display_preview"
            )

            display_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            display_path = (
                display_dir
                / file_name
            )

            display_image.save(
                display_path
            )

            metadata.append({

                "file":
                    file_name,

                "class":
                    class_name,

                "quality_score":
                    quality["score"],

                "mean":
                    quality["mean"],

                "std":
                    quality["std"],

                "edge":
                    quality["edge"],

                "accepted":
                    True,
            })

            generated_count += 1

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = (
        REPORT_ROOT
        / f"{class_name}_synthetic_metadata.csv"
    )

    save_csv(
        metadata_path,
        metadata,
        [
            "file",
            "class",
            "quality_score",
            "mean",
            "std",
            "edge",
            "accepted",
        ],
    )

    # ========================================================
    # CREATE FINAL CONTACT SHEET
    # ========================================================

    preview_images = []

    for item in accepted_images[
        :PREVIEW_COUNT
    ]:

        image = tensor_to_image(
            item
        )

        image = enhance_for_display(
            image
        )

        preview_images.append(
            image
        )

    final_preview = (
        PREVIEW_ROOT
        / class_name
        / "final_synthetic_samples.png"
    )

    final_preview.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_contact_sheet(
        preview_images,
        final_preview,
        columns=4,
    )

    print(
        f"Generated: {generated_count}/{NUM_SYNTHETIC}"
    )

    print(
        f"Raw images: {class_dir}"
    )

    print(
        f"Preview: {final_preview}"
    )

    return {

        "class_name":
            class_name,

        "requested":
            NUM_SYNTHETIC,

        "generated":
            generated_count,

        "attempts":
            attempt,

        "output":
            str(class_dir),

        "preview":
            str(final_preview),
    }


# ============================================================
# 34. SAVE REAL VS SYNTHETIC SUMMARY
# ============================================================

def save_summary(
    training_results,
    generation_results,
    class_names,
):

    summary = {

        "project":
            "MRI Testing GAN V8",

        "created_at":
            datetime.now().isoformat(),

        "device":
            str(DEVICE),

        "testing_dataset":
            str(TESTING_PATH),

        "output_directory":
            str(RUN_DIR),

        "image_size":
            IMAGE_SIZE,

        "latent_dim":
            LATENT_DIM,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "classes_available":
            class_names,

        "pilot_classes":
            PILOT_CLASSES,

        "training":
            training_results,

        "generation":
            generation_results,
    }

    path = (
        REPORT_ROOT
        / "run_summary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    return path


# ============================================================
# 35. MAIN
# ============================================================

def main():

    create_directories()

    print_configuration()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not TESTING_PATH.exists():

        raise FileNotFoundError(
            "\nMRI Testing dataset does not exist:\n"
            f"{TESTING_PATH}\n"
        )

    print(
        "IMPORTANT SAFETY CHECK:"
    )

    print(
        "This script will READ only:"
    )

    print(
        f"  {TESTING_PATH}"
    )

    print(
        "This script will WRITE only:"
    )

    print(
        f"  {RUN_DIR}"
    )

    print(
        "\nTraining data will NOT be modified."
    )

    print(
        "Previous GAN outputs will NOT be deleted.\n"
    )

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    all_classes = discover_classes()

    if not all_classes:

        raise RuntimeError(
            "No MRI classes were found inside Testing."
        )

    print(
        "Classes detected:"
    )

    for class_name in all_classes:

        count = len(
            get_class_images(
                class_name
            )
        )

        print(
            f"  - {class_name}: {count} images"
        )

    # --------------------------------------------------------
    # Select classes
    # --------------------------------------------------------

    if PILOT_CLASSES is None:

        selected_classes = all_classes

    else:

        selected_classes = []

        for class_name in PILOT_CLASSES:

            if class_name in all_classes:

                selected_classes.append(
                    class_name
                )

            else:

                print(
                    f"WARNING: {class_name} "
                    "not found. Skipping."
                )

    if not selected_classes:

        raise RuntimeError(
            "No valid classes selected."
        )

    print(
        "\nClasses selected for V8:"
    )

    for class_name in selected_classes:

        print(
            f"  - {class_name}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_results = []

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(all_classes)
    }

    for class_name in selected_classes:

        result = train_class(
            class_name,
            class_to_index[class_name],
            all_classes,
        )

        if result is not None:

            training_results.append(
                result
            )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_results = []

    for class_name in selected_classes:

        result = generate_synthetic_images(
            class_name,
            class_to_index[class_name],
            all_classes,
        )

        generation_results.append(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = save_summary(
        training_results,
        generation_results,
        all_classes,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("MRI TESTING GAN V8 COMPLETED")

    print("=" * 70)

    print(
        f"Run directory:\n{RUN_DIR}"
    )

    print(
        f"\nSynthetic images:\n{SYNTHETIC_ROOT}"
    )

    print(
        f"\nModels:\n{MODEL_ROOT}"
    )

    print(
        f"\nPreviews:\n{PREVIEW_ROOT}"
    )

    print(
        f"\nReports:\n{REPORT_ROOT}"
    )

    print(
        f"\nSummary:\n{summary_path}"
    )

    print("\nClasses generated:")

    for result in generation_results:

        print(
            f"  {result['class_name']}: "
            f"{result['generated']} images"
        )

    print("\nPrevious V2/V3/V4/V5/V7 outputs were not touched.")

    print(
        "Testing dataset was read-only."
    )

    print("=" * 70)


# ============================================================
# 36. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
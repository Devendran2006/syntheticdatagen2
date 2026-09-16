"""
MRI TESTING DATASET - STABLE GAN V2
===================================

Purpose
-------
Train a completely separate GAN using ONLY:

    data/imaging/MRI/Testing/

Classes:
    glioma
    meningioma
    notumor
    pituitary

IMPORTANT
---------
This script NEVER modifies:

    data/imaging/MRI/Training/
    data/imaging/MRI/Training_Preprocessed/

It also does NOT modify existing:

    outputs/mri/
    outputs/mri/v4/
    outputs/mri/v5/
    outputs/mri/v5_1/
    outputs/mri/v5_2/

All new results are stored under:

    outputs/mri_testing_gan/

GAN METHOD
----------
LSGAN (Least Squares GAN)

Why LSGAN?
----------
The previous BCE GAN became unstable very quickly:

    Generator Loss  -> very high
    Discriminator   -> almost zero

LSGAN generally provides smoother gradients and is less likely
to produce the immediate discriminator domination seen in V1.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# File:
#
# synthetic data/
# └── src/
#     └── mri/
#         └── mri_testing_gan.py
#
# parents[2] = synthetic data/

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# Existing MRI data
MRI_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "MRI"
)


# ONLY INPUT DATA
TESTING_PATH = (
    MRI_ROOT
    / "Testing"
)


# Completely separate output
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_gan"
)


MODEL_ROOT = (
    OUTPUT_ROOT
    / "models"
)


SAMPLE_ROOT = (
    OUTPUT_ROOT
    / "synthetic_samples"
)


PREVIEW_ROOT = (
    OUTPUT_ROOT
    / "previews"
)


HISTORY_ROOT = (
    OUTPUT_ROOT
    / "training_history"
)


CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
)


EVALUATION_ROOT = (
    OUTPUT_ROOT
    / "evaluation"
)


# Create only NEW output folders
for folder in [
    OUTPUT_ROOT,
    MODEL_ROOT,
    SAMPLE_ROOT,
    PREVIEW_ROOT,
    HISTORY_ROOT,
    CHECKPOINT_ROOT,
    EVALUATION_ROOT,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

# Internal GAN resolution.
#
# 64 is intentionally used because you are currently running
# on CPU. It dramatically reduces training time compared with
# 128x128 GAN training.
#
# Generated images are later resized to 128x128 for dashboard use.

GAN_IMAGE_SIZE = 64

OUTPUT_IMAGE_SIZE = 128


CHANNELS = 1


LATENT_DIM = 128


# Start with 10 for validation.
#
# Once the result looks reasonable, increase to 30 or 50.

EPOCHS = 10


BATCH_SIZE = 32


# LSGAN usually behaves better with a conservative LR.

LEARNING_RATE_G = 0.0001

LEARNING_RATE_D = 0.0001


BETA1 = 0.5

BETA2 = 0.999


NUM_WORKERS = 0


# Number of final synthetic images per class.

GENERATE_IMAGES = 100


# Save preview after every N epochs.

SAVE_PREVIEW_EVERY = 2


# Save checkpoint after every N epochs.

SAVE_CHECKPOINT_EVERY = 5


# Random seed.

SEED = 42


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 5. SEED
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed()


# ============================================================
# 6. IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# 7. IMAGE DISCOVERY
# ============================================================

def get_image_files(folder):

    if not folder.exists():

        return []

    files = []

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            files.append(path)

    return sorted(files)


# ============================================================
# 8. DISCOVER CLASSES
# ============================================================

def discover_classes():

    if not TESTING_PATH.exists():

        raise FileNotFoundError(
            f"\nTesting dataset not found:\n"
            f"{TESTING_PATH}\n"
        )

    classes = []

    for folder in sorted(
        TESTING_PATH.iterdir()
    ):

        if not folder.is_dir():

            continue

        image_count = len(
            get_image_files(folder)
        )

        if image_count > 0:

            classes.append(
                folder.name
            )

    if not classes:

        raise RuntimeError(
            "\nNo MRI classes found in:\n"
            f"{TESTING_PATH}\n"
        )

    return classes


# ============================================================
# 9. DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(
        self,
        image_paths
    ):

        self.image_paths = image_paths

        # IMPORTANT:
        #
        # No aggressive augmentation.
        #
        # MRI anatomical structure should not be unnecessarily
        # distorted.

        self.transform = transforms.Compose([

            transforms.Resize(
                (
                    GAN_IMAGE_SIZE,
                    GAN_IMAGE_SIZE
                )
            ),

            transforms.Grayscale(
                num_output_channels=1
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.5],
                [0.5]
            )
        ])


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        path = self.image_paths[index]

        try:

            image = Image.open(
                path
            ).convert("L")

            image = self.transform(
                image
            )

            return image

        except Exception as error:

            print(
                f"\nWarning: unable to read:"
            )

            print(path)

            print(error)

            # Try another image.

            next_index = (
                index + 1
            ) % len(
                self.image_paths
            )

            return self.__getitem__(
                next_index
            )


# ============================================================
# 10. GENERATOR
# ============================================================

class Generator(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()


        self.model = nn.Sequential(

            # ------------------------------------------------
            # 1 x 1 -> 4 x 4
            # ------------------------------------------------

            nn.ConvTranspose2d(
                latent_dim,
                512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 4 x 4 -> 8 x 8
            # ------------------------------------------------

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 8 x 8 -> 16 x 16
            # ------------------------------------------------

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 16 x 16 -> 32 x 32
            # ------------------------------------------------

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),


            # ------------------------------------------------
            # 32 x 32 -> 64 x 64
            # ------------------------------------------------

            nn.ConvTranspose2d(
                64,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()
        )


    def forward(
        self,
        noise
    ):

        return self.model(
            noise
        )


# ============================================================
# 11. DISCRIMINATOR
# ============================================================

class Discriminator(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.model = nn.Sequential(

            # ------------------------------------------------
            # 64 -> 32
            # ------------------------------------------------

            nn.utils.spectral_norm(
                nn.Conv2d(
                    1,
                    64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),


            # ------------------------------------------------
            # 32 -> 16
            # ------------------------------------------------

            nn.utils.spectral_norm(
                nn.Conv2d(
                    64,
                    128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),


            # ------------------------------------------------
            # 16 -> 8
            # ------------------------------------------------

            nn.utils.spectral_norm(
                nn.Conv2d(
                    128,
                    256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),


            # ------------------------------------------------
            # 8 -> 4
            # ------------------------------------------------

            nn.utils.spectral_norm(
                nn.Conv2d(
                    256,
                    512,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # ------------------------------------------------
            # 4 -> 1
            # ------------------------------------------------

            nn.utils.spectral_norm(
                nn.Conv2d(
                    512,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False
                )
            )
        )


    def forward(
        self,
        image
    ):

        return self.model(
            image
        ).view(-1)


# ============================================================
# 12. WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(
    model
):

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

        elif isinstance(
            module,
            nn.BatchNorm2d
        ):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0
            )


# ============================================================
# 13. SAVE PREVIEW
# ============================================================

def save_preview(
    generator,
    fixed_noise,
    class_name,
    epoch
):

    generator.eval()

    with torch.no_grad():

        fake_images = generator(
            fixed_noise
        )

    preview_dir = (
        PREVIEW_ROOT
        / class_name
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    preview_path = (
        preview_dir
        / f"epoch_{epoch:04d}.png"
    )

    save_image(
        fake_images,
        preview_path,
        normalize=True,
        nrow=4
    )

    generator.train()

    return preview_path


# ============================================================
# 14. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    epoch,
    class_name,
    history
):

    checkpoint_path = (
        CHECKPOINT_ROOT
        / f"{class_name}_latest.pth"
    )

    torch.save({

        "epoch": epoch,

        "class_name": class_name,

        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "optimizer_g_state_dict":
            optimizer_g.state_dict(),

        "optimizer_d_state_dict":
            optimizer_d.state_dict(),

        "history": history,

        "config": {

            "image_size":
                GAN_IMAGE_SIZE,

            "latent_dim":
                LATENT_DIM,

            "batch_size":
                BATCH_SIZE,

            "learning_rate_g":
                LEARNING_RATE_G,

            "learning_rate_d":
                LEARNING_RATE_D
        }

    }, checkpoint_path)

    return checkpoint_path


# ============================================================
# 15. GENERATE FINAL IMAGES
# ============================================================

def generate_final_images(
    generator,
    class_name
):

    generator.eval()


    output_dir = (
        SAMPLE_ROOT
        / class_name
        / "generated"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    generated_count = 0


    print(
        f"\nGenerating "
        f"{GENERATE_IMAGES} synthetic "
        f"{class_name} images..."
    )


    with torch.no_grad():

        while (
            generated_count
            < GENERATE_IMAGES
        ):

            remaining = (
                GENERATE_IMAGES
                - generated_count
            )

            current_batch = min(
                BATCH_SIZE,
                remaining
            )


            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )


            fake_images = generator(
                noise
            )


            for i in range(
                current_batch
            ):

                image = (
                    fake_images[i]
                    .detach()
                    .cpu()
                )


                # Convert [-1,1] -> [0,1]

                image = (
                    image + 1
                ) / 2


                image = torch.clamp(
                    image,
                    0,
                    1
                )


                # Upscale to 128x128
                # for dashboard use.

                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(
                        OUTPUT_IMAGE_SIZE,
                        OUTPUT_IMAGE_SIZE
                    ),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)


                image_number = (
                    generated_count
                    + i
                    + 1
                )


                output_path = (
                    output_dir
                    / (
                        f"{class_name}_"
                        f"synthetic_"
                        f"{image_number:05d}.png"
                    )
                )


                save_image(
                    image,
                    output_path
                )


            generated_count += (
                current_batch
            )


    print(
        f"Generated: "
        f"{generated_count} images"
    )


    return output_dir


# ============================================================
# 16. TRAIN ONE CLASS
# ============================================================

def train_class(
    class_name,
    image_paths
):

    print("\n")
    print("=" * 72)
    print(
        f"TRAINING TESTING-DATA GAN: "
        f"{class_name.upper()}"
    )
    print("=" * 72)

    print(
        f"Real Testing images : "
        f"{len(image_paths)}"
    )

    print(
        f"GAN image size      : "
        f"{GAN_IMAGE_SIZE} x {GAN_IMAGE_SIZE}"
    )

    print(
        f"Output image size   : "
        f"{OUTPUT_IMAGE_SIZE} x "
        f"{OUTPUT_IMAGE_SIZE}"
    )

    print(
        f"Epochs              : "
        f"{EPOCHS}"
    )

    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Device              : "
        f"{DEVICE}"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = MRIDataset(
        image_paths
    )


    actual_batch_size = min(
        BATCH_SIZE,
        len(dataset)
    )


    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    generator = Generator().to(
        DEVICE
    )

    discriminator = (
        Discriminator()
        .to(DEVICE)
    )


    initialize_weights(
        generator
    )

    initialize_weights(
        discriminator
    )


    # --------------------------------------------------------
    # LSGAN Loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()


    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE_G,
        betas=(
            BETA1,
            BETA2
        )
    )


    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE_D,
        betas=(
            BETA1,
            BETA2
        )
    )


    # --------------------------------------------------------
    # Fixed noise
    # --------------------------------------------------------

    fixed_noise = torch.randn(
        16,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []


    start_time = time.time()


    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()


        generator.train()

        discriminator.train()


        total_g_loss = 0.0

        total_d_loss = 0.0

        real_score_total = 0.0

        fake_score_total = 0.0

        batch_count = 0


        for real_images in dataloader:

            real_images = (
                real_images
                .to(DEVICE)
            )


            current_batch = (
                real_images.size(0)
            )


            # =================================================
            # TRAIN DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad()


            # Real images

            real_output = (
                discriminator(
                    real_images
                )
            )


            real_targets = torch.ones(
                current_batch,
                device=DEVICE
            )


            real_loss = criterion(
                real_output,
                real_targets
            )


            # Fake images

            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )


            fake_images = generator(
                noise
            )


            fake_output = (
                discriminator(
                    fake_images.detach()
                )
            )


            fake_targets = torch.zeros(
                current_batch,
                device=DEVICE
            )


            fake_loss = criterion(
                fake_output,
                fake_targets
            )


            d_loss = (
                real_loss
                + fake_loss
            ) / 2


            d_loss.backward()


            optimizer_d.step()


            # =================================================
            # TRAIN GENERATOR
            # =================================================

            optimizer_g.zero_grad()


            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )


            generated_images = (
                generator(
                    noise
                )
            )


            generated_output = (
                discriminator(
                    generated_images
                )
            )


            generator_targets = (
                torch.ones(
                    current_batch,
                    device=DEVICE
                )
            )


            g_loss = criterion(
                generated_output,
                generator_targets
            )


            g_loss.backward()


            optimizer_g.step()


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            total_g_loss += (
                g_loss.item()
            )

            total_d_loss += (
                d_loss.item()
            )

            real_score_total += (
                real_output
                .detach()
                .mean()
                .item()
            )

            fake_score_total += (
                fake_output
                .detach()
                .mean()
                .item()
            )


            batch_count += 1


        # ====================================================
        # EPOCH AVERAGES
        # ====================================================

        avg_g_loss = (
            total_g_loss
            / max(
                batch_count,
                1
            )
        )


        avg_d_loss = (
            total_d_loss
            / max(
                batch_count,
                1
            )
        )


        avg_real_score = (
            real_score_total
            / max(
                batch_count,
                1
            )
        )


        avg_fake_score = (
            fake_score_total
            / max(
                batch_count,
                1
            )
        )


        epoch_time = (
            time.time()
            - epoch_start
        )


        history.append({

            "class":
                class_name,

            "epoch":
                epoch,

            "generator_loss":
                round(
                    avg_g_loss,
                    6
                ),

            "discriminator_loss":
                round(
                    avg_d_loss,
                    6
                ),

            "real_score":
                round(
                    avg_real_score,
                    6
                ),

            "fake_score":
                round(
                    avg_fake_score,
                    6
                ),

            "epoch_time_seconds":
                round(
                    epoch_time,
                    2
                )
        })


        # ====================================================
        # CONSOLE
        # ====================================================

        print(
            f"Epoch "
            f"[{epoch:03d}/{EPOCHS}] | "
            f"G: {avg_g_loss:.4f} | "
            f"D: {avg_d_loss:.4f} | "
            f"Real: {avg_real_score:.4f} | "
            f"Fake: {avg_fake_score:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )


        # ====================================================
        # PREVIEW
        # ====================================================

        if (
            epoch == 1
            or epoch % SAVE_PREVIEW_EVERY == 0
            or epoch == EPOCHS
        ):

            preview_path = (
                save_preview(
                    generator,
                    fixed_noise,
                    class_name,
                    epoch
                )
            )

            print(
                f"Preview saved: "
                f"{preview_path}"
            )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (
            epoch % SAVE_CHECKPOINT_EVERY == 0
            or epoch == EPOCHS
        ):

            checkpoint_path = (
                save_checkpoint(
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    epoch,
                    class_name,
                    history
                )
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )


    # ========================================================
    # SAVE FINAL MODELS
    # ========================================================

    generator_path = (
        MODEL_ROOT
        / f"{class_name}_generator.pth"
    )


    discriminator_path = (
        MODEL_ROOT
        / f"{class_name}_discriminator.pth"
    )


    torch.save(
        generator.state_dict(),
        generator_path
    )


    torch.save(
        discriminator.state_dict(),
        discriminator_path
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_path = (
        HISTORY_ROOT
        / f"{class_name}_training_history.csv"
    )


    history_df.to_csv(
        history_path,
        index=False
    )


    # ========================================================
    # GENERATE FINAL IMAGES
    # ========================================================

    generated_dir = (
        generate_final_images(
            generator,
            class_name
        )
    )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )


    print("\n")
    print("-" * 72)
    print(
        f"{class_name.upper()} GAN COMPLETED"
    )
    print("-" * 72)

    print(
        f"Real images       : "
        f"{len(image_paths)}"
    )

    print(
        f"Synthetic images  : "
        f"{GENERATE_IMAGES}"
    )

    print(
        f"Generator model   : "
        f"{generator_path}"
    )

    print(
        f"Discriminator     : "
        f"{discriminator_path}"
    )

    print(
        f"History           : "
        f"{history_path}"
    )

    print(
        f"Training time     : "
        f"{total_time / 60:.2f} minutes"
    )

    print("-" * 72)


    return {

        "class":
            class_name,

        "real_images":
            len(image_paths),

        "synthetic_images":
            GENERATE_IMAGES,

        "epochs":
            EPOCHS,

        "generator_model":
            str(generator_path),

        "discriminator_model":
            str(discriminator_path),

        "training_history":
            str(history_path),

        "synthetic_directory":
            str(generated_dir),

        "training_time_minutes":
            round(
                total_time / 60,
                2
            )
    }


# ============================================================
# 17. TRAIN ALL CLASSES
# ============================================================

def train_all_classes():

    print("\n")
    print("=" * 72)
    print(
        "MRI TESTING DATASET - STABLE GAN V2"
    )
    print("=" * 72)

    print(
        "\nINPUT:"
    )

    print(
        TESTING_PATH
    )

    print(
        "\nOUTPUT:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nDEVICE:"
    )

    print(
        DEVICE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Training dataset will NOT be touched."
    )

    print(
        "Existing MRI GAN outputs will NOT be touched."
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    classes = discover_classes()


    print(
        f"\nClasses detected:"
    )

    for class_name in classes:

        count = len(
            get_image_files(
                TESTING_PATH
                / class_name
            )
        )

        print(
            f"  {class_name:<15} "
            f"{count} images"
        )


    results = []


    # --------------------------------------------------------
    # Train each class
    # --------------------------------------------------------

    for class_name in classes:

        class_path = (
            TESTING_PATH
            / class_name
        )


        image_paths = (
            get_image_files(
                class_path
            )
        )


        if not image_paths:

            print(
                f"\nSkipping "
                f"{class_name}: "
                f"no images."
            )

            continue


        result = train_class(
            class_name,
            image_paths
        )


        results.append(
            result
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    if not results:

        print(
            "\nNo classes were trained."
        )

        return


    summary_df = pd.DataFrame(
        results
    )


    summary_path = (
        OUTPUT_ROOT
        / "training_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 72)
    print(
        "MRI TESTING GAN V2 COMPLETED"
    )
    print("=" * 72)


    print(
        summary_df[
            [
                "class",
                "real_images",
                "synthetic_images",
                "epochs"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nTraining summary:"
    )

    print(
        summary_path
    )


    print(
        "\nModels:"
    )

    print(
        MODEL_ROOT
    )


    print(
        "\nSynthetic MRI:"
    )

    print(
        SAMPLE_ROOT
    )


    print(
        "\nPreviews:"
    )

    print(
        PREVIEW_ROOT
    )


    print(
        "\nTraining history:"
    )

    print(
        HISTORY_ROOT
    )


    print("=" * 72)


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print(
        "MRI Testing GAN V2"
    )

    print(
        "------------------"
    )

    print(
        "Testing dataset ONLY"
    )

    print(
        "Existing training data protected"
    )

    print(
        "Existing GAN outputs protected"
    )


    train_all_classes()
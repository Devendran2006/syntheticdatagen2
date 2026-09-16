
"""
MRI TRAINING SYNTHETIC DATA GENERATION
======================================

Purpose
-------
Generate synthetic MRI images for the TRAINING dataset.

DATA SEPARATION
---------------
REAL TRAINING:
    data/imaging/MRI/Training

REAL TESTING:
    data/imaging/MRI/Testing

This script:
    - Loads the Generator trained using REAL TRAINING data
    - Generates synthetic TRAINING images
    - NEVER uses Testing images for generation
    - NEVER modifies real MRI data
    - Saves synthetic training data separately

Generator checkpoint:
    outputs/mri/checkpoints/mri_generator_final.pth

Output:
    outputs/mri/training_synthetic/synthetic/

Classes:
    glioma
    meningioma
    notumor
    pituitary
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# REAL TRAINING DATA
# ------------------------------------------------------------

TRAINING_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Training"
)

# ------------------------------------------------------------
# REAL TESTING DATA
# ------------------------------------------------------------

TESTING_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)

# ------------------------------------------------------------
# CHECKPOINT
# ------------------------------------------------------------

CHECKPOINT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "checkpoints"
)

GENERATOR_CHECKPOINT = (
    CHECKPOINT_DIR
    / "mri_generator_final.pth"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
)

SYNTHETIC_DIR = (
    OUTPUT_DIR
    / "synthetic"
)

PREVIEW_DIR = (
    OUTPUT_DIR
    / "preview"
)

RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

IMAGE_SIZE = 64

LATENT_DIM = 128

NUM_CLASSES = 4

IMAGE_CHANNELS = 1

BATCH_SIZE = 32

SYNTHETIC_PER_CLASS = 500

SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CPU OPTIMIZATION
# ============================================================

if DEVICE.type == "cpu":

    try:

        torch.set_num_threads(
            min(4, os.cpu_count() or 1)
        )

    except Exception:

        pass


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 70)
print("        MRI TRAINING SYNTHETIC DATA GENERATION")
print("=" * 70)

print()

print(f"Device          : {DEVICE}")
print(f"Image size      : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Latent dimension: {LATENT_DIM}")
print(f"Classes         : {NUM_CLASSES}")
print(f"Images/class    : {SYNTHETIC_PER_CLASS}")
print(
    f"Total images    : "
    f"{SYNTHETIC_PER_CLASS * NUM_CLASSES}"
)

print()

print("IMPORTANT:")
print("This script generates SYNTHETIC TRAINING data only.")
print("Generator was trained using REAL TRAINING data.")
print("Real TESTING data is NEVER used.")
print("Real TESTING data is NEVER modified.")

print()


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):
    """
    Conditional DCGAN Generator.

    IMPORTANT:
    This architecture is matched to the successful
    mri_generator_final.pth checkpoint.

    The final ConvTranspose2d uses:

        bias=False

    This is required because the checkpoint contains:

        main.9.weight

    but does NOT contain:

        main.9.bias
    """

    def __init__(
        self,
        latent_dim=128,
        num_classes=4,
        image_channels=1
    ):

        super().__init__()

        self.latent_dim = latent_dim

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Conditional class embedding
        # ----------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )

        # ----------------------------------------------------
        # Noise + class embedding
        #
        # 128 + 128 = 256
        # ----------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(
                latent_dim * 2,
                512 * 4 * 4
            ),

            nn.BatchNorm1d(
                512 * 4 * 4
            ),

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # Generator network
        #
        # 4x4
        #   ↓
        # 8x8
        #   ↓
        # 16x16
        #   ↓
        # 32x32
        #   ↓
        # 64x64
        # ----------------------------------------------------

        self.main = nn.Sequential(

            # ------------------------------------------------
            # 4x4 -> 8x8
            # ------------------------------------------------

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(True),

            # ------------------------------------------------
            # 8x8 -> 16x16
            # ------------------------------------------------

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(True),

            # ------------------------------------------------
            # 16x16 -> 32x32
            # ------------------------------------------------

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(True),

            # ------------------------------------------------
            # 32x32 -> 64x64
            #
            # IMPORTANT:
            # bias=False
            #
            # This exactly matches the checkpoint.
            # ------------------------------------------------

            nn.ConvTranspose2d(
                64,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        z,
        labels
    ):

        # ----------------------------------------------------
        # Get class embedding
        # ----------------------------------------------------

        label_vector = self.label_embedding(
            labels
        )

        # ----------------------------------------------------
        # Combine noise + class
        # ----------------------------------------------------

        x = torch.cat(
            [
                z,
                label_vector
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        x = self.fc(x)

        # ----------------------------------------------------
        # Reshape
        # ----------------------------------------------------

        x = x.view(
            x.size(0),
            512,
            4,
            4
        )

        # ----------------------------------------------------
        # Generate image
        # ----------------------------------------------------

        return self.main(x)


# ============================================================
# CHECKPOINT INSPECTION
# ============================================================

def inspect_checkpoint():

    print("=" * 70)
    print("             CHECKPOINT INSPECTION")
    print("=" * 70)

    print()

    if not GENERATOR_CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nGenerator checkpoint not found:\n"
            f"{GENERATOR_CHECKPOINT}\n\n"
            "Run train_gan.py first."
        )

    print(
        f"Checkpoint:\n"
        f"{GENERATOR_CHECKPOINT}"
    )

    print()

    checkpoint = torch.load(
        GENERATOR_CHECKPOINT,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "generator_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "generator_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove module prefix
    # --------------------------------------------------------

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    state_dict = cleaned

    print(
        f"Checkpoint parameters: "
        f"{len(state_dict)}"
    )

    print()

    # --------------------------------------------------------
    # Verify important architecture keys
    # --------------------------------------------------------

    required_keys = [

        "label_embedding.weight",

        "fc.0.weight",

        "fc.0.bias",

        "main.0.weight",

        "main.3.weight",

        "main.6.weight",

        "main.9.weight"
    ]

    print("Architecture verification:")

    for key in required_keys:

        if key in state_dict:

            shape = tuple(
                state_dict[key].shape
            )

            print(
                f"  OK   {key:<30} {shape}"
            )

        else:

            print(
                f"  MISS {key}"
            )

    # --------------------------------------------------------
    # Check final bias
    # --------------------------------------------------------

    if "main.9.bias" in state_dict:

        print(
            "\nWARNING: main.9.bias exists in checkpoint."
        )

        print(
            "Checkpoint appears to use a biased final layer."
        )

    else:

        print(
            "\nOK: main.9.bias is absent."
        )

        print(
            "Using bias=False for final layer."
        )

    print()

    return state_dict


# ============================================================
# LOAD GENERATOR
# ============================================================

def load_generator():

    print("=" * 70)
    print("             LOADING TRAINED GENERATOR")
    print("=" * 70)

    print()

    state_dict = inspect_checkpoint()

    print()

    print("Creating Generator architecture...")

    generator = Generator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        image_channels=IMAGE_CHANNELS
    ).to(DEVICE)

    print(
        "Loading checkpoint..."
    )

    try:

        generator.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as error:

        print()

        print("=" * 70)
        print("             CHECKPOINT LOAD ERROR")
        print("=" * 70)

        print()

        print(error)

        print()

        print(
            "The checkpoint still does not match."
        )

        print(
            "The checkpoint file has NOT been deleted."
        )

        print()

        raise

    generator.eval()

    print()

    print(
        "Generator loaded successfully."
    )

    print()

    return generator


# ============================================================
# INSPECT REAL TRAINING DATA
# ============================================================

def inspect_real_training_data():

    print("=" * 70)
    print("             REAL MRI TRAINING DATA")
    print("=" * 70)

    print()

    if not TRAINING_DIR.exists():

        raise FileNotFoundError(
            "\nTraining directory not found:\n"
            f"{TRAINING_DIR}"
        )

    counts = {}

    total = 0

    for class_name in CLASS_NAMES:

        class_dir = (
            TRAINING_DIR
            / class_name
        )

        if not class_dir.exists():

            print(
                f"{class_name:<15}: DIRECTORY MISSING"
            )

            counts[class_name] = 0

            continue

        files = []

        for pattern in [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp"
        ]:

            files.extend(
                class_dir.glob(pattern)
            )

        count = len(files)

        counts[class_name] = count

        total += count

        print(
            f"{class_name:<15}: "
            f"{count:>5} images"
        )

    print()

    print(
        f"Total Real Training Images: "
        f"{total}"
    )

    print()

    return counts


# ============================================================
# CLEAN PREVIOUS OUTPUT
# ============================================================

def clean_previous_output():

    print("=" * 70)
    print("             PREPARING OUTPUT DIRECTORY")
    print("=" * 70)

    print()

    if SYNTHETIC_DIR.exists():

        print(
            "Removing previous synthetic training data..."
        )

        shutil.rmtree(
            SYNTHETIC_DIR
        )

    SYNTHETIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Synthetic output directory ready."
    )

    print()


# ============================================================
# GENERATE ONE CLASS
# ============================================================

@torch.no_grad()
def generate_class_images(
    generator,
    class_index,
    class_name
):

    class_dir = (
        SYNTHETIC_DIR
        / class_name
    )

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Generating {class_name}..."
    )

    preview_images = []

    generated = 0

    while generated < SYNTHETIC_PER_CLASS:

        current_batch = min(
            BATCH_SIZE,
            SYNTHETIC_PER_CLASS - generated
        )

        # ----------------------------------------------------
        # Random latent vectors
        # ----------------------------------------------------

        z = torch.randn(
            current_batch,
            LATENT_DIM,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Class labels
        # ----------------------------------------------------

        labels = torch.full(
            (current_batch,),
            class_index,
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        fake_images = generator(
            z,
            labels
        )

        fake_images_cpu = (
            fake_images.cpu()
        )

        # ----------------------------------------------------
        # Save images
        # ----------------------------------------------------

        for index in range(
            current_batch
        ):

            image = (
                fake_images_cpu[index]
            )

            image_number = (
                generated
                + index
                + 1
            )

            filename = (
                f"{class_name}_"
                f"synthetic_"
                f"{image_number:04d}.png"
            )

            output_path = (
                class_dir
                / filename
            )

            save_image(
                image,
                output_path
            )

            # Keep first 8 for preview

            if len(preview_images) < 8:

                preview_images.append(
                    image
                )

        generated += current_batch

        print(
            f"  {generated}/"
            f"{SYNTHETIC_PER_CLASS}",
            end="\r"
        )

    print()

    print(
        f"  Completed: "
        f"{SYNTHETIC_PER_CLASS} images"
    )

    print()

    return preview_images


# ============================================================
# CREATE PREVIEW
# ============================================================

def create_preview(
    preview_images
):

    print("=" * 70)
    print("             CREATING PREVIEW")
    print("=" * 70)

    print()

    if not preview_images:

        print(
            "No preview images available."
        )

        return

    all_images = []

    for class_name in CLASS_NAMES:

        class_dir = (
            SYNTHETIC_DIR
            / class_name
        )

        files = sorted(
            class_dir.glob("*.png")
        )

        for image_path in files[:8]:

            try:

                image = Image.open(
                    image_path
                ).convert("L")

                array = (
                    np.asarray(
                        image,
                        dtype=np.float32
                    )
                    / 255.0
                )

                tensor = torch.from_numpy(
                    array
                ).unsqueeze(0)

                all_images.append(
                    tensor
                )

            except Exception:

                pass

    if not all_images:

        return

    images = torch.stack(
        all_images
    )

    grid = make_grid(
        images,
        nrow=8,
        padding=2
    )

    preview_path = (
        PREVIEW_DIR
        / "training_synthetic_preview.png"
    )

    save_image(
        grid,
        preview_path
    )

    print(
        f"Preview saved:\n"
        f"{preview_path}"
    )

    print()


# ============================================================
# VERIFY GENERATED DATA
# ============================================================

def verify_generated_data():

    print("=" * 70)
    print("             VERIFYING GENERATED DATA")
    print("=" * 70)

    print()

    rows = []

    total = 0

    valid_total = 0

    for class_name in CLASS_NAMES:

        class_dir = (
            SYNTHETIC_DIR
            / class_name
        )

        if not class_dir.exists():

            print(
                f"{class_name:<15}: "
                f"DIRECTORY MISSING"
            )

            continue

        files = sorted(
            class_dir.glob("*.png")
        )

        valid = 0

        for image_path in files:

            try:

                with Image.open(
                    image_path
                ) as image:

                    image.verify()

                valid += 1

            except Exception:

                pass

        total += len(files)

        valid_total += valid

        print(
            f"{class_name:<15}: "
            f"{valid:>5} valid images"
        )

        rows.append({

            "Class":
                class_name,

            "Synthetic_Images":
                valid,

            "Expected_Images":
                SYNTHETIC_PER_CLASS,

            "Status":
                (
                    "OK"
                    if valid ==
                    SYNTHETIC_PER_CLASS
                    else "CHECK"
                )
        })

    print()

    print(
        f"Total files : {total}"
    )

    print(
        f"Valid files : {valid_total}"
    )

    print()

    expected_total = (
        SYNTHETIC_PER_CLASS
        * NUM_CLASSES
    )

    if valid_total != expected_total:

        raise RuntimeError(
            "\nGenerated image count mismatch.\n"
            f"Expected: {expected_total}\n"
            f"Found: {valid_total}"
        )

    print(
        "All generated images verified successfully."
    )

    print()

    return pd.DataFrame(rows)


# ============================================================
# CREATE DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    real_counts,
    verification_df
):

    rows = []

    for class_name in CLASS_NAMES:

        real_count = (
            real_counts.get(
                class_name,
                0
            )
        )

        matching = verification_df[
            verification_df["Class"]
            == class_name
        ]

        if len(matching) > 0:

            synthetic_count = int(
                matching.iloc[0][
                    "Synthetic_Images"
                ]
            )

        else:

            synthetic_count = 0

        combined_count = (
            real_count
            + synthetic_count
        )

        expansion = (
            synthetic_count
            / real_count
            * 100
            if real_count > 0
            else 0
        )

        rows.append({

            "Class":
                class_name,

            "Real_Training_Images":
                real_count,

            "Synthetic_Training_Images":
                synthetic_count,

            "Combined_Images":
                combined_count,

            "Synthetic_Expansion_Percent":
                expansion
        })

    df = pd.DataFrame(
        rows
    )

    output_path = (
        RESULTS_DIR
        / "training_dataset_summary.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Dataset summary saved:\n"
        f"{output_path}"
    )

    print()

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # Check real training dataset
    # --------------------------------------------------------

    real_counts = (
        inspect_real_training_data()
    )

    # --------------------------------------------------------
    # STEP 2
    # Clean previous synthetic data
    # --------------------------------------------------------

    clean_previous_output()

    # --------------------------------------------------------
    # STEP 3
    # Load generator
    # --------------------------------------------------------

    generator = load_generator()

    # --------------------------------------------------------
    # STEP 4
    # Generate all classes
    # --------------------------------------------------------

    print("=" * 70)
    print("             GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 70)

    print()

    preview_images = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_preview = (
            generate_class_images(
                generator,
                class_index,
                class_name
            )
        )

        preview_images.extend(
            class_preview
        )

    # --------------------------------------------------------
    # STEP 5
    # Preview
    # --------------------------------------------------------

    create_preview(
        preview_images
    )

    # --------------------------------------------------------
    # STEP 6
    # Verify
    # --------------------------------------------------------

    verification_df = (
        verify_generated_data()
    )

    # --------------------------------------------------------
    # STEP 7
    # Dataset summary
    # --------------------------------------------------------

    summary_df = (
        create_dataset_summary(
            real_counts,
            verification_df
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("       MRI TRAINING SYNTHETIC GENERATION COMPLETED")
    print("=" * 70)

    print()

    for class_name in CLASS_NAMES:

        count = int(
            verification_df[
                verification_df["Class"]
                == class_name
            ]["Synthetic_Images"].iloc[0]
        )

        print(
            f"{class_name:<15}: "
            f"{count:,} images"
        )

    print("-" * 70)

    total = int(
        verification_df[
            "Synthetic_Images"
        ].sum()
    )

    print(
        f"Total Synthetic MRI Images: "
        f"{total:,}"
    )

    print()

    print(
        "Saved synthetic training data:"
    )

    print(
        f"{SYNTHETIC_DIR}"
    )

    print()

    print(
        "Preview:"
    )

    print(
        f"{PREVIEW_DIR}"
    )

    print()

    print(
        "Results:"
    )

    print(
        f"{RESULTS_DIR}"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        "  1. training_dataset_summary.csv"
    )

    print(
        "  2. training_synthetic_preview.png"
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()


"""
MRI GAN TESTING / EVALUATION - 128x128
======================================

Purpose:
    Generate and evaluate synthetic MRI Testing images using
    a Generator trained ONLY on the real MRI Training dataset.

IMPORTANT:
    - Real MRI Testing images are NEVER used for GAN training.
    - This script does NOT train the GAN.
    - This script only loads the trained 128x128 Generator.
    - Synthetic Testing images are generated independently.
    - Real Testing images are used only for evaluation/comparison.

Project:
    Synthetic MRI Data Generation

Real Testing Input:
    data/imaging/MRI/Testing

Expected Generator Checkpoint:
    outputs/mri/checkpoints/mri_generator_128_final.pth

Output:
    outputs/mri/gan_testing/
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torchvision.utils import make_grid, save_image


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# REAL MRI TESTING DATA
# ------------------------------------------------------------

TEST_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)


# ------------------------------------------------------------
# GENERATOR CHECKPOINT
# ------------------------------------------------------------

CHECKPOINT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "checkpoints"
)

# IMPORTANT:
# This is the NEW 128x128 generator.
CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "mri_generator_128_final.pth"
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "gan_testing"
)

SYNTHETIC_DIR = (
    OUTPUT_DIR
    / "synthetic_testing"
)

RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
)

PREVIEW_DIR = (
    OUTPUT_DIR
    / "preview"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# IMPORTANT:
# New generator produces 128x128 images.
IMAGE_SIZE = 128

LATENT_DIM = 128

NUM_CLASSES = 4

IMAGE_CHANNELS = 1

BATCH_SIZE = 32

# Real Testing dataset:
# 400 images/class
SAMPLES_PER_CLASS = 400


# ============================================================
# MRI CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# Keep CPU usage reasonable
if DEVICE.type == "cpu":

    try:

        torch.set_num_threads(
            min(
                4,
                os.cpu_count() or 1
            )
        )

    except Exception:

        pass


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SYNTHETIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 75)

print(
    "             MRI GAN TESTING / EVALUATION"
)

print("=" * 75)

print()

print(
    f"Device             : {DEVICE}"
)

print(
    f"Image size         : {IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    f"Latent dimension   : {LATENT_DIM}"
)

print(
    f"Classes            : {NUM_CLASSES}"
)

print(
    f"Samples per class  : {SAMPLES_PER_CLASS}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Real Testing data is NOT used for GAN training."
)

print(
    "The Generator was trained using Training data only."
)

print(
    "Testing data is used only for evaluation."
)

print()


# ============================================================
# 128x128 CONDITIONAL GENERATOR
# ============================================================

class Generator(nn.Module):

    """
    Conditional MRI Generator.

    Input:
        Latent vector: 128
        Class embedding: 128

    Output:
        1-channel 128x128 MRI image
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
        # CLASS EMBEDDING
        # ----------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )


        # ----------------------------------------------------
        # FULLY CONNECTED
        #
        # latent = 128
        # label  = 128
        #
        # total = 256
        #
        # 256 -> 512 * 4 * 4
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
        # GENERATOR
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
        #   ↓
        # 128x128
        # ----------------------------------------------------

        self.main = nn.Sequential(

            # ------------------------------------------------
            # 4 -> 8
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

            nn.ReLU(True),


            # ------------------------------------------------
            # 8 -> 16
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

            nn.ReLU(True),


            # ------------------------------------------------
            # 16 -> 32
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

            nn.ReLU(True),


            # ------------------------------------------------
            # 32 -> 64
            # ------------------------------------------------

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(True),


            # ------------------------------------------------
            # 64 -> 128
            # ------------------------------------------------

            nn.ConvTranspose2d(
                32,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()

        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        z,
        labels
    ):

        # ----------------------------------------------------
        # Class embedding
        # ----------------------------------------------------

        label_vector = (
            self.label_embedding(labels)
        )


        # ----------------------------------------------------
        # Latent + class
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
        # Generate
        # ----------------------------------------------------

        return self.main(x)


# ============================================================
# IMAGE FILE DISCOVERY
# ============================================================

def get_image_files(directory):

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp",
        "*.webp"
    ]

    files = []

    for extension in extensions:

        files.extend(
            directory.glob(extension)
        )

    return sorted(files)


# ============================================================
# LOAD REAL TESTING STATISTICS
# ============================================================

def load_real_testing_statistics():

    print("=" * 75)

    print(
        "             REAL MRI TESTING DATA"
    )

    print("=" * 75)

    print()


    if not TEST_DIR.exists():

        raise FileNotFoundError(

            "\nTesting directory not found:\n"
            f"{TEST_DIR}\n"

        )


    rows = []


    for class_name in CLASS_NAMES:

        class_dir = (
            TEST_DIR
            / class_name
        )


        if not class_dir.exists():

            raise FileNotFoundError(

                f"Missing class directory:\n"
                f"{class_dir}"

            )


        image_files = get_image_files(
            class_dir
        )


        means = []

        stds = []

        sharpness_values = []

        valid_count = 0


        for image_path in image_files:

            try:

                image = (
                    Image.open(
                        image_path
                    )
                    .convert("L")
                )


                image = image.resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    Image.Resampling.LANCZOS
                )


                array = np.asarray(
                    image,
                    dtype=np.float32
                ) / 255.0


                means.append(
                    float(
                        array.mean()
                    )
                )


                stds.append(
                    float(
                        array.std()
                    )
                )


                # --------------------------------------------
                # Simple sharpness indicator
                #
                # Mean squared gradient
                # --------------------------------------------

                gx = np.diff(
                    array,
                    axis=1
                )

                gy = np.diff(
                    array,
                    axis=0
                )


                sharpness = (
                    np.mean(gx ** 2)
                    +
                    np.mean(gy ** 2)
                )


                sharpness_values.append(
                    float(sharpness)
                )


                valid_count += 1


            except Exception:

                continue


        if valid_count == 0:

            raise RuntimeError(

                f"No valid images found for "
                f"class: {class_name}"

            )


        rows.append({

            "Class":
                class_name,

            "Real_Test_Images":
                valid_count,

            "Real_Mean_Pixel":
                float(
                    np.mean(means)
                ),

            "Real_Pixel_Std":
                float(
                    np.mean(stds)
                ),

            "Real_Sharpness":
                float(
                    np.mean(
                        sharpness_values
                    )
                )

        })


        print(

            f"{class_name:<15}: "
            f"{valid_count:>5} images"

        )


    print()


    total = sum(
        row["Real_Test_Images"]
        for row in rows
    )


    print(
        f"Total Real Testing Images: "
        f"{total}"
    )

    print()


    return rows


# ============================================================
# CLEAN OLD SYNTHETIC TESTING DATA
# ============================================================

def clean_old_synthetic_data():

    if SYNTHETIC_DIR.exists():

        print(
            "Removing previous synthetic Testing images..."
        )

        shutil.rmtree(
            SYNTHETIC_DIR
        )


    SYNTHETIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "Synthetic Testing output directory ready."
    )

    print()


# ============================================================
# GENERATE SYNTHETIC TESTING DATA
# ============================================================

@torch.no_grad()
def generate_synthetic_testing(
    generator
):

    print("=" * 75)

    print(
        "             GENERATING 128x128 SYNTHETIC TEST DATA"
    )

    print("=" * 75)

    print()


    rows = []


    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"Generating {class_name}..."
        )


        class_dir = (
            SYNTHETIC_DIR
            / class_name
        )


        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        means = []

        stds = []

        sharpness_values = []

        generated = 0


        while generated < SAMPLES_PER_CLASS:

            current_batch = min(
                BATCH_SIZE,
                SAMPLES_PER_CLASS - generated
            )


            # ------------------------------------------------
            # Random latent vector
            # ------------------------------------------------

            z = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )


            # ------------------------------------------------
            # Same class
            # ------------------------------------------------

            labels = torch.full(
                (
                    current_batch,
                ),
                class_index,
                dtype=torch.long,
                device=DEVICE
            )


            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            fake_images = generator(
                z,
                labels
            )


            # ------------------------------------------------
            # Convert:
            #
            # [-1, 1] -> [0, 1]
            # ------------------------------------------------

            fake_images = (
                fake_images
                .clamp(-1, 1)
                .add(1)
                .div(2)
            )


            fake_images = (
                fake_images
                .cpu()
            )


            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            for i in range(
                current_batch
            ):

                image = (
                    fake_images[i]
                )


                array = (
                    image
                    .squeeze(0)
                    .numpy()
                )


                means.append(
                    float(
                        array.mean()
                    )
                )


                stds.append(
                    float(
                        array.std()
                    )
                )


                # --------------------------------------------
                # Sharpness
                # --------------------------------------------

                gx = np.diff(
                    array,
                    axis=1
                )

                gy = np.diff(
                    array,
                    axis=0
                )


                sharpness = (
                    np.mean(gx ** 2)
                    +
                    np.mean(gy ** 2)
                )


                sharpness_values.append(
                    float(sharpness)
                )


                image_number = (
                    generated
                    + i
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


            generated += current_batch


            print(
                f"  {generated}/{SAMPLES_PER_CLASS}",
                end="\r"
            )


        print()


        rows.append({

            "Class":
                class_name,

            "Synthetic_Images":
                SAMPLES_PER_CLASS,

            "Synthetic_Mean_Pixel":
                float(
                    np.mean(means)
                ),

            "Synthetic_Pixel_Std":
                float(
                    np.mean(stds)
                ),

            "Synthetic_Sharpness":
                float(
                    np.mean(
                        sharpness_values
                    )
                )

        })


        print(
            f"  Completed: "
            f"{SAMPLES_PER_CLASS} images"
        )

        print()


    return rows


# ============================================================
# CREATE PREVIEW
# ============================================================

@torch.no_grad()
def create_preview(
    generator
):

    print(
        "Creating synthetic preview..."
    )


    preview_images = []


    # 8 images per class
    for class_index in range(
        NUM_CLASSES
    ):

        z = torch.randn(
            8,
            LATENT_DIM,
            device=DEVICE
        )


        labels = torch.full(
            (8,),
            class_index,
            dtype=torch.long,
            device=DEVICE
        )


        images = generator(
            z,
            labels
        )


        # [-1,1] -> [0,1]
        images = (
            images
            .clamp(-1, 1)
            .add(1)
            .div(2)
        )


        preview_images.append(
            images.cpu()
        )


    preview_images = torch.cat(
        preview_images,
        dim=0
    )


    grid = make_grid(
        preview_images,
        nrow=8,
        padding=2
    )


    preview_path = (
        PREVIEW_DIR
        / "synthetic_testing_preview_128.png"
    )


    save_image(
        grid,
        preview_path
    )


    print(
        "Preview saved:"
    )

    print(
        preview_path
    )

    print()


# ============================================================
# CALCULATE COMPARISON
# ============================================================

def calculate_comparison(
    real_rows,
    synthetic_rows
):

    real_map = {
        row["Class"]: row
        for row in real_rows
    }


    synthetic_map = {
        row["Class"]: row
        for row in synthetic_rows
    }


    comparison = []


    for class_name in CLASS_NAMES:

        real = (
            real_map[class_name]
        )

        synthetic = (
            synthetic_map[class_name]
        )


        real_mean = (
            real["Real_Mean_Pixel"]
        )

        synthetic_mean = (
            synthetic[
                "Synthetic_Mean_Pixel"
            ]
        )


        real_std = (
            real["Real_Pixel_Std"]
        )

        synthetic_std = (
            synthetic[
                "Synthetic_Pixel_Std"
            ]
        )


        real_sharpness = (
            real["Real_Sharpness"]
        )

        synthetic_sharpness = (
            synthetic[
                "Synthetic_Sharpness"
            ]
        )


        # ----------------------------------------------------
        # Differences
        # ----------------------------------------------------

        mean_difference = abs(
            real_mean
            - synthetic_mean
        )


        std_difference = abs(
            real_std
            - synthetic_std
        )


        sharpness_difference = abs(
            real_sharpness
            - synthetic_sharpness
        )


        # ----------------------------------------------------
        # Mean similarity
        # ----------------------------------------------------

        mean_similarity = max(
            0.0,
            1.0 - mean_difference
        )


        # ----------------------------------------------------
        # Std similarity
        # ----------------------------------------------------

        std_similarity = max(
            0.0,
            1.0 - std_difference
        )


        # ----------------------------------------------------
        # Sharpness similarity
        #
        # Ratio-based comparison.
        # ----------------------------------------------------

        if real_sharpness > 0:

            sharpness_ratio = (
                synthetic_sharpness
                /
                real_sharpness
            )

            sharpness_similarity = max(
                0.0,
                1.0
                -
                abs(
                    1.0
                    -
                    sharpness_ratio
                )
            )

        else:

            sharpness_similarity = 0.0


        # ----------------------------------------------------
        # Overall statistical score
        # ----------------------------------------------------

        distribution_score = (

            (
                mean_similarity
                +
                std_similarity
                +
                sharpness_similarity
            )
            /
            3.0

        )


        comparison.append({

            "Class":
                class_name,

            "Real_Test_Images":
                real[
                    "Real_Test_Images"
                ],

            "Synthetic_Test_Images":
                synthetic[
                    "Synthetic_Images"
                ],

            "Real_Mean_Pixel":
                real_mean,

            "Synthetic_Mean_Pixel":
                synthetic_mean,

            "Mean_Difference":
                mean_difference,

            "Real_Pixel_Std":
                real_std,

            "Synthetic_Pixel_Std":
                synthetic_std,

            "Std_Difference":
                std_difference,

            "Real_Sharpness":
                real_sharpness,

            "Synthetic_Sharpness":
                synthetic_sharpness,

            "Sharpness_Difference":
                sharpness_difference,

            "Mean_Similarity":
                mean_similarity,

            "Std_Similarity":
                std_similarity,

            "Sharpness_Similarity":
                sharpness_similarity,

            "Distribution_Score":
                distribution_score

        })


    return pd.DataFrame(
        comparison
    )


# ============================================================
# SAVE CLASS COMPARISON
# ============================================================

def save_comparison_csv(
    dataframe
):

    output_path = (
        RESULTS_DIR
        / "gan_testing_class_comparison_128.csv"
    )


    dataframe.to_csv(
        output_path,
        index=False
    )


    print(
        "Class comparison saved:"
    )

    print(
        output_path
    )

    print()


    return output_path


# ============================================================
# SAVE OVERALL SCORE
# ============================================================

def save_overall_score(
    dataframe
):

    overall_score = (
        dataframe[
            "Distribution_Score"
        ].mean()
        * 100
    )


    sharpness_similarity = (
        dataframe[
            "Sharpness_Similarity"
        ].mean()
        * 100
    )


    output_path = (
        RESULTS_DIR
        / "gan_testing_overall_score_128.csv"
    )


    result = pd.DataFrame([{

        "Overall_Distribution_Score":
            overall_score,

        "Overall_Sharpness_Similarity":
            sharpness_similarity,

        "Image_Resolution":
            "128x128",

        "Interpretation":
            (
                "Statistical similarity combines "
                "mean pixel similarity, pixel "
                "standard deviation similarity "
                "and sharpness similarity. "
                "This score does NOT prove "
                "medical realism."
            )

    }])


    result.to_csv(
        output_path,
        index=False
    )


    print(
        "Overall score saved:"
    )

    print(
        output_path
    )

    print()


    return overall_score


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

def save_dataset_summary(
    real_rows,
    synthetic_rows
):

    rows = []


    real_map = {
        row["Class"]: row
        for row in real_rows
    }


    synthetic_map = {
        row["Class"]: row
        for row in synthetic_rows
    }


    for class_name in CLASS_NAMES:

        rows.append({

            "Class":
                class_name,

            "Real_Testing_Images":
                real_map[
                    class_name
                ][
                    "Real_Test_Images"
                ],

            "Synthetic_Testing_Images":
                synthetic_map[
                    class_name
                ][
                    "Synthetic_Images"
                ]

        })


    dataframe = pd.DataFrame(
        rows
    )


    dataframe.loc[
        len(dataframe)
    ] = [

        "TOTAL",

        dataframe[
            "Real_Testing_Images"
        ].sum(),

        dataframe[
            "Synthetic_Testing_Images"
        ].sum()

    ]


    output_path = (
        RESULTS_DIR
        / "gan_testing_dataset_summary_128.csv"
    )


    dataframe.to_csv(
        output_path,
        index=False
    )


    print(
        "Dataset summary saved:"
    )

    print(
        output_path
    )

    print()


# ============================================================
# VERIFY SYNTHETIC DATA
# ============================================================

def verify_synthetic_data():

    print("=" * 75)

    print(
        "             VERIFYING SYNTHETIC DATA"
    )

    print("=" * 75)

    print()


    total = 0

    errors = []


    for class_name in CLASS_NAMES:

        class_dir = (
            SYNTHETIC_DIR
            / class_name
        )


        files = get_image_files(
            class_dir
        )


        class_total = len(files)

        total += class_total


        print(
            f"{class_name:<15}: "
            f"{class_total:>5} images"
        )


        if class_total != SAMPLES_PER_CLASS:

            errors.append(

                f"{class_name}: expected "
                f"{SAMPLES_PER_CLASS}, "
                f"found {class_total}"

            )


        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        for image_path in files[:10]:

            try:

                image = Image.open(
                    image_path
                )


                if image.size != (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ):

                    errors.append(

                        f"{image_path.name}: "
                        f"incorrect size "
                        f"{image.size}"

                    )

            except Exception as error:

                errors.append(
                    f"{image_path.name}: "
                    f"{error}"
                )


    print()

    print(
        f"Total synthetic images: {total}"
    )

    print()


    if errors:

        print(
            "Verification warnings:"
        )

        for error in errors:

            print(
                f"  - {error}"
            )

        print()

    else:

        print(
            "Synthetic dataset verification: PASSED"
        )

        print()


    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Starting MRI GAN Testing..."
    )

    print()


    # ========================================================
    # STEP 1
    # Load REAL Testing statistics
    # ========================================================

    real_rows = (
        load_real_testing_statistics()
    )


    # ========================================================
    # STEP 2
    # Clean OLD synthetic Testing images
    # ========================================================

    clean_old_synthetic_data()


    # ========================================================
    # STEP 3
    # Check checkpoint
    # ========================================================

    print("=" * 75)

    print(
        "             LOADING 128x128 GENERATOR"
    )

    print("=" * 75)

    print()


    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(

            "\nNEW 128x128 Generator checkpoint "
            "was not found.\n\n"

            f"Expected:\n"
            f"{CHECKPOINT_PATH}\n\n"

            "You must train the new 128x128 GAN "
            "before running this testing script.\n\n"

            "Do NOT use the old 64x64 checkpoint "
            "for this script."

        )


    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print()


    # ========================================================
    # STEP 4
    # Create generator
    # ========================================================

    generator = Generator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        image_channels=IMAGE_CHANNELS
    ).to(DEVICE)


    # ========================================================
    # STEP 5
    # Load checkpoint
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "generator_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "generator_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

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
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}


    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key.replace(
                "module.",
                "",
                1
            )


        cleaned_state_dict[
            key
        ] = value


    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    try:

        generator.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError as error:

        print()

        print("=" * 75)

        print(
            "             CHECKPOINT MISMATCH"
        )

        print("=" * 75)

        print()

        print(
            str(error)
        )

        print()

        print(
            "The checkpoint does not match "
            "the new 128x128 Generator."
        )

        print()

        print(
            "Make sure the checkpoint was created "
            "using the new 128x128 training script."
        )

        print()

        raise


    generator.eval()


    print(
        "128x128 Generator loaded successfully."
    )

    print()


    # ========================================================
    # STEP 6
    # Generate synthetic Testing images
    # ========================================================

    synthetic_rows = (
        generate_synthetic_testing(
            generator
        )
    )


    # ========================================================
    # STEP 7
    # Verify generated dataset
    # ========================================================

    verify_synthetic_data()


    # ========================================================
    # STEP 8
    # Create preview
    # ========================================================

    create_preview(
        generator
    )


    # ========================================================
    # STEP 9
    # Compare Real vs Synthetic
    # ========================================================

    comparison_df = (
        calculate_comparison(
            real_rows,
            synthetic_rows
        )
    )


    # ========================================================
    # STEP 10
    # Save comparison
    # ========================================================

    save_comparison_csv(
        comparison_df
    )


    # ========================================================
    # STEP 11
    # Overall score
    # ========================================================

    overall_score = (
        save_overall_score(
            comparison_df
        )
    )


    # ========================================================
    # STEP 12
    # Dataset summary
    # ========================================================

    save_dataset_summary(
        real_rows,
        synthetic_rows
    )


    # ========================================================
    # STEP 13
    # DISPLAY RESULTS
    # ========================================================

    print()

    print("=" * 75)

    print(
        "             MRI GAN TESTING SUMMARY"
    )

    print("=" * 75)

    print()


    display_columns = [

        "Class",

        "Real_Test_Images",

        "Synthetic_Test_Images",

        "Real_Mean_Pixel",

        "Synthetic_Mean_Pixel",

        "Real_Sharpness",

        "Synthetic_Sharpness",

        "Sharpness_Similarity",

        "Distribution_Score"

    ]


    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print()

    print("-" * 75)


    print(
        f"Overall Distribution Score: "
        f"{overall_score:.2f}%"
    )


    print()


    print(
        "IMPORTANT:"
    )

    print(
        "This score measures statistical similarity."
    )

    print(
        "It does NOT prove that synthetic MRI images "
        "are medically realistic."
    )

    print()


    print(
        "Visual inspection and downstream ML validation "
        "are required before using the synthetic dataset."
    )


    print()


    print("=" * 75)

    print(
        "             MRI GAN TESTING COMPLETED"
    )

    print("=" * 75)

    print()


    print(
        "Real Testing data:"
    )

    print(
        f"  {TEST_DIR}"
    )

    print()


    print(
        "Synthetic Testing data:"
    )

    print(
        f"  {SYNTHETIC_DIR}"
    )

    print()


    print(
        "Results:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print()


    print(
        "Preview:"
    )

    print(
        f"  {PREVIEW_DIR / 'synthetic_testing_preview_128.png'}"
    )

    print()


    print(
        "Generated files:"
    )

    print(
        "  1. gan_testing_class_comparison_128.csv"
    )

    print(
        "  2. gan_testing_overall_score_128.csv"
    )

    print(
        "  3. gan_testing_dataset_summary_128.csv"
    )

    print(
        "  4. synthetic_testing_preview_128.png"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
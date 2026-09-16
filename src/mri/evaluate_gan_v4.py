import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/imaging/MRI/Training_Preprocessed"

OUTPUT_DIR = "outputs/mri/v4/evaluation"

CHECKPOINT_PATH = (
    "outputs/mri/v4/checkpoints/"
    "mri_gan_v4_epoch_010.pth"
)

IMAGE_SIZE = 128
CHANNELS = 1

LATENT_DIM = 128
NUM_CLASSES = 4

SYNTHETIC_PER_CLASS = 500
REAL_PER_CLASS = 100

DIVERSITY_SAMPLE = 50

BATCH_SIZE = 32

CPU_THREADS = min(
    4,
    os.cpu_count() or 1
)

SEED = 42


# ============================================================
# CLASS NAMES
# ============================================================

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

torch.set_num_threads(
    CPU_THREADS
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

SYNTHETIC_DIR = os.path.join(
    OUTPUT_DIR,
    "synthetic_samples"
)

PREVIEW_DIR = os.path.join(
    OUTPUT_DIR,
    "previews"
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

os.makedirs(
    SYNTHETIC_DIR,
    exist_ok=True
)

os.makedirs(
    PREVIEW_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MRI GAN V4 EVALUATION")
print("=" * 70)

print(
    f"Device                 : {DEVICE}"
)

print(
    f"Image Size             : "
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    f"Synthetic/Class        : "
    f"{SYNTHETIC_PER_CLASS}"
)

print(
    f"Real/Class             : "
    f"{REAL_PER_CLASS}"
)

print(
    f"Diversity Sample       : "
    f"{DIVERSITY_SAMPLE}"
)

print("=" * 70)


# ============================================================
# CHECKPOINT CHECK
# ============================================================

if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        "\nV4 Epoch 10 checkpoint was not found.\n\n"
        f"Expected:\n{CHECKPOINT_PATH}\n\n"
        "Make sure Epoch 10 training completed."
    )


# ============================================================
# V4 GENERATOR
# ============================================================

class GeneratorV4(nn.Module):

    def __init__(
        self,
        latent_dim=128,
        num_classes=4,
        image_channels=1
    ):

        super().__init__()

        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Conditional label embedding
        # ----------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )

        # ----------------------------------------------------
        # Initial projection
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
        # 4 -> 8
        # ----------------------------------------------------

        self.block1 = nn.Sequential(

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

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # 8 -> 16
        # ----------------------------------------------------

        self.block2 = nn.Sequential(

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

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # 16 -> 32
        # ----------------------------------------------------

        self.block3 = nn.Sequential(

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

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # 32 -> 64
        # ----------------------------------------------------

        self.block4 = nn.Sequential(

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

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # 64 -> 128
        # ----------------------------------------------------

        self.block5 = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # Final image layer
        # ----------------------------------------------------

        self.final = nn.Sequential(

            nn.Conv2d(
                16,
                image_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Tanh()
        )

        # ----------------------------------------------------
        # Multi-level noise injection
        # ----------------------------------------------------

        self.noise_8 = nn.Linear(
            latent_dim,
            256
        )

        self.noise_16 = nn.Linear(
            latent_dim,
            128
        )

        self.noise_32 = nn.Linear(
            latent_dim,
            64
        )

        self.noise_64 = nn.Linear(
            latent_dim,
            32
        )

        self.noise_128 = nn.Linear(
            latent_dim,
            16
        )


    def forward(
        self,
        z,
        labels
    ):

        # ----------------------------------------------------
        # Label embedding
        # ----------------------------------------------------

        label_embedding = (
            self.label_embedding(labels)
        )

        # ----------------------------------------------------
        # Combine noise + label
        # ----------------------------------------------------

        combined = torch.cat(
            [
                z,
                label_embedding
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Project to 4x4
        # ----------------------------------------------------

        x = self.fc(
            combined
        )

        x = x.view(
            -1,
            512,
            4,
            4
        )

        # ----------------------------------------------------
        # 4 -> 8
        # ----------------------------------------------------

        x = self.block1(
            x
        )

        noise = self.noise_8(
            z
        )

        noise = noise.view(
            -1,
            256,
            1,
            1
        )

        x = x + noise

        # ----------------------------------------------------
        # 8 -> 16
        # ----------------------------------------------------

        x = self.block2(
            x
        )

        noise = self.noise_16(
            z
        )

        noise = noise.view(
            -1,
            128,
            1,
            1
        )

        x = x + noise

        # ----------------------------------------------------
        # 16 -> 32
        # ----------------------------------------------------

        x = self.block3(
            x
        )

        noise = self.noise_32(
            z
        )

        noise = noise.view(
            -1,
            64,
            1,
            1
        )

        x = x + noise

        # ----------------------------------------------------
        # 32 -> 64
        # ----------------------------------------------------

        x = self.block4(
            x
        )

        noise = self.noise_64(
            z
        )

        noise = noise.view(
            -1,
            32,
            1,
            1
        )

        x = x + noise

        # ----------------------------------------------------
        # 64 -> 128
        # ----------------------------------------------------

        x = self.block5(
            x
        )

        noise = self.noise_128(
            z
        )

        noise = noise.view(
            -1,
            16,
            1,
            1
        )

        x = x + noise

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        x = self.final(
            x
        )

        return x


# ============================================================
# LOAD GENERATOR
# ============================================================

print(
    "\nLoading V4 generator checkpoint..."
)

generator = GeneratorV4(
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES,
    image_channels=CHANNELS
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


generator.load_state_dict(
    checkpoint[
        "generator_state_dict"
    ]
)

generator.eval()


trained_epoch = checkpoint.get(
    "epoch",
    10
)


print(
    f"Checkpoint loaded     : "
    f"{CHECKPOINT_PATH}"
)

print(
    f"Trained Epoch         : "
    f"{trained_epoch}"
)


# ============================================================
# REAL DATASET
# ============================================================

print(
    "\nLoading real preprocessed MRI data..."
)


# ============================================================
# IMPORTANT:
# DO NOT NORMALIZE REAL IMAGES TO [-1,1]
#
# Real images remain [0,1]
# Synthetic images will also be converted to [0,1]
#
# Therefore the evaluation compares:
#
# REAL       [0,1]
# SYNTHETIC  [0,1]
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)


print(
    f"Real dataset size     : "
    f"{len(dataset)}"
)

print(
    f"Classes               : "
    f"{dataset.classes}"
)


# ============================================================
# COLLECT REAL IMAGES
# ============================================================

def collect_real_images(
    dataset,
    num_per_class
):

    class_images = {
        name: []
        for name in CLASS_NAMES
    }

    counters = {
        name: 0
        for name in CLASS_NAMES
    }

    print(
        "\nCollecting real images..."
    )

    for image, label in dataset:

        class_name = CLASS_NAMES[
            label
        ]

        if (
            counters[class_name]
            < num_per_class
        ):

            class_images[
                class_name
            ].append(
                image
            )

            counters[
                class_name
            ] += 1

        if all(
            counters[name]
            >= num_per_class
            for name in CLASS_NAMES
        ):

            break


    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<12}: "
            f"{len(class_images[class_name])}"
        )


    return class_images


real_images = collect_real_images(
    dataset,
    REAL_PER_CLASS
)


# ============================================================
# GENERATE SYNTHETIC IMAGES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GENERATING SYNTHETIC MRI IMAGES"
)

print(
    "=" * 70
)


synthetic_images = {}


with torch.no_grad():

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"\nGenerating {class_name}..."
        )

        generated_batches = []

        remaining = (
            SYNTHETIC_PER_CLASS
        )


        while remaining > 0:

            current_batch = min(
                BATCH_SIZE,
                remaining
            )


            # Random latent vectors
            z = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )


            # Class labels
            labels = torch.full(
                (
                    current_batch,
                ),
                class_index,
                device=DEVICE,
                dtype=torch.long
            )


            # Generate
            fake = generator(
                z,
                labels
            )


            # ------------------------------------------------
            # Generator output:
            #
            # [-1,1]
            #
            # Convert to:
            #
            # [0,1]
            # ------------------------------------------------

            fake = (
                fake + 1.0
            ) / 2.0


            fake = fake.clamp(
                0.0,
                1.0
            )


            generated_batches.append(
                fake.cpu()
            )


            remaining -= (
                current_batch
            )


        synthetic_images[
            class_name
        ] = torch.cat(
            generated_batches,
            dim=0
        )


        print(
            f"Generated            : "
            f"{len(synthetic_images[class_name])}"
        )


# ============================================================
# SAVE SYNTHETIC IMAGES
# ============================================================

print(
    "\nSaving synthetic images..."
)


for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        SYNTHETIC_DIR,
        class_name
    )

    os.makedirs(
        class_dir,
        exist_ok=True
    )


    images = synthetic_images[
        class_name
    ]


    for index, image in enumerate(
        images
    ):

        image_path = os.path.join(
            class_dir,
            f"{class_name}_{index + 1:04d}.png"
        )


        save_image(
            image,
            image_path
        )


print(
    "Synthetic images saved."
)


# ============================================================
# GRID DISPLAY HELPER
# ============================================================

def grid_to_display_array(
    grid
):

    grid = (
        grid.detach()
        .cpu()
    )


    if grid.dim() == 2:

        return grid.numpy()


    if grid.dim() == 3:

        channels = grid.shape[0]


        if channels == 1:

            return (
                grid
                .squeeze(0)
                .numpy()
            )


        if channels == 3:

            return (
                grid
                .permute(
                    1,
                    2,
                    0
                )
                .numpy()
            )


    raise ValueError(
        "Unsupported grid shape: "
        f"{tuple(grid.shape)}"
    )


# ============================================================
# CLASS-WISE PREVIEWS
# ============================================================

print(
    "\nCreating class-wise previews..."
)


for class_name in CLASS_NAMES:

    real = torch.stack(
        real_images[
            class_name
        ]
    )


    fake = synthetic_images[
        class_name
    ]


    real_preview = real[
        :16
    ]


    fake_preview = fake[
        :16
    ]


    real_grid = make_grid(
        real_preview,
        nrow=4,
        normalize=False
    )


    fake_grid = make_grid(
        fake_preview,
        nrow=4,
        normalize=False
    )


    fig = plt.figure(
        figsize=(10, 5)
    )


    ax1 = plt.subplot(
        1,
        2,
        1
    )


    ax1.imshow(
        grid_to_display_array(
            real_grid
        ),
        cmap="gray",
        vmin=0,
        vmax=1
    )


    ax1.set_title(
        f"{class_name.upper()} - REAL"
    )


    ax1.axis(
        "off"
    )


    ax2 = plt.subplot(
        1,
        2,
        2
    )


    ax2.imshow(
        grid_to_display_array(
            fake_grid
        ),
        cmap="gray",
        vmin=0,
        vmax=1
    )


    ax2.set_title(
        f"{class_name.upper()} - SYNTHETIC"
    )


    ax2.axis(
        "off"
    )


    plt.tight_layout()


    preview_path = os.path.join(
        PREVIEW_DIR,
        f"{class_name}_real_vs_synthetic_v4.png"
    )


    plt.savefig(
        preview_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Preview saved        : "
        f"{preview_path}"
    )


# ============================================================
# COMBINED PREVIEW
# ============================================================

print(
    "\nCreating combined preview..."
)


fig = plt.figure(
    figsize=(12, 16)
)


plot_index = 1


for class_name in CLASS_NAMES:

    real = torch.stack(
        real_images[
            class_name
        ][:8]
    )


    fake = synthetic_images[
        class_name
    ][:8]


    real_grid = make_grid(
        real,
        nrow=4,
        normalize=False
    )


    fake_grid = make_grid(
        fake,
        nrow=4,
        normalize=False
    )


    ax1 = plt.subplot(
        8,
        2,
        plot_index
    )


    ax1.imshow(
        grid_to_display_array(
            real_grid
        ),
        cmap="gray",
        vmin=0,
        vmax=1
    )


    ax1.set_title(
        f"{class_name.upper()} - REAL"
    )


    ax1.axis(
        "off"
    )


    plot_index += 1


    ax2 = plt.subplot(
        8,
        2,
        plot_index
    )


    ax2.imshow(
        grid_to_display_array(
            fake_grid
        ),
        cmap="gray",
        vmin=0,
        vmax=1
    )


    ax2.set_title(
        f"{class_name.upper()} - SYNTHETIC"
    )


    ax2.axis(
        "off"
    )


    plot_index += 1


plt.tight_layout()


combined_preview_path = os.path.join(
    PREVIEW_DIR,
    "real_vs_synthetic_v4.png"
)


plt.savefig(
    combined_preview_path,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    f"Combined preview      : "
    f"{combined_preview_path}"
)


# ============================================================
# MEAN
# ============================================================

def calculate_mean(
    images
):

    return images.mean().item()


# ============================================================
# STANDARD DEVIATION
# ============================================================

def calculate_std(
    images
):

    return images.std().item()


# ============================================================
# DISTRIBUTION SCORE
# ============================================================

def calculate_distribution_score(
    real_mean,
    fake_mean,
    real_std,
    fake_std
):

    mean_difference = abs(
        real_mean -
        fake_mean
    )


    std_difference = abs(
        real_std -
        fake_std
    )


    mean_score = max(
        0.0,
        1.0 -
        (
            mean_difference /
            (
                abs(real_mean)
                + 1e-6
            )
        )
    )


    std_score = max(
        0.0,
        1.0 -
        (
            std_difference /
            (
                abs(real_std)
                + 1e-6
            )
        )
    )


    score = (
        0.5 * mean_score +
        0.5 * std_score
    )


    return min(
        1.0,
        score
    ) * 100.0


# ============================================================
# SOBEL SHARPNESS
# ============================================================

def calculate_sharpness(
    images
):

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=torch.float32
    ).view(
        1,
        1,
        3,
        3
    )


    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=torch.float32
    ).view(
        1,
        1,
        3,
        3
    )


    sobel_x = sobel_x.to(
        images.device
    )


    sobel_y = sobel_y.to(
        images.device
    )


    grad_x = F.conv2d(
        images,
        sobel_x,
        padding=1
    )


    grad_y = F.conv2d(
        images,
        sobel_y,
        padding=1
    )


    magnitude = torch.sqrt(
        grad_x ** 2 +
        grad_y ** 2 +
        1e-8
    )


    return magnitude.mean().item()


# ============================================================
# SHARPNESS SCORE
# ============================================================

def calculate_sharpness_score(
    real_sharpness,
    fake_sharpness
):

    ratio = (
        fake_sharpness /
        (
            real_sharpness
            + 1e-8
        )
    )


    return min(
        ratio,
        1.0
    ) * 100.0


# ============================================================
# PER-IMAGE NORMALIZATION
# ============================================================

def normalize_each_image(
    images
):

    minimum = images.amin(
        dim=(1, 2, 3),
        keepdim=True
    )


    maximum = images.amax(
        dim=(1, 2, 3),
        keepdim=True
    )


    normalized = (
        images - minimum
    ) / (
        maximum -
        minimum +
        1e-8
    )


    return normalized


# ============================================================
# DIVERSITY
# ============================================================

def calculate_diversity(
    images,
    sample_size=50
):

    if len(images) > sample_size:

        indices = torch.linspace(
            0,
            len(images) - 1,
            sample_size
        ).long()

        images = images[
            indices
        ]


    images = normalize_each_image(
        images
    )


    n = len(images)


    if n < 2:

        return 0.0


    total_distance = 0.0
    pair_count = 0


    for i in range(n):

        current = images[i]

        others = images[
            i + 1:
        ]


        if len(others) == 0:

            continue


        distances = torch.mean(
            torch.abs(
                current.unsqueeze(0)
                - others
            ),
            dim=(1, 2, 3)
        )


        total_distance += (
            distances.sum().item()
        )


        pair_count += len(
            distances
        )


    return (
        total_distance /
        max(
            pair_count,
            1
        )
    )


# ============================================================
# DIVERSITY SCORE
# ============================================================

def calculate_diversity_score(
    real_diversity,
    synthetic_diversity
):

    score = (
        synthetic_diversity /
        (
            real_diversity +
            1e-8
        )
    )


    return min(
        score,
        1.0
    ) * 100.0


# ============================================================
# CALCULATE ALL METRICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CALCULATING IMAGE STATISTICS"
)

print(
    "=" * 70
)


results = []


for class_name in CLASS_NAMES:

    print(
        "\n" + "-" * 70
    )


    print(
        f"Class: {class_name}"
    )


    real = torch.stack(
        real_images[
            class_name
        ]
    )


    fake = synthetic_images[
        class_name
    ]


    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    real_mean = calculate_mean(
        real
    )


    fake_mean = calculate_mean(
        fake
    )


    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    real_std = calculate_std(
        real
    )


    fake_std = calculate_std(
        fake
    )


    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    distribution_score = (
        calculate_distribution_score(
            real_mean,
            fake_mean,
            real_std,
            fake_std
        )
    )


    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    real_sharpness = (
        calculate_sharpness(
            real
        )
    )


    fake_sharpness = (
        calculate_sharpness(
            fake
        )
    )


    sharpness_score = (
        calculate_sharpness_score(
            real_sharpness,
            fake_sharpness
        )
    )


    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    real_diversity = (
        calculate_diversity(
            real,
            DIVERSITY_SAMPLE
        )
    )


    fake_diversity = (
        calculate_diversity(
            fake,
            DIVERSITY_SAMPLE
        )
    )


    diversity_score = (
        calculate_diversity_score(
            real_diversity,
            fake_diversity
        )
    )


    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_score = (
        0.40 * distribution_score +
        0.30 * sharpness_score +
        0.30 * diversity_score
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Real Mean           : "
        f"{real_mean:.6f}"
    )


    print(
        f"Synthetic Mean      : "
        f"{fake_mean:.6f}"
    )


    print(
        f"Real Std            : "
        f"{real_std:.6f}"
    )


    print(
        f"Synthetic Std       : "
        f"{fake_std:.6f}"
    )


    print(
        f"Distribution Score  : "
        f"{distribution_score:.2f}%"
    )


    print(
        f"Real Sharpness      : "
        f"{real_sharpness:.6f}"
    )


    print(
        f"Synthetic Sharpness : "
        f"{fake_sharpness:.6f}"
    )


    print(
        f"Sharpness Score     : "
        f"{sharpness_score:.2f}%"
    )


    print(
        f"Real Diversity      : "
        f"{real_diversity:.6f}"
    )


    print(
        f"Synthetic Diversity : "
        f"{fake_diversity:.6f}"
    )


    print(
        f"Diversity Score     : "
        f"{diversity_score:.2f}%"
    )


    print(
        f"Overall Diagnostic  : "
        f"{overall_score:.2f}%"
    )


    results.append(
        {
            "class": class_name,

            "real_mean":
                real_mean,

            "synthetic_mean":
                fake_mean,

            "real_std":
                real_std,

            "synthetic_std":
                fake_std,

            "distribution_score":
                distribution_score,

            "real_sharpness":
                real_sharpness,

            "synthetic_sharpness":
                fake_sharpness,

            "sharpness_score":
                sharpness_score,

            "real_diversity":
                real_diversity,

            "synthetic_diversity":
                fake_diversity,

            "diversity_score":
                diversity_score,

            "overall_diagnostic":
                overall_score
        }
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

df_results = pd.DataFrame(
    results
)


overall_distribution = (
    df_results[
        "distribution_score"
    ].mean()
)


overall_sharpness = (
    df_results[
        "sharpness_score"
    ].mean()
)


overall_diversity = (
    df_results[
        "diversity_score"
    ].mean()
)


overall_diagnostic = (
    0.40 * overall_distribution +
    0.30 * overall_sharpness +
    0.30 * overall_diversity
)


# ============================================================
# PRINT OVERALL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "OVERALL MRI GAN V4 RESULTS"
)

print(
    "=" * 70
)


print(
    f"Distribution Score   : "
    f"{overall_distribution:.2f}%"
)


print(
    f"Sharpness Score      : "
    f"{overall_sharpness:.2f}%"
)


print(
    f"Diversity Score      : "
    f"{overall_diversity:.2f}%"
)


print(
    f"Overall Diagnostic   : "
    f"{overall_diagnostic:.2f}%"
)


print(
    "=" * 70
)


# ============================================================
# SAVE DETAILED CSV
# ============================================================

detailed_csv = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_evaluation.csv"
)


df_results.to_csv(
    detailed_csv,
    index=False
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary = pd.DataFrame(
    [
        {
            "checkpoint":
                CHECKPOINT_PATH,

            "epoch":
                trained_epoch,

            "distribution_score":
                overall_distribution,

            "sharpness_score":
                overall_sharpness,

            "diversity_score":
                overall_diversity,

            "overall_diagnostic":
                overall_diagnostic
        }
    ]
)


summary_csv = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_summary.csv"
)


summary.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# SCORE CHART
# ============================================================

print(
    "\nCreating score chart..."
)


metrics = [
    "Distribution",
    "Sharpness",
    "Diversity",
    "Overall"
]


scores = [
    overall_distribution,
    overall_sharpness,
    overall_diversity,
    overall_diagnostic
]


plt.figure(
    figsize=(9, 6)
)


bars = plt.bar(
    metrics,
    scores
)


plt.ylim(
    0,
    100
)


plt.ylabel(
    "SCORE (%)"
)


plt.title(
    "MRI GAN V4 - Epoch 10 Evaluation"
)


for bar, score in zip(
    bars,
    scores
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )


plt.tight_layout()


score_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_scores.png"
)


plt.savefig(
    score_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# CLASS COMPARISON CHART
# ============================================================

print(
    "Creating class comparison chart..."
)


x = np.arange(
    len(CLASS_NAMES)
)


width = 0.25


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width,
    df_results[
        "distribution_score"
    ],
    width,
    label="Distribution"
)


plt.bar(
    x,
    df_results[
        "sharpness_score"
    ],
    width,
    label="Sharpness"
)


plt.bar(
    x + width,
    df_results[
        "diversity_score"
    ],
    width,
    label="Diversity"
)


plt.xticks(
    x,
    CLASS_NAMES
)


plt.ylim(
    0,
    105
)


plt.ylabel(
    "SCORE (%)"
)


plt.title(
    "MRI GAN V4 - Class-wise Evaluation"
)


plt.legend()


plt.tight_layout()


class_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_class_comparison.png"
)


plt.savefig(
    class_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# DIVERSITY COMPARISON
# ============================================================

print(
    "Creating diversity comparison chart..."
)


width = 0.35


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width / 2,
    df_results[
        "real_diversity"
    ],
    width,
    label="Real"
)


plt.bar(
    x + width / 2,
    df_results[
        "synthetic_diversity"
    ],
    width,
    label="Synthetic"
)


plt.xticks(
    x,
    CLASS_NAMES
)


plt.ylabel(
    "AVERAGE PAIRWISE MAE"
)


plt.title(
    "MRI GAN V4 - Real vs Synthetic Diversity"
)


plt.legend()


plt.tight_layout()


diversity_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_diversity_comparison.png"
)


plt.savefig(
    diversity_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SHARPNESS COMPARISON
# ============================================================

print(
    "Creating sharpness comparison chart..."
)


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width / 2,
    df_results[
        "real_sharpness"
    ],
    width,
    label="Real"
)


plt.bar(
    x + width / 2,
    df_results[
        "synthetic_sharpness"
    ],
    width,
    label="Synthetic"
)


plt.xticks(
    x,
    CLASS_NAMES
)


plt.ylabel(
    "SOBEL SHARPNESS"
)


plt.title(
    "MRI GAN V4 - Real vs Synthetic Sharpness"
)


plt.legend()


plt.tight_layout()


sharpness_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_sharpness_comparison.png"
)


plt.savefig(
    sharpness_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# DISTRIBUTION COMPARISON
# ============================================================

print(
    "Creating distribution comparison chart..."
)


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width / 2,
    df_results[
        "real_mean"
    ],
    width,
    label="Real Mean"
)


plt.bar(
    x + width / 2,
    df_results[
        "synthetic_mean"
    ],
    width,
    label="Synthetic Mean"
)


plt.xticks(
    x,
    CLASS_NAMES
)


plt.ylabel(
    "PIXEL INTENSITY MEAN"
)


plt.title(
    "MRI GAN V4 - Real vs Synthetic Pixel Distribution"
)


plt.legend()


plt.tight_layout()


distribution_chart = os.path.join(
    RESULT_DIR,
    "mri_gan_v4_distribution_comparison.png"
)


plt.savefig(
    distribution_chart,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MRI GAN V4 INTERPRETATION"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# Diversity
# ------------------------------------------------------------

if overall_diversity < 5:

    diversity_status = "CRITICAL"

    diversity_message = (
        "Very low diversity. "
        "Strong evidence of mode collapse."
    )

elif overall_diversity < 10:

    diversity_status = "LOW"

    diversity_message = (
        "Diversity improved but remains limited."
    )

elif overall_diversity < 20:

    diversity_status = "MODERATE"

    diversity_message = (
        "Meaningful diversity improvement."
    )

else:

    diversity_status = "GOOD"

    diversity_message = (
        "Strong improvement in diversity."
    )


# ------------------------------------------------------------
# Sharpness
# ------------------------------------------------------------

if overall_sharpness >= 90:

    sharpness_status = "GOOD"

elif overall_sharpness >= 70:

    sharpness_status = "MODERATE"

else:

    sharpness_status = "LOW"


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

if overall_distribution >= 90:

    distribution_status = "GOOD"

elif overall_distribution >= 75:

    distribution_status = "MODERATE"

else:

    distribution_status = "LOW"


print(
    f"Diversity Status     : "
    f"{diversity_status}"
)


print(
    f"                     "
    f"{diversity_message}"
)


print(
    f"Sharpness Status     : "
    f"{sharpness_status}"
)


print(
    f"Distribution Status  : "
    f"{distribution_status}"
)


# ============================================================
# DISCLAIMER
# ============================================================

print(
    "\nIMPORTANT:"
)


print(
    "These metrics are engineering diagnostics."
)


print(
    "They do NOT establish medical realism."
)


print(
    "They do NOT establish clinical validity."
)


print(
    "They do NOT establish diagnostic safety."
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MRI GAN V4 EVALUATION COMPLETE"
)

print(
    "=" * 70
)


print(
    f"Checkpoint            : "
    f"{CHECKPOINT_PATH}"
)


print(
    f"Evaluated Epoch       : "
    f"{trained_epoch}"
)


print(
    f"Evaluation directory  : "
    f"{OUTPUT_DIR}"
)


print(
    f"Synthetic images      : "
    f"{SYNTHETIC_DIR}"
)


print(
    f"Visual previews       : "
    f"{PREVIEW_DIR}"
)


print(
    f"Detailed results      : "
    f"{detailed_csv}"
)


print(
    f"Summary               : "
    f"{summary_csv}"
)


print(
    f"Score chart           : "
    f"{score_chart}"
)


print(
    f"Class chart           : "
    f"{class_chart}"
)


print(
    f"Diversity chart       : "
    f"{diversity_chart}"
)


print(
    f"Sharpness chart       : "
    f"{sharpness_chart}"
)


print(
    f"Distribution chart    : "
    f"{distribution_chart}"
)


print(
    "=" * 70
)
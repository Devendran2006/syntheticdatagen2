import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/imaging/MRI/Training_Preprocessed"
OUTPUT_DIR = "outputs/mri/v4"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "generated_samples")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "training_history")

IMAGE_SIZE = 128
CHANNELS = 1

BATCH_SIZE = 16

# Start with 20 epochs.
# We will evaluate Epoch 5 and Epoch 10 before deciding further.
TOTAL_EPOCHS = 20

LATENT_DIM = 128
NUM_CLASSES = 4

LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002

BETA1 = 0.0
BETA2 = 0.9

CPU_THREADS = min(4, os.cpu_count() or 1)

NUM_WORKERS = 0

# ------------------------------------------------------------
# V4 MODE-SEEKING DIVERSITY LOSS
# ------------------------------------------------------------

# Minimum desired image variation relative to latent variation.
# This is an engineering regularization value, NOT a medical metric.
MODE_SEEKING_TARGET = 0.10

# Weight of diversity loss.
MODE_SEEKING_WEIGHT = 2.0

# ------------------------------------------------------------
# CHECKPOINT / PREVIEW
# ------------------------------------------------------------

SAVE_EVERY_EPOCH = 1
PREVIEW_EVERY = 5

RESUME_TRAINING = True

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(CPU_THREADS)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DIRECTORIES
# ============================================================

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)


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
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ============================================================
# DATASET
# ============================================================

print("=" * 70)
print("MRI CONDITIONAL GAN V4")
print("=" * 70)

print(f"Device              : {DEVICE}")
print(f"CPU Threads         : {CPU_THREADS}")
print(f"Image Size          : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch Size          : {BATCH_SIZE}")
print(f"Total Epochs        : {TOTAL_EPOCHS}")
print(f"Latent Dimension    : {LATENT_DIM}")
print(f"Number of Classes   : {NUM_CLASSES}")
print(f"Generator LR        : {LEARNING_RATE_G}")
print(f"Discriminator LR    : {LEARNING_RATE_D}")
print(f"Mode Seeking Target : {MODE_SEEKING_TARGET}")
print(f"Mode Seeking Weight : {MODE_SEEKING_WEIGHT}")
print(f"Resume Training     : {RESUME_TRAINING}")
print("=" * 70)


print("\nLoading preprocessed MRI dataset...")
print(f"Dataset path       : {DATA_DIR}")

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print(f"Total images       : {len(dataset)}")

print("\nClasses:")
print(dataset.classes)

print("\nClass mapping:")
for idx, name in enumerate(dataset.classes):
    print(f"  {idx} -> {name}")


# ============================================================
# DATA LOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    drop_last=True
)


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(module):

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:

        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

    elif classname.find("BatchNorm") != -1:

        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)

        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


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
        # Class embedding
        # ----------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )

        # ----------------------------------------------------
        # Initial projection
        # ----------------------------------------------------

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )

        # ----------------------------------------------------
        # Main generator
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
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(16),
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
        # Noise injection layers
        #
        # These inject additional latent information at
        # multiple resolutions to encourage variation.
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

    def forward(self, z, labels):

        # ----------------------------------------------------
        # Class conditioning
        # ----------------------------------------------------

        label_embedding = self.label_embedding(labels)

        combined = torch.cat(
            [z, label_embedding],
            dim=1
        )

        x = self.fc(combined)

        x = x.view(
            -1,
            512,
            4,
            4
        )

        # ----------------------------------------------------
        # 4 -> 8
        # ----------------------------------------------------

        x = self.block1(x)

        # Noise injection
        noise = self.noise_8(z)
        noise = noise.view(-1, 256, 1, 1)
        x = x + noise

        # ----------------------------------------------------
        # 8 -> 16
        # ----------------------------------------------------

        x = self.block2(x)

        noise = self.noise_16(z)
        noise = noise.view(-1, 128, 1, 1)
        x = x + noise

        # ----------------------------------------------------
        # 16 -> 32
        # ----------------------------------------------------

        x = self.block3(x)

        noise = self.noise_32(z)
        noise = noise.view(-1, 64, 1, 1)
        x = x + noise

        # ----------------------------------------------------
        # 32 -> 64
        # ----------------------------------------------------

        x = self.block4(x)

        noise = self.noise_64(z)
        noise = noise.view(-1, 32, 1, 1)
        x = x + noise

        # ----------------------------------------------------
        # 64 -> 128
        # ----------------------------------------------------

        x = self.block5(x)

        noise = self.noise_128(z)
        noise = noise.view(-1, 16, 1, 1)
        x = x + noise

        # ----------------------------------------------------
        # Final image
        # ----------------------------------------------------

        x = self.final(x)

        return x


# ============================================================
# V4 PROJECTION DISCRIMINATOR
# ============================================================

class DiscriminatorV4(nn.Module):

    def __init__(
        self,
        num_classes=4,
        image_channels=1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Spectral normalized convolution blocks
        # ----------------------------------------------------

        self.block1 = nn.utils.spectral_norm(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.block2 = nn.utils.spectral_norm(
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.block3 = nn.utils.spectral_norm(
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.block4 = nn.utils.spectral_norm(
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.block5 = nn.utils.spectral_norm(
            nn.Conv2d(
                512,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        # ----------------------------------------------------
        # Minibatch standard deviation
        # ----------------------------------------------------

        self.final = nn.utils.spectral_norm(
            nn.Conv2d(
                513,
                512,
                kernel_size=4,
                stride=1,
                padding=0
            )
        )

        # ----------------------------------------------------
        # Projection discriminator
        # ----------------------------------------------------

        self.output = nn.utils.spectral_norm(
            nn.Linear(
                512,
                1
            )
        )

        self.label_embedding = nn.utils.spectral_norm(
            nn.Embedding(
                num_classes,
                512
            )
        )

    def minibatch_std(self, x):

        batch_size = x.size(0)

        if batch_size <= 1:
            std = torch.zeros(
                1,
                device=x.device,
                dtype=x.dtype
            )

        else:

            std = torch.std(
                x,
                dim=0,
                keepdim=True
            )

            std = torch.mean(std)

        std_map = std.expand(
            batch_size,
            1,
            x.size(2),
            x.size(3)
        )

        return torch.cat(
            [x, std_map],
            dim=1
        )

    def forward(self, x, labels):

        # ----------------------------------------------------
        # Convolution blocks
        # ----------------------------------------------------

        x = F.leaky_relu(
            self.block1(x),
            0.2
        )

        x = F.leaky_relu(
            self.block2(x),
            0.2
        )

        x = F.leaky_relu(
            self.block3(x),
            0.2
        )

        x = F.leaky_relu(
            self.block4(x),
            0.2
        )

        x = F.leaky_relu(
            self.block5(x),
            0.2
        )

        # ----------------------------------------------------
        # Minibatch standard deviation
        # ----------------------------------------------------

        x = self.minibatch_std(x)

        x = F.leaky_relu(
            self.final(x),
            0.2
        )

        x = x.view(
            x.size(0),
            -1
        )

        # ----------------------------------------------------
        # Unconditional score
        # ----------------------------------------------------

        score = self.output(x)

        # ----------------------------------------------------
        # Projection class score
        # ----------------------------------------------------

        label_vector = self.label_embedding(labels)

        projection = torch.sum(
            x * label_vector,
            dim=1,
            keepdim=True
        )

        return score + projection


# ============================================================
# CREATE MODELS
# ============================================================

print("\nInitializing V4 models...")

generator = GeneratorV4(
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES,
    image_channels=CHANNELS
).to(DEVICE)

discriminator = DiscriminatorV4(
    num_classes=NUM_CLASSES,
    image_channels=CHANNELS
).to(DEVICE)

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)


# ============================================================
# OPTIMIZERS
# ============================================================

optimizer_g = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE_G,
    betas=(BETA1, BETA2)
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE_D,
    betas=(BETA1, BETA2)
)


# ============================================================
# MODE SEEKING LOSS
# ============================================================

def mode_seeking_loss(
    fake_1,
    fake_2,
    z_1,
    z_2
):
    """
    Encourages different latent vectors to produce
    different generated images.

    This is an engineering diversity regularizer.
    It is NOT a medical quality metric.
    """

    # --------------------------------------------------------
    # Image distance
    # --------------------------------------------------------

    image_distance = torch.mean(
        torch.abs(fake_1 - fake_2),
        dim=(1, 2, 3)
    )

    # --------------------------------------------------------
    # Latent distance
    # --------------------------------------------------------

    latent_distance = torch.mean(
        torch.abs(z_1 - z_2),
        dim=1
    )

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    ratio = image_distance / (
        latent_distance + 1e-6
    )

    # --------------------------------------------------------
    # Encourage ratio above target
    # --------------------------------------------------------

    loss = F.relu(
        MODE_SEEKING_TARGET - ratio
    )

    return loss.mean(), ratio.mean().detach()


# ============================================================
# FIXED PREVIEW LATENTS
# ============================================================

fixed_noise = torch.randn(
    NUM_CLASSES * 4,
    LATENT_DIM,
    device=DEVICE
)

fixed_labels = torch.tensor(
    [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3
    ],
    device=DEVICE,
    dtype=torch.long
)


# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def get_checkpoint_path(epoch):

    return os.path.join(
        CHECKPOINT_DIR,
        f"mri_gan_v4_epoch_{epoch:03d}.pth"
    )


def find_latest_checkpoint():

    checkpoints = list(
        Path(CHECKPOINT_DIR).glob(
            "mri_gan_v4_epoch_*.pth"
        )
    )

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda p: p.stat().st_mtime
    )

    return str(checkpoints[-1])


# ============================================================
# RESUME
# ============================================================

start_epoch = 1

if RESUME_TRAINING:

    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint is not None:

        print("\n" + "=" * 70)
        print("RESUMING V4 TRAINING")
        print("=" * 70)

        print(
            f"Checkpoint: {latest_checkpoint}"
        )

        checkpoint = torch.load(
            latest_checkpoint,
            map_location=DEVICE,
            weights_only=False
        )

        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )

        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )

        optimizer_g.load_state_dict(
            checkpoint["optimizer_g_state_dict"]
        )

        optimizer_d.load_state_dict(
            checkpoint["optimizer_d_state_dict"]
        )

        completed_epochs = checkpoint["epoch"]

        start_epoch = completed_epochs + 1

        print(
            f"Completed epochs : {completed_epochs}"
        )

        print(
            f"Starting epoch   : {start_epoch}"
        )

        print(
            f"Remaining epochs : "
            f"{max(0, TOTAL_EPOCHS - completed_epochs)}"
        )

        print("=" * 70)

    else:

        print("\nNo V4 checkpoint found.")
        print("Starting V4 training from scratch.")

else:

    print("\nStarting V4 training from scratch.")


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_path = os.path.join(
    OUTPUT_DIR,
    "v4_config.txt"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("MRI CONDITIONAL GAN V4 CONFIGURATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset: {DATA_DIR}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Total Epochs: {TOTAL_EPOCHS}\n")
    f.write(f"Latent Dimension: {LATENT_DIM}\n")
    f.write(f"Classes: {NUM_CLASSES}\n")
    f.write(f"Generator LR: {LEARNING_RATE_G}\n")
    f.write(f"Discriminator LR: {LEARNING_RATE_D}\n")
    f.write(
        f"Mode Seeking Target: "
        f"{MODE_SEEKING_TARGET}\n"
    )
    f.write(
        f"Mode Seeking Weight: "
        f"{MODE_SEEKING_WEIGHT}\n"
    )
    f.write("GAN Loss: Hinge\n")
    f.write(
        "Discriminator: Projection + Spectral Normalization\n"
    )
    f.write(
        "Generator: Conditional + Multi-stage Noise Injection\n"
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    HISTORY_DIR,
    "mri_gan_v4_training_history.csv"
)

if not os.path.exists(history_path):

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "epoch,generator_loss,"
            "discriminator_loss,"
            "mode_seeking_loss,"
            "mode_seeking_ratio,"
            "epoch_time_minutes\n"
        )


# ============================================================
# PREVIEW FUNCTION
# ============================================================

def generate_preview(epoch):

    generator.eval()

    with torch.no_grad():

        images = generator(
            fixed_noise,
            fixed_labels
        )

    images = (images + 1) / 2

    preview_path = os.path.join(
        SAMPLE_DIR,
        f"epoch_{epoch:03d}.png"
    )

    save_image(
        images,
        preview_path,
        nrow=4,
        normalize=False
    )

    generator.train()

    print(
        f"Preview saved      : {preview_path}"
    )


# ============================================================
# TRAINING
# ============================================================

if start_epoch > TOTAL_EPOCHS:

    print("\nTraining already completed.")

else:

    print("\n" + "=" * 70)
    print(
        f"TRAINING V4 EPOCH {start_epoch} -> "
        f"{TOTAL_EPOCHS}"
    )
    print("=" * 70)

    for epoch in range(
        start_epoch,
        TOTAL_EPOCHS + 1
    ):

        epoch_start = time.time()

        generator.train()
        discriminator.train()

        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        mode_loss_total = 0.0
        mode_ratio_total = 0.0

        batches = 0

        for real_images, labels in loader:

            real_images = real_images.to(
                DEVICE,
                non_blocking=False
            )

            labels = labels.to(
                DEVICE,
                non_blocking=False
            )

            current_batch = real_images.size(0)

            # =================================================
            # TRAIN DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Real
            # -------------------------------------------------

            real_scores = discriminator(
                real_images,
                labels
            )

            # -------------------------------------------------
            # Fake
            # -------------------------------------------------

            z = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )

            with torch.no_grad():

                fake_images = generator(
                    z,
                    labels
                )

            fake_scores = discriminator(
                fake_images,
                labels
            )

            # -------------------------------------------------
            # Hinge discriminator loss
            # -------------------------------------------------

            d_real_loss = torch.mean(
                F.relu(1.0 - real_scores)
            )

            d_fake_loss = torch.mean(
                F.relu(1.0 + fake_scores)
            )

            d_loss = (
                d_real_loss +
                d_fake_loss
            )

            d_loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0
            )

            optimizer_d.step()

            # =================================================
            # TRAIN GENERATOR
            # =================================================

            optimizer_g.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Two different latent vectors
            # -------------------------------------------------

            z1 = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )

            z2 = torch.randn(
                current_batch,
                LATENT_DIM,
                device=DEVICE
            )

            # Same class labels.
            # Only latent vector changes.
            fake_1 = generator(
                z1,
                labels
            )

            fake_2 = generator(
                z2,
                labels
            )

            # -------------------------------------------------
            # Adversarial generator loss
            # -------------------------------------------------

            fake_scores_1 = discriminator(
                fake_1,
                labels
            )

            fake_scores_2 = discriminator(
                fake_2,
                labels
            )

            g_adv_loss = -torch.mean(
                fake_scores_1
            )

            # -------------------------------------------------
            # Diversity / Mode Seeking Loss
            # -------------------------------------------------

            ms_loss, ms_ratio = mode_seeking_loss(
                fake_1,
                fake_2,
                z1,
                z2
            )

            # -------------------------------------------------
            # Total generator loss
            # -------------------------------------------------

            g_loss = (
                g_adv_loss +
                MODE_SEEKING_WEIGHT * ms_loss
            )

            g_loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=5.0
            )

            optimizer_g.step()

            # =================================================
            # TRACK METRICS
            # =================================================

            generator_loss_total += (
                g_loss.item()
            )

            discriminator_loss_total += (
                d_loss.item()
            )

            mode_loss_total += (
                ms_loss.item()
            )

            mode_ratio_total += (
                ms_ratio.item()
            )

            batches += 1

        # =====================================================
        # EPOCH METRICS
        # =====================================================

        avg_g_loss = (
            generator_loss_total / batches
        )

        avg_d_loss = (
            discriminator_loss_total / batches
        )

        avg_mode_loss = (
            mode_loss_total / batches
        )

        avg_mode_ratio = (
            mode_ratio_total / batches
        )

        epoch_time = (
            time.time() - epoch_start
        ) / 60.0

        print("\n" + "-" * 70)

        print(
            f"Epoch [{epoch}/{TOTAL_EPOCHS}]"
        )

        print(
            f"Generator Loss      : "
            f"{avg_g_loss:.4f}"
        )

        print(
            f"Discriminator Loss  : "
            f"{avg_d_loss:.4f}"
        )

        print(
            f"Mode Seeking Loss   : "
            f"{avg_mode_loss:.4f}"
        )

        print(
            f"Mode Seeking Ratio  : "
            f"{avg_mode_ratio:.4f}"
        )

        print(
            f"Epoch Time          : "
            f"{epoch_time:.2f} minutes"
        )

        # =====================================================
        # SAVE HISTORY
        # =====================================================

        with open(
            history_path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{epoch},"
                f"{avg_g_loss:.6f},"
                f"{avg_d_loss:.6f},"
                f"{avg_mode_loss:.6f},"
                f"{avg_mode_ratio:.6f},"
                f"{epoch_time:.2f}\n"
            )

        # =====================================================
        # SAVE CHECKPOINT
        # =====================================================

        if (
            epoch % SAVE_EVERY_EPOCH == 0
        ):

            checkpoint_path = (
                get_checkpoint_path(epoch)
            )

            torch.save(
                {
                    "epoch": epoch,

                    "generator_state_dict":
                        generator.state_dict(),

                    "discriminator_state_dict":
                        discriminator.state_dict(),

                    "optimizer_g_state_dict":
                        optimizer_g.state_dict(),

                    "optimizer_d_state_dict":
                        optimizer_d.state_dict(),

                    "generator_loss":
                        avg_g_loss,

                    "discriminator_loss":
                        avg_d_loss,

                    "mode_seeking_loss":
                        avg_mode_loss,

                    "mode_seeking_ratio":
                        avg_mode_ratio,

                    "config": {
                        "image_size":
                            IMAGE_SIZE,

                        "batch_size":
                            BATCH_SIZE,

                        "latent_dim":
                            LATENT_DIM,

                        "num_classes":
                            NUM_CLASSES,

                        "mode_seeking_target":
                            MODE_SEEKING_TARGET,

                        "mode_seeking_weight":
                            MODE_SEEKING_WEIGHT
                    }
                },
                checkpoint_path
            )

            print(
                f"Checkpoint saved   : "
                f"{checkpoint_path}"
            )

        # =====================================================
        # PREVIEW
        # =====================================================

        if (
            epoch % PREVIEW_EVERY == 0
        ):

            generate_preview(epoch)

        print("-" * 70)


# ============================================================
# FINAL CHECKPOINT
# ============================================================

final_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "mri_generator_v4_final.pth"
)

torch.save(
    {
        "epoch": TOTAL_EPOCHS,

        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "optimizer_g_state_dict":
            optimizer_g.state_dict(),

        "optimizer_d_state_dict":
            optimizer_d.state_dict(),

        "config": {
            "image_size":
                IMAGE_SIZE,

            "latent_dim":
                LATENT_DIM,

            "num_classes":
                NUM_CLASSES,

            "mode_seeking_target":
                MODE_SEEKING_TARGET,

            "mode_seeking_weight":
                MODE_SEEKING_WEIGHT
        }
    },
    final_checkpoint
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MRI GAN V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Final checkpoint : {final_checkpoint}"
)

print(
    f"Samples          : {SAMPLE_DIR}"
)

print(
    f"History          : {history_path}"
)

print(
    f"Config           : {config_path}"
)

print("=" * 70)
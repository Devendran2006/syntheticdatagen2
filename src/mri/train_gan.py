# ============================================================
# MRI CONDITIONAL DCGAN - TRAINING WITH RESUME SUPPORT
# ============================================================
#
# Purpose:
#   Train a conditional DCGAN using ONLY real MRI Training data.
#
# Important:
#   MRI Testing data is NEVER used for GAN training.
#
# Features:
#   - 128x128 grayscale MRI images
#   - 4 MRI classes
#   - Checkpoint every 5 epochs
#   - Resume training from latest checkpoint
#   - Saves Generator + Discriminator + Optimizers
#   - Fixed preview images
#   - Training history CSV
#   - CPU-friendly configuration
#
# ============================================================

import os
import random
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = "data/imaging/MRI/Training"

OUTPUT_DIR = "outputs/mri"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

SAMPLE_DIR = os.path.join(
    OUTPUT_DIR,
    "generated_samples_128"
)

HISTORY_DIR = os.path.join(
    OUTPUT_DIR,
    "training_history"
)

HISTORY_FILE = os.path.join(
    HISTORY_DIR,
    "mri_gan_128_training_history.csv"
)

CONFIG_FILE = os.path.join(
    HISTORY_DIR,
    "mri_gan_128_config.txt"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 128

# Safer for your i3 + 8GB RAM laptop
BATCH_SIZE = 16

# Total target epochs
TOTAL_EPOCHS = 100

LATENT_DIM = 128

NUM_CLASSES = 4

LEARNING_RATE = 0.0002

BETA1 = 0.5
BETA2 = 0.999

NUM_WORKERS = 0

LABEL_SMOOTHING = 0.90

CHECKPOINT_INTERVAL = 5

SAMPLE_INTERVAL = 5


# ============================================================
# RESUME SETTINGS
# ============================================================

# True = continue from latest checkpoint
# False = start a completely new training
RESUME_TRAINING = True


# If you want to manually specify a checkpoint,
# put its path here.
#
# Example:
#
# RESUME_CHECKPOINT = (
#     "outputs/mri/checkpoints/"
#     "mri_gan_128_epoch_025.pth"
# )
#
# Keep None to automatically find the latest checkpoint.

RESUME_CHECKPOINT = None


# ============================================================
# CPU THREAD SETTINGS
# ============================================================

CPU_THREADS = min(
    4,
    os.cpu_count() if os.cpu_count() else 2
)

torch.set_num_threads(CPU_THREADS)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True
)

os.makedirs(
    HISTORY_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MRI CONDITIONAL DCGAN TRAINING")
print("=" * 70)

print(f"Device          : {DEVICE}")
print(f"CPU Threads     : {CPU_THREADS}")
print(f"Image Size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch Size      : {BATCH_SIZE}")
print(f"Total Epochs    : {TOTAL_EPOCHS}")
print(f"Latent Dim      : {LATENT_DIM}")
print(f"Number Classes  : {NUM_CLASSES}")
print(f"Resume Training : {RESUME_TRAINING}")

print("=" * 70)


# ============================================================
# 2. TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5],
        [0.5]
    )
])


# ============================================================
# 3. DATASET
# ============================================================

print("\nLoading MRI Training dataset...")

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print(f"Dataset path    : {DATA_DIR}")
print(f"Total images    : {len(dataset)}")
print(f"Classes         : {dataset.classes}")

print("\nClass mapping:")

for class_name, class_id in dataset.class_to_idx.items():
    print(
        f"  {class_id} -> {class_name}"
    )


# ============================================================
# VALIDATE CLASSES
# ============================================================

if len(dataset.classes) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(dataset.classes)} classes."
    )


# ============================================================
# 4. DATA LOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


# ============================================================
# 5. GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim,
        num_classes
    ):

        super().__init__()

        self.label_embedding = nn.Embedding(
            num_classes,
            latent_dim
        )

        self.fc = nn.Sequential(

            nn.Linear(
                latent_dim * 2,
                1024 * 4 * 4
            ),

            nn.BatchNorm1d(
                1024 * 4 * 4
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.main = nn.Sequential(

            # 4 -> 8
            nn.ConvTranspose2d(
                1024,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(
                inplace=True
            ),

            # 8 -> 16
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(
                inplace=True
            ),

            # 16 -> 32
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(
                inplace=True
            ),

            # 32 -> 64
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),

            # 64 -> 128
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),

            # Final image
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Tanh()
        )


    def forward(
        self,
        noise,
        labels
    ):

        label_vector = self.label_embedding(
            labels
        )

        x = torch.cat(
            [noise, label_vector],
            dim=1
        )

        x = self.fc(x)

        x = x.view(
            x.size(0),
            1024,
            4,
            4
        )

        return self.main(x)


# ============================================================
# 6. DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.label_embedding = nn.Embedding(
            num_classes,
            IMAGE_SIZE * IMAGE_SIZE
        )

        self.main = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                2,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 64 -> 32
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 32 -> 16
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16 -> 8
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8 -> 4
            nn.Conv2d(
                512,
                1024,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(1024),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 4 -> 1
            nn.Conv2d(
                1024,
                1,
                kernel_size=4,
                stride=1,
                padding=0
            )
        )


    def forward(
        self,
        images,
        labels
    ):

        label_vector = self.label_embedding(
            labels
        )

        label_vector = label_vector.view(
            labels.size(0),
            1,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        x = torch.cat(
            [images, label_vector],
            dim=1
        )

        output = self.main(x)

        return output.view(-1)


# ============================================================
# 7. INITIALIZE MODELS
# ============================================================

generator = Generator(
    LATENT_DIM,
    NUM_CLASSES
).to(DEVICE)

discriminator = Discriminator(
    NUM_CLASSES
).to(DEVICE)


# ============================================================
# 8. LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# 9. OPTIMIZERS
# ============================================================

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


# ============================================================
# 10. WEIGHT INITIALIZATION
# ============================================================

def weights_init(module):

    classname = module.__class__.__name__

    if "Conv" in classname:

        if hasattr(module, "weight"):

            nn.init.normal_(
                module.weight.data,
                0.0,
                0.02
            )

    elif "BatchNorm" in classname:

        if hasattr(module, "weight"):

            nn.init.normal_(
                module.weight.data,
                1.0,
                0.02
            )

        if hasattr(module, "bias"):

            nn.init.constant_(
                module.bias.data,
                0
            )


# ============================================================
# 11. FIND LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint():

    checkpoints = []

    for filename in os.listdir(
        CHECKPOINT_DIR
    ):

        if filename.startswith(
            "mri_gan_128_epoch_"
        ) and filename.endswith(
            ".pth"
        ):

            checkpoints.append(filename)


    if not checkpoints:

        return None


    def extract_epoch(filename):

        try:

            number = filename.replace(
                "mri_gan_128_epoch_",
                ""
            )

            number = number.replace(
                ".pth",
                ""
            )

            return int(number)

        except ValueError:

            return -1


    checkpoints.sort(
        key=extract_epoch
    )

    latest = checkpoints[-1]

    return os.path.join(
        CHECKPOINT_DIR,
        latest
    )


# ============================================================
# 12. LOAD / RESUME
# ============================================================

start_epoch = 1

history = []


if RESUME_TRAINING:

    if RESUME_CHECKPOINT is not None:

        checkpoint_path = RESUME_CHECKPOINT

    else:

        checkpoint_path = find_latest_checkpoint()


    if checkpoint_path is not None and os.path.exists(
        checkpoint_path
    ):

        print("\n" + "=" * 70)
        print("RESUMING PREVIOUS TRAINING")
        print("=" * 70)

        print(
            f"Checkpoint: {checkpoint_path}"
        )


        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
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


        completed_epoch = checkpoint[
            "epoch"
        ]

        start_epoch = completed_epoch + 1


        if "history" in checkpoint:

            history = checkpoint[
                "history"
            ]


        print(
            f"Completed epochs : {completed_epoch}"
        )

        print(
            f"Starting epoch   : {start_epoch}"
        )

        print(
            f"Remaining epochs : "
            f"{max(0, TOTAL_EPOCHS - completed_epoch)}"
        )

        print("=" * 70)


    else:

        print("\nNo previous checkpoint found.")

        print(
            "Starting new training..."
        )

        generator.apply(
            weights_init
        )

        discriminator.apply(
            weights_init
        )

else:

    print("\nStarting NEW training...")

    generator.apply(
        weights_init
    )

    discriminator.apply(
        weights_init
    )


# ============================================================
# 13. FIXED SAMPLE NOISE
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
    dtype=torch.long,
    device=DEVICE
)


# ============================================================
# 14. SAVE TRAINING HISTORY
# ============================================================

def save_history():

    with open(
        HISTORY_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "generator_loss",
            "discriminator_loss",
            "epoch_time_seconds"
        ])

        for row in history:

            writer.writerow([
                row["epoch"],
                row["generator_loss"],
                row["discriminator_loss"],
                row["epoch_time_seconds"]
            ])


# ============================================================
# 15. SAVE CONFIGURATION
# ============================================================

with open(
    CONFIG_FILE,
    "w"
) as file:

    file.write(
        "MRI Conditional DCGAN Configuration\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(
        f"Data Directory: {DATA_DIR}\n"
    )

    file.write(
        f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
    )

    file.write(
        f"Batch Size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Total Epochs: {TOTAL_EPOCHS}\n"
    )

    file.write(
        f"Latent Dimension: {LATENT_DIM}\n"
    )

    file.write(
        f"Number of Classes: {NUM_CLASSES}\n"
    )

    file.write(
        f"Learning Rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Device: {DEVICE}\n"
    )

    file.write(
        f"CPU Threads: {CPU_THREADS}\n"
    )


# ============================================================
# 16. TRAINING LOOP
# ============================================================

if start_epoch > TOTAL_EPOCHS:

    print("\nTraining already completed.")

else:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING FROM EPOCH "
        f"{start_epoch} TO {TOTAL_EPOCHS}"
    )
    print("=" * 70)


    for epoch in range(
        start_epoch,
        TOTAL_EPOCHS + 1
    ):

        epoch_start_time = time.time()

        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        batch_count = 0


        for real_images, labels in loader:

            real_images = real_images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            current_batch_size = (
                real_images.size(0)
            )


            # ====================================================
            # TRAIN DISCRIMINATOR
            # ====================================================

            optimizer_d.zero_grad()


            real_targets = torch.full(
                (
                    current_batch_size,
                ),
                LABEL_SMOOTHING,
                device=DEVICE
            )


            real_output = discriminator(
                real_images,
                labels
            )


            d_real_loss = criterion(
                real_output,
                real_targets
            )


            # Generate fake images
            noise = torch.randn(
                current_batch_size,
                LATENT_DIM,
                device=DEVICE
            )


            fake_images = generator(
                noise,
                labels
            )


            fake_targets = torch.zeros(
                current_batch_size,
                device=DEVICE
            )


            fake_output = discriminator(
                fake_images.detach(),
                labels
            )


            d_fake_loss = criterion(
                fake_output,
                fake_targets
            )


            d_loss = (
                d_real_loss +
                d_fake_loss
            ) / 2


            d_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=5.0
            )


            optimizer_d.step()


            # ====================================================
            # TRAIN GENERATOR
            # ====================================================

            optimizer_g.zero_grad()


            noise = torch.randn(
                current_batch_size,
                LATENT_DIM,
                device=DEVICE
            )


            fake_images = generator(
                noise,
                labels
            )


            generator_targets = torch.ones(
                current_batch_size,
                device=DEVICE
            )


            fake_output = discriminator(
                fake_images,
                labels
            )


            g_loss = criterion(
                fake_output,
                generator_targets
            )


            g_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=5.0
            )


            optimizer_g.step()


            total_g_loss += (
                g_loss.item()
            )

            total_d_loss += (
                d_loss.item()
            )

            batch_count += 1


        # ========================================================
        # EPOCH STATISTICS
        # ========================================================

        avg_g_loss = (
            total_g_loss /
            batch_count
        )

        avg_d_loss = (
            total_d_loss /
            batch_count
        )


        epoch_time = (
            time.time() -
            epoch_start_time
        )


        history.append({
            "epoch": epoch,
            "generator_loss": avg_g_loss,
            "discriminator_loss": avg_d_loss,
            "epoch_time_seconds": epoch_time
        })


        # ========================================================
        # PRINT PROGRESS
        # ========================================================

        print(
            f"\nEpoch [{epoch}/{TOTAL_EPOCHS}]"
        )

        print(
            f"Generator Loss     : "
            f"{avg_g_loss:.4f}"
        )

        print(
            f"Discriminator Loss  : "
            f"{avg_d_loss:.4f}"
        )

        print(
            f"Epoch Time         : "
            f"{epoch_time / 60:.2f} minutes"
        )


        # ========================================================
        # GENERATE PREVIEW
        # ========================================================

        if (
            epoch % SAMPLE_INTERVAL == 0
            or epoch == 1
            or epoch == TOTAL_EPOCHS
        ):

            generator.eval()

            with torch.no_grad():

                samples = generator(
                    fixed_noise,
                    fixed_labels
                )


            sample_path = os.path.join(
                SAMPLE_DIR,
                f"epoch_{epoch:03d}.png"
            )


            save_image(
                samples,
                sample_path,
                normalize=True,
                nrow=4
            )


            print(
                f"Preview saved      : "
                f"{sample_path}"
            )


        # ========================================================
        # SAVE CHECKPOINT
        # ========================================================

        if (
            epoch % CHECKPOINT_INTERVAL == 0
            or epoch == TOTAL_EPOCHS
        ):

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"mri_gan_128_epoch_{epoch:03d}.pth"
            )


            checkpoint = {

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

                "history":
                    history,

                "config": {

                    "image_size":
                        IMAGE_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "latent_dim":
                        LATENT_DIM,

                    "num_classes":
                        NUM_CLASSES,

                    "learning_rate":
                        LEARNING_RATE
                }
            }


            torch.save(
                checkpoint,
                checkpoint_path
            )


            print(
                f"Checkpoint saved   : "
                f"{checkpoint_path}"
            )


            # Save history
            save_history()


        # ========================================================
        # CLEAR CACHE
        # ========================================================

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()


# ============================================================
# 17. FINAL GENERATOR
# ============================================================

final_generator_path = os.path.join(
    CHECKPOINT_DIR,
    "mri_generator_128_final.pth"
)

final_discriminator_path = os.path.join(
    CHECKPOINT_DIR,
    "mri_discriminator_128_final.pth"
)


torch.save(
    generator.state_dict(),
    final_generator_path
)

torch.save(
    discriminator.state_dict(),
    final_discriminator_path
)


# ============================================================
# 18. FINAL HISTORY SAVE
# ============================================================

save_history()


# ============================================================
# 19. TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MRI GAN TRAINING COMPLETE")
print("=" * 70)

print(
    f"Final Generator     : "
    f"{final_generator_path}"
)

print(
    f"Final Discriminator  : "
    f"{final_discriminator_path}"
)

print(
    f"Training History     : "
    f"{HISTORY_FILE}"
)

print(
    f"Generated Samples    : "
    f"{SAMPLE_DIR}"
)

print("=" * 70)
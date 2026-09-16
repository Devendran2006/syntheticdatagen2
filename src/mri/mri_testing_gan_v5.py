"""
MRI TESTING GAN V5
==================

Testing-data-only MRI synthetic image generation.

V5 approach:
    1. Dataset inspection
    2. Global MRI intensity normalization
    3. Autoencoder pretraining
    4. Decoder used as GAN generator initialization
    5. LSGAN adversarial training
    6. EMA generator
    7. Class-wise synthetic generation
    8. Preview and checkpoint saving

PROTECTED:
    data/imaging/MRI/Training
    data/imaging/MRI/Training_Preprocessed
    outputs/mri
    outputs/mri_testing_gan
    outputs/mri_testing_gan_v3
    outputs/mri_testing_gan_v4

NEW OUTPUT ONLY:
    outputs/mri_testing_gan_v5

IMPORTANT:
This script generates synthetic images for experimentation.
Generated images are NOT medical diagnoses and must be evaluated
before being used for any medical/clinical purpose.
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageFile, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.utils import save_image


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MRI_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "MRI"
)

# ONLY TESTING DATA
TESTING_ROOT = MRI_ROOT / "Testing"

# COMPLETELY NEW V5 OUTPUT
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_gan_v5"
)

MODEL_ROOT = OUTPUT_ROOT / "models"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
SAMPLE_ROOT = OUTPUT_ROOT / "synthetic_samples"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
HISTORY_ROOT = OUTPUT_ROOT / "training_history"
AE_ROOT = OUTPUT_ROOT / "autoencoder"
EVALUATION_ROOT = OUTPUT_ROOT / "evaluation"


for folder in [
    OUTPUT_ROOT,
    MODEL_ROOT,
    PREVIEW_ROOT,
    SAMPLE_ROOT,
    CHECKPOINT_ROOT,
    HISTORY_ROOT,
    AE_ROOT,
    EVALUATION_ROOT,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

CHANNELS = 1

LATENT_DIM = 128


# ------------------------------------------------------------
# FIRST RUN
# ------------------------------------------------------------
#
# Keep this as "glioma".
#
# Once Glioma looks good, change to:
#
# TARGET_CLASS = None
#
# That will train:
#   glioma
#   meningioma
#   notumor
#   pituitary
#
# ------------------------------------------------------------

TARGET_CLASS = "glioma"


# ------------------------------------------------------------
# Autoencoder
# ------------------------------------------------------------

AE_EPOCHS = 5

AE_BATCH_SIZE = 16

AE_LEARNING_RATE = 0.0002


# ------------------------------------------------------------
# GAN
# ------------------------------------------------------------

GAN_EPOCHS = 10

GAN_BATCH_SIZE = 16

G_LEARNING_RATE = 0.0001

D_LEARNING_RATE = 0.0001


BETA1 = 0.5

BETA2 = 0.999


# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------

EMA_DECAY = 0.995


# ------------------------------------------------------------
# Synthetic output
# ------------------------------------------------------------

SYNTHETIC_IMAGES = 100


# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------

PREVIEW_COUNT = 16

PREVIEW_EVERY = 2

CHECKPOINT_EVERY = 5


# ------------------------------------------------------------
# Workers
# ------------------------------------------------------------

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
# ============================================================

SEED = 42


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# IMAGE EXTENSIONS
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
# IMAGE DISCOVERY
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
# DISCOVER CLASSES
# ============================================================

def discover_classes():

    if not TESTING_ROOT.exists():

        raise FileNotFoundError(
            f"\nTesting dataset not found:\n"
            f"{TESTING_ROOT}"
        )

    classes = []

    for folder in sorted(
        TESTING_ROOT.iterdir()
    ):

        if not folder.is_dir():
            continue

        count = len(
            get_image_files(folder)
        )

        if count > 0:
            classes.append(
                folder.name
            )

    if not classes:

        raise RuntimeError(
            f"\nNo MRI classes found in:\n"
            f"{TESTING_ROOT}"
        )

    return classes


# ============================================================
# LOAD RAW IMAGE
# ============================================================

def load_raw_image(path):

    image = Image.open(path)

    image = ImageOps.exif_transpose(image)

    image = image.convert("L")

    return image


# ============================================================
# RESIZE WITHOUT CROPPING
# ============================================================

def resize_preserve(image):

    image = ImageOps.contain(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        method=Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        0
    )

    left = (
        IMAGE_SIZE
        - image.width
    ) // 2

    top = (
        IMAGE_SIZE
        - image.height
    ) // 2

    canvas.paste(
        image,
        (
            left,
            top
        )
    )

    return canvas


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_image(image):

    """
    Robust percentile normalization.

    Unlike aggressive autocontrast, this attempts to preserve
    the relative MRI intensity structure.
    """

    array = np.asarray(
        image,
        dtype=np.float32
    )

    low = np.percentile(
        array,
        1
    )

    high = np.percentile(
        array,
        99
    )

    if high <= low:

        low = array.min()

        high = array.max()

    if high <= low:

        normalized = np.zeros_like(
            array
        )

    else:

        normalized = (
            array - low
        ) / (
            high - low
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0
        )

    normalized = (
        normalized * 255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        normalized,
        mode="L"
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(path):

    image = load_raw_image(
        path
    )

    image = resize_preserve(
        image
    )

    image = normalize_image(
        image
    )

    return image


# ============================================================
# DATASET
# ============================================================

class MRIDataset(Dataset):

    def __init__(
        self,
        paths
    ):

        self.paths = paths

        self.transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                [0.5],
                [0.5]
            ),

        ])


    def __len__(self):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        index
    ):

        path = self.paths[index]

        try:

            image = preprocess_image(
                path
            )

            tensor = self.transform(
                image
            )

            return tensor

        except Exception as error:

            print(
                f"Warning: {path}"
            )

            print(
                error
            )

            new_index = (
                index + 1
            ) % len(self.paths)

            return self.__getitem__(
                new_index
            )


# ============================================================
# AUTOENCODER ENCODER
# ============================================================

class Encoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        self.features = nn.Sequential(

            # 128 -> 64

            nn.Conv2d(
                1,
                64,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 64 -> 32

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 32 -> 16

            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16 -> 8

            nn.Conv2d(
                256,
                512,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8 -> 4

            nn.Conv2d(
                512,
                512,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

        )

        self.latent = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512 * 4 * 4,
                latent_dim
            ),

        )


    def forward(self, x):

        x = self.features(x)

        z = self.latent(x)

        return z


# ============================================================
# DECODER
# ============================================================

class Decoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        self.start = nn.Sequential(

            nn.Linear(
                latent_dim,
                512 * 4 * 4
            ),

            nn.ReLU(
                inplace=True
            ),

        )

        self.reshape_channels = 512


        self.blocks = nn.Sequential(

            # 4 -> 8

            nn.ConvTranspose2d(
                512,
                512,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            # 8 -> 16

            nn.ConvTranspose2d(
                512,
                256,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            # 16 -> 32

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 32 -> 64

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 64 -> 128

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

        )

        self.output = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                3,
                1,
                1
            ),

            nn.BatchNorm2d(
                16
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                16,
                1,
                3,
                1,
                1
            ),

            nn.Tanh()

        )


    def forward(self, z):

        x = self.start(z)

        x = x.view(
            -1,
            512,
            4,
            4
        )

        x = self.blocks(x)

        x = self.output(x)

        return x


# ============================================================
# AUTOENCODER
# ============================================================

class AutoEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()


    def forward(self, x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        return reconstruction


# ============================================================
# GAN GENERATOR
# ============================================================

class Generator(
    nn.Module
):

    def __init__(
        self,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        self.decoder = Decoder(
            latent_dim
        )


    def forward(self, z):

        return self.decoder(z)


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 128 -> 64

            nn.Conv2d(
                1,
                64,
                4,
                2,
                1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),

            # 64 -> 32

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),

            # 32 -> 16

            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Dropout2d(
                0.10
            ),

            # 16 -> 8

            nn.Conv2d(
                256,
                512,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8 -> 4

            nn.Conv2d(
                512,
                512,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

        )

        self.classifier = nn.Sequential(

            nn.Conv2d(
                512,
                1,
                4,
                1,
                0
            ),

            nn.Flatten(),

            nn.Sigmoid()

        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(
    model
):

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.Linear
            )
        ):

            if hasattr(
                module,
                "weight"
            ):

                nn.init.normal_(
                    module.weight,
                    0.0,
                    0.02
                )

            if getattr(
                module,
                "bias",
                None
            ) is not None:

                nn.init.constant_(
                    module.bias,
                    0
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
# AUTOENCODER PRETRAINING
# ============================================================

def train_autoencoder(
    image_paths,
    class_name
):

    print("\n")

    print("=" * 72)

    print(
        f"AUTOENCODER PRETRAINING: "
        f"{class_name.upper()}"
    )

    print("=" * 72)

    print(
        f"Images : {len(image_paths)}"
    )

    print(
        f"Resolution : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Epochs : {AE_EPOCHS}"
    )

    print(
        f"Device : {DEVICE}"
    )

    print("=" * 72)


    dataset = MRIDataset(
        image_paths
    )


    loader = DataLoader(

        dataset,

        batch_size=min(
            AE_BATCH_SIZE,
            len(dataset)
        ),

        shuffle=True,

        num_workers=NUM_WORKERS,

        drop_last=True

    )


    autoencoder = AutoEncoder().to(
        DEVICE
    )


    initialize_weights(
        autoencoder
    )


    optimizer = optim.Adam(

        autoencoder.parameters(),

        lr=AE_LEARNING_RATE,

        betas=(

            0.5,

            0.999

        )

    )


    criterion = nn.L1Loss()


    history = []


    for epoch in range(
        1,
        AE_EPOCHS + 1
    ):

        start = time.time()

        total_loss = 0.0

        batches = 0


        autoencoder.train()


        for real_images in loader:

            real_images = (
                real_images.to(
                    DEVICE
                )
            )


            optimizer.zero_grad()


            reconstructed = (
                autoencoder(
                    real_images
                )
            )


            loss = criterion(

                reconstructed,

                real_images

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                autoencoder.parameters(),

                5.0

            )


            optimizer.step()


            total_loss += (
                loss.item()
            )

            batches += 1


        avg_loss = (

            total_loss
            / max(
                batches,
                1
            )

        )


        elapsed = (
            time.time()
            - start
        )


        history.append({

            "epoch":
                epoch,

            "reconstruction_loss":
                avg_loss,

            "time_seconds":
                elapsed

        })


        print(

            f"AE Epoch "
            f"[{epoch:02d}/{AE_EPOCHS}] | "
            f"Reconstruction Loss: "
            f"{avg_loss:.5f} | "
            f"Time: "
            f"{elapsed:.1f}s"

        )


        # Save reconstruction preview

        if (
            epoch == 1
            or epoch == AE_EPOCHS
        ):

            autoencoder.eval()

            with torch.no_grad():

                preview = autoencoder(
                    real_images[:16]
                )


            preview_path = (

                AE_ROOT
                / class_name
                / f"epoch_{epoch:04d}.png"

            )


            preview_path.parent.mkdir(

                parents=True,

                exist_ok=True

            )


            save_image(

                preview,

                preview_path,

                normalize=True,

                value_range=(-1, 1),

                nrow=4

            )


            print(

                f"AE preview saved: "
                f"{preview_path}"

            )


    # Save AE

    ae_path = (

        AE_ROOT
        / f"{class_name}_autoencoder.pth"

    )


    torch.save(

        autoencoder.state_dict(),

        ae_path

    )


    history_path = (

        HISTORY_ROOT
        / f"{class_name}_autoencoder_history.csv"

    )


    pd.DataFrame(
        history
    ).to_csv(

        history_path,

        index=False

    )


    print(
        f"Autoencoder saved: "
        f"{ae_path}"
    )


    return autoencoder


# ============================================================
# CREATE GENERATOR FROM AUTOENCODER DECODER
# ============================================================

def create_generator_from_autoencoder(
    autoencoder
):

    generator = Generator().to(
        DEVICE
    )


    generator.decoder.load_state_dict(

        autoencoder.decoder.state_dict()

    )


    return generator


# ============================================================
# EMA
# ============================================================

def create_ema(
    generator
):

    ema = Generator().to(
        DEVICE
    )

    ema.load_state_dict(
        generator.state_dict()
    )


    for parameter in ema.parameters():

        parameter.requires_grad = False


    return ema


@torch.no_grad()
def update_ema(
    generator,
    ema
):

    for source, target in zip(

        generator.parameters(),

        ema.parameters()

    ):

        target.data.mul_(
            EMA_DECAY
        )

        target.data.add_(

            source.data,

            alpha=1.0 - EMA_DECAY

        )


# ============================================================
# SAVE GAN PREVIEW
# ============================================================

def save_gan_preview(

    generator,

    noise,

    class_name,

    epoch

):

    generator.eval()


    with torch.no_grad():

        images = generator(
            noise
        )


    path = (

        PREVIEW_ROOT
        / class_name
        / f"epoch_{epoch:04d}.png"

    )


    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    save_image(

        images,

        path,

        normalize=True,

        value_range=(-1, 1),

        nrow=4

    )


    generator.train()


    return path


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(

    generator,

    discriminator,

    ema,

    optimizer_g,

    optimizer_d,

    epoch,

    class_name,

    history

):

    path = (

        CHECKPOINT_ROOT
        / f"{class_name}_latest.pth"

    )


    torch.save({

        "epoch":
            epoch,

        "class_name":
            class_name,

        "generator":
            generator.state_dict(),

        "ema_generator":
            ema.state_dict(),

        "discriminator":
            discriminator.state_dict(),

        "optimizer_g":
            optimizer_g.state_dict(),

        "optimizer_d":
            optimizer_d.state_dict(),

        "history":
            history,

        "image_size":
            IMAGE_SIZE,

        "latent_dim":
            LATENT_DIM,

    }, path)


    return path


# ============================================================
# GAN TRAINING
# ============================================================

def train_gan(

    generator,

    discriminator,

    image_paths,

    class_name

):

    print("\n")

    print("=" * 72)

    print(
        f"GAN TRAINING: "
        f"{class_name.upper()}"
    )

    print("=" * 72)

    print(
        f"Real Testing images : "
        f"{len(image_paths)}"
    )

    print(
        f"Resolution          : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"GAN Epochs          : "
        f"{GAN_EPOCHS}"
    )

    print(
        f"Batch size          : "
        f"{GAN_BATCH_SIZE}"
    )

    print(
        "Objective            : LSGAN"
    )

    print(
        "Generator init      : Autoencoder decoder"
    )

    print(
        "EMA                 : Enabled"
    )

    print(
        f"Device              : "
        f"{DEVICE}"
    )

    print("=" * 72)


    dataset = MRIDataset(
        image_paths
    )


    loader = DataLoader(

        dataset,

        batch_size=min(
            GAN_BATCH_SIZE,
            len(dataset)
        ),

        shuffle=True,

        num_workers=NUM_WORKERS,

        drop_last=True

    )


    discriminator = (
        discriminator.to(
            DEVICE
        )
    )


    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    optimizer_g = optim.Adam(

        generator.parameters(),

        lr=G_LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )

    )


    optimizer_d = optim.Adam(

        discriminator.parameters(),

        lr=D_LEARNING_RATE,

        betas=(
            BETA1,
            BETA2
        )

    )


    # --------------------------------------------------------
    # LSGAN loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()


    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema = create_ema(
        generator
    )


    # --------------------------------------------------------
    # Fixed preview
    # --------------------------------------------------------

    fixed_noise = torch.randn(

        PREVIEW_COUNT,

        LATENT_DIM,

        device=DEVICE

    )


    history = []


    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(

        1,

        GAN_EPOCHS + 1

    ):

        epoch_start = time.time()


        generator.train()

        discriminator.train()


        g_total = 0.0

        d_total = 0.0

        real_total = 0.0

        fake_total = 0.0

        batches = 0


        # ====================================================
        # BATCHES
        # ====================================================

        for real_images in loader:

            real_images = (

                real_images
                .to(DEVICE)

            )


            batch_size = (
                real_images.size(0)
            )


            # ------------------------------------------------
            # REAL / FAKE LABELS
            # ------------------------------------------------

            real_labels = torch.ones(

                batch_size,

                1,

                device=DEVICE

            )


            fake_labels = torch.zeros(

                batch_size,

                1,

                device=DEVICE

            )


            # =================================================
            # DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad()


            real_output = discriminator(
                real_images
            )


            real_loss = criterion(

                real_output,

                real_labels

            )


            noise = torch.randn(

                batch_size,

                LATENT_DIM,

                device=DEVICE

            )


            fake_images = generator(
                noise
            )


            fake_output = discriminator(

                fake_images.detach()

            )


            fake_loss = criterion(

                fake_output,

                fake_labels

            )


            d_loss = (

                real_loss
                + fake_loss

            ) * 0.5


            d_loss.backward()


            torch.nn.utils.clip_grad_norm_(

                discriminator.parameters(),

                5.0

            )


            optimizer_d.step()


            # =================================================
            # GENERATOR
            # =================================================

            optimizer_g.zero_grad()


            noise = torch.randn(

                batch_size,

                LATENT_DIM,

                device=DEVICE

            )


            generated = generator(
                noise
            )


            generated_output = discriminator(
                generated
            )


            g_loss = criterion(

                generated_output,

                real_labels

            )


            g_loss.backward()


            torch.nn.utils.clip_grad_norm_(

                generator.parameters(),

                5.0

            )


            optimizer_g.step()


            # =================================================
            # EMA
            # =================================================

            update_ema(

                generator,

                ema

            )


            # =================================================
            # METRICS
            # =================================================

            g_total += g_loss.item()

            d_total += d_loss.item()

            real_total += (

                real_output
                .mean()
                .detach()
                .item()

            )

            fake_total += (

                fake_output
                .mean()
                .detach()
                .item()

            )

            batches += 1


        # ====================================================
        # AVERAGE
        # ====================================================

        avg_g = (

            g_total
            / max(
                batches,
                1
            )

        )


        avg_d = (

            d_total
            / max(
                batches,
                1
            )

        )


        avg_real = (

            real_total
            / max(
                batches,
                1
            )

        )


        avg_fake = (

            fake_total
            / max(
                batches,
                1
            )

        )


        elapsed = (

            time.time()
            - epoch_start

        )


        history.append({

            "class":
                class_name,

            "epoch":
                epoch,

            "generator_loss":
                avg_g,

            "discriminator_loss":
                avg_d,

            "real_score":
                avg_real,

            "fake_score":
                avg_fake,

            "time_seconds":
                elapsed,

        })


        # ====================================================
        # TERMINAL
        # ====================================================

        print(

            f"Epoch "
            f"[{epoch:02d}/{GAN_EPOCHS}] | "

            f"G: "
            f"{avg_g:.4f} | "

            f"D: "
            f"{avg_d:.4f} | "

            f"Real: "
            f"{avg_real:.4f} | "

            f"Fake: "
            f"{avg_fake:.4f} | "

            f"Time: "
            f"{elapsed:.1f}s"

        )


        # ====================================================
        # PREVIEW
        # ====================================================

        if (

            epoch == 1

            or epoch % PREVIEW_EVERY == 0

            or epoch == GAN_EPOCHS

        ):

            preview_path = save_gan_preview(

                ema,

                fixed_noise,

                class_name,

                epoch

            )


            print(

                f"Preview saved: "
                f"{preview_path}"

            )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (

            epoch % CHECKPOINT_EVERY == 0

            or epoch == GAN_EPOCHS

        ):

            checkpoint_path = save_checkpoint(

                generator,

                discriminator,

                ema,

                optimizer_g,

                optimizer_d,

                epoch,

                class_name,

                history

            )


            print(

                f"Checkpoint saved: "
                f"{checkpoint_path}"

            )


        # ====================================================
        # HISTORY
        # ====================================================

        history_path = (

            HISTORY_ROOT
            / f"{class_name}_gan_history.csv"

        )


        pd.DataFrame(
            history
        ).to_csv(

            history_path,

            index=False

        )


    return ema, history


# ============================================================
# FINAL SYNTHETIC IMAGES
# ============================================================

def generate_synthetic_images(

    generator,

    class_name

):

    generator.eval()


    output_dir = (

        SAMPLE_ROOT
        / class_name

    )


    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # Clear ONLY V5 class output
    # --------------------------------------------------------

    for old_file in output_dir.glob(
        "*.png"
    ):

        try:

            old_file.unlink()

        except Exception:

            pass


    generated = 0


    print("\n")

    print(

        f"Generating "
        f"{SYNTHETIC_IMAGES} "
        f"{class_name} images..."

    )


    with torch.no_grad():

        while generated < SYNTHETIC_IMAGES:

            current = min(

                GAN_BATCH_SIZE,

                SYNTHETIC_IMAGES
                - generated

            )


            noise = torch.randn(

                current,

                LATENT_DIM,

                device=DEVICE

            )


            fake = generator(
                noise
            )


            fake = (

                fake + 1.0

            ) / 2.0


            fake = torch.clamp(

                fake,

                0.0,

                1.0

            )


            for index in range(
                current
            ):

                number = (

                    generated
                    + index
                    + 1

                )


                path = (

                    output_dir
                    / (
                        f"{class_name}_"
                        f"synthetic_"
                        f"{number:05d}.png"
                    )

                )


                save_image(

                    fake[index],

                    path

                )


            generated += current


            print(

                f"\rGenerated "
                f"{generated}/"
                f"{SYNTHETIC_IMAGES}",

                end=""

            )


    print("\n")


    return output_dir


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(
    image_paths,
    class_name
):

    print("\n")

    print(
        "Calculating source image statistics..."
    )


    values = []

    widths = []

    heights = []


    for path in image_paths:

        try:

            image = load_raw_image(
                path
            )

            widths.append(
                image.width
            )

            heights.append(
                image.height
            )

            array = np.asarray(

                image,

                dtype=np.float32

            )

            values.append(
                array.mean()
            )

        except Exception:

            continue


    if values:

        statistics = {

            "class":
                class_name,

            "images":
                len(image_paths),

            "average_original_width":
                float(np.mean(widths)),

            "average_original_height":
                float(np.mean(heights)),

            "mean_pixel":
                float(np.mean(values)),

            "std_pixel":
                float(np.std(values)),

            "min_pixel":
                float(np.min(values)),

            "max_pixel":
                float(np.max(values)),

        }


    else:

        statistics = {

            "class":
                class_name,

            "images":
                len(image_paths),

        }


    path = (

        EVALUATION_ROOT
        / f"{class_name}_source_statistics.json"

    )


    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            statistics,

            file,

            indent=4

        )


    return statistics


# ============================================================
# CLASS TRAINING
# ============================================================

def train_class(
    class_name
):

    class_path = (
        TESTING_ROOT
        / class_name
    )


    image_paths = get_image_files(
        class_path
    )


    if not image_paths:

        print(
            f"No images found for {class_name}"
        )

        return None


    print("\n")

    print("#" * 72)

    print(
        f"STARTING V5: "
        f"{class_name.upper()}"
    )

    print("#" * 72)


    calculate_dataset_statistics(

        image_paths,

        class_name

    )


    # ========================================================
    # AUTOENCODER
    # ========================================================

    autoencoder = train_autoencoder(

        image_paths,

        class_name

    )


    # ========================================================
    # GENERATOR FROM DECODER
    # ========================================================

    generator = (
        create_generator_from_autoencoder(
            autoencoder
        )
    )


    # ========================================================
    # DISCRIMINATOR
    # ========================================================

    discriminator = Discriminator().to(
        DEVICE
    )


    initialize_weights(
        discriminator
    )


    # ========================================================
    # GAN
    # ========================================================

    ema, history = train_gan(

        generator,

        discriminator,

        image_paths,

        class_name

    )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    generator_path = (

        MODEL_ROOT
        / f"{class_name}_generator.pth"

    )


    ema_path = (

        MODEL_ROOT
        / f"{class_name}_generator_ema.pth"

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

        ema.state_dict(),

        ema_path

    )


    torch.save(

        discriminator.state_dict(),

        discriminator_path

    )


    # ========================================================
    # SYNTHETIC IMAGES
    # ========================================================

    synthetic_dir = generate_synthetic_images(

        ema,

        class_name

    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "class":
            class_name,

        "real_images":
            len(image_paths),

        "synthetic_images":
            SYNTHETIC_IMAGES,

        "resolution":
            f"{IMAGE_SIZE}x{IMAGE_SIZE}",

        "ae_epochs":
            AE_EPOCHS,

        "gan_epochs":
            GAN_EPOCHS,

        "generator":
            str(generator_path),

        "ema_generator":
            str(ema_path),

        "discriminator":
            str(discriminator_path),

        "synthetic_directory":
            str(synthetic_dir),

    }


    result_path = (

        OUTPUT_ROOT
        / f"{class_name}_result.json"

    )


    with open(

        result_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            result,

            file,

            indent=4

        )


    print("\n")

    print("-" * 72)

    print(
        f"{class_name.upper()} V5 COMPLETED"
    )

    print("-" * 72)

    print(
        f"Real images      : "
        f"{len(image_paths)}"
    )

    print(
        f"Synthetic images : "
        f"{SYNTHETIC_IMAGES}"
    )

    print(
        f"Generator        : "
        f"{generator_path}"
    )

    print(
        f"EMA Generator    : "
        f"{ema_path}"
    )

    print(
        f"Synthetic folder : "
        f"{synthetic_dir}"
    )

    print("-" * 72)


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print(
        "MRI TESTING GAN V5"
    )

    print(
        "=================="
    )

    print(
        "Testing dataset ONLY"
    )

    print(
        "Autoencoder pretraining + LSGAN"
    )

    print(
        "Native 128x128"
    )

    print(
        "EMA Generator"
    )

    print(
        "Existing data protected"
    )

    print("\n")


    # ========================================================
    # SAFETY DISPLAY
    # ========================================================

    print("=" * 72)

    print(
        "PROTECTED DATA"
    )

    print("=" * 72)

    print(
        "Training                  : NOT MODIFIED"
    )

    print(
        "Training_Preprocessed     : NOT MODIFIED"
    )

    print(
        "outputs/mri               : NOT MODIFIED"
    )

    print(
        "mri_testing_gan           : NOT MODIFIED"
    )

    print(
        "mri_testing_gan_v3        : NOT MODIFIED"
    )

    print(
        "mri_testing_gan_v4        : NOT MODIFIED"
    )

    print("=" * 72)


    # ========================================================
    # PATHS
    # ========================================================

    print("\nTESTING INPUT:")

    print(
        TESTING_ROOT
    )


    print("\nV5 OUTPUT:")

    print(
        OUTPUT_ROOT
    )


    print("\nDEVICE:")

    print(
        DEVICE
    )


    # ========================================================
    # CLASSES
    # ========================================================

    classes = discover_classes()


    print("\nDetected classes:")


    for class_name in classes:

        count = len(

            get_image_files(

                TESTING_ROOT
                / class_name

            )

        )


        print(

            f"  {class_name:<15}"
            f"{count} images"

        )


    # ========================================================
    # SELECT CLASSES
    # ========================================================

    if TARGET_CLASS is not None:

        if TARGET_CLASS not in classes:

            raise ValueError(

                f"\nTARGET_CLASS "
                f"'{TARGET_CLASS}' "
                f"not found.\n"

                f"Available classes: "
                f"{classes}"

            )


        selected_classes = [
            TARGET_CLASS
        ]


    else:

        selected_classes = classes


    print("\nTraining classes:")

    for class_name in selected_classes:

        print(
            f"  - {class_name}"
        )


    # ========================================================
    # RUN
    # ========================================================

    results = []


    for class_name in selected_classes:

        try:

            result = train_class(
                class_name
            )

            if result is not None:

                results.append(
                    result
                )


        except KeyboardInterrupt:

            print("\n")

            print("=" * 72)

            print(
                "TRAINING STOPPED BY USER"
            )

            print("=" * 72)

            print(
                "Latest V5 checkpoint "
                "has been preserved."
            )

            print(
                CHECKPOINT_ROOT
            )

            return


        except Exception as error:

            print("\n")

            print("=" * 72)

            print(
                f"ERROR IN "
                f"{class_name.upper()}"
            )

            print("=" * 72)

            print(
                repr(error)
            )

            print("=" * 72)

            raise


    # ========================================================
    # SUMMARY
    # ========================================================

    if results:

        summary = pd.DataFrame(
            results
        )


        summary_path = (

            OUTPUT_ROOT
            / "training_summary.csv"

        )


        summary.to_csv(

            summary_path,

            index=False

        )


        print("\n")

        print("=" * 72)

        print(
            "MRI TESTING GAN V5 FINISHED"
        )

        print("=" * 72)


        print(
            summary[
                [
                    "class",
                    "real_images",
                    "synthetic_images",
                    "resolution",
                    "ae_epochs",
                    "gan_epochs"
                ]
            ].to_string(
                index=False
            )
        )


        print("\nV5 output:")

        print(
            OUTPUT_ROOT
        )


        print("\nSynthetic MRI:")

        print(
            SAMPLE_ROOT
        )


        print("\nModels:")

        print(
            MODEL_ROOT
        )


        print("\nPreviews:")

        print(
            PREVIEW_ROOT
        )


        print("=" * 72)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
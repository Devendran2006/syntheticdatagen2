"""
X-RAY CONDITIONAL GAN
=====================

Purpose:
    Conditional GAN architecture for synthetic chest X-ray generation.

Classes:
    0 -> NORMAL
    1 -> PNEUMONIA

Input:
    Grayscale X-ray images
    Shape: [B, 1, 128, 128]
    Range: [-1, 1]

Output:
    Synthetic grayscale X-ray images
    Shape: [B, 1, 128, 128]
    Range: [-1, 1]

IMPORTANT:
    This project is for machine-learning research and engineering validation.
    Synthetic images must NOT be treated as medically diagnostic images.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
IMAGE_CHANNELS = 1

NUM_CLASSES = 2

LATENT_DIM = 128
CLASS_EMBED_DIM = 16

GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(module):
    """
    Initialize GAN layers using standard DCGAN-style initialization.
    """

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)

        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


# ============================================================
# CONDITIONAL GENERATOR
# ============================================================

class XRayGenerator(nn.Module):
    """
    Conditional generator.

    Inputs:
        noise  -> [B, LATENT_DIM]
        labels -> [B]

    Output:
        images -> [B, 1, 128, 128]
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        class_embed_dim=CLASS_EMBED_DIM,
        base_channels=GENERATOR_BASE_CHANNELS,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        # Learnable class representation
        self.class_embedding = nn.Embedding(
            num_classes,
            class_embed_dim
        )

        # Start from a 4x4 feature map
        self.initial_size = 4

        input_dim = latent_dim + class_embed_dim

        self.project = nn.Sequential(
            nn.Linear(
                input_dim,
                base_channels * 8 * 4 * 4
            ),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # 4x4 -> 8x8
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # 8x8 -> 16x16
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        # 16x16 -> 32x32
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 32x32 -> 64x64
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels,
                base_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )

        # 64x64 -> 128x128
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels // 2,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.apply(initialize_weights)

    def forward(self, noise, labels):
        """
        Generate X-ray images conditioned on class labels.
        """

        # [B, class_embed_dim]
        class_vector = self.class_embedding(labels)

        # [B, latent_dim + class_embed_dim]
        combined = torch.cat(
            [noise, class_vector],
            dim=1
        )

        # Project to feature vector
        x = self.project(combined)

        # [B, C, 4, 4]
        x = x.view(
            x.size(0),
            GENERATOR_BASE_CHANNELS * 8,
            4,
            4
        )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # [B, 1, 128, 128]
        x = self.output_layer(x)

        return x


# ============================================================
# CONDITIONAL DISCRIMINATOR
# ============================================================

class XRayDiscriminator(nn.Module):
    """
    Conditional discriminator.

    Inputs:
        images -> [B, 1, 128, 128]
        labels -> [B]

    Output:
        logits -> [B]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        class_embed_dim=CLASS_EMBED_DIM,
        base_channels=DISCRIMINATOR_BASE_CHANNELS,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        # Convert class label into a spatial feature map.
        self.class_embedding = nn.Embedding(
            num_classes,
            class_embed_dim
        )

        # Image branch
        self.image_features = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(
                IMAGE_CHANNELS,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.Conv2d(
                base_channels * 8,
                base_channels * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # The image feature map is:
        #
        # [B, base_channels*8, 4, 4]
        #
        # Class embedding:
        #
        # [B, class_embed_dim]
        #
        # We project the class embedding into the same spatial
        # feature dimension and combine it with the image features.

        self.class_projection = nn.Linear(
            class_embed_dim,
            base_channels * 8 * 4 * 4
        )

        self.output_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                base_channels * 8 * 4 * 4,
                base_channels * 4
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(
                base_channels * 4,
                1
            )
        )

        self.apply(initialize_weights)

    def forward(self, images, labels):
        """
        Decide whether an image is real/fake
        while considering the requested class.
        """

        # Image features
        image_features = self.image_features(images)

        # Class features
        class_vector = self.class_embedding(labels)

        class_features = self.class_projection(class_vector)

        class_features = class_features.view(
            images.size(0),
            DISCRIMINATOR_BASE_CHANNELS * 8,
            4,
            4
        )

        # Condition image representation on class
        combined = image_features + class_features

        # Raw logits
        logits = self.output_layer(combined)

        return logits.view(-1)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_models(device=DEVICE):
    """
    Create generator and discriminator.
    """

    generator = XRayGenerator().to(device)

    discriminator = XRayDiscriminator().to(device)

    return generator, discriminator


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# MODEL SANITY CHECK
# ============================================================

def run_model_sanity_check():
    """
    Verify that generator and discriminator shapes work correctly.

    No training happens here.
    """

    print("=" * 70)
    print("X-RAY GAN MODEL SANITY CHECK")
    print("=" * 70)

    print(f"Device           : {DEVICE}")
    print(f"Image size       : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Latent dimension : {LATENT_DIM}")
    print(f"Classes          : {NUM_CLASSES}")
    print()

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    generator, discriminator = create_models()

    generator.eval()
    discriminator.eval()

    print(
        f"Generator parameters     : "
        f"{count_parameters(generator):,}"
    )

    print(
        f"Discriminator parameters : "
        f"{count_parameters(discriminator):,}"
    )

    print()

    # --------------------------------------------------------
    # Create test batch
    # --------------------------------------------------------

    batch_size = 4

    noise = torch.randn(
        batch_size,
        LATENT_DIM,
        device=DEVICE
    )

    # 0 = NORMAL
    # 1 = PNEUMONIA
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
        device=DEVICE
    )

    print("Generating test batch...")

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    with torch.no_grad():

        fake_images = generator(
            noise,
            labels
        )

    print(
        f"Generated shape  : "
        f"{tuple(fake_images.shape)}"
    )

    print(
        f"Generated range  : "
        f"{fake_images.min().item():.4f} "
        f"to "
        f"{fake_images.max().item():.4f}"
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert fake_images.shape == expected_shape, (
        f"Generator output shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(fake_images.shape)}"
    )

    # --------------------------------------------------------
    # Discriminator
    # --------------------------------------------------------

    print()
    print("Testing discriminator...")

    with torch.no_grad():

        logits = discriminator(
            fake_images,
            labels
        )

    print(
        f"Discriminator shape : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Discriminator values: "
        f"{logits.detach().cpu().numpy()}"
    )

    expected_logits_shape = (batch_size,)

    assert logits.shape == expected_logits_shape, (
        f"Discriminator output shape mismatch. "
        f"Expected {expected_logits_shape}, "
        f"got {tuple(logits.shape)}"
    )

    # --------------------------------------------------------
    # Range validation
    # --------------------------------------------------------

    assert torch.all(fake_images >= -1.0)
    assert torch.all(fake_images <= 1.0)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SANITY CHECK PASSED")
    print("=" * 70)
    print()
    print("Generator : noise + class -> synthetic X-ray")
    print("Classes   : NORMAL / PNEUMONIA")
    print("Resolution: 128 x 128")
    print("Channels  : 1 (grayscale)")
    print()
    print("No training was performed.")
    print("No real test images were used.")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_model_sanity_check()
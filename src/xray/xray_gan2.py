"""
Conditional X-ray GAN model definitions.

IMPORTANT:
- This architecture is a new V2 model and is NOT compatible with checkpoints
  created by the earlier xray_gan.py architecture.
- Back up the existing file and checkpoints before switching architectures.
- Outputs are research-only synthetic images, not medical/diagnostic evidence.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 128
IMAGE_CHANNELS = 1
NUM_CLASSES = 2
LATENT_DIM = 128
CLASS_EMBED_DIM = 32

GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _validate_labels(labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError(
            f"labels must have shape ({batch_size},), got {tuple(labels.shape)}"
        )
    return labels.long()


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------

class XRayGenerator(nn.Module):
    """
    Conditional DCGAN-style generator.

    Input:
        noise:  (B, LATENT_DIM)
        labels: (B,) with 0=NORMAL and 1=PNEUMONIA

    Output:
        images: (B, 1, 128, 128), pixel range approximately [-1, 1]
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
        self.base_channels = base_channels

        self.label_embedding = nn.Embedding(
            num_classes, class_embed_dim
        )

        input_dim = latent_dim + class_embed_dim

        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.project = nn.Sequential(
            nn.Linear(input_dim, base_channels * 16 * 4 * 4),
            nn.BatchNorm1d(base_channels * 16 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.network = nn.Sequential(
            nn.Unflatten(1, (base_channels * 16, 4, 4)),

            nn.ConvTranspose2d(
                base_channels * 16, base_channels * 8,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                base_channels, IMAGE_CHANNELS,
                kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Tanh(),
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"noise must have shape (B, {self.latent_dim}), "
                f"got {tuple(noise.shape)}"
            )

        labels = _validate_labels(labels, noise.shape[0]).to(noise.device)
        label_features = self.label_embedding(labels)
        joined = torch.cat((noise, label_features), dim=1)

        projected = self.project(joined)
        return self.network(projected)


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------

class XRayDiscriminator(nn.Module):
    """
    Conditional discriminator returning one logit per image.

    Compatible with the training loop:
        discriminator(images, labels) -> shape (B,)
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        base_channels=DISCRIMINATOR_BASE_CHANNELS,
        image_channels=IMAGE_CHANNELS,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.base_channels = base_channels
        self.image_channels = image_channels

        self.image_features = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(
                image_channels, base_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(
                base_channels, base_channels * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(
                base_channels * 2, base_channels * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(
                base_channels * 4, base_channels * 8,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.Conv2d(
                base_channels * 8, base_channels * 16,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        feature_dim = base_channels * 16

        self.image_head = nn.utils.spectral_norm(
            nn.Linear(feature_dim, 1)
        )

        self.label_embedding = nn.Embedding(
            num_classes, feature_dim
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Spectral norm wraps the linear weight, but initialization
            # is safely applied to the available weight tensor.
            if hasattr(module, "weight_orig"):
                nn.init.normal_(module.weight_orig, mean=0.0, std=0.02)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, images, labels):
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape (B, C, H, W), got {tuple(images.shape)}"
            )
        if images.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} image channel(s), "
                f"got {images.shape[1]}"
            )

        labels = _validate_labels(labels, images.shape[0]).to(images.device)

        features = self.image_features(images)
        unconditional_score = self.image_head(features).squeeze(1)

        # Projection conditioning makes the discriminator score depend
        # on both image content and the requested class.
        label_features = self.label_embedding(labels)
        conditional_score = (features * label_features).sum(dim=1)
        conditional_score = conditional_score / (features.shape[1] ** 0.5)

        return unconditional_score + conditional_score


# ---------------------------------------------------------------------
# Standalone architecture smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    generator = XRayGenerator().to(DEVICE)
    discriminator = XRayDiscriminator().to(DEVICE)

    batch_size = 4
    noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = generator(noise, labels)
        scores = discriminator(generated, labels)

    print("=" * 68)
    print("X-RAY GAN V2 ARCHITECTURE CHECK")
    print("=" * 68)
    print(f"Device              : {DEVICE}")
    print(f"Generated shape     : {tuple(generated.shape)}")
    print(f"Generated range     : {generated.min().item():.4f} to {generated.max().item():.4f}")
    print(f"Discriminator shape : {tuple(scores.shape)}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("=" * 68)
    print("Architecture check passed. No training was performed.")

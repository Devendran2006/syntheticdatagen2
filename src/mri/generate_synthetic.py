import os
import shutil
import torch
import torch.nn as nn
from torchvision.utils import save_image


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
LATENT_DIM = 128
NUM_CLASSES = 4
EMBED_DIM = 128

# IMPORTANT:
# Change this if you want more/less synthetic images per class
IMAGES_PER_CLASS = 500

BATCH_SIZE = 64

OUTPUT_DIR = "outputs/mri/synthetic"

GENERATOR_PATH = (
    "outputs/mri/checkpoints/mri_generator_final.pth"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# CONDITIONAL GENERATOR
# MUST MATCH TRAINING CHECKPOINT
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        latent_dim=128,
        num_classes=4,
        embed_dim=128
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Class embedding
        self.label_embedding = nn.Embedding(
            num_classes,
            embed_dim
        )

        # latent 128 + embedding 128 = 256
        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim + embed_dim,
                512 * 4 * 4
            ),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )

        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.main = nn.Sequential(

            # 4x4 -> 8x8
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

            # 8x8 -> 16x16
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

            # 16x16 -> 32x32
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

            # 32x32 -> 64x64
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

    def forward(self, z, labels):

        # Get class embedding
        label_embedding = self.label_embedding(labels)

        # Combine noise + class information
        x = torch.cat(
            [z, label_embedding],
            dim=1
        )

        # Fully connected layer
        x = self.fc(x)

        # Reshape to feature map
        x = x.view(
            x.size(0),
            512,
            4,
            4
        )

        # Generate image
        image = self.main(x)

        return image


# ============================================================
# LOAD GENERATOR
# ============================================================

def load_generator():

    print("Loading trained generator...")

    if not os.path.exists(GENERATOR_PATH):

        raise FileNotFoundError(
            f"\nGenerator checkpoint not found:\n"
            f"{GENERATOR_PATH}\n\n"
            f"Run train_gan.py first."
        )

    generator = Generator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    checkpoint = torch.load(
        GENERATOR_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "generator_state_dict" in checkpoint:

            state_dict = checkpoint[
                "generator_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key.replace(
                "module.",
                "",
                1
            )

        cleaned_state_dict[key] = value

    # Load weights
    generator.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    generator.eval()

    print("Generator loaded successfully.")

    return generator


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )


# ============================================================
# GENERATE ONE CLASS
# ============================================================

@torch.no_grad()
def generate_class_images(
    generator,
    class_id,
    class_name,
    num_images
):

    print(
        f"\nGenerating {class_name}..."
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    generated = 0

    while generated < num_images:

        current_batch = min(
            BATCH_SIZE,
            num_images - generated
        )

        # Random latent vectors
        z = torch.randn(
            current_batch,
            LATENT_DIM,
            device=DEVICE
        )

        # Same class label
        labels = torch.full(
            (current_batch,),
            class_id,
            dtype=torch.long,
            device=DEVICE
        )

        # Generate images
        fake_images = generator(
            z,
            labels
        )

        # Convert [-1, 1] -> [0, 1]
        fake_images = (
            fake_images + 1
        ) / 2

        fake_images = torch.clamp(
            fake_images,
            0,
            1
        )

        # Save individual images
        for i in range(current_batch):

            image_number = (
                generated + i + 1
            )

            file_path = os.path.join(
                output_path,
                f"{class_name}_synthetic_{image_number:05d}.png"
            )

            save_image(
                fake_images[i],
                file_path
            )

        generated += current_batch

        print(
            f"  {generated}/{num_images} generated",
            end="\r"
        )

    print(
        f"\n  Completed: {num_images} images"
    )


# ============================================================
# GENERATE PREVIEW
# ============================================================

@torch.no_grad()
def generate_preview(generator):

    print(
        "\nCreating preview..."
    )

    preview_images = []

    for class_id in range(NUM_CLASSES):

        z = torch.randn(
            4,
            LATENT_DIM,
            device=DEVICE
        )

        labels = torch.full(
            (4,),
            class_id,
            dtype=torch.long,
            device=DEVICE
        )

        images = generator(
            z,
            labels
        )

        images = (
            images + 1
        ) / 2

        images = torch.clamp(
            images,
            0,
            1
        )

        preview_images.append(images)

    preview_images = torch.cat(
        preview_images,
        dim=0
    )

    preview_path = (
        "outputs/mri/"
        "generated_samples/"
        "final_generation_preview.png"
    )

    os.makedirs(
        os.path.dirname(preview_path),
        exist_ok=True
    )

    save_image(
        preview_images,
        preview_path,
        nrow=4
    )

    print(
        f"Preview saved:\n{preview_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 68)
    print(
        "             MRI SYNTHETIC IMAGE GENERATION"
    )
    print("=" * 68)

    print(
        f"\nDevice          : {DEVICE}"
    )

    print(
        f"Image size      : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Latent dimension: {LATENT_DIM}"
    )

    print(
        f"Classes         : {NUM_CLASSES}"
    )

    print(
        f"Images/class    : {IMAGES_PER_CLASS}"
    )

    print(
        f"Total images    : "
        f"{IMAGES_PER_CLASS * NUM_CLASSES}"
    )

    print("\n" + "=" * 68)

    # --------------------------------------------------------
    # Prepare folders
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    generator = load_generator()

    # --------------------------------------------------------
    # Generate images
    # --------------------------------------------------------

    total_generated = 0

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        generate_class_images(
            generator,
            class_id,
            class_name,
            IMAGES_PER_CLASS
        )

        total_generated += (
            IMAGES_PER_CLASS
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    generate_preview(
        generator
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print(
        "       MRI SYNTHETIC GENERATION COMPLETED"
    )
    print("=" * 68)

    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        count = len([
            f
            for f in os.listdir(class_path)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        print(
            f"{class_name:<15}: "
            f"{count:,} images"
        )

    print("-" * 68)

    print(
        f"Total Synthetic MRI Images: "
        f"{total_generated:,}"
    )

    print("\nSaved to:")
    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print("=" * 68)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
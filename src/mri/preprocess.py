import os
from PIL import Image
from torchvision import transforms


# =========================================================
# PATHS
# =========================================================

BASE_DIR = "data/imaging/MRI"

TRAINING_DIR = os.path.join(
    BASE_DIR,
    "Training"
)

TESTING_DIR = os.path.join(
    BASE_DIR,
    "Testing"
)

OUTPUT_DIR = "outputs/mri/processed"


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_SIZE = 128

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5],
            [0.5]
        )
    ]
)


# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

def create_directories():

    for dataset_type in [
        "Training",
        "Testing"
    ]:

        for class_name in CLASSES:

            path = os.path.join(
                OUTPUT_DIR,
                dataset_type,
                class_name
            )

            os.makedirs(
                path,
                exist_ok=True
            )


# =========================================================
# PROCESS IMAGES
# =========================================================

def process_dataset(
    source_dir,
    dataset_type
):

    total = 0

    print()
    print(
        f"========== {dataset_type.upper()} PROCESSING =========="
    )

    for class_name in CLASSES:

        class_path = os.path.join(
            source_dir,
            class_name
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            dataset_type,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                f"{class_name:<15}: folder not found"
            )

            continue

        files = []

        for file in os.listdir(class_path):

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            ):

                files.append(file)

        processed = 0

        for file in files:

            source_file = os.path.join(
                class_path,
                file
            )

            try:

                image = Image.open(
                    source_file
                )

                image = transform(
                    image
                )

                image = (
                    image
                    .clamp(-1, 1)
                    .add(1)
                    .div(2)
                    .mul(255)
                    .byte()
                )

                image = image.squeeze(
                    0
                )

                image = Image.fromarray(
                    image.numpy()
                )

                output_file = os.path.join(
                    output_path,
                    os.path.splitext(file)[0]
                    + ".png"
                )

                image.save(
                    output_file
                )

                processed += 1
                total += 1

            except Exception as e:

                print(
                    f"Error processing {file}: {e}"
                )

        print(
            f"{class_name:<15}: {processed:,} images processed"
        )

    print(
        f"Total {dataset_type}: {total:,}"
    )

    return total


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print(
        "          MRI IMAGE PREPROCESSING"
    )
    print("=" * 55)

    create_directories()

    training_total = process_dataset(
        TRAINING_DIR,
        "Training"
    )

    testing_total = process_dataset(
        TESTING_DIR,
        "Testing"
    )

    print()
    print("=" * 55)
    print(
        "PREPROCESSING COMPLETED"
    )
    print("=" * 55)

    print(
        f"Training images : {training_total:,}"
    )

    print(
        f"Testing images  : {testing_total:,}"
    )

    print(
        f"Total images    : {training_total + testing_total:,}"
    )

    print()
    print(
        f"Processed data saved to: {OUTPUT_DIR}"
    )
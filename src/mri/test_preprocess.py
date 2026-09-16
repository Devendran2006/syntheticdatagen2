import os
import cv2


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "imaging",
    "MRI",
    "Testing"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "mri",
    "processed_testing"
)


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_SIZE = 64

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

for class_name in CLASSES:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            class_name
        ),
        exist_ok=True
    )


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(
    input_path,
    output_path
):

    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        return False

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    cv2.imwrite(
        output_path,
        image
    )

    return True


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 60)
print("          MRI TESTING PREPROCESSING")
print("=" * 60)
print()


if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"Testing directory not found:\n{TEST_DIR}"
    )


total_processed = 0


for class_name in CLASSES:

    input_class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    print(
        f"{class_name:<15}: ",
        end=""
    )

    if not os.path.exists(
        input_class_dir
    ):

        print("folder not found")

        continue


    files = [

        file

        for file in os.listdir(
            input_class_dir
        )

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        )
    ]


    processed = 0


    for index, filename in enumerate(
        files
    ):

        input_path = os.path.join(
            input_class_dir,
            filename
        )

        output_filename = (
            f"{class_name}_"
            f"{index + 1:05d}.png"
        )

        output_path = os.path.join(
            output_class_dir,
            output_filename
        )


        if process_image(
            input_path,
            output_path
        ):

            processed += 1


    total_processed += processed


    print(
        f"{processed:,} images processed"
    )


print()
print("-" * 60)

print(
    f"Total Testing Images: "
    f"{total_processed:,}"
)

print()

print(
    "Processed testing data saved to:"
)

print(
    OUTPUT_DIR
)

print()

print("=" * 60)
print("       TEST PREPROCESSING COMPLETED")
print("=" * 60)
print()
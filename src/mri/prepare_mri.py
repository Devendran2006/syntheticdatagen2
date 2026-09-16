import os
import shutil
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = "data/imaging/MRI/Training"

OUTPUT_DIR = "data/imaging/MRI/Training_Preprocessed"

PREVIEW_DIR = "outputs/mri/preprocessing_preview"

IMAGE_SIZE = 128

PADDING_RATIO = 0.08

MAX_IMAGES_PER_CLASS_PREVIEW = 4

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PREVIEW_DIR,
    exist_ok=True
)


# ============================================================
# FIND BOUNDING BOX
# ============================================================

def find_brain_bbox(image):

    array = np.array(image)

    # Small threshold to remove pure black background
    threshold = max(
        5,
        int(np.percentile(array, 10))
    )

    mask = array > threshold

    rows = np.where(
        np.any(mask, axis=1)
    )[0]

    cols = np.where(
        np.any(mask, axis=0)
    )[0]

    if len(rows) == 0 or len(cols) == 0:

        return (
            0,
            0,
            image.width,
            image.height
        )

    x_min = cols.min()
    x_max = cols.max()

    y_min = rows.min()
    y_max = rows.max()

    width = x_max - x_min
    height = y_max - y_min

    padding = int(
        max(width, height)
        * PADDING_RATIO
    )

    x_min = max(
        0,
        x_min - padding
    )

    y_min = max(
        0,
        y_min - padding
    )

    x_max = min(
        image.width,
        x_max + padding
    )

    y_max = min(
        image.height,
        y_max + padding
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path,
    output_path
):

    image = Image.open(
        image_path
    ).convert("L")

    # Crop unnecessary background
    bbox = find_brain_bbox(
        image
    )

    image = image.crop(
        bbox
    )

    # Make square
    width, height = image.size

    max_dimension = max(
        width,
        height
    )

    padding_left = (
        max_dimension - width
    ) // 2

    padding_top = (
        max_dimension - height
    ) // 2

    padding_right = (
        max_dimension
        - width
        - padding_left
    )

    padding_bottom = (
        max_dimension
        - height
        - padding_top
    )

    image = ImageOps.expand(
        image,
        border=(
            padding_left,
            padding_top,
            padding_right,
            padding_bottom
        ),
        fill=0
    )

    # High-quality resize
    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.LANCZOS
    )

    # Mild contrast enhancement
    image = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    image.save(
        output_path
    )


# ============================================================
# PROCESS DATASET
# ============================================================

print("=" * 70)
print("MRI PREPROCESSING")
print("=" * 70)

print(
    f"Source      : {SOURCE_DIR}"
)

print(
    f"Output      : {OUTPUT_DIR}"
)

print(
    f"Image Size  : {IMAGE_SIZE}x{IMAGE_SIZE}"
)

print("=" * 70)


classes = sorted(
    [
        folder
        for folder in os.listdir(
            SOURCE_DIR
        )
        if os.path.isdir(
            os.path.join(
                SOURCE_DIR,
                folder
            )
        )
    ]
)


print(
    f"\nClasses found: {classes}"
)


preview_images = []


for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_class_dir,
        exist_ok=True
    )


    image_files = sorted(
        [
            filename
            for filename in os.listdir(
                source_class_dir
            )
            if filename.lower().endswith(
                VALID_EXTENSIONS
            )
        ]
    )


    print(
        f"\n{class_name}: "
        f"{len(image_files)} images"
    )


    for index, filename in enumerate(
        image_files
    ):

        source_path = os.path.join(
            source_class_dir,
            filename
        )

        output_filename = (
            f"{index:05d}.png"
        )

        output_path = os.path.join(
            output_class_dir,
            output_filename
        )


        try:

            preprocess_image(
                source_path,
                output_path
            )


            if (
                len(
                    [
                        x
                        for x in preview_images
                        if x["class"] == class_name
                    ]
                )
                < MAX_IMAGES_PER_CLASS_PREVIEW
            ):

                preview_images.append({
                    "class": class_name,
                    "path": output_path
                })


        except Exception as error:

            print(
                f"Error: {source_path}"
            )

            print(
                error
            )


# ============================================================
# CREATE PREVIEW
# ============================================================

print(
    "\nCreating preprocessing preview..."
)


fig, axes = plt.subplots(
    len(classes),
    MAX_IMAGES_PER_CLASS_PREVIEW,
    figsize=(10, 10)
)


if len(classes) == 1:

    axes = np.expand_dims(
        axes,
        axis=0
    )


for row, class_name in enumerate(
    classes
):

    class_images = [
        item["path"]
        for item in preview_images
        if item["class"] == class_name
    ]


    for column in range(
        MAX_IMAGES_PER_CLASS_PREVIEW
    ):

        ax = axes[
            row,
            column
        ]

        ax.axis("off")


        if column < len(
            class_images
        ):

            image = Image.open(
                class_images[column]
            )

            ax.imshow(
                image,
                cmap="gray"
            )

            if column == 0:

                ax.set_title(
                    class_name
                )


plt.tight_layout()


preview_path = os.path.join(
    PREVIEW_DIR,
    "preprocessed_mri_preview.png"
)


plt.savefig(
    preview_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Output dataset : {OUTPUT_DIR}"
)

print(
    f"Preview        : {preview_path}"
)

print("\nProcessed classes:")

for class_name in classes:

    class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    count = len(
        [
            x
            for x in os.listdir(
                class_dir
            )
            if x.endswith(".png")
        ]
    )

    print(
        f"  {class_name}: {count}"
    )

print("=" * 70)
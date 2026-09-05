from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_DIR = Path(
    r"D:\AgriMind\agents\disease_agent\data\raw\tomato_5_classes"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

def inspect_dataset():
    print("=" * 70)
    print("AgriMind - Tomato Disease Dataset Inspection")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset directory not found.")
        print(DATASET_DIR)
        return

    total_images = 0
    corrupted_images = []
    image_sizes = Counter()
    formats = Counter()

    class_counts = {}

    

    class_directories = sorted(
        [
            directory
            for directory in DATASET_DIR.iterdir()
            if directory.is_dir()
        ]
    )

    print(f"\nClasses found: {len(class_directories)}")

    for class_dir in class_directories:

        class_name = class_dir.name

        image_files = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[class_name] = len(image_files)

        print(
            f"{class_name:<40} {len(image_files)} images"
        )

        # ----------------------------------------------------
        # Inspect individual images
        # ----------------------------------------------------

        for image_file in image_files:

            total_images += 1

            formats[image_file.suffix.lower()] += 1

            try:

                with Image.open(image_file) as image:

                    # Force PIL to actually read the image
                    image.verify()

                # Open again after verify
                with Image.open(image_file) as image:

                    image_sizes[image.size] += 1

            except Exception:

                corrupted_images.append(
                    str(image_file)
                )

    # ========================================================
    # Results
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTotal images: {total_images}")

    print(f"Total classes: {len(class_directories)}")

    print(
        f"Corrupted images: {len(corrupted_images)}"
    )

    # --------------------------------------------------------
    # Formats
    # --------------------------------------------------------

    print("\nImage formats:")

    for image_format, count in formats.items():

        print(
            f"  {image_format}: {count}"
        )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print("\nMost common image dimensions:")

    for size, count in image_sizes.most_common(10):

        print(
            f"  {size}: {count} images"
        )

    # --------------------------------------------------------
    # Corrupted files
    # --------------------------------------------------------

    if corrupted_images:

        print("\nCorrupted images:")

        for image in corrupted_images[:20]:

            print(f"  {image}")

        if len(corrupted_images) > 20:

            print(
                f"\n... and "
                f"{len(corrupted_images) - 20} more."
            )

    else:

        print("\nNo corrupted images found. ✅")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_name, count in class_counts.items():

        percentage = (
            count / total_images
        ) * 100

        print(
            f"{class_name:<40} "
            f"{count:>5} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    inspect_dataset()
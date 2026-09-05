from pathlib import Path
import zipfile
import shutil


# ============================================================
# Configuration
# ============================================================

ZIP_PATH = Path(
    r"C:\Users\RISHAV\Downloads\archive (2).zip"
)

TARGET_DIR = Path(
    r"D:\AgriMind\agents\disease_agent\data\raw\tomato_5_classes"
)

SELECTED_CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]


# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# Main function
# ============================================================

def prepare_dataset():

    print("=" * 60)
    print("AgriMind - Tomato Disease Dataset Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check ZIP file
    # --------------------------------------------------------

    if not ZIP_PATH.exists():

        print("\nERROR: ZIP file not found!")
        print(f"Expected location:\n{ZIP_PATH}")

        return

    print(f"\nZIP file found:")
    print(ZIP_PATH)

    # --------------------------------------------------------
    # Create target directory
    # --------------------------------------------------------

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nTarget directory:")
    print(TARGET_DIR)

    # --------------------------------------------------------
    # Open ZIP
    # --------------------------------------------------------

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

        all_files = zip_file.namelist()

        print(f"\nTotal files inside ZIP: {len(all_files)}")

        total_images = 0

        # ----------------------------------------------------
        # Process each selected class
        # ----------------------------------------------------

        for class_name in SELECTED_CLASSES:

            print("\n" + "-" * 60)
            print(f"Processing: {class_name}")
            print("-" * 60)

            class_files = []

            # Find files belonging to this class
            for file_name in all_files:

                path = Path(file_name)

                # Check if this file belongs to the class
                if class_name in path.parts:

                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        class_files.append(file_name)

            if not class_files:

                print("WARNING: No images found for this class.")
                continue

            # Create class directory
            class_target_dir = TARGET_DIR / class_name

            class_target_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Extract images
            # ------------------------------------------------

            extracted_count = 0

            for file_name in class_files:

                source_path = Path(file_name)

                # Original filename
                image_name = source_path.name

                destination = (
                    class_target_dir / image_name
                )

                # Avoid overwriting duplicate filenames
                if destination.exists():

                    stem = destination.stem
                    suffix = destination.suffix

                    counter = 1

                    while destination.exists():

                        destination = (
                            class_target_dir
                            / f"{stem}_{counter}{suffix}"
                        )

                        counter += 1

                # Extract file directly from ZIP
                with zip_file.open(file_name) as source:

                    with open(destination, "wb") as target:

                        shutil.copyfileobj(
                            source,
                            target
                        )

                extracted_count += 1

            print(
                f"Images extracted: {extracted_count}"
            )

            total_images += extracted_count

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nDataset location:")
    print(TARGET_DIR)

    print(f"\nNumber of classes: {len(SELECTED_CLASSES)}")

    print(f"Total images extracted: {total_images}")

    print("\nClass distribution:")

    for class_name in SELECTED_CLASSES:

        class_dir = TARGET_DIR / class_name

        if class_dir.exists():

            image_count = len([
                file
                for file in class_dir.iterdir()
                if file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ])

            print(
                f"{class_name:<45} {image_count}"
            )

    print("\nDone!")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    prepare_dataset()
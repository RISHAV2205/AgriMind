from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    r"D:\AgriMind\agents\disease_agent\data\raw\tomato_5_classes"
)

OUTPUT_DIR = Path(
    r"D:\AgriMind\agents\disease_agent\data\processed"
)

OUTPUT_FILE = OUTPUT_DIR / "dataset_split.csv"


# ============================================================
# Classes and labels
# ============================================================

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]

CLASS_TO_LABEL = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Collect images
# ============================================================

def collect_images():

    records = []

    print("=" * 70)
    print("AgriMind - Creating Stratified Dataset Split")
    print("=" * 70)

    print("\nCollecting images...")

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():

            print(
                f"ERROR: Class directory not found: "
                f"{class_dir}"
            )

            continue

        label = CLASS_TO_LABEL[class_name]

        image_files = sorted([
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ])

        print(
            f"{class_name:<40} "
            f"{len(image_files)} images"
        )

        for image_file in image_files:

            records.append({
                "image_path": str(
                    image_file.relative_to(DATASET_DIR)
                ),
                "class_name": class_name,
                "label": label
            })

    return pd.DataFrame(records)


# ============================================================
# Create stratified split
# ============================================================

def create_split(df):

    print("\nCreating stratified split...")

    # --------------------------------------------------------
    # First split:
    #
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Temporary 30%
    #
    # 50% → Validation = 15% total
    # 50% → Test       = 15% total
    # --------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Add split column
    # --------------------------------------------------------

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ],
        ignore_index=True
    )

    # Shuffle final dataframe
    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    return final_df


# ============================================================
# Display split statistics
# ============================================================

def display_statistics(df):

    print("\n")
    print("=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Overall split
    # --------------------------------------------------------

    print("\nOverall distribution:")

    split_counts = df["split"].value_counts()

    for split in ["train", "val", "test"]:

        count = split_counts.get(split, 0)

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"{split:<10} "
            f"{count:>6} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Per-class distribution
    # --------------------------------------------------------

    print("\nPer-class distribution:")

    distribution = pd.crosstab(
        df["class_name"],
        df["split"]
    )

    # Make sure columns appear in desired order
    distribution = distribution.reindex(
        columns=["train", "val", "test"],
        fill_value=0
    )

    print(distribution)

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    print("\nPer-class percentages:")

    for class_name in CLASS_NAMES:

        class_df = df[
            df["class_name"] == class_name
        ]

        total = len(class_df)

        train_count = len(
            class_df[class_df["split"] == "train"]
        )

        val_count = len(
            class_df[class_df["split"] == "val"]
        )

        test_count = len(
            class_df[class_df["split"] == "test"]
        )

        print(
            f"\n{class_name}"
        )

        print(
            f"  Train: "
            f"{train_count} "
            f"({train_count / total * 100:.2f}%)"
        )

        print(
            f"  Val:   "
            f"{val_count} "
            f"({val_count / total * 100:.2f}%)"
        )

        print(
            f"  Test:  "
            f"{test_count} "
            f"({test_count / total * 100:.2f}%)"
        )


# ============================================================
# Verify no duplicate images across splits
# ============================================================

def verify_no_overlap(df):

    print("\n")
    print("=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)

    train_paths = set(
        df[df["split"] == "train"]["image_path"]
    )

    val_paths = set(
        df[df["split"] == "val"]["image_path"]
    )

    test_paths = set(
        df[df["split"] == "test"]["image_path"]
    )

    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths

    print(
        f"\nTrain ∩ Validation: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(val_test_overlap)}"
    )

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):

        print(
            "\nNo overlap detected. ✅"
        )

    else:

        print(
            "\nWARNING: Data overlap detected!"
        )


# ============================================================
# Main
# ============================================================

def main():

    # Check dataset
    if not DATASET_DIR.exists():

        print(
            f"ERROR: Dataset not found:\n"
            f"{DATASET_DIR}"
        )

        return

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Collect images
    df = collect_images()

    if df.empty:

        print("\nERROR: No images found.")

        return

    print(
        f"\nTotal images collected: "
        f"{len(df)}"
    )

    # Create split
    df = create_split(df)

    # Statistics
    display_statistics(df)

    # Leakage check
    verify_no_overlap(df)

    # Save CSV
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("DATASET SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"\nCSV saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nColumns:"
    )

    print(
        "  image_path"
    )

    print(
        "  class_name"
    )

    print(
        "  label"
    )

    print(
        "  split"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
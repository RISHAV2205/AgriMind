
"""
AgriMind - Disease Classification Agent

Diagnostic:
    Check train/validation/test split for exact and near-duplicate image leakage.

Dataset structure:
    agents/
        disease_agent/
            data/
                raw/
                    tomato_5_classes/
                        Tomato_Early_blight/
                        Tomato_healthy/
                        Tomato_Late_blight/
                        Tomato_Leaf_Mold/
                        Tomato_Septoria_leaf_spot/

CSV:
    The CSV contains relative image paths such as:

        Tomato_Early_blight/xxxxx.JPG

    The paths are resolved relative to DATASET_ROOT.

Checks:
    1. Exact duplicates using MD5.
    2. Near duplicates using perceptual hash (pHash).

Important:
    Missing files are treated as an error.
    We do NOT silently skip missing images because that could make
    the leakage analysis incomplete.

Usage from D:\\AgriMind:

    python -m agents.disease_agent.src.check_split_leakage \
        --csv agents/disease_agent/data/processed/dataset_split.csv
"""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Your actual dataset root
DATASET_ROOT = Path(
    "agents/disease_agent/data/raw/tomato_5_classes"
)

# CSV column names
COL_PATH = "image_path"
COL_LABEL = "label"
COL_SPLIT = "split"

# Split values
TRAIN_VALUE = "train"
VAL_VALUE = "val"
TEST_VALUE = "test"

# pHash Hamming distance threshold
#
# 0 = identical perceptual hash
# Lower values = stronger visual similarity
#
# We use 5 as a strict initial screening threshold.
PHASH_DISTANCE_THRESHOLD = 5

# Number of examples printed for manual inspection
MAX_EXAMPLES_PRINTED = 8


# ---------------------------------------------------------------------------
# PATH HANDLING
# ---------------------------------------------------------------------------

def resolve_image_path(relative_path: str) -> Path:
    """
    Convert the relative image path stored in the CSV into
    the actual path inside the AgriMind dataset.

    Example:

        CSV:
        Tomato_Early_blight/image.JPG

        becomes:

        agents/disease_agent/data/raw/tomato_5_classes/
        Tomato_Early_blight/image.JPG
    """

    path = Path(relative_path)

    # If CSV accidentally contains an absolute path,
    # keep it unchanged.
    if path.is_absolute():
        return path

    return DATASET_ROOT / path


# ---------------------------------------------------------------------------
# HASHING
# ---------------------------------------------------------------------------

def md5_of_file(path: Path) -> str:
    """
    Calculate MD5 hash of the complete file.

    Same file content -> same MD5.
    """

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def phash_of_image(path: Path):
    """
    Calculate perceptual hash.

    Unlike MD5, pHash can identify visually similar images even
    when the files are not byte-for-byte identical.
    """

    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def compute_hashes(
    paths: List[Path],
) -> Tuple[
    Dict[Path, str],
    Dict[Path, "imagehash.ImageHash"],
]:
    """
    Calculate MD5 and pHash for every image.

    Missing or unreadable images are treated as errors.
    """

    md5_map: Dict[Path, str] = {}
    phash_map: Dict[Path, imagehash.ImageHash] = {}

    total = len(paths)

    for i, path in enumerate(paths, start=1):

        # ---------------------------------------------------------------
        # Check that file exists
        # ---------------------------------------------------------------

        if not path.exists():
            raise FileNotFoundError(
                f"\nMissing image:\n{path}\n\n"
                "The leakage check has been stopped because the dataset "
                "paths do not match the files on disk."
            )

        try:
            md5_map[path] = md5_of_file(path)
            phash_map[path] = phash_of_image(path)

        except Exception as e:
            raise RuntimeError(
                f"\nCould not process image:\n{path}\n"
                f"Error: {e}"
            ) from e

        # Progress
        if i % 500 == 0 or i == total:
            print(
                f"  Hashed {i}/{total} images..."
            )

    return md5_map, phash_map


# ---------------------------------------------------------------------------
# EXACT DUPLICATE CHECK
# ---------------------------------------------------------------------------

def find_exact_duplicates(
    train_md5: Dict[Path, str],
    other_md5: Dict[Path, str],
) -> List[Tuple[Path, Path]]:
    """
    Find images in another split that are byte-for-byte identical
    to an image in the training split.

    Returns:

        (other_split_image, matching_train_image)
    """

    train_by_hash: Dict[str, Path] = {}

    for path, h in train_md5.items():
        train_by_hash.setdefault(h, path)

    leaks = []

    for path, h in other_md5.items():

        if h in train_by_hash:
            leaks.append(
                (path, train_by_hash[h])
            )

    return leaks


# ---------------------------------------------------------------------------
# NEAR DUPLICATE CHECK
# ---------------------------------------------------------------------------

def find_near_duplicates(
    train_phash: Dict[Path, "imagehash.ImageHash"],
    other_phash: Dict[Path, "imagehash.ImageHash"],
    threshold: int,
) -> List[Tuple[Path, Path, int]]:
    """
    Find visually near-identical images between another split
    and the training split.

    pHash distance:

        0 -> extremely similar / same visual hash
        small value -> highly similar
        larger value -> less similar

    Returns:

        (other_image, matching_train_image, distance)

    """

    train_items = list(train_phash.items())

    leaks = []

    for other_path, other_hash in other_phash.items():

        best_train_path = None
        best_distance = None

        for train_path, train_hash in train_items:

            distance = other_hash - train_hash

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_train_path = train_path

        if (
            best_distance is not None
            and best_distance <= threshold
        ):
            leaks.append(
                (
                    other_path,
                    best_train_path,
                    best_distance,
                )
            )

    return leaks


# ---------------------------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------------------------

def load_split_paths(
    csv_path: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Load the split CSV and validate required columns.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Validate columns
    for col in (
        COL_PATH,
        COL_LABEL,
        COL_SPLIT,
    ):
        if col not in df.columns:
            raise ValueError(
                f"\nExpected column '{col}' not found.\n"
                f"Columns present: {list(df.columns)}\n\n"
                "Update COL_PATH / COL_LABEL / COL_SPLIT "
                "at the top of this script."
            )

    return {
        "train": df[
            df[COL_SPLIT] == TRAIN_VALUE
        ],

        "val": df[
            df[COL_SPLIT] == VAL_VALUE
        ],

        "test": df[
            df[COL_SPLIT] == TEST_VALUE
        ],
    }


# ---------------------------------------------------------------------------
# VALIDATE ALL PATHS BEFORE HASHING
# ---------------------------------------------------------------------------

def validate_paths(
    paths: List[Path],
    split_name: str,
) -> None:
    """
    Make sure every image referenced by the CSV exists.

    We stop before hashing if even one image is missing.
    """

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:

        print(
            f"\nERROR: {len(missing)} "
            f"{split_name} image(s) are missing."
        )

        print("\nFirst missing images:")

        for path in missing[:10]:
            print(f"  {path}")

        raise FileNotFoundError(
            "\nDataset path validation failed.\n"
            "Fix the image paths before running the leakage check."
        )


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def report_leaks(
    split_name: str,
    exact_leaks,
    near_leaks,
    other_total: int,
) -> None:

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} vs TRAIN")
    print("=" * 65)

    print(
        f"Images checked:                 {other_total}"
    )

    exact_percentage = (
        100 * len(exact_leaks)
        / max(other_total, 1)
    )

    near_percentage = (
        100 * len(near_leaks)
        / max(other_total, 1)
    )

    print(
        f"Exact duplicates:               "
        f"{len(exact_leaks)} "
        f"({exact_percentage:.2f}%)"
    )

    print(
        f"Near duplicates (pHash <= "
        f"{PHASH_DISTANCE_THRESHOLD}): "
        f"{len(near_leaks)} "
        f"({near_percentage:.2f}%)"
    )

    # ---------------------------------------------------------------
    # Exact duplicate examples
    # ---------------------------------------------------------------

    if exact_leaks:

        print(
            "\nExample exact duplicates:"
        )

        for (
            other_path,
            train_path,
        ) in exact_leaks[
            :MAX_EXAMPLES_PRINTED
        ]:

            print(
                f"  {other_path}"
                f"\n      == "
                f"{train_path}"
            )

    # ---------------------------------------------------------------
    # Near duplicate examples
    # ---------------------------------------------------------------

    if near_leaks:

        print(
            "\nExample near duplicates:"
        )

        sorted_near = sorted(
            near_leaks,
            key=lambda x: x[2],
        )

        for (
            other_path,
            train_path,
            distance,
        ) in sorted_near[
            :MAX_EXAMPLES_PRINTED
        ]:

            print(
                f"  {other_path}"
                f"\n      ~~ "
                f"{train_path}"
                f"\n      pHash distance = "
                f"{distance}"
            )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Check AgriMind train/val/test image "
            "splits for exact and near-duplicate leakage."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Path to dataset_split.csv "
            "(image_path, label, split)"
        ),
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    # ---------------------------------------------------------------
    # Load CSV
    # ---------------------------------------------------------------

    splits = load_split_paths(csv_path)

    # ---------------------------------------------------------------
    # Resolve actual dataset paths
    # ---------------------------------------------------------------

    train_paths = [
        resolve_image_path(p)
        for p in splits["train"][COL_PATH].tolist()
    ]

    val_paths = [
        resolve_image_path(p)
        for p in splits["val"][COL_PATH].tolist()
    ]

    test_paths = [
        resolve_image_path(p)
        for p in splits["test"][COL_PATH].tolist()
    ]

    # ---------------------------------------------------------------
    # Dataset summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 65)
    print("AGRIMIND DATASET LEAKAGE CHECK")
    print("=" * 65)

    print(
        f"\nDataset root:\n"
        f"  {DATASET_ROOT.resolve()}"
    )

    print(
        f"\nCSV:\n"
        f"  {csv_path.resolve()}"
    )

    print(
        f"\nSplit sizes:"
        f"\n  Train: {len(train_paths)}"
        f"\n  Val:   {len(val_paths)}"
        f"\n  Test:  {len(test_paths)}"
        f"\n  Total: "
        f"{len(train_paths) + len(val_paths) + len(test_paths)}"
    )

    # ---------------------------------------------------------------
    # Validate paths BEFORE hashing
    # ---------------------------------------------------------------

    print("\nValidating image paths...")

    validate_paths(
        train_paths,
        "train",
    )

    validate_paths(
        val_paths,
        "val",
    )

    validate_paths(
        test_paths,
        "test",
    )

    print(
        "\n✓ All CSV image paths exist."
    )

    # ---------------------------------------------------------------
    # Hash train
    # ---------------------------------------------------------------

    print("\nHashing TRAIN split...")

    train_md5, train_phash = compute_hashes(
        train_paths
    )

    # ---------------------------------------------------------------
    # Hash validation
    # ---------------------------------------------------------------

    print("\nHashing VAL split...")

    val_md5, val_phash = compute_hashes(
        val_paths
    )

    # ---------------------------------------------------------------
    # Hash test
    # ---------------------------------------------------------------

    print("\nHashing TEST split...")

    test_md5, test_phash = compute_hashes(
        test_paths
    )

    # ---------------------------------------------------------------
    # VAL vs TRAIN
    # ---------------------------------------------------------------

    val_exact = find_exact_duplicates(
        train_md5,
        val_md5,
    )

    val_near = find_near_duplicates(
        train_phash,
        val_phash,
        PHASH_DISTANCE_THRESHOLD,
    )

    report_leaks(
        "val",
        val_exact,
        val_near,
        len(val_paths),
    )

    # ---------------------------------------------------------------
    # TEST vs TRAIN
    # ---------------------------------------------------------------

    test_exact = find_exact_duplicates(
        train_md5,
        test_md5,
    )

    test_near = find_near_duplicates(
        train_phash,
        test_phash,
        PHASH_DISTANCE_THRESHOLD,
    )

    report_leaks(
        "test",
        test_exact,
        test_near,
        len(test_paths),
    )

    # ---------------------------------------------------------------
    # FINAL INTERPRETATION
    # ---------------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL INTERPRETATION")
    print("=" * 65)

    total_exact = (
        len(val_exact)
        + len(test_exact)
    )

    total_near = (
        len(val_near)
        + len(test_near)
    )

    if total_exact == 0 and total_near == 0:

        print(
            "\n✓ No exact or pHash near-duplicate "
            "matches were found between TRAIN and "
            "VAL/TEST."
        )

        print(
            "\nThis provides evidence against "
            "image-content leakage in the current split."
        )

        print(
            "\nHowever, this does NOT prove that the "
            "dataset is completely independent."
        )

        print(
            "PlantVillage can still contain images "
            "with strong visual similarity or source-level "
            "relationships that pHash does not detect."
        )

    else:

        print(
            f"\n⚠ Potential leakage detected."
        )

        print(
            f"Exact duplicate matches: {total_exact}"
        )

        print(
            f"Near-duplicate matches:   {total_near}"
        )

        print(
            "\nInspect the reported image pairs."
        )

        print(
            "\nIf the same physical leaf appears in "
            "different splits, the dataset should be "
            "re-split so related images remain in the "
            "same split."
        )

    print("\n" + "=" * 65)
    print("CHECK COMPLETE")
    print("=" * 65)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()


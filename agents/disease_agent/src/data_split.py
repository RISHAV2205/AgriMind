from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("agents/disease_agent/data/raw/tomato_5_classes")
OUT = "agents/disease_agent/data/processed/dataset_split_leakage_free.csv"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


# Collect images
data = []

for class_dir in RAW_DIR.iterdir():
    if class_dir.is_dir():
        for img in class_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                data.append({
                    "image_path": str(img.relative_to(RAW_DIR)),
                    "label": class_dir.name,
                    "group": md5(img)
                })

df = pd.DataFrame(data)

# One group = all exact duplicate files
groups = df.groupby("group").first().reset_index()

# 70% train, 30% temporary
train_groups, temp_groups = train_test_split(
    groups,
    test_size=0.30,
    stratify=groups["label"],
    random_state=42
)

# 15% validation, 15% test
val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    stratify=temp_groups["label"],
    random_state=42
)

# Assign splits
split_map = {}

for g in train_groups["group"]:
    split_map[g] = "train"

for g in val_groups["group"]:
    split_map[g] = "val"

for g in test_groups["group"]:
    split_map[g] = "test"

df["split"] = df["group"].map(split_map)

# Save
df[["image_path", "label", "split"]].to_csv(OUT, index=False)

print(df["split"].value_counts())
print(f"\nSaved to: {OUT}")
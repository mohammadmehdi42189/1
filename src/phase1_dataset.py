from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from src.phase1_config import IMAGE_EXTENSIONS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED


def scan_image_folder(data_dir):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    rows = []
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        raise ValueError(
            "No class folders found. Expected structure: dataset/ClassName/image.jpg"
        )

    for class_dir in class_dirs:
        label = class_dir.name
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": str(image_path.as_posix()),
                        "label": label,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No image files found in the dataset folder.")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def get_class_distribution(df):
    distribution = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    return distribution


def split_dataset(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    labels = df["label"]
    can_stratify = labels.value_counts().min() >= 2

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=SEED,
        shuffle=True,
        stratify=labels if can_stratify else None,
    )

    temp_ratio = val_ratio + test_ratio
    val_size_inside_temp = val_ratio / temp_ratio

    temp_labels = temp_df["label"]
    can_stratify_temp = temp_labels.value_counts().min() >= 2

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_inside_temp,
        random_state=SEED,
        shuffle=True,
        stratify=temp_labels if can_stratify_temp else None,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def load_split_csv(path):
    return pd.read_csv(path)


def sample_images_by_class(df, max_per_class=1):
    samples = []
    rng = random.Random(SEED)

    for label in sorted(df["label"].unique()):
        class_paths = df[df["label"] == label]["path"].tolist()
        rng.shuffle(class_paths)
        for path in class_paths[:max_per_class]:
            samples.append({"label": label, "path": path})

    return pd.DataFrame(samples)

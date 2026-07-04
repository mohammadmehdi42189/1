from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image

from src.phase1_dataset import get_class_distribution, sample_images_by_class


def plot_class_distribution(df, save_path=None):
    distribution = get_class_distribution(df)

    fig, ax = plt.subplots(figsize=(max(8, len(distribution) * 0.6), 5))
    ax.bar(distribution["label"], distribution["count"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, distribution


def plot_sample_grid(df, save_path=None, max_per_class=1):
    samples = sample_images_by_class(df, max_per_class=max_per_class)
    n = len(samples)

    if n == 0:
        raise ValueError("No samples available for visualization.")

    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax, (_, row) in zip(axes, samples.iterrows()):
        image = Image.open(row["path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(row["label"])
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Random Sample from Each Class", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes, samples

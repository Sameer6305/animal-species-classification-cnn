import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config

OUT_DIR = config.PROJECT_ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame):
    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(counts.sum(axis=1).sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 20))
    counts.plot(kind="barh", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("Number of images")
    ax.set_title("Animal-80 Class Distribution (train / val / test)")
    ax.legend(title="Split")
    plt.tight_layout()
    out_path = OUT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_sample_grid(df: pd.DataFrame, n_classes=16, n_per_class=1):
    classes = sorted(df["label"].unique())
    chosen = random.sample(classes, n_classes)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, class_name in zip(axes.flatten(), chosen):
        candidates = df[
            (df["label"] == class_name)
            & (df["split"] == "train")
            & (~df["filepath"].str.contains("aug_"))   # exclude augmented copies
        ]
        row = candidates.sample(1).iloc[0]
        img_path = config.PROJECT_ROOT / row["filepath"]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(class_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "sample_crops_grid.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_augmentation_examples(df: pd.DataFrame, n_examples=4):
    aug_rows = df[df["filepath"].str.contains("aug_")].sample(n_examples, random_state=1)

    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 8))
    for i, (_, row) in enumerate(aug_rows.iterrows()):
        aug_path = config.PROJECT_ROOT / row["filepath"]

        # Match the EXACT original crop this augmented image was derived from,
        # via the shared source_image field (not a random same-class image).
        original_match = df[
            (df["label"] == row["label"])
            & (df["source_image"] == row["source_image"])
            & (~df["filepath"].str.contains("aug_"))
        ]

        if original_match.empty:
            continue  # skip if we can't find the true source (shouldn't normally happen)

        orig_row = original_match.iloc[0]
        orig_path = config.PROJECT_ROOT / orig_row["filepath"]

        axes[0, i].imshow(Image.open(orig_path))
        axes[0, i].set_title(f"{row['label']} (original)", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(Image.open(aug_path))
        axes[1, i].set_title(f"{row['label']} (augmented)", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "augmentation_examples.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    df = pd.read_csv(config.MANIFEST_PATH)
    plot_class_distribution(df)
    plot_sample_grid(df)
    plot_augmentation_examples(df)


if __name__ == "__main__":
    main()
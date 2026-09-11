import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config

VALID_EXT = {".jpg", ".jpeg", ".png"}


def compute_raw_image_dimensions():
    """Sample raw image dimensions before resizing, across train+test."""
    widths, heights = [], []
    sample_count = 0

    for split_dir in [config.TRAIN_DIR, config.TEST_DIR]:
        for class_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXT]
            for img_path in images[:20]:  # sample per class, not exhaustive — for speed
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        sample_count += 1
                except Exception:
                    continue

    stats = pd.DataFrame({"width": widths, "height": heights})
    print(f"Sampled {sample_count} raw images")
    print(stats.describe())
    stats.to_csv(config.PROJECT_ROOT / "data" / "raw_image_dimension_stats.csv", index=False)
    return stats


def report_cleaning_summary():
    """Report how many source images/annotations were skipped during crop_and_resize."""
    skipped_no_annotation = 0
    skipped_unreadable = 0
    total_source_images = 0

    for split_dir in [config.TRAIN_DIR, config.TEST_DIR]:
        for class_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXT]
            for img_path in images:
                total_source_images += 1
                ann_path = img_path.parent / "Label" / (img_path.stem + ".txt")
                if not ann_path.exists():
                    skipped_no_annotation += 1
                    continue
                try:
                    Image.open(img_path).convert("RGB")
                except Exception:
                    skipped_unreadable += 1

    print(f"\nTotal source images scanned: {total_source_images}")
    print(f"Skipped (no annotation file): {skipped_no_annotation}")
    print(f"Skipped (unreadable/corrupt): {skipped_unreadable}")
    print(f"Usable images: {total_source_images - skipped_no_annotation - skipped_unreadable}")


def compute_normalization_stats(sample_size=2000):
    """Compute actual per-channel mean/std from processed training crops."""
    df = pd.read_csv(config.MANIFEST_PATH)
    train_df = df[df["split"] == "train"].sample(min(sample_size, len(df)), random_state=42)

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0

    for _, row in train_df.iterrows():
        img_path = config.PROJECT_ROOT / row["filepath"]
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        n_pixels += img.shape[0] * img.shape[1]

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
    print(f"Dataset mean (RGB): {mean}")
    print(f"Dataset std (RGB): {std}")
    return mean, std


if __name__ == "__main__":
    compute_raw_image_dimensions()
    report_cleaning_summary()
    compute_normalization_stats()
import csv
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config
from src.preprocessing.parse_annotations import parse_annotation_file, get_annotation_path_for_image

VALID_EXT = {".jpg", ".jpeg", ".png"}


def process_split(split_dir: Path, split_name: str, writer):
    for class_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
        class_name = class_dir.name
        out_class_dir = config.PROCESSED_DIR / split_name / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXT]

        for img_path in image_paths:
            ann_path = get_annotation_path_for_image(img_path)
            if not ann_path.exists():
                continue

            boxes = parse_annotation_file(ann_path)
            if not boxes:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [warn] could not open {img_path}: {e}")
                continue

            w, h = img.size
            for i, (box_class, xmin, ymin, xmax, ymax) in enumerate(boxes):
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)
                if xmax <= xmin or ymax <= ymin:
                    continue

                crop = img.crop((xmin, ymin, xmax, ymax)).resize(
                    (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
                )
                out_path = out_class_dir / f"{img_path.stem}_{i}.jpg"
                crop.save(out_path, quality=95)

                writer.writerow({
                    "filepath": str(out_path.relative_to(config.PROJECT_ROOT)),
                    "label": class_name,
                    "split": split_name,
                    "source_image": str(img_path.relative_to(config.PROJECT_ROOT)),
                })

        print(f"[{split_name}] done: {class_name} ({len(image_paths)} source images)")


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_manifest_path = config.PROJECT_ROOT / "data" / "processed_manifest_raw.csv"

    with open(raw_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split", "source_image"])
        writer.writeheader()
        process_split(config.TRAIN_DIR, "train", writer)
        process_split(config.TEST_DIR, "test", writer)

    print(f"\nDone. Raw manifest written to {raw_manifest_path}")


if __name__ == "__main__":
    main()
"""
Parses Open-Images-style Label .txt files.
Each line: <ClassName> <xmin> <ymin> <xmax> <ymax>  (absolute pixel coords)
A file can have multiple lines if multiple animals appear in one image.
"""
from pathlib import Path
from typing import List, Tuple

BBox = Tuple[str, float, float, float, float]


def parse_annotation_file(txt_path: Path) -> List[BBox]:
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            *class_tokens, xmin, ymin, xmax, ymax = parts  # class name may have spaces
            class_name = " ".join(class_tokens)
            boxes.append((class_name, float(xmin), float(ymin), float(xmax), float(ymax)))
    return boxes


def get_annotation_path_for_image(image_path: Path) -> Path:
    # .../<split>/<Class>/<image>.jpg -> .../<split>/<Class>/Label/<image>.txt
    return image_path.parent / "Label" / (image_path.stem + ".txt")
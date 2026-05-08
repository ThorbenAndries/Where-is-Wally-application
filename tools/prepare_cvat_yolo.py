#!/usr/bin/env python3
"""Prepare a CVAT YOLO export for Ultralytics YOLO training."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_class_names(source: Path, fallback: str) -> list[str]:
    for name_file in ("obj.names", "classes.txt"):
        matches = list(source.rglob(name_file))
        if matches:
            names = [line.strip() for line in matches[0].read_text().splitlines() if line.strip()]
            if names:
                return names
    return [fallback]


def find_labeled_images(source: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in source.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))

    return sorted(pairs)


def reset_output(output: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            split_dir = output / kind / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for item in split_dir.iterdir():
                if item.is_file():
                    item.unlink()


def copy_split(items: list[tuple[Path, Path]], output: Path, split: str) -> None:
    for image_path, label_path in items:
        shutil.copy2(image_path, output / "images" / split / image_path.name)
        shutil.copy2(label_path, output / "labels" / split / label_path.name)


def write_data_yaml(output: Path, class_names: list[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    (output / "data.yaml").write_text(yaml)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CVAT YOLO export folder into an Ultralytics dataset."
    )
    parser.add_argument("source", type=Path, help="Folder containing the extracted CVAT YOLO export.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wally-yolo"),
        help="Output dataset folder. Default: data/wally-yolo",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    parser.add_argument(
        "--fallback-class",
        default="wally",
        help="Class name to use when the CVAT export has no obj.names/classes.txt.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source folder not found: {args.source}")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")

    pairs = find_labeled_images(args.source)
    if not pairs:
        raise SystemExit(
            "No image/label pairs found. Export from CVAT as YOLO and extract the zip first."
        )

    class_names = read_class_names(args.source, args.fallback_class)

    random.seed(args.seed)
    random.shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * args.val_ratio))) if len(pairs) > 1 else 0
    val_items = pairs[:val_count]
    train_items = pairs[val_count:]

    reset_output(args.output)
    copy_split(train_items, args.output, "train")
    copy_split(val_items, args.output, "val")
    write_data_yaml(args.output, class_names)

    print(f"Prepared {len(train_items)} train and {len(val_items)} val images.")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Dataset YAML: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()

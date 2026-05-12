#!/usr/bin/env python3
"""Generate a synthetic hidden-mascot object-detection dataset for YOLO."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


CANVAS_SIZE = 640
CLASS_NAME = "mascot"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DIFFICULTY_SETTINGS = {
    "easy": {
        "mascot_size": (46, 82),
        "shapes": (90, 150),
        "people": (8, 18),
        "lookalikes": (2, 5),
        "occluders": (4, 10),
        "blur_chance": 0.25,
        "rotation": (-25, 25),
        "scale_jitter": (0.85, 1.2),
        "perspective_chance": 0.25,
        "noise_chance": 0.25,
        "brightness": (0.82, 1.18),
    },
    "medium": {
        "mascot_size": (32, 62),
        "shapes": (170, 260),
        "people": (25, 45),
        "lookalikes": (8, 16),
        "occluders": (12, 24),
        "blur_chance": 0.35,
        "rotation": (-38, 38),
        "scale_jitter": (0.75, 1.35),
        "perspective_chance": 0.38,
        "noise_chance": 0.38,
        "brightness": (0.72, 1.28),
    },
    "hard": {
        "mascot_size": (24, 46),
        "shapes": (260, 390),
        "people": (55, 95),
        "lookalikes": (18, 34),
        "occluders": (22, 42),
        "blur_chance": 0.45,
        "rotation": (-55, 55),
        "scale_jitter": (0.62, 1.55),
        "perspective_chance": 0.55,
        "noise_chance": 0.5,
        "brightness": (0.6, 1.42),
    },
}


def make_mascot(size: int) -> Image.Image:
    mascot = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mascot)

    unit = size / 100
    red = (214, 48, 49, 255)
    cream = (255, 235, 205, 255)
    navy = (30, 39, 73, 255)
    white = (255, 255, 255, 255)
    black = (20, 20, 20, 255)

    draw.ellipse((32 * unit, 10 * unit, 68 * unit, 46 * unit), fill=cream, outline=black, width=max(1, int(2 * unit)))
    draw.polygon(
        [(28 * unit, 16 * unit), (72 * unit, 16 * unit), (62 * unit, 2 * unit), (38 * unit, 2 * unit)],
        fill=red,
        outline=black,
    )
    draw.rectangle((26 * unit, 16 * unit, 74 * unit, 23 * unit), fill=white, outline=black)

    draw.ellipse((39 * unit, 25 * unit, 47 * unit, 33 * unit), fill=white, outline=black)
    draw.ellipse((53 * unit, 25 * unit, 61 * unit, 33 * unit), fill=white, outline=black)
    draw.ellipse((42 * unit, 28 * unit, 45 * unit, 31 * unit), fill=black)
    draw.ellipse((56 * unit, 28 * unit, 59 * unit, 31 * unit), fill=black)
    draw.arc((42 * unit, 33 * unit, 58 * unit, 43 * unit), start=10, end=170, fill=black, width=max(1, int(2 * unit)))

    draw.rounded_rectangle((30 * unit, 45 * unit, 70 * unit, 82 * unit), radius=int(8 * unit), fill=red, outline=black)
    for y in (52, 64, 76):
        draw.rectangle((31 * unit, y * unit, 69 * unit, (y + 5) * unit), fill=white)

    draw.line((31 * unit, 52 * unit, 15 * unit, 70 * unit), fill=navy, width=max(2, int(5 * unit)))
    draw.line((69 * unit, 52 * unit, 85 * unit, 70 * unit), fill=navy, width=max(2, int(5 * unit)))
    draw.line((41 * unit, 82 * unit, 34 * unit, 98 * unit), fill=navy, width=max(2, int(6 * unit)))
    draw.line((59 * unit, 82 * unit, 66 * unit, 98 * unit), fill=navy, width=max(2, int(6 * unit)))

    return mascot


def load_custom_mascot(path: Path, size: int) -> Image.Image:
    mascot = Image.open(path).convert("RGBA")
    width, height = mascot.size
    scale = size / max(width, height)
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    return mascot.resize(resized, Image.Resampling.LANCZOS)


def crop_alpha(sprite: Image.Image) -> Image.Image:
    bbox = sprite.getchannel("A").getbbox()
    if bbox is None:
        return sprite
    return sprite.crop(bbox)


def apply_scale(sprite: Image.Image, rng: random.Random, settings: dict[str, object]) -> Image.Image:
    min_scale, max_scale = settings["scale_jitter"]
    scale = rng.uniform(min_scale, max_scale)
    width = max(1, int(sprite.width * scale))
    height = max(1, int(sprite.height * scale))
    return sprite.resize((width, height), Image.Resampling.LANCZOS)


def apply_perspective(sprite: Image.Image, rng: random.Random) -> Image.Image:
    pad = max(6, int(max(sprite.size) * 0.18))
    canvas = Image.new("RGBA", (sprite.width + pad * 2, sprite.height + pad * 2), (0, 0, 0, 0))
    canvas.paste(sprite, (pad, pad), sprite)

    width, height = canvas.size
    max_shift = int(min(width, height) * rng.uniform(0.04, 0.16))
    source_quad = (
        rng.randint(0, max_shift),
        rng.randint(0, max_shift),
        width - rng.randint(0, max_shift),
        rng.randint(0, max_shift),
        width - rng.randint(0, max_shift),
        height - rng.randint(0, max_shift),
        rng.randint(0, max_shift),
        height - rng.randint(0, max_shift),
    )
    warped = canvas.transform(
        canvas.size,
        Image.Transform.QUAD,
        source_quad,
        resample=Image.Resampling.BICUBIC,
    )
    return crop_alpha(warped)


def augment_mascot(sprite: Image.Image, rng: random.Random, settings: dict[str, object]) -> Image.Image:
    sprite = apply_scale(sprite, rng, settings)

    min_angle, max_angle = settings["rotation"]
    if rng.random() < 0.9:
        sprite = sprite.rotate(
            rng.uniform(min_angle, max_angle),
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )

    if rng.random() < settings["perspective_chance"]:
        sprite = apply_perspective(sprite, rng)

    if rng.random() < settings["blur_chance"]:
        sprite = sprite.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.15)))

    return crop_alpha(sprite)


def apply_image_augmentations(image: Image.Image, rng: random.Random, settings: dict[str, object]) -> Image.Image:
    min_brightness, max_brightness = settings["brightness"]
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(min_brightness, max_brightness))

    if rng.random() < 0.65:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.22))

    if rng.random() < settings["blur_chance"] * 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.65)))

    if rng.random() < settings["noise_chance"]:
        pixels = np.asarray(image).astype(np.int16)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
            0,
            rng.uniform(4, 14),
            pixels.shape,
        )
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")

    return image


def list_backgrounds(path: Path | None) -> list[Path]:
    if path is None:
        return []
    return sorted(file_path for file_path in path.rglob("*") if file_path.suffix.lower() in IMAGE_EXTENSIONS)


def make_base_image(rng: random.Random, backgrounds: list[Path]) -> Image.Image:
    if not backgrounds:
        return Image.new(
            "RGB",
            (CANVAS_SIZE, CANVAS_SIZE),
            rng.choice([(232, 236, 239), (246, 244, 239), (230, 238, 230)]),
        )

    background = Image.open(rng.choice(backgrounds)).convert("RGB")
    width, height = background.size
    scale = max(CANVAS_SIZE / width, CANVAS_SIZE / height)
    resized = background.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    left = rng.randint(0, resized.width - CANVAS_SIZE)
    top = rng.randint(0, resized.height - CANVAS_SIZE)
    return resized.crop((left, top, left + CANVAS_SIZE, top + CANVAS_SIZE))


def add_background_noise(draw: ImageDraw.ImageDraw, rng: random.Random, settings: dict[str, object]) -> None:
    palette = [
        (244, 241, 222),
        (129, 178, 154),
        (242, 204, 143),
        (61, 64, 91),
        (224, 122, 95),
        (93, 138, 168),
        (247, 247, 247),
    ]

    min_shapes, max_shapes = settings["shapes"]
    for _ in range(rng.randint(min_shapes, max_shapes)):
        color = rng.choice(palette)
        x = rng.randint(0, CANVAS_SIZE)
        y = rng.randint(0, CANVAS_SIZE)
        w = rng.randint(12, 65)
        h = rng.randint(12, 65)
        if rng.random() < 0.5:
            draw.rectangle((x, y, x + w, y + h), fill=color)
        else:
            draw.ellipse((x, y, x + w, y + h), fill=color)

    for _ in range(rng.randint(min_shapes // 2, max_shapes // 2)):
        color = rng.choice(palette)
        x1 = rng.randint(0, CANVAS_SIZE)
        y1 = rng.randint(0, CANVAS_SIZE)
        angle = rng.random() * math.tau
        length = rng.randint(20, 90)
        x2 = int(x1 + math.cos(angle) * length)
        y2 = int(y1 + math.sin(angle) * length)
        draw.line((x1, y1, x2, y2), fill=color, width=rng.randint(2, 7))


def make_decoy_person(size: int, rng: random.Random, lookalike: bool = False) -> Image.Image:
    person = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(person)
    unit = size / 100

    skin = rng.choice([(250, 218, 185, 255), (224, 172, 105, 255), (141, 85, 36, 255)])
    shirt = rng.choice([(75, 123, 236, 255), (32, 191, 107, 255), (253, 150, 68, 255), (165, 94, 234, 255)])
    dark = (35, 35, 45, 255)

    if lookalike:
        shirt = rng.choice([(214, 48, 49, 255), (240, 70, 70, 255), (250, 250, 250, 255)])
        draw.polygon(
            [(30 * unit, 18 * unit), (70 * unit, 18 * unit), (62 * unit, 5 * unit), (38 * unit, 5 * unit)],
            fill=rng.choice([(214, 48, 49, 255), (255, 255, 255, 255)]),
            outline=dark,
        )

    draw.ellipse((34 * unit, 14 * unit, 66 * unit, 43 * unit), fill=skin, outline=dark)
    draw.rounded_rectangle((31 * unit, 43 * unit, 69 * unit, 78 * unit), radius=int(6 * unit), fill=shirt, outline=dark)

    if lookalike and rng.random() < 0.75:
        for y in (50, 63):
            draw.rectangle((32 * unit, y * unit, 68 * unit, (y + 5) * unit), fill=(255, 255, 255, 255))

    arm_color = rng.choice([(31, 55, 120, 255), (45, 45, 65, 255), shirt])
    draw.line((32 * unit, 50 * unit, 16 * unit, 70 * unit), fill=arm_color, width=max(2, int(5 * unit)))
    draw.line((68 * unit, 50 * unit, 84 * unit, 70 * unit), fill=arm_color, width=max(2, int(5 * unit)))
    draw.line((42 * unit, 78 * unit, 36 * unit, 96 * unit), fill=dark, width=max(2, int(5 * unit)))
    draw.line((58 * unit, 78 * unit, 64 * unit, 96 * unit), fill=dark, width=max(2, int(5 * unit)))

    return person


def paste_sprite(image: Image.Image, sprite: Image.Image, x: int, y: int, rng: random.Random) -> None:
    if rng.random() < 0.65:
        sprite = sprite.rotate(rng.uniform(-22, 22), expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(sprite, (x, y), sprite)


def scatter_people(image: Image.Image, rng: random.Random, settings: dict[str, object]) -> None:
    min_people, max_people = settings["people"]
    min_lookalikes, max_lookalikes = settings["lookalikes"]
    people_count = rng.randint(min_people, max_people)
    lookalike_count = rng.randint(min_lookalikes, max_lookalikes)

    for index in range(people_count):
        size = rng.randint(24, 58)
        sprite = make_decoy_person(size, rng, lookalike=index < lookalike_count)
        x = rng.randint(-10, CANVAS_SIZE - 16)
        y = rng.randint(-10, CANVAS_SIZE - 16)
        paste_sprite(image, sprite, x, y, rng)


def add_occluders(image: Image.Image, rng: random.Random, settings: dict[str, object]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    palette = [
        (245, 203, 92, 235),
        (88, 129, 87, 235),
        (69, 123, 157, 235),
        (42, 47, 79, 235),
        (231, 111, 81, 235),
        (245, 245, 245, 230),
    ]
    min_occ, max_occ = settings["occluders"]
    for _ in range(rng.randint(min_occ, max_occ)):
        color = rng.choice(palette)
        x = rng.randint(0, CANVAS_SIZE)
        y = rng.randint(0, CANVAS_SIZE)
        w = rng.randint(10, 72)
        h = rng.randint(8, 55)
        if rng.random() < 0.55:
            draw.rectangle((x, y, x + w, y + h), fill=color)
        else:
            draw.ellipse((x, y, x + w, y + h), fill=color)


def generate_image(
    rng: random.Random,
    difficulty: str,
    mascot_path: Path | None,
    backgrounds: list[Path],
    clean_backgrounds: bool,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    settings = DIFFICULTY_SETTINGS[difficulty]
    image = make_base_image(rng, backgrounds)
    if not clean_backgrounds:
        draw = ImageDraw.Draw(image)
        add_background_noise(draw, rng, settings)
        scatter_people(image, rng, settings)

    min_size, max_size = settings["mascot_size"]
    size = rng.randint(min_size, max_size)
    mascot = load_custom_mascot(mascot_path, size) if mascot_path else make_mascot(size)
    mascot = augment_mascot(mascot, rng, settings)

    x = rng.randint(0, CANVAS_SIZE - mascot.width)
    y = rng.randint(0, CANVAS_SIZE - mascot.height)

    image.paste(mascot, (x, y), mascot)
    if not clean_backgrounds:
        add_occluders(image, rng, settings)

    image = apply_image_augmentations(image, rng, settings)

    cx = (x + mascot.width / 2) / CANVAS_SIZE
    cy = (y + mascot.height / 2) / CANVAS_SIZE
    width = mascot.width / CANVAS_SIZE
    height = mascot.height / CANVAS_SIZE
    return image, (cx, cy, width, height)


def write_yaml(output: Path) -> None:
    yaml = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (output / "data.yaml").write_text(yaml)


def clear_split(output: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            directory = output / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a YOLO dataset with a hidden custom mascot.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic-mascot"))
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTY_SETTINGS),
        default="hard",
        help="How hidden and crowded the generated scenes should be.",
    )
    parser.add_argument(
        "--mascot",
        type=Path,
        help="Optional transparent PNG/WebP mascot to paste into generated scenes.",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        help="Optional folder with background images. Images are center-cropped to 640x640.",
    )
    parser.add_argument(
        "--clean-backgrounds",
        action="store_true",
        help="Only paste the mascot on the background image. Do not add generated clutter or decoys.",
    )
    args = parser.parse_args()

    if args.mascot and not args.mascot.exists():
        raise SystemExit(f"Mascot image not found: {args.mascot}")
    if args.backgrounds and not args.backgrounds.exists():
        raise SystemExit(f"Background folder not found: {args.backgrounds}")

    rng = random.Random(args.seed)
    backgrounds = list_backgrounds(args.backgrounds)
    if args.backgrounds and not backgrounds:
        raise SystemExit(f"No background images found in: {args.backgrounds}")

    clear_split(args.output)

    for split, count in (("train", args.train), ("val", args.val)):
        for index in range(count):
            image, bbox = generate_image(
                rng,
                args.difficulty,
                args.mascot,
                backgrounds,
                args.clean_backgrounds,
            )
            stem = f"{split}_{index:04d}"
            image.save(args.output / "images" / split / f"{stem}.jpg", quality=92)
            label = "0 " + " ".join(f"{value:.6f}" for value in bbox)
            (args.output / "labels" / split / f"{stem}.txt").write_text(label + "\n")
            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f"{split}: generated {index + 1}/{count}")

    write_yaml(args.output)
    print(f"Generated {args.train} training and {args.val} validation images.")
    print(f"Difficulty: {args.difficulty}")
    if args.mascot:
        print(f"Mascot: {args.mascot}")
    if args.backgrounds:
        print(f"Backgrounds: {len(backgrounds)} images from {args.backgrounds}")
    if args.clean_backgrounds:
        print("Clean backgrounds: enabled")
    print(f"Dataset YAML: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()

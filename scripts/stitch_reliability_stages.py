#!/usr/bin/env python3
"""Join one reliability-analysis image from S00, S04, S08, and S11."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STAGES = ("S00", "S04", "S08", "S11")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Horizontally join one map from reliability stages S00/S04/S08/S11."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Reliability analysis directory containing maps/.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Directory containing ref_<id>/S* stage folders. Defaults to "
            "<analysis-root>/maps; use the reliability_study root for reliability.png."
        ),
    )
    parser.add_argument(
        "--view",
        required=True,
        help="Reference id, for example 0, 00000000, or ref_00000000.",
    )
    parser.add_argument(
        "--image-name",
        required=True,
        help="Map filename, with or without the .png suffix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG. Defaults to <analysis-root>/comparisons/ref_<id>/.",
    )
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--label-height", type=int, default=54)
    args = parser.parse_args()
    if args.gap < 0 or args.label_height < 0:
        parser.error("--gap and --label-height must be non-negative")
    return args


def normalize_view(value):
    text = value.strip()
    if text.startswith("ref_"):
        text = text[4:]
    try:
        return f"ref_{int(text):08d}"
    except ValueError as error:
        raise ValueError(f"Invalid reference view: {value}") from error


def normalize_image_name(value):
    name = Path(value).name
    return name if Path(name).suffix else f"{name}.png"


def find_stage_directory(view_root, stage):
    matches = sorted(path for path in view_root.glob(f"{stage}_*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {stage}_* directory below {view_root}, found {len(matches)}"
        )
    return matches[0]


def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    args = parse_args()
    reference = normalize_view(args.view)
    image_name = normalize_image_name(args.image_name)
    source_root = args.source_root or (args.analysis_root / "maps")
    view_root = source_root / reference

    stage_directories = [find_stage_directory(view_root, stage) for stage in STAGES]
    input_paths = [directory / image_name for directory in stage_directories]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Stage image is not available yet:\n{formatted}")

    images = []
    for path in input_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    canvas_width = sum(image.width for image in images) + args.gap * (len(images) - 1)
    canvas_height = args.label_height + max(image.height for image in images)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(14, min(32, args.label_height - 12)))

    x = 0
    for stage, directory, image in zip(STAGES, stage_directories, images):
        label = f"{stage} | {directory.name[len(stage) + 1:].replace('_', ' ')}"
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        text_x = x + max(0, (image.width - text_width) // 2)
        text_y = max(0, (args.label_height - text_height) // 2 - box[1])
        draw.text((text_x, text_y), label, fill="black", font=font)
        canvas.paste(image, (x, args.label_height))
        x += image.width
        if stage != STAGES[-1]:
            draw.rectangle((x, 0, x + args.gap - 1, canvas_height), fill="#d0d0d0")
            x += args.gap

    output = args.output
    if output is None:
        stem = Path(image_name).stem
        output = (
            args.analysis_root
            / "comparisons"
            / reference
            / f"{stem}_S00_S04_S08_S11.png"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Written: {output}")
    for stage, path in zip(STAGES, input_paths):
        print(f"  {stage}: {path}")


if __name__ == "__main__":
    main()

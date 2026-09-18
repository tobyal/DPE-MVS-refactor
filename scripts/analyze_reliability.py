#!/usr/bin/env python3
"""Analyze DPE reliability-study snapshots against aligned GT geometry."""

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


WEAK = 0
STRONG = 1
UNKNOWN = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze exported DPE reliability evolution snapshots."
    )
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gt-depth-pattern")
    parser.add_argument("--gt-normal-pattern")
    parser.add_argument("--gt-mask-pattern")
    parser.add_argument("--camera-root", type=Path)
    parser.add_argument("--derive-gt-normals", action="store_true")
    parser.add_argument(
        "--gt-normal-coordinates", choices=("world", "camera"), default="world"
    )
    parser.add_argument(
        "--normal-sign-mode",
        choices=("signed", "unsigned"),
        default="signed",
        help="Use unsigned only when GT and DPE normals have an unresolved sign convention.",
    )
    parser.add_argument("--depth-thresholds", type=float, nargs="+", default=[0.02, 0.10])
    parser.add_argument("--normal-thresholds", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument(
        "--depth-correct-threshold", "--reliability-depth-threshold",
        dest="depth_correct_threshold", type=float, default=0.02,
    )
    parser.add_argument(
        "--normal-correct-threshold", "--reliability-normal-threshold",
        dest="normal_correct_threshold", type=float, default=10.0,
    )
    parser.add_argument("--depth-error-vmax", type=float, default=0.10)
    parser.add_argument("--normal-error-vmax", type=float, default=30.0)
    parser.add_argument("--skip-error-maps", action="store_true")
    return parser.parse_args()


def read_dmb(path):
    with path.open("rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid DMB header: {path}")
        version, rows, cols, cv_type = struct.unpack("=4i", header)
        if version != 1 or rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid DMB dimensions: {path}")
        depth_type = cv_type & 7
        dtype_by_depth = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.int32,
            5: np.float32,
            6: np.float64,
        }
        if depth_type not in dtype_by_depth:
            raise ValueError(f"Unsupported OpenCV type {cv_type}: {path}")
        channels = 1 + (cv_type >> 3)
        values = np.fromfile(handle, dtype=dtype_by_depth[depth_type])
    expected = rows * cols * channels
    if values.size != expected:
        raise ValueError(f"DMB payload size mismatch: {path}")
    shape = (rows, cols) if channels == 1 else (rows, cols, channels)
    return values.reshape(shape)


def format_pattern(pattern, gt_root, ref_id):
    return Path(
        pattern.format(
            gt_root=str(gt_root), ref_id=ref_id, ref_id8=f"{ref_id:08d}"
        )
    )


def find_gt_file(gt_root, ref_id, pattern, kind, required):
    if pattern:
        path = format_pattern(pattern, gt_root, ref_id)
        if path.exists():
            return path
        if required:
            raise FileNotFoundError(path)
        return None

    ref_id8 = f"{ref_id:08d}"
    names = {
        "depth": ("depth.dmb", "gt_depth.dmb"),
        "normal": ("normal.dmb", "gt_normal.dmb"),
        "mask": ("valid_mask.dmb", "mask.dmb"),
    }[kind]
    candidates = []
    for name in names:
        candidates.extend(
            [
                gt_root / f"ref_{ref_id8}" / name,
                gt_root / "views" / ref_id8 / name,
                gt_root / ref_id8 / name,
            ]
        )
    if kind == "depth":
        candidates.append(gt_root / f"{ref_id8}.dmb")
    for path in candidates:
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"No aligned GT {kind} found for reference {ref_id8}")
    return None


def nearest_resize(array, height, width):
    if array.shape[:2] == (height, width):
        return array.copy()
    y = np.minimum(
        array.shape[0] - 1,
        ((np.arange(height) + 0.5) * array.shape[0] / height).astype(np.int64),
    )
    x = np.minimum(
        array.shape[1] - 1,
        ((np.arange(width) + 0.5) * array.shape[1] / width).astype(np.int64),
    )
    return array[y[:, None], x[None, :]]


def parse_camera(path):
    tokens = path.read_text(encoding="utf-8").split()
    ext = tokens.index("extrinsic") + 1
    extrinsic = np.asarray(tokens[ext : ext + 16], dtype=np.float64).reshape(4, 4)
    intr = tokens.index("intrinsic") + 1
    intrinsic = np.asarray(tokens[intr : intr + 9], dtype=np.float64).reshape(3, 3)
    return intrinsic, extrinsic[:3, :3]


def find_camera(camera_root, ref_id):
    if camera_root is None:
        raise ValueError("--camera-root is required for camera-space or derived normals")
    ref_id8 = f"{ref_id:08d}"
    candidates = [
        camera_root / f"{ref_id8}_cam.txt",
        camera_root / "cams" / f"{ref_id8}_cam.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No camera file found for reference {ref_id8}")


def normalize_normals(normal):
    length = np.linalg.norm(normal, axis=2)
    valid = np.isfinite(normal).all(axis=2) & (length > 1e-8)
    result = np.zeros_like(normal, dtype=np.float32)
    result[valid] = normal[valid] / length[valid, None]
    return result, valid


def derive_world_normals(depth, valid, intrinsic, rotation):
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    z = depth.astype(np.float64)
    points = np.stack(
        (
            z * (xx - intrinsic[0, 2]) / intrinsic[0, 0],
            z * (yy - intrinsic[1, 2]) / intrinsic[1, 1],
            z,
        ),
        axis=2,
    )
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    camera_normal = np.cross(dx, dy)
    view_direction = points[1:-1, 1:-1]
    facing_away = np.sum(camera_normal * view_direction, axis=2) > 0.0
    camera_normal[facing_away] *= -1.0
    world_normal = camera_normal @ rotation
    result = np.zeros((height, width, 3), dtype=np.float32)
    result[1:-1, 1:-1] = world_normal.astype(np.float32)
    neighbor_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
    )
    normalized, normal_valid = normalize_normals(result)
    normal_valid[1:-1, 1:-1] &= neighbor_valid
    normal_valid[[0, -1], :] = False
    normal_valid[:, [0, -1]] = False
    return normalized, normal_valid


def angular_error(predicted, ground_truth, valid, sign_mode):
    predicted, predicted_valid = normalize_normals(predicted.astype(np.float32))
    ground_truth, gt_valid = normalize_normals(ground_truth.astype(np.float32))
    valid = valid & predicted_valid & gt_valid
    dot = np.sum(predicted * ground_truth, axis=2)
    if sign_mode == "unsigned":
        dot = np.abs(dot)
    angle = np.full(valid.shape, np.nan, dtype=np.float32)
    angle[valid] = np.degrees(np.arccos(np.clip(dot[valid], -1.0, 1.0)))
    return angle, valid


def finite_stats(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.median(values))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def threshold_label(value, angular=False):
    if angular:
        return str(int(round(value)))
    centimeters = value * 100.0
    if abs(centimeters - round(centimeters)) < 1e-6:
        return f"{int(round(centimeters))}cm"
    return f"{int(round(value * 1000.0))}mm"


def confusion(reliability, correct, valid):
    strong = (reliability == STRONG) & valid
    weak = (reliability == WEAK) & valid
    true_strong = int(np.count_nonzero(strong & correct))
    false_strong = int(np.count_nonzero(strong & ~correct))
    true_weak = int(np.count_nonzero(weak & ~correct))
    false_weak = int(np.count_nonzero(weak & correct))
    return {
        "true_strong": true_strong,
        "false_strong": false_strong,
        "true_weak": true_weak,
        "false_weak": false_weak,
        "fsr": ratio(false_strong, true_strong + false_strong),
        "fwr": ratio(false_weak, true_weak + false_weak),
    }


def load_gt(args, ref_id):
    depth_path = find_gt_file(
        args.gt_root, ref_id, args.gt_depth_pattern, "depth", required=True
    )
    depth = read_dmb(depth_path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"GT depth must be single-channel: {depth_path}")
    valid = np.isfinite(depth) & (depth > 0.0)

    mask_path = find_gt_file(
        args.gt_root, ref_id, args.gt_mask_pattern, "mask", required=False
    )
    if mask_path:
        mask = read_dmb(mask_path).astype(bool)
        if mask.shape != depth.shape:
            raise ValueError(f"GT valid mask shape does not match depth: {mask_path}")
        valid &= mask

    normal_path = find_gt_file(
        args.gt_root, ref_id, args.gt_normal_pattern, "normal", required=False
    )
    normal = None
    normal_valid = np.zeros(depth.shape, dtype=bool)
    if normal_path:
        normal = read_dmb(normal_path).astype(np.float32)
        if normal.shape != depth.shape + (3,):
            raise ValueError(f"GT normal shape does not match depth: {normal_path}")
        normal, normal_valid = normalize_normals(normal)
        if args.gt_normal_coordinates == "camera":
            _, rotation = parse_camera(find_camera(args.camera_root, ref_id))
            normal = normal @ rotation
            normal, normal_valid = normalize_normals(normal)
        normal_valid &= valid
    elif args.derive_gt_normals:
        intrinsic, rotation = parse_camera(find_camera(args.camera_root, ref_id))
        normal, normal_valid = derive_world_normals(depth, valid, intrinsic, rotation)
    return depth, valid, normal, normal_valid


def masked_values(values, valid):
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[valid] = values[valid]
    return result


def save_scalar_map(path, values, maximum, cmap_name, unit, title):
    figure, axis = plt.subplots(figsize=(12, 8))
    cmap = copy(plt.get_cmap(cmap_name))
    cmap.set_bad(color="#303030", alpha=1.0)
    image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=maximum)
    axis.set_title(title)
    axis.axis("off")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label(unit)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_categorical_map(path, classes, colors, labels, title):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(classes, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_title(title)
    axis.axis("off")
    handles = [Patch(facecolor=colors[index], label=label)
               for index, label in enumerate(labels)]
    axis.legend(handles=handles, loc="upper right", framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def correctness_classes(reliability_mask, valid, correct):
    classes = np.zeros(valid.shape, dtype=np.uint8)
    classes[reliability_mask & valid & correct] = 1
    classes[reliability_mask & valid & ~correct] = 2
    return classes


def reliability_gt_classes(reliability, valid, correct):
    classes = np.zeros(valid.shape, dtype=np.uint8)
    classes[(reliability == STRONG) & valid & correct] = 1
    classes[(reliability == STRONG) & valid & ~correct] = 2
    classes[(reliability == WEAK) & valid & correct] = 3
    classes[(reliability == WEAK) & valid & ~correct] = 4
    return classes


def save_error_maps(output_root, ref_name, stage_name, reliability,
                    depth_pre, depth_post, normal_pre, normal_post,
                    depth_valid, normal_valid, depth_correct, normal_correct,
                    joint_valid, joint_correct, depth_vmax, normal_vmax):
    directory = output_root / "maps" / ref_name / stage_name
    directory.mkdir(parents=True, exist_ok=True)
    save_scalar_map(directory / "depth_error_pre.png", depth_pre, depth_vmax,
                    "magma", "depth error (meter)", "Pre-R depth error")
    save_scalar_map(directory / "depth_error_post.png", depth_post, depth_vmax,
                    "magma", "depth error (meter)", "Post-refinement depth error")
    save_scalar_map(directory / "normal_error_pre.png", normal_pre, normal_vmax,
                    "viridis", "normal error (degree)", "Pre-R normal error")
    save_scalar_map(directory / "normal_error_post.png", normal_post, normal_vmax,
                    "viridis", "normal error (degree)", "Post-refinement normal error")

    strong = reliability == STRONG
    weak = reliability == WEAK
    for name, values, valid, maximum, cmap, unit, title in (
        ("strong_depth_error.png", depth_pre, strong & depth_valid, depth_vmax,
         "magma", "depth error (meter)", "STRONG depth error"),
        ("weak_depth_error.png", depth_pre, weak & depth_valid, depth_vmax,
         "magma", "depth error (meter)", "WEAK depth error"),
        ("strong_normal_error.png", normal_pre, strong & normal_valid, normal_vmax,
         "viridis", "normal error (degree)", "STRONG normal error"),
        ("weak_normal_error.png", normal_pre, weak & normal_valid, normal_vmax,
         "viridis", "normal error (degree)", "WEAK normal error"),
    ):
        save_scalar_map(directory / name, masked_values(values, valid), maximum,
                        cmap, unit, title)

    correctness_colors = ["#4a4a4a", "#2ca02c", "#d62728"]
    correctness_labels = ["Other / Invalid GT", "Correct", "Wrong"]
    for name, reliability_mask, valid, correct, title in (
        ("strong_depth_correct.png", strong, depth_valid, depth_correct,
         "STRONG depth correctness"),
        ("weak_depth_correct.png", weak, depth_valid, depth_correct,
         "WEAK depth correctness"),
        ("strong_normal_correct.png", strong, normal_valid, normal_correct,
         "STRONG normal correctness"),
        ("weak_normal_correct.png", weak, normal_valid, normal_correct,
         "WEAK normal correctness"),
    ):
        save_categorical_map(
            directory / name,
            correctness_classes(reliability_mask, valid, correct),
            correctness_colors, correctness_labels, title,
        )

    comparison_colors = ["#4a4a4a", "#2ca02c", "#d627a8", "#17becf", "#ff7f0e"]
    comparison_labels = [
        "UNKNOWN / Invalid GT", "STRONG + Correct", "STRONG + Wrong",
        "WEAK + Correct", "WEAK + Wrong",
    ]
    for name, valid, correct, title in (
        ("reliability_vs_depth_gt.png", depth_valid, depth_correct,
         "Reliability vs depth GT"),
        ("reliability_vs_normal_gt.png", normal_valid, normal_correct,
         "Reliability vs normal GT"),
        ("reliability_vs_joint_gt.png", joint_valid, joint_correct,
         "Reliability vs joint geometry GT"),
    ):
        save_categorical_map(
            directory / name, reliability_gt_classes(reliability, valid, correct),
            comparison_colors, comparison_labels, title,
        )

    false_strong = strong & joint_valid & ~joint_correct
    false_weak = weak & joint_valid & joint_correct
    save_categorical_map(
        directory / "false_strong_joint.png", false_strong.astype(np.uint8),
        ["#000000", "#ff00ff"], ["Other", "False Strong"], "False Strong (joint)",
    )
    save_categorical_map(
        directory / "false_weak_joint.png", false_weak.astype(np.uint8),
        ["#000000", "#00ffff"], ["Other", "False Weak"], "False Weak (joint)",
    )


def analyze_stage(args, output_root, ref_directory, stage_directory, gt_cache):
    metadata = json.loads((stage_directory / "stage.json").read_text(encoding="utf-8"))
    ref_id = int(metadata["ref_image_id"])
    if ref_id not in gt_cache:
        gt_cache[ref_id] = load_gt(args, ref_id)
    gt_depth_source, gt_valid_source, gt_normal_source, gt_normal_valid_source = gt_cache[ref_id]

    depth_pre = read_dmb(stage_directory / "depth_pre.dmb").astype(np.float32)
    depth_post = read_dmb(stage_directory / "depth_post.dmb").astype(np.float32)
    normal_pre = read_dmb(stage_directory / "normal_pre.dmb").astype(np.float32)
    normal_post = read_dmb(stage_directory / "normal_post.dmb").astype(np.float32)
    reliability = read_dmb(stage_directory / "reliability.bin").astype(np.uint8)
    height, width = depth_pre.shape

    gt_depth = nearest_resize(gt_depth_source, height, width)
    gt_valid = nearest_resize(gt_valid_source, height, width).astype(bool)
    depth_valid_pre = gt_valid & np.isfinite(depth_pre) & (depth_pre > 0.0)
    depth_valid_post = gt_valid & np.isfinite(depth_post) & (depth_post > 0.0)
    depth_error_pre = np.full((height, width), np.nan, dtype=np.float32)
    depth_error_post = np.full((height, width), np.nan, dtype=np.float32)
    depth_error_pre[depth_valid_pre] = np.abs(depth_pre[depth_valid_pre] - gt_depth[depth_valid_pre])
    depth_error_post[depth_valid_post] = np.abs(depth_post[depth_valid_post] - gt_depth[depth_valid_post])

    normal_error_pre = np.full((height, width), np.nan, dtype=np.float32)
    normal_error_post = np.full((height, width), np.nan, dtype=np.float32)
    normal_valid_pre = np.zeros((height, width), dtype=bool)
    normal_valid_post = np.zeros((height, width), dtype=bool)
    if gt_normal_source is not None:
        gt_normal = nearest_resize(gt_normal_source, height, width)
        gt_normal_valid = nearest_resize(gt_normal_valid_source, height, width).astype(bool)
        normal_error_pre, normal_valid_pre = angular_error(
            normal_pre, gt_normal, gt_normal_valid, args.normal_sign_mode
        )
        normal_error_post, normal_valid_post = angular_error(
            normal_post, gt_normal, gt_normal_valid, args.normal_sign_mode
        )

    row = {
        "ref_image_id": ref_id,
        "stage_id": metadata["stage_id"],
        "level": metadata["pyramid_level"],
        "scale": metadata["scale"],
        "run_state": metadata["run_state"],
        "outer_refine": metadata["outer_refine"],
    }
    row["depth_pre_mean"], row["depth_pre_median"] = finite_stats(depth_error_pre)
    row["depth_post_mean"], row["depth_post_median"] = finite_stats(depth_error_post)
    for threshold in args.depth_thresholds:
        label = threshold_label(threshold)
        row[f"depth_pre_acc_{label}"] = ratio(
            np.count_nonzero(depth_error_pre[depth_valid_pre] <= threshold),
            np.count_nonzero(depth_valid_pre),
        )
        row[f"depth_post_acc_{label}"] = ratio(
            np.count_nonzero(depth_error_post[depth_valid_post] <= threshold),
            np.count_nonzero(depth_valid_post),
        )

    row["normal_pre_mean"], row["normal_pre_median"] = finite_stats(normal_error_pre)
    row["normal_post_mean"], row["normal_post_median"] = finite_stats(normal_error_post)
    for threshold in args.normal_thresholds:
        label = threshold_label(threshold, angular=True)
        row[f"normal_pre_acc_{label}"] = ratio(
            np.count_nonzero(normal_error_pre[normal_valid_pre] <= threshold),
            np.count_nonzero(normal_valid_pre),
        )
        row[f"normal_post_acc_{label}"] = ratio(
            np.count_nonzero(normal_error_post[normal_valid_post] <= threshold),
            np.count_nonzero(normal_valid_post),
        )

    total = reliability.size
    row["strong_ratio"] = np.count_nonzero(reliability == STRONG) / total
    row["weak_ratio"] = np.count_nonzero(reliability == WEAK) / total
    row["unknown_ratio"] = np.count_nonzero(reliability == UNKNOWN) / total

    depth_correct = depth_error_pre < args.depth_correct_threshold
    normal_correct = normal_error_pre < args.normal_correct_threshold
    joint_valid = depth_valid_pre & normal_valid_pre
    joint_correct = depth_correct & normal_correct

    strong = reliability == STRONG
    weak = reliability == WEAK
    conditioned_masks = {
        "strong_depth": strong & depth_valid_pre,
        "weak_depth": weak & depth_valid_pre,
        "strong_normal": strong & normal_valid_pre,
        "weak_normal": weak & normal_valid_pre,
        "strong_joint": strong & joint_valid,
        "weak_joint": weak & joint_valid,
    }
    for prefix in ("strong_depth", "weak_depth"):
        mask = conditioned_masks[prefix]
        row[f"{prefix}_mean_error"], row[f"{prefix}_median_error"] = finite_stats(
            depth_error_pre[mask]
        )
        row[f"{prefix}_valid_count"] = int(np.count_nonzero(mask))
    for prefix in ("strong_normal", "weak_normal"):
        mask = conditioned_masks[prefix]
        row[f"{prefix}_mean_error"], row[f"{prefix}_median_error"] = finite_stats(
            normal_error_pre[mask]
        )
        row[f"{prefix}_valid_count"] = int(np.count_nonzero(mask))
    row["strong_joint_valid_count"] = int(np.count_nonzero(conditioned_masks["strong_joint"]))
    row["weak_joint_valid_count"] = int(np.count_nonzero(conditioned_masks["weak_joint"]))

    row["strong_depth_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["strong_depth"] & depth_correct),
        row["strong_depth_valid_count"],
    )
    row["weak_depth_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["weak_depth"] & depth_correct),
        row["weak_depth_valid_count"],
    )
    row["strong_normal_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["strong_normal"] & normal_correct),
        row["strong_normal_valid_count"],
    )
    row["weak_normal_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["weak_normal"] & normal_correct),
        row["weak_normal_valid_count"],
    )
    row["strong_joint_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["strong_joint"] & joint_correct),
        row["strong_joint_valid_count"],
    )
    row["weak_joint_correct_ratio"] = ratio(
        np.count_nonzero(conditioned_masks["weak_joint"] & joint_correct),
        row["weak_joint_valid_count"],
    )
    confusion_sets = {
        "depth": confusion(reliability, depth_correct, depth_valid_pre),
        "normal": confusion(reliability, normal_correct, normal_valid_pre),
        "joint": confusion(reliability, joint_correct, joint_valid),
    }
    for name, values in confusion_sets.items():
        row[f"fsr_{name}"] = values["fsr"]
        row[f"fwr_{name}"] = values["fwr"]
        for key in ("true_strong", "false_strong", "true_weak", "false_weak"):
            row[f"{key}_{name}"] = values[key]
    row["valid_count"] = int(np.count_nonzero(joint_valid))
    row["depth_valid_count"] = int(np.count_nonzero(depth_valid_pre))
    row["normal_valid_count"] = int(np.count_nonzero(normal_valid_pre))
    row["pixel_count"] = int(total)

    if not args.skip_error_maps:
        save_error_maps(
            output_root, ref_directory.name, stage_directory.name, reliability,
            depth_error_pre, depth_error_post, normal_error_pre, normal_error_post,
            depth_valid_pre, normal_valid_pre, depth_correct, normal_correct,
            joint_valid, joint_correct, args.depth_error_vmax, args.normal_error_vmax,
        )
    return row


def weighted_value(rows, key, weight_key):
    values = []
    weights = []
    for row in rows:
        value = row.get(key, math.nan)
        weight = row.get(weight_key, 0)
        if np.isfinite(value) and weight > 0:
            values.append(value)
            weights.append(weight)
    return float(np.average(values, weights=weights)) if values else math.nan


def aggregate_stage(rows):
    result = {
        key: rows[0][key]
        for key in ("stage_id", "level", "scale", "run_state", "outer_refine")
    }
    for key in rows[0]:
        if key in result or key == "ref_image_id":
            continue
        if key.endswith("_count") or key.startswith(("true_", "false_")):
            result[key] = sum(row.get(key, 0) for row in rows)
            continue
        if key.startswith(("strong_depth_", "weak_depth_")):
            prefix = "strong_depth" if key.startswith("strong_depth_") else "weak_depth"
            result[key] = weighted_value(rows, key, f"{prefix}_valid_count")
        elif key.startswith(("strong_normal_", "weak_normal_")):
            prefix = "strong_normal" if key.startswith("strong_normal_") else "weak_normal"
            result[key] = weighted_value(rows, key, f"{prefix}_valid_count")
        elif key.startswith(("strong_joint_", "weak_joint_")):
            prefix = "strong_joint" if key.startswith("strong_joint_") else "weak_joint"
            result[key] = weighted_value(rows, key, f"{prefix}_valid_count")
        elif key.startswith("depth_"):
            result[key] = weighted_value(rows, key, "depth_valid_count")
        elif key.startswith("normal_"):
            result[key] = weighted_value(rows, key, "normal_valid_count")
        else:
            result[key] = weighted_value(rows, key, "pixel_count")
    for name in ("depth", "normal", "joint"):
        true_strong = result[f"true_strong_{name}"]
        false_strong = result[f"false_strong_{name}"]
        true_weak = result[f"true_weak_{name}"]
        false_weak = result[f"false_weak_{name}"]
        result[f"fsr_{name}"] = ratio(false_strong, true_strong + false_strong)
        result[f"fwr_{name}"] = ratio(false_weak, true_weak + false_weak)
        result[f"strong_{name}_correct_ratio"] = ratio(
            true_strong, true_strong + false_strong
        )
        result[f"weak_{name}_correct_ratio"] = ratio(
            false_weak, true_weak + false_weak
        )
    return result


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_evolution(output_root, stage_rows, depth_labels, normal_labels):
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(stage_rows))
    labels = [row["stage_id"] for row in stage_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, [r["depth_pre_mean"] for r in stage_rows], marker="o", label="pre mean error")
    axes[0].plot(x, [r["depth_post_mean"] for r in stage_rows], marker="o", label="post mean error")
    axes[0].set_ylabel("absolute depth error")
    axes[0].legend()
    for label in depth_labels:
        axes[1].plot(x, [r[f"depth_pre_acc_{label}"] for r in stage_rows], label=f"pre {label}")
        axes[1].plot(x, [r[f"depth_post_acc_{label}"] for r in stage_rows], linestyle="--", label=f"post {label}")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(plot_root / "depth_evolution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, [r["normal_pre_mean"] for r in stage_rows], marker="o", label="pre mean angle")
    axes[0].plot(x, [r["normal_post_mean"] for r in stage_rows], marker="o", label="post mean angle")
    axes[0].set_ylabel("angular error (deg)")
    axes[0].legend()
    preferred = "10" if "10" in normal_labels else normal_labels[0]
    axes[1].plot(x, [r[f"normal_pre_acc_{preferred}"] for r in stage_rows], marker="o", label=f"pre < {preferred} deg")
    axes[1].plot(x, [r[f"normal_post_acc_{preferred}"] for r in stage_rows], marker="o", label=f"post < {preferred} deg")
    axes[1].set_ylabel("accuracy")
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_root / "normal_evolution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for key in ("fsr_joint", "fwr_joint", "strong_ratio", "weak_ratio"):
        ax.plot(x, [r[key] for r in stage_rows], marker="o", label=key)
    ax.set_xticks(x, labels, rotation=45)
    ax.set_ylabel("ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "reliability_evolution.png", dpi=160)
    plt.close(fig)

    fig, ax_geometry = plt.subplots(figsize=(12, 5))
    ax_reliability = ax_geometry.twinx()
    ax_geometry.plot(x, [r["depth_pre_mean"] for r in stage_rows], color="tab:blue", marker="o", label="pre depth error")
    ax_reliability.plot(x, [r["fsr_joint"] for r in stage_rows], color="tab:red", marker="s", label="FSR joint")
    ax_geometry.set_ylabel("pre-R mean depth error", color="tab:blue")
    ax_reliability.set_ylabel("joint false-strong rate", color="tab:red")
    ax_geometry.set_xticks(x, labels, rotation=45)
    fig.tight_layout()
    fig.savefig(plot_root / "geometry_vs_reliability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, [r["strong_depth_mean_error"] for r in stage_rows],
            marker="o", label="STRONG mean depth error")
    ax.plot(x, [r["weak_depth_mean_error"] for r in stage_rows],
            marker="o", label="WEAK mean depth error")
    ax.set_ylabel("depth error (meter)")
    ax.set_xticks(x, labels, rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "strong_vs_weak_depth_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, [r["strong_normal_mean_error"] for r in stage_rows],
            marker="o", label="STRONG mean normal error")
    ax.plot(x, [r["weak_normal_mean_error"] for r in stage_rows],
            marker="o", label="WEAK mean normal error")
    ax.set_ylabel("normal error (degree)")
    ax.set_xticks(x, labels, rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "strong_vs_weak_normal_error.png", dpi=160)
    plt.close(fig)

    for geometry, title in (
        ("depth", "Depth correctness by reliability"),
        ("normal", "Normal correctness by reliability"),
        ("joint", "Joint geometry correctness by reliability"),
    ):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, [r[f"strong_{geometry}_correct_ratio"] for r in stage_rows],
                marker="o", label="STRONG correct ratio")
        ax.plot(x, [r[f"weak_{geometry}_correct_ratio"] for r in stage_rows],
                marker="o", label="WEAK correct ratio")
        ax.set_title(title)
        ax.set_ylabel("correct ratio")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, labels, rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_root / f"strong_vs_weak_{geometry}_correctness.png", dpi=160)
        plt.close(fig)


def write_summary(path, stage_rows):
    if not stage_rows:
        return
    first, last = stage_rows[0], stage_rows[-1]
    metrics = [
        "depth_pre_mean", "depth_post_mean", "normal_pre_mean", "normal_post_mean",
        "fsr_joint", "fwr_joint", "strong_ratio", "weak_ratio",
        "strong_depth_mean_error", "weak_depth_mean_error",
        "strong_normal_mean_error", "weak_normal_mean_error",
        "strong_depth_correct_ratio", "weak_depth_correct_ratio",
        "strong_normal_correct_ratio", "weak_normal_correct_ratio",
        "strong_joint_correct_ratio", "weak_joint_correct_ratio",
    ]
    rows = []
    for metric in metrics:
        first_value = first.get(metric, math.nan)
        last_value = last.get(metric, math.nan)
        rows.append(
            {
                "metric": metric,
                "first_stage": first["stage_id"],
                "last_stage": last["stage_id"],
                "first_value": first_value,
                "last_value": last_value,
                "delta": last_value - first_value,
            }
        )
    write_csv(path, rows)


def display_value(value, unit=""):
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.4f}{unit}"


def print_stage_summaries(stage_rows):
    for row in stage_rows:
        print(row["stage_id"])
        print(f"  STRONG ratio         : {display_value(row['strong_ratio'])}")
        print(f"  WEAK ratio           : {display_value(row['weak_ratio'])}")
        print(f"  STRONG depth error   : {display_value(row['strong_depth_mean_error'], ' m')}")
        print(f"  WEAK depth error     : {display_value(row['weak_depth_mean_error'], ' m')}")
        print(f"  STRONG normal error  : {display_value(row['strong_normal_mean_error'], ' deg')}")
        print(f"  WEAK normal error    : {display_value(row['weak_normal_mean_error'], ' deg')}")
        print(f"  STRONG joint correct : {display_value(row['strong_joint_correct_ratio'])}")
        print(f"  WEAK joint correct   : {display_value(row['weak_joint_correct_ratio'])}")
        print(f"  FSR joint            : {display_value(row['fsr_joint'])}")
        print(f"  FWR joint            : {display_value(row['fwr_joint'])}")
        print()


def main():
    args = parse_args()
    if not args.study_root.is_dir():
        raise FileNotFoundError(args.study_root)
    output_root = args.output or (args.study_root / "analysis")
    output_root.mkdir(parents=True, exist_ok=True)

    configuration = {
        "study_root": str(args.study_root),
        "gt_root": str(args.gt_root),
        "depth_thresholds": args.depth_thresholds,
        "normal_thresholds_deg": args.normal_thresholds,
        "depth_correct_threshold": args.depth_correct_threshold,
        "normal_correct_threshold_deg": args.normal_correct_threshold,
        "depth_error_vmax": args.depth_error_vmax,
        "normal_error_vmax_deg": args.normal_error_vmax,
        "normal_sign_mode": args.normal_sign_mode,
        "normal_sign_note": (
            "dot(n_pred,n_gt)" if args.normal_sign_mode == "signed"
            else "abs(dot(n_pred,n_gt)); enabled explicitly for sign-ambiguous normals"
        ),
        "aggregate_median_note": "per-stage medians are valid-count-weighted per-view medians",
    }
    (output_root / "analysis_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    gt_cache = {}
    view_rows = []
    for ref_directory in sorted(args.study_root.glob("ref_*")):
        if not ref_directory.is_dir():
            continue
        for stage_directory in sorted(ref_directory.glob("S*_L*")):
            view_rows.append(
                analyze_stage(args, output_root, ref_directory, stage_directory, gt_cache)
            )
    if not view_rows:
        raise RuntimeError(f"No reliability-study stages found below {args.study_root}")

    grouped = defaultdict(list)
    for row in view_rows:
        grouped[row["stage_id"]].append(row)
    stage_rows = [aggregate_stage(grouped[key]) for key in sorted(grouped)]

    metrics_root = output_root / "metrics"
    write_csv(metrics_root / "per_view.csv", view_rows)
    write_csv(metrics_root / "per_stage.csv", stage_rows)
    write_summary(metrics_root / "summary.csv", stage_rows)
    depth_labels = [threshold_label(value) for value in args.depth_thresholds]
    normal_labels = [threshold_label(value, angular=True) for value in args.normal_thresholds]
    plot_evolution(output_root, stage_rows, depth_labels, normal_labels)
    print_stage_summaries(stage_rows)
    print(f"Reliability analysis written to {output_root}")


if __name__ == "__main__":
    main()

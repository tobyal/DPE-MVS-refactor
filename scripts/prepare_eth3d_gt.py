#!/usr/bin/env python3
"""Rasterize official ETH3D laser scans into per-view DPE depth maps."""

import argparse
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass
class View:
    image_id: int
    width: int
    height: int
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    depth: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project official ETH3D scan*.ply files through DPE cameras and "
            "write aligned per-view gt_depth.dmb files."
        )
    )
    parser.add_argument("--dense-folder", type=Path, required=True)
    parser.add_argument("--ground-truth-mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=0,
        help="Circular point radius in pixels; 0 preserves raw scan sampling.",
    )
    parser.add_argument("--chunk-size", type=int, default=500000)
    parser.add_argument(
        "--view",
        type=int,
        action="append",
        dest="views",
        help="Only rasterize this reference id; may be repeated.",
    )
    args = parser.parse_args()
    if args.splat_radius < 0:
        parser.error("--splat-radius must be non-negative")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def read_reference_ids(path):
    tokens = path.read_text(encoding="utf-8").split()
    cursor = 0
    count = int(tokens[cursor])
    cursor += 1
    result = []
    for _ in range(count):
        reference_id = int(tokens[cursor])
        source_count = int(tokens[cursor + 1])
        cursor += 2 + 2 * source_count
        result.append(reference_id)
    return result


def parse_camera(path):
    tokens = path.read_text(encoding="utf-8").split()
    ext = tokens.index("extrinsic") + 1
    extrinsic = np.asarray(tokens[ext : ext + 16], dtype=np.float64).reshape(4, 4)
    intr = tokens.index("intrinsic") + 1
    intrinsic = np.asarray(tokens[intr : intr + 9], dtype=np.float64).reshape(3, 3)
    return intrinsic, extrinsic[:3, :3], extrinsic[:3, 3]


def load_views(dense_folder, selected_ids):
    views = []
    for image_id in read_reference_ids(dense_folder / "pair.txt"):
        if selected_ids is not None and image_id not in selected_ids:
            continue
        image_id8 = f"{image_id:08d}"
        image_path = dense_folder / "images" / f"{image_id8}.jpg"
        camera_path = dense_folder / "cams" / f"{image_id8}_cam.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        intrinsic, rotation, translation = parse_camera(camera_path)
        views.append(
            View(
                image_id=image_id,
                width=width,
                height=height,
                intrinsic=intrinsic,
                rotation=rotation,
                translation=translation,
                depth=np.full(width * height, np.inf, dtype=np.float32),
            )
        )
    if not views:
        raise ValueError("No selected reference views were found in pair.txt")
    return views


def read_scan_entries(mlp_path):
    root = ET.parse(mlp_path).getroot()
    entries = []
    for mesh in root.findall(".//MLMesh"):
        filename = mesh.get("filename")
        matrix_node = mesh.find("MLMatrix44")
        if not filename or matrix_node is None or not matrix_node.text:
            raise ValueError(f"Invalid MLMesh entry in {mlp_path}")
        values = np.fromstring(matrix_node.text, sep=" ", dtype=np.float64)
        if values.size != 16:
            raise ValueError(f"Invalid MLMatrix44 for {filename}")
        scan_path = Path(filename)
        if not scan_path.is_absolute():
            scan_path = mlp_path.parent / scan_path
        entries.append((scan_path, values.reshape(4, 4)))
    if not entries:
        raise ValueError(f"No scans found in {mlp_path}")
    return entries


def read_vertex_layout(path):
    properties = []
    vertex_count = None
    current_element = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            text = line.decode("ascii").strip()
            parts = text.split()
            if parts[:2] == ["format", "binary_little_endian"]:
                pass
            elif parts and parts[0] == "format":
                raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
            elif parts[:1] == ["element"]:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError(f"List-valued vertex properties are unsupported: {path}")
                if parts[1] not in PLY_DTYPES:
                    raise ValueError(f"Unsupported PLY property type {parts[1]}: {path}")
                properties.append((parts[2], PLY_DTYPES[parts[1]]))
            elif text == "end_header":
                data_offset = handle.tell()
                break
    if vertex_count is None or not {"x", "y", "z"}.issubset(name for name, _ in properties):
        raise ValueError(f"PLY vertex x/y/z properties are required: {path}")
    return vertex_count, np.dtype(properties), data_offset


def iter_scan_chunks(path, chunk_size):
    vertex_count, dtype, offset = read_vertex_layout(path)
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,))
    for start in range(0, vertex_count, chunk_size):
        chunk = vertices[start : min(start + chunk_size, vertex_count)]
        yield np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
            np.float64, copy=False
        )


def splat_offsets(radius):
    return [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]


def project_chunk(world_points, view, offsets):
    camera_points = world_points @ view.rotation.T + view.translation
    camera_z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (camera_z > 0.0)
    if not np.any(valid):
        return 0
    camera_points = camera_points[valid]
    camera_z = camera_z[valid]
    homogeneous = camera_points @ view.intrinsic.T
    denominator = homogeneous[:, 2]
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(denominator) > 1e-12)
    if not np.any(valid):
        return 0
    homogeneous = homogeneous[valid]
    camera_z = camera_z[valid]
    pixel_x = np.rint(homogeneous[:, 0] / homogeneous[:, 2]).astype(np.int64)
    pixel_y = np.rint(homogeneous[:, 1] / homogeneous[:, 2]).astype(np.int64)
    projected = 0
    for offset_x, offset_y in offsets:
        x = pixel_x + offset_x
        y = pixel_y + offset_y
        inside = (x >= 0) & (x < view.width) & (y >= 0) & (y < view.height)
        if not np.any(inside):
            continue
        indices = y[inside] * view.width + x[inside]
        np.minimum.at(view.depth, indices, camera_z[inside].astype(np.float32))
        if offset_x == 0 and offset_y == 0:
            projected = int(np.count_nonzero(inside))
    return projected


def write_dmb(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(array, dtype=np.float32)
    with path.open("wb") as handle:
        handle.write(struct.pack("=4i", 1, values.shape[0], values.shape[1], 5))
        values.tofile(handle)


def main():
    args = parse_args()
    dense_folder = args.dense_folder.resolve()
    mlp_path = args.ground_truth_mlp.resolve()
    selected_ids = set(args.views) if args.views else None
    views = load_views(dense_folder, selected_ids)
    scan_entries = read_scan_entries(mlp_path)
    offsets = splat_offsets(args.splat_radius)
    projected_counts = {view.image_id: 0 for view in views}

    print(f"Reference views: {len(views)}")
    print(f"Official scans: {len(scan_entries)}")
    print(f"Splat radius: {args.splat_radius} px")
    for scan_path, global_from_scan in scan_entries:
        print(f"Loading {scan_path}")
        for local_points in iter_scan_chunks(scan_path, args.chunk_size):
            world_points = (
                local_points @ global_from_scan[:3, :3].T + global_from_scan[:3, 3]
            )
            for view in views:
                projected_counts[view.image_id] += project_chunk(world_points, view, offsets)

    for view in views:
        depth = view.depth.reshape(view.height, view.width)
        depth[~np.isfinite(depth)] = 0.0
        output = args.output / f"ref_{view.image_id:08d}" / "gt_depth.dmb"
        write_dmb(output, depth)
        valid_count = int(np.count_nonzero(depth > 0.0))
        valid_ratio = valid_count / depth.size
        print(
            f"{view.image_id:08d}: projected={projected_counts[view.image_id]}, "
            f"valid_pixels={valid_count}/{depth.size} ({valid_ratio:.2%})"
        )


if __name__ == "__main__":
    main()

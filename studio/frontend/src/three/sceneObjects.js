import * as THREE from 'three';

const CAMERA_COLOR = 0x6fa8dc;
const SELECTED_CAMERA_COLOR = 0xffcc33;

export function parseAsciiPly(text) {
  const lines = text.split(/\r?\n/);
  let vertexCount = 0;
  let dataStart = -1;

  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].startsWith('element vertex ')) {
      vertexCount = Number(lines[index].split(' ').pop());
    }
    if (lines[index] === 'end_header') {
      dataStart = index + 1;
      break;
    }
  }

  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  let actualCount = 0;
  for (let index = 0; index < vertexCount && dataStart + index < lines.length; index += 1) {
    const values = lines[dataStart + index].trim().split(/\s+/).map(Number);
    if (values.length < 3 || !Number.isFinite(values[0])) continue;
    const offset = actualCount * 3;
    positions[offset] = values[0];
    positions[offset + 1] = values[1];
    positions[offset + 2] = values[2];
    colors[offset] = (values[3] ?? 200) / 255;
    colors[offset + 1] = (values[4] ?? 200) / 255;
    colors[offset + 2] = (values[5] ?? 200) / 255;
    actualCount += 1;
  }
  return {
    positions: positions.slice(0, actualCount * 3),
    colors: colors.slice(0, actualCount * 3),
  };
}

export function createPointCloudGeometry(text) {
  const {positions, colors} = parseAsciiPly(text);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  return geometry;
}

export function createPointCloud(geometry, name, size) {
  const material = new THREE.PointsMaterial({
    size,
    vertexColors: true,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, material);
  points.name = name;
  return points;
}

export function createCameraFrustum(view) {
  const center = new THREE.Vector3(...view.c);
  const depth = Math.max((view.depth_min || 0.1) * 0.015, 0.04);
  const corners = [[0, 0], [view.width, 0], [view.width, view.height], [0, view.height]];
  const points = [center];

  for (const [u, v] of corners) {
    const x = ((u - view.K[2]) / view.K[0]) * depth;
    const y = ((v - view.K[5]) / view.K[4]) * depth;
    points.push(new THREE.Vector3(
      view.R[0] * x + view.R[3] * y + view.R[6] * depth + center.x,
      view.R[1] * x + view.R[4] * y + view.R[7] * depth + center.y,
      view.R[2] * x + view.R[5] * y + view.R[8] * depth + center.z,
    ));
  }

  const segments = [];
  for (let index = 1; index <= 4; index += 1) {
    segments.push(points[0], points[index]);
    segments.push(points[index], points[index === 4 ? 1 : index + 1]);
  }
  const frustum = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(segments),
    new THREE.LineBasicMaterial({color: CAMERA_COLOR}),
  );
  frustum.userData = {type: 'camera', id: view.id};
  return frustum;
}

export function updateCameraSelection(cameraObjects, selectedId) {
  for (const object of cameraObjects) {
    object.material.color.setHex(object.userData.id === selectedId
      ? SELECTED_CAMERA_COLOR
      : CAMERA_COLOR);
  }
}

function worldPoint(view, u, v, depth) {
  if (!view || !Number.isFinite(depth) || depth <= 0) return null;
  const x = depth * (u - view.K[2]) / view.K[0];
  const y = depth * (v - view.K[5]) / view.K[4];
  return new THREE.Vector3(
    view.R[0] * x + view.R[3] * y + view.R[6] * depth + view.c[0],
    view.R[1] * x + view.R[4] * y + view.R[7] * depth + view.c[1],
    view.R[2] * x + view.R[5] * y + view.R[8] * depth + view.c[2],
  );
}

function worldNormal(view, normal) {
  if (!view || !normal || normal.length < 3) return null;
  const result = new THREE.Vector3(
    view.R[0] * normal[0] + view.R[3] * normal[1] + view.R[6] * normal[2],
    view.R[1] * normal[0] + view.R[4] * normal[1] + view.R[7] * normal[2],
    view.R[2] * normal[0] + view.R[5] * normal[1] + view.R[8] * normal[2],
  );
  return result.lengthSq() > 1e-12 ? result.normalize() : null;
}

function addMarker(group, position, color, radius) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 12, 8),
    new THREE.MeshBasicMaterial({color}),
  );
  marker.position.copy(position);
  group.add(marker);
}

function addLine(group, from, to, color, opacity = 1) {
  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([from, to]),
    new THREE.LineBasicMaterial({color, transparent: opacity < 1, opacity}),
  ));
}

export function drawPixelDebug(group, view, pixel) {
  const point = worldPoint(view, pixel.x, pixel.y, Number(pixel.depth));
  if (!point) return;

  addMarker(group, point, 0xffdf59, 0.012);
  addLine(group, new THREE.Vector3(...view.c), point, 0xffdf59);

  for (const anchor of pixel.anchor_details ?? []) {
    const anchorPoint = worldPoint(view, anchor.x, anchor.y, Number(anchor.depth));
    if (!anchorPoint) continue;
    const color = anchor.same_surface ? 0x55d68b : 0xff6b6b;
    addMarker(group, anchorPoint, color, 0.01);
    addLine(group, point, anchorPoint, color, 0.65);
  }

  if (pixel.fitted_plane?.plane) {
    const normal = worldNormal(view, pixel.fitted_plane.plane.slice(0, 3));
    if (normal) {
      const size = Math.max(0.05, Number(pixel.depth || 1) * 0.03);
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(size, size),
        new THREE.MeshBasicMaterial({
          color: 0x62b6ff,
          transparent: true,
          opacity: 0.28,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      plane.position.copy(point);
      plane.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
      group.add(plane);
    }
  }

  if (Number(pixel.gt_depth) > 0) {
    const gtPoint = worldPoint(view, pixel.x, pixel.y, Number(pixel.gt_depth));
    if (gtPoint) addMarker(group, gtPoint, 0xff7c43, 0.009);
  }
}

export function clearGroup(group, disposeGeometry = true) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((object) => {
      if (disposeGeometry) object.geometry?.dispose();
      if (Array.isArray(object.material)) {
        object.material.forEach((material) => material.dispose());
      } else {
        object.material?.dispose();
      }
    });
  }
}

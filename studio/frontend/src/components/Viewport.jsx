import React, {useEffect, useRef} from 'react';
import * as THREE from 'three';
import {requestResource, resourceKey, versionedUrl} from '../cache/resourceCache';
import {
  clearGroup,
  createCameraFrustum,
  createPointCloud,
  createPointCloudGeometry,
  drawPixelDebug,
  updateCameraSelection,
} from '../three/sceneObjects';

function removeCloud(group, cloud) {
  if (!cloud) return;
  group.remove(cloud);
  cloud.material.dispose();
}

export default function Viewport({
  caseName,
  cacheToken,
  manifest,
  selectedId,
  onSelect,
  showDpe,
  showGt,
  showCameras,
  pixelData,
}) {
  const hostRef = useRef(null);
  const runtimeRef = useRef(null);
  const geometryCacheRef = useRef(new Map());
  const onSelectRef = useRef(onSelect);

  useEffect(() => { onSelectRef.current = onSelect; }, [onSelect]);

  useEffect(() => {
    if (!hostRef.current) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101419);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10000);
    camera.position.set(2, 2, 2);
    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    hostRef.current.replaceChildren(renderer.domElement);

    const staticGroup = new THREE.Group();
    const cameraGroup = new THREE.Group();
    const debugGroup = new THREE.Group();
    staticGroup.name = 'static';
    cameraGroup.name = 'cameras';
    debugGroup.name = 'debug';
    scene.add(staticGroup, cameraGroup, debugGroup, new THREE.AxesHelper(0.5));

    const runtime = {
      scene,
      camera,
      renderer,
      target: new THREE.Vector3(),
      staticGroup,
      cameraGroup,
      debugGroup,
      cameraObjects: [],
      dpeCloud: null,
      gtCloud: null,
      showDpe,
      showGt,
      cloudGeneration: 0,
      destroyed: false,
    };
    runtimeRef.current = runtime;

    let dragging = false;
    let pointerX = 0;
    let pointerY = 0;
    let moved = 0;
    const pointerDown = (event) => {
      dragging = true;
      pointerX = event.clientX;
      pointerY = event.clientY;
      moved = 0;
    };
    const pointerUp = () => { dragging = false; };
    const pointerMove = (event) => {
      if (!dragging) return;
      const deltaX = event.clientX - pointerX;
      const deltaY = event.clientY - pointerY;
      moved += Math.abs(deltaX) + Math.abs(deltaY);
      pointerX = event.clientX;
      pointerY = event.clientY;
      const offset = camera.position.clone().sub(runtime.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= deltaX * 0.005;
      spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi - deltaY * 0.005));
      camera.position.copy(runtime.target.clone().add(new THREE.Vector3().setFromSpherical(spherical)));
      camera.lookAt(runtime.target);
    };
    const wheel = (event) => {
      const scale = Math.exp(event.deltaY * 0.001);
      camera.position.copy(runtime.target.clone().add(
        camera.position.clone().sub(runtime.target).multiplyScalar(scale),
      ));
    };
    const click = (event) => {
      if (moved > 6) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
        -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.params.Line.threshold = 0.03;
      raycaster.setFromCamera(mouse, camera);
      const [hit] = raycaster.intersectObjects(runtime.cameraObjects, false);
      if (hit) onSelectRef.current(hit.object.userData.id);
    };

    renderer.domElement.addEventListener('pointerdown', pointerDown);
    renderer.domElement.addEventListener('wheel', wheel, {passive: true});
    renderer.domElement.addEventListener('click', click);
    window.addEventListener('pointerup', pointerUp);
    window.addEventListener('pointermove', pointerMove);

    const resize = () => {
      const width = hostRef.current?.clientWidth || 1;
      const height = hostRef.current?.clientHeight || 1;
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(hostRef.current);
    resize();

    let animationFrame;
    const render = () => {
      animationFrame = requestAnimationFrame(render);
      renderer.render(scene, camera);
    };
    render();

    return () => {
      runtime.destroyed = true;
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointerdown', pointerDown);
      renderer.domElement.removeEventListener('wheel', wheel);
      renderer.domElement.removeEventListener('click', click);
      window.removeEventListener('pointerup', pointerUp);
      window.removeEventListener('pointermove', pointerMove);
      clearGroup(staticGroup, false);
      clearGroup(cameraGroup);
      clearGroup(debugGroup);
      for (const entry of geometryCacheRef.current.values()) entry.geometry?.dispose();
      renderer.dispose();
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime || !caseName || !cacheToken || !manifest) return undefined;

    const generation = ++runtime.cloudGeneration;
    removeCloud(runtime.staticGroup, runtime.dpeCloud);
    removeCloud(runtime.staticGroup, runtime.gtCloud);
    runtime.dpeCloud = null;
    runtime.gtCloud = null;

    const loadGeometry = (name, path) => {
      const key = resourceKey(cacheToken, caseName, null, name);
      let entry = geometryCacheRef.current.get(key);
      if (!entry) {
        entry = {geometry: null, promise: null};
        entry.promise = requestResource(key, versionedUrl(path, cacheToken))
          .then((blob) => blob.text())
          .then((text) => {
            if (runtime.destroyed) return null;
            entry.geometry = createPointCloudGeometry(text);
            return entry.geometry;
          })
          .catch(() => null);
        geometryCacheRef.current.set(key, entry);
      }
      return entry.promise;
    };

    const encodedCase = encodeURIComponent(caseName);
    const dpeGeometry = loadGeometry('dpe-point-cloud', `/api/cases/${encodedCase}/point-cloud`);
    const gtGeometry = manifest.telemetry
      ? loadGeometry('gt-point-cloud', `/api/cases/${encodedCase}/gt-point-cloud`)
      : Promise.resolve(null);

    Promise.all([dpeGeometry, gtGeometry]).then(([dpe, gt]) => {
      if (runtime.destroyed || generation !== runtime.cloudGeneration) return;
      if (dpe) {
        runtime.dpeCloud = createPointCloud(dpe, 'DPE', 0.006);
        runtime.dpeCloud.visible = runtime.showDpe;
        runtime.staticGroup.add(runtime.dpeCloud);
      }
      if (gt) {
        runtime.gtCloud = createPointCloud(gt, 'GT', 0.004);
        runtime.gtCloud.visible = runtime.showGt;
        runtime.staticGroup.add(runtime.gtCloud);
      }

      const sphere = dpe?.boundingSphere ?? gt?.boundingSphere;
      if (sphere) {
        runtime.camera.position.copy(sphere.center).add(new THREE.Vector3(
          sphere.radius * 1.4,
          sphere.radius * 0.9,
          sphere.radius * 1.4,
        ));
        runtime.camera.lookAt(sphere.center);
        runtime.target.copy(sphere.center);
      }
    });

    return () => { runtime.cloudGeneration += 1; };
  }, [caseName, cacheToken, manifest?.telemetry]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    clearGroup(runtime.cameraGroup);
    runtime.cameraObjects = (manifest?.views ?? []).map((view) => {
      const frustum = createCameraFrustum(view);
      runtime.cameraGroup.add(frustum);
      return frustum;
    });
    runtime.cameraGroup.visible = showCameras;
    updateCameraSelection(runtime.cameraObjects, selectedId);
  }, [manifest]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (runtime) updateCameraSelection(runtime.cameraObjects, selectedId);
  }, [selectedId]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.showDpe = showDpe;
    if (runtime.dpeCloud) runtime.dpeCloud.visible = showDpe;
  }, [showDpe]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.showGt = showGt;
    if (runtime.gtCloud) runtime.gtCloud.visible = showGt;
  }, [showGt]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (runtime) runtime.cameraGroup.visible = showCameras;
  }, [showCameras]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    clearGroup(runtime.debugGroup);
    const view = manifest?.views?.find((item) => item.id === selectedId);
    if (view && pixelData?._viewId === selectedId) {
      drawPixelDebug(runtime.debugGroup, view, pixelData);
    }
  }, [manifest, selectedId, pixelData]);

  return <div className="viewport" ref={hostRef}/>;
}

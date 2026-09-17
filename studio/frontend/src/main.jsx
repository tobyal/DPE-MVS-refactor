import React, {useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import ImageInspector from './components/ImageInspector';
import Viewport from './components/Viewport';
import {prefetchView} from './cache/prefetch';
import './style.css';

async function api(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

const formatNumber = (value, digits = 4) => (
  value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits)
);

function PixelInspector({view, pixel}) {
  if (!view) {
    return <div className="inspector"><h3>Inspector</h3><p>No camera selected.</p></div>;
  }
  const state = ['WEAK', 'STRONG', 'UNKNOWN'][Number(pixel?.state)] ?? '—';
  return (
    <div className="inspector">
      <h3>Camera {String(view.id).padStart(8, '0')}</h3>
      <dl>
        <dt>Size</dt><dd>{view.width} × {view.height}</dd>
        <dt>Center</dt><dd>{view.c.map((value) => value.toFixed(3)).join(', ')}</dd>
        <dt>Sources</dt><dd>{view.sources.join(', ')}</dd>
      </dl>
      <h3>Pixel</h3>
      {!pixel ? <p>Click the image to inspect one pixel.</p> : (
        <dl>
          <dt>xy</dt><dd>{pixel.x}, {pixel.y}</dd>
          <dt>State</dt><dd>{state}</dd>
          <dt>DPE depth</dt><dd>{formatNumber(pixel.depth)} m</dd>
          <dt>GT depth</dt><dd>{formatNumber(pixel.gt_depth)} m</dd>
          <dt>Depth error</dt><dd>{formatNumber(pixel.final_depth_error)} m</dd>
          <dt>GT surface</dt><dd>{pixel.gt_surface_label ?? '—'}</dd>
          <dt>Geometry edge</dt><dd>{pixel.gt_geometry_edge ? 'yes' : 'no'}</dd>
          <dt>Matching cost</dt><dd>{formatNumber(pixel.matching_cost)}</dd>
          <dt>PRE candidates</dt><dd>{pixel.candidate_count ?? '—'} / same {pixel.same_surface_candidates ?? '—'}</dd>
          <dt>Anchors</dt><dd>{pixel.anchor_count ?? '—'} / same {pixel.same_surface_anchors ?? '—'}</dd>
          <dt>Plane depth err</dt><dd>{formatNumber(pixel.plane_depth_error)} m</dd>
          <dt>Plane normal err</dt><dd>{formatNumber(pixel.plane_normal_error, 2)}°</dd>
          <dt>Adaptive radius</dt><dd>{pixel.adaptive_radius ?? '—'} px</dd>
          <dt>Radius violation</dt>
          <dd>{pixel.radius_violation == null ? '—' : `${(Number(pixel.radius_violation) * 100).toFixed(1)}%`}</dd>
        </dl>
      )}
    </div>
  );
}

function Metrics({row}) {
  if (!row) return <div className="metrics"><span>Telemetry unavailable for this view</span></div>;
  return (
    <div className="metrics">
      <div><b>{(row.within_2cm * 100).toFixed(1)}%</b><span>GT pixels ≤2 cm</span></div>
      <div><b>{(row.within_10cm * 100).toFixed(1)}%</b><span>GT pixels ≤10 cm</span></div>
      <div><b>{(row.anchor_same_surface_ratio * 100).toFixed(1)}%</b><span>same-surface anchors</span></div>
      <div><b>{row.plane_normal_error_deg.toFixed(1)}°</b><span>plane normal error</span></div>
      <div><b>{(row.radius_violation * 100).toFixed(1)}%</b><span>patch violation</span></div>
    </div>
  );
}

function App() {
  const [cases, setCases] = useState([]);
  const [caseName, setCaseName] = useState('');
  const [manifest, setManifest] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [status, setStatus] = useState({phase: 'idle', completed: 0, total: 0});
  const [pixel, setPixel] = useState(null);
  const [summary, setSummary] = useState({views: []});
  const [showDpe, setShowDpe] = useState(true);
  const [showGt, setShowGt] = useState(true);
  const [showCameras, setShowCameras] = useState(true);
  const pixelCacheRef = useRef(new Map());
  const selectionRef = useRef(null);

  useEffect(() => {
    api('/api/cases').then((items) => {
      setCases(items);
      if (items[0]) setCaseName(items[0].name);
    });
  }, []);

  useEffect(() => {
    if (!caseName) return undefined;
    let active = true;
    setManifest(null);
    setSelectedId(null);
    setPixel(null);
    Promise.all([
      api(`/api/cases/${encodeURIComponent(caseName)}/manifest`),
      api(`/api/cases/${encodeURIComponent(caseName)}/summary`),
    ]).then(([nextManifest, nextSummary]) => {
      if (!active) return;
      setManifest(nextManifest);
      setSummary(nextSummary);
      setSelectedId(nextManifest.views?.[0]?.id ?? null);
    }).catch((error) => console.error(error));
    return () => { active = false; };
  }, [caseName]);

  useEffect(() => { setPixel(null); }, [selectedId]);

  useEffect(() => {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${location.host}/ws/status`);
    socket.onmessage = (event) => {
      try { setStatus(JSON.parse(event.data)); } catch { /* Ignore partial status writes. */ }
    };
    return () => socket.close();
  }, []);

  const view = useMemo(
    () => manifest?.views?.find((item) => item.id === selectedId),
    [manifest, selectedId],
  );
  const metric = useMemo(
    () => summary.views?.find((item) => item.id === selectedId),
    [summary, selectedId],
  );
  const cacheToken = manifest?.cache_token ?? 'unversioned';
  const layers = manifest?.layers ?? ['rgb', 'depth', 'normal', 'state'];
  const progress = status.total ? Math.round((status.completed / status.total) * 100) : 0;
  selectionRef.current = {caseName, selectedId, cacheToken};

  useEffect(() => {
    if (!caseName || selectedId == null || !manifest) return undefined;
    return prefetchView({caseName, viewId: selectedId, layers, cacheToken});
  }, [caseName, selectedId, manifest, cacheToken]);

  const inspectPixel = async (x, y) => {
    if (!caseName || !view) return;
    const requested = {caseName, viewId: view.id, cacheToken};
    const key = [cacheToken, caseName, view.id, x, y].join(':');
    let request = pixelCacheRef.current.get(key);
    if (!request) {
      request = api(`/api/cases/${encodeURIComponent(caseName)}/views/${view.id}/pixel?x=${x}&y=${y}`)
        .then((result) => ({...result, _viewId: view.id}));
      pixelCacheRef.current.set(key, request);
      request.catch(() => pixelCacheRef.current.delete(key));
    }
    try {
      const result = await request;
      const current = selectionRef.current;
      if (current.caseName === requested.caseName
          && current.selectedId === requested.viewId
          && current.cacheToken === requested.cacheToken) {
        setPixel(result);
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="app">
      <header>
        <div className="brand">DPE Studio <small>GT Telemetry</small></div>
        <select value={caseName} onChange={(event) => setCaseName(event.target.value)}>
          {cases.map((item) => <option key={item.name}>{item.name}</option>)}
        </select>
        <div className="run">
          <span>{status.case || 'idle'} · {status.phase}</span>
          <div className="bar"><i style={{width: `${progress}%`}}/></div>
          <b>{progress}%</b>
        </div>
      </header>
      <Metrics row={metric}/>
      <main>
        <aside className="tree">
          <h3>Scene</h3>
          <section>
            <b>Cameras</b>
            <div className="cameraList">
              {manifest?.views?.map((item) => (
                <button
                  key={item.id}
                  className={item.id === selectedId ? 'selected' : ''}
                  onClick={() => setSelectedId(item.id)}
                >
                  {String(item.id).padStart(8, '0')}
                </button>
              ))}
            </div>
          </section>
          <section>
            <b>Geometry</b>
            <label><input type="checkbox" checked={showDpe} onChange={(event) => setShowDpe(event.target.checked)}/> DPE point cloud</label>
            <label><input type="checkbox" checked={showGt} onChange={(event) => setShowGt(event.target.checked)} disabled={!manifest?.telemetry}/> ETH3D GT scan</label>
            <label><input type="checkbox" checked={showCameras} onChange={(event) => setShowCameras(event.target.checked)}/> Camera frusta</label>
          </section>
          <section>
            <b>Legend</b>
            <small>
              <i className="dot yellow"/> selected pixel<br/>
              <i className="dot green"/> same-surface anchor<br/>
              <i className="dot red"/> cross-surface anchor<br/>
              <i className="dot orange"/> GT point
            </small>
          </section>
        </aside>
        <div className="center">
          <Viewport
            caseName={caseName}
            cacheToken={cacheToken}
            manifest={manifest}
            selectedId={selectedId}
            onSelect={setSelectedId}
            showDpe={showDpe}
            showGt={showGt}
            showCameras={showCameras}
            pixelData={pixel}
          />
          <ImageInspector
            caseName={caseName}
            cacheToken={cacheToken}
            view={view}
            layers={layers}
            pixelData={pixel}
            onPixel={inspectPixel}
          />
        </div>
        <PixelInspector view={view} pixel={pixel}/>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App/>);

import React, {useEffect, useMemo, useState} from 'react';
import {requestResource, resourceKey, versionedUrl} from '../cache/resourceCache';

function useResourceUrl(key, url) {
  const [objectUrl, setObjectUrl] = useState('');

  useEffect(() => {
    let active = true;
    let createdUrl = '';
    setObjectUrl('');
    if (!key || !url) return () => { active = false; };
    requestResource(key, url)
      .then((blob) => {
        if (!active) return;
        createdUrl = URL.createObjectURL(blob);
        setObjectUrl(createdUrl);
      })
      .catch((error) => console.error(error));

    return () => {
      active = false;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [key, url]);

  return objectUrl;
}

export default function ImageInspector({
  caseName,
  cacheToken,
  view,
  layers,
  pixelData,
  onPixel,
}) {
  const availableLayers = useMemo(() => layers?.length ? layers : ['rgb', 'depth', 'normal', 'state'], [layers]);
  const [layer, setLayer] = useState('rgb');

  useEffect(() => {
    setLayer((current) => availableLayers.includes(current) ? current : 'rgb');
  }, [caseName, availableLayers]);

  const encodedCase = encodeURIComponent(caseName);
  const path = !view ? '' : layer === 'rgb'
    ? `/api/cases/${encodedCase}/views/${view.id}/image`
    : `/api/cases/${encodedCase}/views/${view.id}/layer/${encodeURIComponent(layer)}`;
  const key = view ? resourceKey(cacheToken, caseName, view.id, layer) : '';
  const src = useResourceUrl(key, path ? versionedUrl(path, cacheToken) : '');

  if (!view) return <div className="imageEmpty">Select a camera</div>;

  const inspectPixel = (event) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(
      view.width - 1,
      Math.round(((event.clientX - bounds.left) / bounds.width) * view.width),
    ));
    const y = Math.max(0, Math.min(
      view.height - 1,
      Math.round(((event.clientY - bounds.top) / bounds.height) * view.height),
    ));
    onPixel(x, y);
  };

  const radius = Number(pixelData?.adaptive_radius || pixelData?.fitted_plane?.radius || 0);
  const pixelX = pixelData?.x;
  const pixelY = pixelData?.y;
  const pixelMatchesView = pixelData?._viewId === view.id;

  return (
    <div className="imagePanel">
      <div className="tabs layerTabs">
        {availableLayers.map((name) => (
          <button
            className={layer === name ? 'active' : ''}
            onClick={() => setLayer(name)}
            key={name}
          >
            {name.replaceAll('_', ' ')}
          </button>
        ))}
      </div>
      <div className="imgWrap" onClick={inspectPixel}>
        {src && <img src={src}/>}
        {pixelMatchesView && (
          <svg
            className="overlay"
            viewBox={`0 0 ${view.width} ${view.height}`}
            preserveAspectRatio="none"
          >
            <circle cx={pixelX} cy={pixelY} r={Math.max(3, view.width / 500)} className="selPixel"/>
            {radius > 0 && (
              <rect
                x={pixelX - radius}
                y={pixelY - radius}
                width={radius * 2}
                height={radius * 2}
                className="patchBox"
              />
            )}
            {(pixelData.anchor_details ?? []).map((anchor, index) => (
              <g key={index}>
                <line
                  x1={pixelX}
                  y1={pixelY}
                  x2={anchor.x}
                  y2={anchor.y}
                  className={anchor.same_surface ? 'anchorGood' : 'anchorBad'}
                />
                <circle
                  cx={anchor.x}
                  cy={anchor.y}
                  r={Math.max(3, view.width / 650)}
                  className={anchor.same_surface ? 'anchorGoodFill' : 'anchorBadFill'}
                />
              </g>
            ))}
          </svg>
        )}
        {pixelMatchesView && <div className="pixelBadge">({pixelX}, {pixelY})</div>}
      </div>
    </div>
  );
}

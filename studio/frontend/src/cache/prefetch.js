import {requestResource, resourceKey, versionedUrl} from './resourceCache';

const MAX_CONCURRENT_REQUESTS = 3;
const queue = [];
let activeRequests = 0;

function drainQueue() {
  while (activeRequests < MAX_CONCURRENT_REQUESTS && queue.length > 0) {
    const task = queue.shift();
    if (task.cancelled()) continue;

    activeRequests += 1;
    requestResource(task.key, task.url)
      .catch(() => {})
      .finally(() => {
        activeRequests -= 1;
        drainQueue();
      });
  }
}

export function prefetchView({caseName, viewId, layers, cacheToken}) {
  let cancelled = false;
  const available = new Set(layers ?? []);
  const priority = ['rgb', 'depth', 'state', 'final_depth_error'];
  const orderedLayers = [
    ...priority.filter((layer) => available.has(layer)),
    ...[...available].filter((layer) => !priority.includes(layer)),
  ];

  for (const layer of orderedLayers) {
    const path = layer === 'rgb'
      ? `/api/cases/${encodeURIComponent(caseName)}/views/${viewId}/image`
      : `/api/cases/${encodeURIComponent(caseName)}/views/${viewId}/layer/${encodeURIComponent(layer)}`;
    queue.push({
      key: resourceKey(cacheToken, caseName, viewId, layer),
      url: versionedUrl(path, cacheToken),
      cancelled: () => cancelled,
    });
  }
  drainQueue();
  return () => { cancelled = true; };
}

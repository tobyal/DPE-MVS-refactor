import {readPersistentResource, writePersistentResource} from './persistentCache';

const memoryCache = new Map();

export function resourceKey(cacheToken, caseName, viewId, resourceName) {
  return [cacheToken, caseName, viewId ?? 'case', resourceName].join(':');
}

export function versionedUrl(url, cacheToken) {
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}v=${encodeURIComponent(cacheToken)}`;
}

export function requestResource(key, url) {
  const cached = memoryCache.get(key);
  if (cached) return cached;

  const request = (async () => {
    const persistent = await readPersistentResource(key);
    if (persistent) return persistent;

    const response = await fetch(url);
    if (!response.ok) throw new Error(await response.text());
    const blob = await response.blob();
    void writePersistentResource(key, blob);
    return blob;
  })();

  memoryCache.set(key, request);
  request.catch(() => memoryCache.delete(key));
  return request;
}

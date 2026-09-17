const DATABASE_NAME = 'DPEStudioCache';
const STORE_NAME = 'resources';
const DATABASE_VERSION = 1;

let databasePromise;

function openDatabase() {
  if (!('indexedDB' in globalThis)) {
    return Promise.resolve(null);
  }
  if (!databasePromise) {
    databasePromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME, {keyPath: 'key'});
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    }).catch((error) => {
      console.warn('IndexedDB is unavailable; using memory cache only.', error);
      return null;
    });
  }
  return databasePromise;
}

export async function readPersistentResource(key) {
  const database = await openDatabase();
  if (!database) return null;

  try {
    return await new Promise((resolve) => {
      const request = database.transaction(STORE_NAME, 'readonly')
        .objectStore(STORE_NAME)
        .get(key);
      request.onsuccess = () => resolve(request.result?.blob ?? null);
      request.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

export async function writePersistentResource(key, blob) {
  const database = await openDatabase();
  if (!database) return;

  try {
    await new Promise((resolve) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite');
      transaction.objectStore(STORE_NAME).put({key, blob, timestamp: Date.now()});
      transaction.oncomplete = resolve;
      transaction.onerror = () => resolve();
      transaction.onabort = () => resolve();
    });
  } catch {
    // Persistent caching is opportunistic; the RAM cache remains usable.
  }
}

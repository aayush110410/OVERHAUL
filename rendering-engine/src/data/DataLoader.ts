/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Data Loader
 *
 * Lazy-loading system for GeoJSON, tiles, and streaming data.
 * Manages fetch lifecycle and caching.
 * ────────────────────────────────────────────────────────────────────────── */

import type { DataSource, TileCoord } from '../core/types';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  size: number;
}

export class DataLoader {
  private cache = new Map<string, CacheEntry<unknown>>();
  private inflight = new Map<string, Promise<unknown>>();
  private maxCacheSize = 200; // entries

  /* ── Fetch a data source ────────────────────────────────────────────── */

  async load<T>(source: DataSource<T>): Promise<T[]> {
    const key = source.id;

    // Return cached if fresh
    const cached = this.cache.get(key) as CacheEntry<T[]> | undefined;
    if (cached && (!source.refreshInterval || Date.now() - cached.timestamp < source.refreshInterval)) {
      return cached.data;
    }

    // Deduplicate concurrent requests
    if (this.inflight.has(key)) {
      return this.inflight.get(key) as Promise<T[]>;
    }

    const promise = this.fetchAndTransform(source);
    this.inflight.set(key, promise);

    try {
      const data = await promise;
      this.cache.set(key, { data, timestamp: Date.now(), size: 1 });
      this.evictIfNeeded();
      return data;
    } finally {
      this.inflight.delete(key);
    }
  }

  /* ── Tile fetching ──────────────────────────────────────────────────── */

  async loadTile(baseUrl: string, coord: TileCoord): Promise<Blob> {
    const key = `${baseUrl}/${coord.z}/${coord.x}/${coord.y}`;
    const cached = this.cache.get(key) as CacheEntry<Blob> | undefined;
    if (cached) return cached.data;

    const url = `${baseUrl}/${coord.z}/${coord.x}/${coord.y}.png`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Tile fetch failed: ${response.status}`);
    const blob = await response.blob();
    this.cache.set(key, { data: blob, timestamp: Date.now(), size: 1 });
    this.evictIfNeeded();
    return blob;
  }

  /* ── Cache management ───────────────────────────────────────────────── */

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }

  /* ── Internals ──────────────────────────────────────────────────────── */

  private async fetchAndTransform<T>(source: DataSource<T>): Promise<T[]> {
    const response = await fetch(source.url);
    if (!response.ok) throw new Error(`Data fetch failed: ${response.status} for ${source.url}`);

    let raw: unknown;
    switch (source.type) {
      case 'geojson':
      case 'stream':
        raw = await response.json();
        break;
      case 'csv':
        raw = await response.text();
        break;
      case 'binary':
        raw = await response.arrayBuffer();
        break;
      default:
        raw = await response.json();
    }

    if (source.transform) {
      return source.transform(raw);
    }
    return Array.isArray(raw) ? raw : [raw] as T[];
  }

  private evictIfNeeded(): void {
    if (this.cache.size <= this.maxCacheSize) return;
    // LRU eviction: delete oldest entries
    const entries = [...this.cache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
    const toEvict = entries.slice(0, this.cache.size - this.maxCacheSize);
    for (const [key] of toEvict) {
      this.cache.delete(key);
    }
  }
}

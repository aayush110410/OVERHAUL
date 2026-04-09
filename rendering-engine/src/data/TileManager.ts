/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Tile Manager
 *
 * Manages level-of-detail map tile loading / unloading based on the
 * Cesium camera viewport.
 * ────────────────────────────────────────────────────────────────────────── */

import type { TileCoord, GeoBounds } from '../core/types';
import { MAX_TILE_ZOOM, MIN_TILE_ZOOM, TILE_SIZE } from '../core/constants';
import { DataLoader } from './DataLoader';

export class TileManager {
  private loader = new DataLoader();
  private loadedTiles = new Set<string>();
  private baseUrl: string;

  constructor(baseUrl = 'https://tile.openstreetmap.org') {
    this.baseUrl = baseUrl;
  }

  /** Compute which tiles are needed for the given bounds and zoom. */
  getVisibleTiles(bounds: GeoBounds, zoom: number): TileCoord[] {
    const z = Math.max(MIN_TILE_ZOOM, Math.min(MAX_TILE_ZOOM, Math.round(zoom)));
    const n = Math.pow(2, z);

    const xMin = Math.floor(((bounds.west + 180) / 360) * n);
    const xMax = Math.floor(((bounds.east + 180) / 360) * n);
    const yMin = Math.floor(
      ((1 - Math.log(Math.tan((bounds.north * Math.PI) / 180) + 1 / Math.cos((bounds.north * Math.PI) / 180)) / Math.PI) / 2) * n,
    );
    const yMax = Math.floor(
      ((1 - Math.log(Math.tan((bounds.south * Math.PI) / 180) + 1 / Math.cos((bounds.south * Math.PI) / 180)) / Math.PI) / 2) * n,
    );

    const tiles: TileCoord[] = [];
    for (let x = Math.max(0, xMin); x <= Math.min(n - 1, xMax); x++) {
      for (let y = Math.max(0, yMin); y <= Math.min(n - 1, yMax); y++) {
        tiles.push({ x, y, z });
      }
    }
    return tiles;
  }

  /** Load tiles that aren't already cached. */
  async loadTiles(tiles: TileCoord[]): Promise<void> {
    const needed = tiles.filter((t) => !this.loadedTiles.has(this.key(t)));
    await Promise.allSettled(
      needed.map(async (t) => {
        await this.loader.loadTile(this.baseUrl, t);
        this.loadedTiles.add(this.key(t));
      }),
    );
  }

  /** Unload tiles that aren't in the given set. */
  pruneOutside(keepTiles: TileCoord[]): void {
    const keepSet = new Set(keepTiles.map((t) => this.key(t)));
    for (const key of this.loadedTiles) {
      if (!keepSet.has(key)) {
        this.loadedTiles.delete(key);
      }
    }
  }

  getLoadedCount(): number {
    return this.loadedTiles.size;
  }

  private key(t: TileCoord): string {
    return `${t.z}/${t.x}/${t.y}`;
  }
}

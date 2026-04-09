/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Level-of-Detail Manager
 *
 * Determines which LOD level each object should use based on the
 * camera's current altitude / distance.
 * ────────────────────────────────────────────────────────────────────────── */

import { LOD_LEVELS } from '../core/constants';
import type { LODLevel } from '../core/types';

export class LODManager {
  private levels: LODLevel[] = LOD_LEVELS;
  private currentLevel = 0;

  /** Call once per frame with the camera height (metres above ground). */
  update(cameraHeight: number): void {
    for (let i = 0; i < this.levels.length; i++) {
      if (cameraHeight < this.levels[i].distance) {
        this.currentLevel = i;
        return;
      }
    }
    this.currentLevel = this.levels.length - 1;
  }

  getCurrentLevel(): LODLevel {
    return this.levels[this.currentLevel];
  }

  getLevelIndex(): number {
    return this.currentLevel;
  }

  /** Get the recommended detail factor (0-1) for the current camera distance. */
  getDetailFactor(): number {
    return this.levels[this.currentLevel].geometryDetail;
  }

  /** Whether shadows should be enabled at the current LOD. */
  shouldEnableShadows(): boolean {
    return this.levels[this.currentLevel].shadowsEnabled;
  }

  /** Recommended max texture dimension for the current LOD. */
  getTextureSize(): number {
    return this.levels[this.currentLevel].textureSize;
  }
}

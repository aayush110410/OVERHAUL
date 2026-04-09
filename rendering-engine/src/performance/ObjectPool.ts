/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Object Pool
 *
 * Reusable-object pool to avoid GC pressure when thousands of
 * renderable objects are created & destroyed each frame.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';

interface PoolEntry {
  mesh: THREE.Mesh;
  inUse: boolean;
}

export class ObjectPool {
  private pools = new Map<string, PoolEntry[]>();

  /** Acquire a mesh from the named pool (or create a new one). */
  acquire(
    poolName: string,
    factory: () => THREE.Mesh,
  ): THREE.Mesh {
    let pool = this.pools.get(poolName);
    if (!pool) {
      pool = [];
      this.pools.set(poolName, pool);
    }

    // Find a free entry
    for (const entry of pool) {
      if (!entry.inUse) {
        entry.inUse = true;
        entry.mesh.visible = true;
        return entry.mesh;
      }
    }

    // None available — grow the pool
    const mesh = factory();
    pool.push({ mesh, inUse: true });
    return mesh;
  }

  /** Return a mesh to its pool. */
  release(poolName: string, mesh: THREE.Mesh): void {
    const pool = this.pools.get(poolName);
    if (!pool) return;
    const entry = pool.find((e) => e.mesh === mesh);
    if (entry) {
      entry.inUse = false;
      entry.mesh.visible = false;
    }
  }

  /** Pre-allocate N objects in a pool. */
  preallocate(poolName: string, count: number, factory: () => THREE.Mesh): void {
    let pool = this.pools.get(poolName);
    if (!pool) {
      pool = [];
      this.pools.set(poolName, pool);
    }
    for (let i = 0; i < count; i++) {
      const mesh = factory();
      mesh.visible = false;
      pool.push({ mesh, inUse: false });
    }
  }

  /** How many objects are allocated vs in-use. */
  getStats(poolName: string): { total: number; inUse: number } {
    const pool = this.pools.get(poolName);
    if (!pool) return { total: 0, inUse: 0 };
    return {
      total: pool.length,
      inUse: pool.filter((e) => e.inUse).length,
    };
  }

  /** Dispose all meshes in all pools. */
  disposeAll(): void {
    this.pools.forEach((pool) => {
      for (const entry of pool) {
        entry.mesh.geometry.dispose();
        const mat = entry.mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    });
    this.pools.clear();
  }
}

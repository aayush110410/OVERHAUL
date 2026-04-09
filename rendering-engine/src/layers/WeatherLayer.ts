/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Weather Layer
 *
 * Renders weather patterns as animated particle clouds and
 * colour-coded grid cells on the globe.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface WeatherRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  properties?: Record<string, unknown>;
}

const PARTICLE_COUNT = 5000;
const GRID_RES = 24; // grid cells per side

export class WeatherLayer extends BaseLayer {
  private particles: THREE.Points | null = null;
  private gridMesh: THREE.Mesh | null = null;
  private particleSpeeds: Float32Array = new Float32Array(PARTICLE_COUNT * 3);
  private time = 0;
  private liveRecords: WeatherRecord[] = [];

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    this.buildParticles();
    this.buildGrid();
    this.setExternalStats({ objectCount: PARTICLE_COUNT, source: 'procedural' });
  }

  protected onUpdate(dt: number): void {
    this.time += dt;
    this.animateParticles(dt);
    this.animateGrid();
  }

  protected onUnload(): void {
    if (this.particles) {
      this.scene.remove(this.particles);
      this.particles.geometry.dispose();
      (this.particles.material as THREE.Material).dispose();
      this.particles = null;
    }
    if (this.gridMesh) {
      this.scene.remove(this.gridMesh);
      this.gridMesh.geometry.dispose();
      (this.gridMesh.material as THREE.Material).dispose();
      this.gridMesh = null;
    }
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.particles) this.particles.visible = visible;
    if (this.gridMesh) this.gridMesh.visible = visible;
  }

  /* ── Particles (wind / cloud) ───────────────────────────────────────── */

  private buildParticles(): void {
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const colors = new Float32Array(PARTICLE_COUNT * 3);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Random positions on globe surface
      const lon = Math.random() * 360 - 180;
      const lat = Math.random() * 180 - 90;
      const alt = 500 + Math.random() * 15_000;

      const cart = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
      positions[i * 3]     = cart.x;
      positions[i * 3 + 1] = cart.y;
      positions[i * 3 + 2] = cart.z;

      // Wind velocities
      this.particleSpeeds[i * 3]     = (Math.random() - 0.5) * 50000;
      this.particleSpeeds[i * 3 + 1] = (Math.random() - 0.5) * 50000;
      this.particleSpeeds[i * 3 + 2] = (Math.random() - 0.5) * 10000;

      // Whitish-blue
      colors[i * 3]     = 0.8 + Math.random() * 0.2;
      colors[i * 3 + 1] = 0.9 + Math.random() * 0.1;
      colors[i * 3 + 2] = 1.0;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setDrawRange(0, PARTICLE_COUNT);

    const material = new THREE.PointsMaterial({
      size: 25_000,
      vertexColors: true,
      transparent: true,
      opacity: 0.35,
      sizeAttenuation: true,
      depthWrite: false,
    });

    this.particles = new THREE.Points(geometry, material);
    this.particles.frustumCulled = false;
    this.scene.add(this.particles);
  }

  private animateParticles(dt: number): void {
    if (!this.particles) return;
    const pos = this.particles.geometry.attributes.position as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;

    if (this.liveRecords.length > 0) {
      const colors = this.particles.geometry.attributes.color as THREE.BufferAttribute;
      const colorArr = colors.array as Float32Array;
      const drawCount = Math.min(this.liveRecords.length, PARTICLE_COUNT);
      this.particles.geometry.setDrawRange(0, drawCount);

      for (let i = 0; i < drawCount; i++) {
        const record = this.liveRecords[i];
        const position = record.position;
        if (!position) continue;
        const altitude = position.altitude ?? 1200;
        const cart = Cesium.Cartesian3.fromDegrees(position.longitude, position.latitude, altitude);
        arr[i * 3] = cart.x;
        arr[i * 3 + 1] = cart.y;
        arr[i * 3 + 2] = cart.z;

        const precipitation = Number(record.properties?.precipitation ?? 0);
        const cloudCover = Number(record.properties?.cloud_cover ?? 0);
        const temperature = Number(record.properties?.temperature_c ?? 0);
        const hue = THREE.MathUtils.clamp(0.64 - (temperature + 10) / 100, 0.03, 0.66);
        const lightness = THREE.MathUtils.clamp(0.35 + cloudCover / 220, 0.3, 0.72);
        const color = new THREE.Color().setHSL(hue, 0.72, lightness);
        colorArr[i * 3] = color.r + precipitation * 0.02;
        colorArr[i * 3 + 1] = color.g;
        colorArr[i * 3 + 2] = color.b + cloudCover * 0.002;
      }

      pos.needsUpdate = true;
      colors.needsUpdate = true;
      return;
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      arr[i * 3]     += this.particleSpeeds[i * 3] * dt;
      arr[i * 3 + 1] += this.particleSpeeds[i * 3 + 1] * dt;
      arr[i * 3 + 2] += this.particleSpeeds[i * 3 + 2] * dt;

      // Wrap around (keep on globe-ish shell)
      const len = Math.sqrt(arr[i * 3] ** 2 + arr[i * 3 + 1] ** 2 + arr[i * 3 + 2] ** 2);
      const targetR = 6_371_000 + 500 + Math.random() * 15_000;
      const scale = targetR / Math.max(len, 1);
      arr[i * 3]     *= scale;
      arr[i * 3 + 1] *= scale;
      arr[i * 3 + 2] *= scale;
    }
    pos.needsUpdate = true;
  }

  /* ── Grid overlay (temperature / precipitation) ─────────────────────── */

  private buildGrid(): void {
    const EARTH_R = 6_371_000;
    const vertices: number[] = [];
    const vertColors: number[] = [];
    const indices: number[] = [];

    for (let yi = 0; yi < GRID_RES; yi++) {
      for (let xi = 0; xi < GRID_RES; xi++) {
        const lon0 = -180 + (xi / GRID_RES) * 360;
        const lon1 = -180 + ((xi + 1) / GRID_RES) * 360;
        const lat0 = -90 + (yi / GRID_RES) * 180;
        const lat1 = -90 + ((yi + 1) / GRID_RES) * 180;
        const alt = 200;

        const corners: [number, number][] = [
          [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1],
        ];

        const base = vertices.length / 3;
        for (const [lon, lat] of corners) {
          const c = Cesium.Cartesian3.fromDegrees(lon, lat, alt);
          vertices.push(c.x, c.y, c.z);
          // Colour by pseudo temperature
          const temp = (Math.sin((lat + 90) / 180 * Math.PI) + Math.sin(lon / 45)) * 0.5;
          const color = new THREE.Color().setHSL(0.66 - temp * 0.66, 0.7, 0.5);
          vertColors.push(color.r, color.g, color.b);
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(vertColors, 3));
    geometry.setIndex(indices);

    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this.gridMesh = new THREE.Mesh(geometry, material);
    this.gridMesh.frustumCulled = false;
    this.scene.add(this.gridMesh);
  }

  private animateGrid(): void {
    if (!this.gridMesh) return;
    if (this.liveRecords.length > 0) {
      this.gridMesh.visible = false;
      return;
    }
    this.gridMesh.visible = this.config.visible;
    const colors = this.gridMesh.geometry.attributes.color as THREE.BufferAttribute;
    const arr = colors.array as Float32Array;

    for (let i = 0; i < arr.length / 3; i++) {
      const phase = this.time * 0.3 + i * 0.01;
      const temp = (Math.sin(phase) + 1) * 0.5;
      const color = new THREE.Color().setHSL(0.66 - temp * 0.66, 0.7, 0.5);
      arr[i * 3] = color.r;
      arr[i * 3 + 1] = color.g;
      arr[i * 3 + 2] = color.b;
    }
    colors.needsUpdate = true;
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: WeatherRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
    };
    this.liveRecords = payload?.items ?? [];
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.liveRecords.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }
}

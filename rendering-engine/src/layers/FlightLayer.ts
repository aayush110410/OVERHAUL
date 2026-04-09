/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Flight Layer
 *
 * Renders simulated commercial flights as instanced cones
 * following great-circle routes.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface FlightData {
  from: { lon: number; lat: number };
  to: { lon: number; lat: number };
  altitude: number;
  progress: number;  // 0-1
  speed: number;
}

interface FlightRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  properties?: Record<string, unknown>;
}

const FLIGHT_COUNT = 1800;
const DEG2RAD = Math.PI / 180;

export class FlightLayer extends BaseLayer {
  private instancedMesh: THREE.InstancedMesh | null = null;
  private flights: FlightData[] = [];
  private dummy = new THREE.Object3D();
  private trailLines: THREE.LineSegments | null = null;
  private liveFlights: FlightRecord[] = [];

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    // Cone geometry for aircraft markers
    const geometry = new THREE.ConeGeometry(15_000, 60_000, 4);
    geometry.rotateX(Math.PI / 2);
    const material = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.9,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, FLIGHT_COUNT);
    this.instancedMesh.frustumCulled = true;
    this.instancedMesh.count = 0;
    this.scene.add(this.instancedMesh);

    // Generate random flights
    const hubs = [
      { lon: 77.1, lat: 28.55 },   // DEL
      { lon: -0.46, lat: 51.47 },   // LHR
      { lon: -73.78, lat: 40.64 },  // JFK
      { lon: 103.99, lat: 1.36 },   // SIN
      { lon: 139.78, lat: 35.55 },  // NRT
      { lon: 55.36, lat: 25.25 },   // DXB
      { lon: -43.17, lat: -22.81 }, // GIG
      { lon: 151.18, lat: -33.95 }, // SYD
      { lon: 116.60, lat: 40.08 },  // PEK
      { lon: 2.55, lat: 49.01 },    // CDG
    ];

    for (let i = 0; i < 500; i++) {
      const from = hubs[Math.floor(Math.random() * hubs.length)];
      let to = from;
      while (to === from) to = hubs[Math.floor(Math.random() * hubs.length)];

      this.flights.push({
        from: { lon: from.lon + (Math.random() - 0.5) * 10, lat: from.lat + (Math.random() - 0.5) * 10 },
        to: { lon: to.lon + (Math.random() - 0.5) * 10, lat: to.lat + (Math.random() - 0.5) * 10 },
        altitude: 9_000 + Math.random() * 3_000,
        progress: Math.random(),
        speed: 0.01 + Math.random() * 0.02,
      });
    }

    // Trail lines
    this.buildTrails();
    this.setExternalStats({ objectCount: this.flights.length, source: 'procedural' });
  }

  protected onUpdate(dt: number): void {
    if (!this.instancedMesh) return;

    if (this.liveFlights.length > 0) {
      for (let i = 0; i < this.liveFlights.length && i < FLIGHT_COUNT; i++) {
        const flight = this.liveFlights[i];
        const position = flight.position;
        if (!position) continue;
        const cart = Cesium.Cartesian3.fromDegrees(
          position.longitude,
          position.latitude,
          position.altitude ?? 10_000,
        );

        this.dummy.position.set(cart.x, cart.y, cart.z);
        const heading = Number(flight.properties?.heading_deg ?? 0);
        this.dummy.rotation.set(0, 0, THREE.MathUtils.degToRad(heading));
        this.dummy.updateMatrix();
        this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
      }
      this.instancedMesh.count = Math.min(this.liveFlights.length, FLIGHT_COUNT);
      this.instancedMesh.instanceMatrix.needsUpdate = true;
      if (this.trailLines) this.trailLines.visible = false;
      return;
    }

    if (this.trailLines) this.trailLines.visible = this.config.visible;
    for (let i = 0; i < this.flights.length; i++) {
      const f = this.flights[i];
      f.progress += dt * f.speed;
      if (f.progress > 1) f.progress -= 1;

      // Interpolate great-circle position
      const pos = this.interpolateGreatCircle(f.from.lat, f.from.lon, f.to.lat, f.to.lon, f.progress);
      const cart = Cesium.Cartesian3.fromDegrees(pos.lon, pos.lat, f.altitude);

      this.dummy.position.set(cart.x, cart.y, cart.z);
      // Orient cone along trajectory
      const nextPos = this.interpolateGreatCircle(f.from.lat, f.from.lon, f.to.lat, f.to.lon, Math.min(1, f.progress + 0.01));
      const nextCart = Cesium.Cartesian3.fromDegrees(nextPos.lon, nextPos.lat, f.altitude);
      this.dummy.lookAt(nextCart.x, nextCart.y, nextCart.z);
      this.dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.instancedMesh.count = this.flights.length;
    this.instancedMesh.instanceMatrix.needsUpdate = true;
  }

  protected onUnload(): void {
    if (this.instancedMesh) {
      this.scene.remove(this.instancedMesh);
      this.instancedMesh.geometry.dispose();
      (this.instancedMesh.material as THREE.Material).dispose();
      this.instancedMesh = null;
    }
    if (this.trailLines) {
      this.scene.remove(this.trailLines);
      this.trailLines.geometry.dispose();
      (this.trailLines.material as THREE.Material).dispose();
      this.trailLines = null;
    }
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.instancedMesh) this.instancedMesh.visible = visible;
    if (this.trailLines) this.trailLines.visible = visible && this.liveFlights.length === 0;
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: FlightRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
    };
    this.liveFlights = payload?.items ?? [];
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.liveFlights.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }

  /* ── Great-circle interpolation ─────────────────────────────────────── */

  private interpolateGreatCircle(
    lat1: number, lon1: number, lat2: number, lon2: number, t: number,
  ): { lat: number; lon: number } {
    const φ1 = lat1 * DEG2RAD, λ1 = lon1 * DEG2RAD;
    const φ2 = lat2 * DEG2RAD, λ2 = lon2 * DEG2RAD;
    const d = 2 * Math.asin(Math.sqrt(
      Math.sin((φ2 - φ1) / 2) ** 2 +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin((λ2 - λ1) / 2) ** 2,
    ));
    if (d < 1e-10) return { lat: lat1, lon: lon1 };
    const A = Math.sin((1 - t) * d) / Math.sin(d);
    const B = Math.sin(t * d) / Math.sin(d);
    const x = A * Math.cos(φ1) * Math.cos(λ1) + B * Math.cos(φ2) * Math.cos(λ2);
    const y = A * Math.cos(φ1) * Math.sin(λ1) + B * Math.cos(φ2) * Math.sin(λ2);
    const z = A * Math.sin(φ1) + B * Math.sin(φ2);
    return { lat: Math.atan2(z, Math.sqrt(x * x + y * y)) / DEG2RAD, lon: Math.atan2(y, x) / DEG2RAD };
  }

  /* ── Trail arcs ─────────────────────────────────────────────────────── */

  private buildTrails(): void {
    const positions: number[] = [];
    const SEGMENTS = 20;
    for (const f of this.flights) {
      for (let s = 0; s < SEGMENTS; s++) {
        const t0 = s / SEGMENTS;
        const t1 = (s + 1) / SEGMENTS;
        const p0 = this.interpolateGreatCircle(f.from.lat, f.from.lon, f.to.lat, f.to.lon, t0);
        const p1 = this.interpolateGreatCircle(f.from.lat, f.from.lon, f.to.lat, f.to.lon, t1);
        const arc = f.altitude * Math.sin(t0 * Math.PI);
        const arc1 = f.altitude * Math.sin(t1 * Math.PI);
        const c0 = Cesium.Cartesian3.fromDegrees(p0.lon, p0.lat, arc);
        const c1 = Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, arc1);
        positions.push(c0.x, c0.y, c0.z, c1.x, c1.y, c1.z);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.15 });
    this.trailLines = new THREE.LineSegments(geometry, material);
    this.trailLines.frustumCulled = false;
    this.scene.add(this.trailLines);
  }
}

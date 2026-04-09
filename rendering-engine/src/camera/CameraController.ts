/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Camera Controller
 *
 * Provides orbit, pan, tilt, free-look and cinematic-transition modes
 * on top of the Cesium camera system.
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import type { CameraMode, CameraState, CameraTransition, GeoPosition, EasingFunction } from '../core/types';
import { DEFAULT_FOV } from '../core/constants';

export class CameraController {
  private viewer: Cesium.Viewer;
  private mode: CameraMode = 'orbit';
  private transitions: CameraTransition[] = [];
  private transitionProgress = 0;
  private transitionStart: CameraState | null = null;

  constructor(viewer: Cesium.Viewer) {
    this.viewer = viewer;
    // Enable default screen-space camera controller
    const sscController = this.viewer.scene.screenSpaceCameraController;
    sscController.enableRotate = true;
    sscController.enableTranslate = true;
    sscController.enableZoom = true;
    sscController.enableTilt = true;
    sscController.enableLook = true;
  }

  /* ── Mode switching ─────────────────────────────────────────────────── */

  setMode(mode: CameraMode): void {
    this.mode = mode;
    const ctrl = this.viewer.scene.screenSpaceCameraController;

    switch (mode) {
      case 'orbit':
        ctrl.enableRotate = true;
        ctrl.enableTranslate = false;
        ctrl.enableTilt = true;
        ctrl.enableZoom = true;
        break;
      case 'pan':
        ctrl.enableRotate = false;
        ctrl.enableTranslate = true;
        ctrl.enableTilt = false;
        ctrl.enableZoom = true;
        break;
      case 'tilt':
        ctrl.enableRotate = false;
        ctrl.enableTranslate = false;
        ctrl.enableTilt = true;
        ctrl.enableZoom = false;
        break;
      case 'free':
        ctrl.enableRotate = true;
        ctrl.enableTranslate = true;
        ctrl.enableTilt = true;
        ctrl.enableZoom = true;
        ctrl.enableLook = true;
        break;
      case 'cinematic':
        // Automated — disable manual controls
        ctrl.enableRotate = false;
        ctrl.enableTranslate = false;
        ctrl.enableTilt = false;
        ctrl.enableZoom = false;
        break;
    }
  }

  getMode(): CameraMode { return this.mode; }

  /* ── State ──────────────────────────────────────────────────────────── */

  getState(): CameraState {
    const cam = this.viewer.camera;
    const carto = cam.positionCartographic;
    return {
      position: {
        longitude: Cesium.Math.toDegrees(carto.longitude),
        latitude: Cesium.Math.toDegrees(carto.latitude),
        altitude: carto.height,
      },
      heading: Cesium.Math.toDegrees(cam.heading),
      pitch: Cesium.Math.toDegrees(cam.pitch),
      roll: Cesium.Math.toDegrees(cam.roll),
      fov: DEFAULT_FOV,
      mode: this.mode,
    };
  }

  /* ── Cinematic transitions ──────────────────────────────────────────── */

  queueTransition(transition: CameraTransition): void {
    this.transitions.push(transition);
  }

  flyTo(pos: GeoPosition, duration = 2, easing: EasingFunction = 'ease-in-out'): void {
    this.queueTransition({
      target: {
        position: pos,
      },
      duration,
      easing,
    });
  }

  cinematicOrbit(pos: GeoPosition, altitude: number, revolutions = 1, duration = 20): void {
    // Orbit around a point — queues multiple waypoints
    const STEPS = 36;
    const stepDuration = duration / STEPS;
    for (let i = 0; i <= STEPS; i++) {
      const angle = (i / STEPS) * 360 * revolutions;
      this.queueTransition({
        target: {
          position: pos,
          heading: angle,
        },
        duration: stepDuration,
        easing: 'linear',
      });
    }
  }

  cinematicZoom(from: GeoPosition, to: GeoPosition, duration = 5): void {
    this.queueTransition({
      target: { position: from },
      duration: duration * 0.1,
      easing: 'ease-out',
    });
    this.queueTransition({
      target: { position: to },
      duration: duration * 0.9,
      easing: 'ease-in-out-cubic',
    });
  }

  /* ── Per-frame update ───────────────────────────────────────────────── */

  update(dt: number): void {
    if (this.transitions.length === 0) return;

    const transition = this.transitions[0];
    if (!this.transitionStart) {
      this.transitionStart = this.getState();
      this.transitionProgress = 0;
    }

    this.transitionProgress += dt / transition.duration;
    const t = Math.min(1, this.transitionProgress);
    const easedT = this.ease(t, transition.easing);

    // Interpolate position
    if (transition.target.position && this.transitionStart) {
      const sp = this.transitionStart.position;
      const tp = transition.target.position;
      const lon = sp.longitude + (tp.longitude - sp.longitude) * easedT;
      const lat = sp.latitude + (tp.latitude - sp.latitude) * easedT;
      const alt = sp.altitude + (tp.altitude - sp.altitude) * easedT;

      const heading = this.lerpAngle(
        this.transitionStart.heading,
        transition.target.heading ?? this.transitionStart.heading,
        easedT,
      );
      const pitch = this.transitionStart.pitch +
        ((transition.target.pitch ?? this.transitionStart.pitch) - this.transitionStart.pitch) * easedT;

      this.viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
        orientation: {
          heading: Cesium.Math.toRadians(heading),
          pitch: Cesium.Math.toRadians(pitch),
          roll: 0,
        },
      });
    }

    if (t >= 1) {
      this.transitions.shift();
      this.transitionStart = null;
      this.transitionProgress = 0;
      if (transition.onComplete) transition.onComplete();
    }
  }

  /* ── Easing functions ───────────────────────────────────────────────── */

  private ease(t: number, fn: EasingFunction): number {
    switch (fn) {
      case 'linear': return t;
      case 'ease-in': return t * t;
      case 'ease-out': return t * (2 - t);
      case 'ease-in-out': return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      case 'ease-in-cubic': return t * t * t;
      case 'ease-out-cubic': return (--t) * t * t + 1;
      case 'ease-in-out-cubic':
        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
      default: return t;
    }
  }

  private lerpAngle(a: number, b: number, t: number): number {
    let diff = b - a;
    while (diff > 180) diff -= 360;
    while (diff < -180) diff += 360;
    return a + diff * t;
  }
}

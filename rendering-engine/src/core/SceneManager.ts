/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Three.js Scene Manager
 *
 * Manages a Three.js WebGLRenderer + Scene that is composited on top
 * of the Cesium globe.  All dynamic 3D objects (instanced meshes,
 * particle systems, etc.) live in this scene.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import * as Cesium from 'cesium';
import type { EngineConfig } from './types';

export class SceneManager {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;

  private canvas: HTMLCanvasElement;
  private cesiumViewer: Cesium.Viewer;

  constructor(cesiumViewer: Cesium.Viewer, config: EngineConfig) {
    this.cesiumViewer = cesiumViewer;

    // Create a transparent canvas overlay
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText =
      'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
    config.container.appendChild(this.canvas);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(config.container.clientWidth, config.container.clientHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = false;
    this.renderer.autoClear = false;

    // Scene
    this.scene = new THREE.Scene();

    // Camera (kept in sync with Cesium each frame)
    this.camera = new THREE.PerspectiveCamera(
      60,
      config.container.clientWidth / config.container.clientHeight,
      0.1,
      1e9,
    );
    this.scene.add(this.camera);

    // Ambient light
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(1, 1, 1).normalize();
    this.scene.add(directional);
  }

  /* ── Per-frame sync with Cesium camera ──────────────────────────────── */

  syncWithCesium(): void {
    const cesiumCamera = this.cesiumViewer.camera;

    // Copy Cesium's view matrix into the Three.js camera
    const vm = cesiumCamera.viewMatrix;
    const threeMatrix = new THREE.Matrix4();
    // Cesium uses column-major, Three.js uses column-major too
    threeMatrix.set(
      vm[0], vm[4], vm[8],  vm[12],
      vm[1], vm[5], vm[9],  vm[13],
      vm[2], vm[6], vm[10], vm[14],
      vm[3], vm[7], vm[11], vm[15],
    );

    this.camera.matrixAutoUpdate = false;
    this.camera.matrixWorld.copy(threeMatrix).invert();
    this.camera.matrixWorldInverse.copy(threeMatrix);

    // Copy projection
    const pm = cesiumCamera.frustum.projectionMatrix;
    this.camera.projectionMatrix.set(
      pm[0], pm[4], pm[8],  pm[12],
      pm[1], pm[5], pm[9],  pm[13],
      pm[2], pm[6], pm[10], pm[14],
      pm[3], pm[7], pm[11], pm[15],
    );
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }

  /* ── Render ─────────────────────────────────────────────────────────── */

  render(): void {
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);
  }

  /* ── Lifecycle ──────────────────────────────────────────────────────── */

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  getInfo(): THREE.WebGLInfo {
    return this.renderer.info;
  }

  destroy(): void {
    this.renderer.dispose();
    this.canvas.remove();
  }
}

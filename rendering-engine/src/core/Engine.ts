/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Main Rendering Engine
 *
 * Orchestrates the Cesium globe, Three.js scene, layer manager,
 * shader pipeline, camera controller and performance systems.
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import { Globe } from './Globe';
import { SceneManager } from './SceneManager';
import { LayerManager } from '../layers/LayerManager';
import { ShaderPipeline } from '../shaders/ShaderPipeline';
import { CameraController } from '../camera/CameraController';
import { LODManager } from '../performance/LODManager';
import { SpatialIndex } from '../performance/SpatialIndex';
import { ObjectPool } from '../performance/ObjectPool';
import type { EngineConfig, EngineState, PerformanceMetrics } from './types';

export class Engine {
  private config: EngineConfig;
  private globe!: Globe;
  private sceneManager!: SceneManager;
  private layerManager!: LayerManager;
  private shaderPipeline!: ShaderPipeline;
  private cameraController!: CameraController;
  private lodManager!: LODManager;
  private spatialIndex!: SpatialIndex;
  private objectPool!: ObjectPool;
  private running = false;
  private rafId = 0;
  private lastFrameTime = 0;
  private frameCount = 0;
  private fpsAccumulator = 0;
  private currentFps = 60;
  private resizeObserver?: ResizeObserver;

  // Exposed read-only state for React UI
  state: EngineState = {
    viewer: null,
    scene: null,
    renderer: null,
    camera: null,
    running: false,
    fps: 60,
    frameTime: 0,
    objectCount: 0,
    drawCalls: 0,
    triangles: 0,
  };

  constructor(config: EngineConfig) {
    this.config = config;
  }

  /* ── Initialisation ─────────────────────────────────────────────────── */

  async init(): Promise<void> {
    // 1. Cesium globe
    this.globe = new Globe(this.config);
    const viewer = await this.globe.init();

    // 2. Three.js overlay scene
    this.sceneManager = new SceneManager(viewer, this.config);

    // 3. Layer manager
    this.layerManager = new LayerManager(viewer, this.sceneManager);

    // 4. Post-processing pipeline
    this.shaderPipeline = new ShaderPipeline(
      this.sceneManager.renderer,
      this.sceneManager.scene,
      this.sceneManager.camera,
    );

    // 5. Camera controller
    this.cameraController = new CameraController(viewer);

    // 6. Performance systems
    this.lodManager = new LODManager();
    this.spatialIndex = new SpatialIndex();
    this.objectPool = new ObjectPool();

    // 7. Resize handling
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.config.container);

    // Populate state
    this.state.viewer = viewer;
    this.state.scene = this.sceneManager.scene;
    this.state.renderer = this.sceneManager.renderer;
    this.state.camera = this.sceneManager.camera;
  }

  /* ── Render loop ────────────────────────────────────────────────────── */

  start(): void {
    if (this.running) return;
    this.running = true;
    this.state.running = true;
    this.lastFrameTime = performance.now();
    this.tick();
  }

  stop(): void {
    this.running = false;
    this.state.running = false;
    cancelAnimationFrame(this.rafId);
  }

  private tick = (): void => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this.tick);

    const now = performance.now();
    const dt = (now - this.lastFrameTime) / 1000; // seconds
    this.lastFrameTime = now;

    // FPS tracking
    this.frameCount++;
    this.fpsAccumulator += dt;
    if (this.fpsAccumulator >= 1) {
      this.currentFps = this.frameCount / this.fpsAccumulator;
      this.frameCount = 0;
      this.fpsAccumulator = 0;
    }

    // 1. Update camera controller
    this.cameraController.update(dt);

    // 2. Frustum-cull & LOD
    const cameraHeight = this.getCameraHeight();
    this.lodManager.update(cameraHeight);

    // 3. Update layers
    this.layerManager.update(dt, cameraHeight);

    // 4. Sync Three.js camera with Cesium
    this.sceneManager.syncWithCesium();

    // 5. Render Three.js scene through shader pipeline
    this.shaderPipeline.render();

    // 6. Update state
    this.updateState();
  };

  /* ── Accessors ──────────────────────────────────────────────────────── */

  getGlobe(): Globe { return this.globe; }
  getSceneManager(): SceneManager { return this.sceneManager; }
  getLayers(): LayerManager { return this.layerManager; }
  getShaders(): ShaderPipeline { return this.shaderPipeline; }
  getCamera(): CameraController { return this.cameraController; }
  getLOD(): LODManager { return this.lodManager; }
  getSpatialIndex(): SpatialIndex { return this.spatialIndex; }
  getObjectPool(): ObjectPool { return this.objectPool; }

  getMetrics(): PerformanceMetrics {
    const info = this.sceneManager.getInfo();
    return {
      fps: this.currentFps,
      frameTime: 1000 / Math.max(this.currentFps, 1),
      gpuTime: 0, // requires EXT_disjoint_timer_query
      cpuTime: 0,
      objectCount: info.render.calls,
      visibleObjects: this.layerManager.getVisibleObjectCount(),
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
      textureMemory: info.memory.textures,
      geometryMemory: info.memory.geometries,
    };
  }

  /* ── Internals ──────────────────────────────────────────────────────── */

  private getCameraHeight(): number {
    const cart = this.globe.viewer.camera.positionCartographic;
    return cart ? cart.height : 15_000_000;
  }

  private updateState(): void {
    const info = this.sceneManager.getInfo();
    this.state.fps = Math.round(this.currentFps);
    this.state.frameTime = +(1000 / Math.max(this.currentFps, 1)).toFixed(1);
    this.state.objectCount = this.layerManager.getVisibleObjectCount();
    this.state.drawCalls = info.render.calls;
    this.state.triangles = info.render.triangles;
  }

  private handleResize(): void {
    const { clientWidth: w, clientHeight: h } = this.config.container;
    this.globe.resize();
    this.sceneManager.resize(w, h);
    this.shaderPipeline.resize(w, h);
  }

  /* ── Cleanup ────────────────────────────────────────────────────────── */

  destroy(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.layerManager.removeAll();
    this.shaderPipeline.dispose();
    this.sceneManager.destroy();
    this.globe.destroy();
  }
}

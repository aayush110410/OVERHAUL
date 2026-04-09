/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Globe Renderer (CesiumJS)
 *
 * Wraps a Cesium.Viewer with terrain, imagery, atmosphere, and lighting.
 * Provides a clean API consumed by the higher-level Engine.
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import {
  DEFAULT_VIEW,
  DEFAULT_HEADING,
  DEFAULT_PITCH,
} from './constants';
import type { EngineConfig, GeoPosition } from './types';

export class Globe {
  viewer!: Cesium.Viewer;
  private container: HTMLElement;
  private config: EngineConfig;

  constructor(config: EngineConfig) {
    this.config = config;
    this.container = config.container;
  }

  /* ── Initialisation ─────────────────────────────────────────────────── */

  async init(): Promise<Cesium.Viewer> {
    // Cesium Ion token (optional — OSM tiles work without one)
    if (this.config.cesiumToken) {
      Cesium.Ion.defaultAccessToken = this.config.cesiumToken;
    }

    // Build terrain provider
    const terrainProvider = await this.buildTerrainProvider();

    this.viewer = new Cesium.Viewer(this.container, {
      terrainProvider,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      vrButton: false,
      selectionIndicator: false,
      infoBox: false,
      creditContainer: document.createElement('div'), // hide credits from viewport
      msaaSamples: this.config.msaaSamples ?? 4,
      requestRenderMode: false,
      maximumRenderTimeChange: Infinity,
      targetFrameRate: this.config.maxFps ?? 60,
      orderIndependentTranslucency: true,
      shadows: false,
      shouldAnimate: true,
    });

    this.configureScene();
    this.addImageryLayer();
    this.setInitialView();

    return this.viewer;
  }

  /* ── Terrain ────────────────────────────────────────────────────────── */

  private async buildTerrainProvider(): Promise<Cesium.TerrainProvider> {
    const mode = this.config.terrainProvider ?? 'cesium-world';
    switch (mode) {
      case 'cesium-world':
        try {
          return await Cesium.CesiumTerrainProvider.fromIonAssetId(1);
        } catch {
          // Fallback if no Ion token
          return new Cesium.EllipsoidTerrainProvider();
        }
      case 'ellipsoid':
      default:
        return new Cesium.EllipsoidTerrainProvider();
    }
  }

  /* ── Imagery ────────────────────────────────────────────────────────── */

  private addImageryLayer(): void {
    const layers = this.viewer.imageryLayers;
    const mode = this.config.imageryProvider ?? 'osm';

    switch (mode) {
      case 'osm':
        layers.addImageryProvider(
          new Cesium.OpenStreetMapImageryProvider({
            url: 'https://tile.openstreetmap.org/',
          }),
        );
        break;
      // Additional providers can be wired here
      default:
        break;
    }
  }

  /* ── Scene settings ─────────────────────────────────────────────────── */

  private configureScene(): void {
    const scene = this.viewer.scene;
    const globe = scene.globe;

    // Atmosphere & sky
    if (scene.skyAtmosphere) {
      scene.skyAtmosphere.show = this.config.enableAtmosphere ?? true;
    }
    scene.fog.enabled = this.config.enableFog ?? true;
    scene.fog.density = 2.0e-4;

    // Lighting
    globe.enableLighting = this.config.enableLighting ?? true;

    // Tile loading
    globe.tileCacheSize = 1000;
    globe.maximumScreenSpaceError = 1.5; // higher quality

    // Depth testing against terrain
    globe.depthTestAgainstTerrain = true;

    // Performance: enable request render mode for idle scenes
    scene.requestRenderMode = false;

    // High-DPI
    this.viewer.resolutionScale = Math.min(window.devicePixelRatio, 2);

    // Debug
    if (this.config.debugMode) {
      scene.debugShowFramesPerSecond = true;
    }
  }

  /* ── Camera ─────────────────────────────────────────────────────────── */

  private setInitialView(): void {
    const pos = this.config.initialView ?? DEFAULT_VIEW;
    const heading = Cesium.Math.toRadians(this.config.initialHeading ?? DEFAULT_HEADING);
    const pitch = Cesium.Math.toRadians(this.config.initialPitch ?? DEFAULT_PITCH);

    this.viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(pos.longitude, pos.latitude, pos.altitude),
      orientation: { heading, pitch, roll: 0 },
    });
  }

  flyTo(pos: GeoPosition, duration = 2): void {
    this.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(pos.longitude, pos.latitude, pos.altitude),
      duration,
    });
  }

  /* ── Lifecycle ──────────────────────────────────────────────────────── */

  resize(): void {
    this.viewer.resize();
  }

  destroy(): void {
    if (!this.viewer.isDestroyed()) {
      this.viewer.destroy();
    }
  }
}

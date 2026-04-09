/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Shader Pipeline
 *
 * GPU-accelerated post-processing pipeline using Three.js EffectComposer
 * pattern built from raw WebGL shader passes.
 *
 * Passes: Bloom → Contrast → Sharpness → Ambient Glow → Fog Depth
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import type { ShaderPreset, VisualizationMode } from '../core/types';
import { SHADER_PRESETS } from '../core/constants';

// GLSL (loaded via vite-plugin-glsl)
import passthroughVert from './glsl/passthrough.vert';
import bloomFrag from './glsl/bloom.frag';
import contrastFrag from './glsl/contrast.frag';
import sharpnessFrag from './glsl/sharpness.frag';
import glowFrag from './glsl/glow.frag';
import fogFrag from './glsl/fog.frag';

/* ── Generic full-screen quad pass ────────────────────────────────────── */

class ShaderPass {
  material: THREE.ShaderMaterial;
  quad: THREE.Mesh;
  enabled: boolean;

  constructor(
    public name: string,
    fragmentShader: string,
    uniforms: Record<string, THREE.IUniform>,
    enabled = true,
  ) {
    this.enabled = enabled;
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        ...uniforms,
      },
      vertexShader: passthroughVert,
      fragmentShader,
      depthTest: false,
      depthWrite: false,
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    this.quad = new THREE.Mesh(geometry, this.material);
  }

  dispose(): void {
    this.material.dispose();
    this.quad.geometry.dispose();
  }
}

/* ── Pipeline ─────────────────────────────────────────────────────────── */

export class ShaderPipeline {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.Camera;
  private passScene = new THREE.Scene();
  private passCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  private rtA: THREE.WebGLRenderTarget;
  private rtB: THREE.WebGLRenderTarget;

  private passes: ShaderPass[] = [];
  private currentPreset: ShaderPreset;
  private settings = {
    exposure: 1,
    bloomStrength: 0,
    bloomRadius: 0,
    bloomThreshold: 0,
    contrast: 1,
    brightness: 0,
    sharpness: 0,
    glowIntensity: 0,
    fogNear: 1000,
    fogFar: 50000,
  };

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;

    const { width, height } = renderer.getSize(new THREE.Vector2());
    const opts: THREE.RenderTargetOptions = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
    };
    this.rtA = new THREE.WebGLRenderTarget(width, height, opts);
    this.rtB = new THREE.WebGLRenderTarget(width, height, opts);

    this.currentPreset = SHADER_PRESETS.standard;
    this.buildPasses();
  }

  /* ── Pass construction ──────────────────────────────────────────────── */

  private buildPasses(): void {
    this.passes.forEach((p) => p.dispose());
    this.passes = [];
    const p = this.currentPreset;
    this.settings = {
      exposure: this.renderer.toneMappingExposure,
      bloomStrength: p.bloom.strength,
      bloomRadius: p.bloom.radius,
      bloomThreshold: p.bloom.threshold,
      contrast: p.contrast.amount,
      brightness: p.contrast.brightness,
      sharpness: p.sharpness.amount,
      glowIntensity: p.glow.intensity,
      fogNear: p.fog.near,
      fogFar: p.fog.far,
    };

    // 1. Bloom
    this.passes.push(new ShaderPass('bloom', bloomFrag, {
      bloomStrength: { value: p.bloom.strength },
      bloomRadius: { value: p.bloom.radius },
      bloomThreshold: { value: p.bloom.threshold },
    }, p.bloom.enabled));

    // 2. Contrast & Brightness
    this.passes.push(new ShaderPass('contrast', contrastFrag, {
      contrast: { value: p.contrast.amount },
      brightness: { value: p.contrast.brightness },
    }, p.contrast.enabled));

    // 3. Sharpness
    const size = this.renderer.getSize(new THREE.Vector2());
    this.passes.push(new ShaderPass('sharpness', sharpnessFrag, {
      resolution: { value: new THREE.Vector2(size.x, size.y) },
      amount: { value: p.sharpness.amount },
    }, p.sharpness.enabled));

    // 4. Ambient Glow
    this.passes.push(new ShaderPass('glow', glowFrag, {
      glowIntensity: { value: p.glow.intensity },
      glowColor: { value: new THREE.Vector3(...p.glow.color) },
    }, p.glow.enabled));

    // 5. Fog Depth
    this.passes.push(new ShaderPass('fog', fogFrag, {
      tDepth: { value: null },
      fogNear: { value: p.fog.near },
      fogFar: { value: p.fog.far },
      fogColor: { value: new THREE.Vector3(...p.fog.color) },
      cameraNear: { value: 0.1 },
      cameraFar: { value: 1e9 },
    }, p.fog.enabled));
  }

  /* ── Render ─────────────────────────────────────────────────────────── */

  render(): void {
    const activePasses = this.passes.filter((p) => p.enabled);
    if (activePasses.length === 0) {
      // No post-processing — render directly
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Render scene into first RT
    this.renderer.setRenderTarget(this.rtA);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);

    // Ping-pong through passes
    let readRT = this.rtA;
    let writeRT = this.rtB;

    for (let i = 0; i < activePasses.length; i++) {
      const pass = activePasses[i];
      const isLast = i === activePasses.length - 1;

      pass.material.uniforms.tDiffuse.value = readRT.texture;

      if (isLast) {
        // Final pass outputs to screen
        this.renderer.setRenderTarget(null);
      } else {
        this.renderer.setRenderTarget(writeRT);
      }

      this.renderer.clear();

      // Render full-screen quad
      this.passScene.children.length = 0;
      this.passScene.add(pass.quad);
      this.renderer.render(this.passScene, this.passCamera);

      // Swap
      [readRT, writeRT] = [writeRT, readRT];
    }
  }

  /* ── Preset switching ───────────────────────────────────────────────── */

  setPreset(mode: VisualizationMode): void {
    const preset = SHADER_PRESETS[mode];
    if (!preset) return;
    this.currentPreset = preset;
    this.buildPasses();
  }

  getPreset(): ShaderPreset {
    return this.currentPreset;
  }

  /* ── Individual pass controls ───────────────────────────────────────── */

  setPassEnabled(name: string, enabled: boolean): void {
    const pass = this.passes.find((p) => p.name === name);
    if (pass) pass.enabled = enabled;
  }

  setPassUniform(name: string, uniform: string, value: number | THREE.Vector2 | THREE.Vector3): void {
    const pass = this.passes.find((p) => p.name === name);
    if (pass && pass.material.uniforms[uniform]) {
      pass.material.uniforms[uniform].value = value;
    }
  }

  getSettings(): Record<string, number> {
    return { ...this.settings };
  }

  setToneMappingExposure(exposure: number): void {
    this.settings.exposure = exposure;
    this.renderer.toneMappingExposure = exposure;
  }

  setBloom(strength: number, radius = this.settings.bloomRadius, threshold = this.settings.bloomThreshold): void {
    this.settings.bloomStrength = strength;
    this.settings.bloomRadius = radius;
    this.settings.bloomThreshold = threshold;
    this.setPassEnabled('bloom', strength > 0.01);
    this.setPassUniform('bloom', 'bloomStrength', strength);
    this.setPassUniform('bloom', 'bloomRadius', radius);
    this.setPassUniform('bloom', 'bloomThreshold', threshold);
  }

  setContrast(amount: number, brightness = this.settings.brightness): void {
    this.settings.contrast = amount;
    this.settings.brightness = brightness;
    this.setPassEnabled('contrast', Math.abs(amount - 1) > 0.01 || Math.abs(brightness) > 0.01);
    this.setPassUniform('contrast', 'contrast', amount);
    this.setPassUniform('contrast', 'brightness', brightness);
  }

  setSharpness(amount: number): void {
    this.settings.sharpness = amount;
    this.setPassEnabled('sharpness', amount > 0.01);
    this.setPassUniform('sharpness', 'amount', amount);
  }

  setGlow(intensity: number): void {
    this.settings.glowIntensity = intensity;
    this.setPassEnabled('glow', intensity > 0.01);
    this.setPassUniform('glow', 'glowIntensity', intensity);
  }

  setFog(near: number, far: number): void {
    this.settings.fogNear = near;
    this.settings.fogFar = far;
    this.setPassEnabled('fog', far > near);
    this.setPassUniform('fog', 'fogNear', near);
    this.setPassUniform('fog', 'fogFar', far);
  }

  /* ── Resize ─────────────────────────────────────────────────────────── */

  resize(width: number, height: number): void {
    this.rtA.setSize(width, height);
    this.rtB.setSize(width, height);

    const sharpPass = this.passes.find((p) => p.name === 'sharpness');
    if (sharpPass) {
      sharpPass.material.uniforms.resolution.value = new THREE.Vector2(width, height);
    }
  }

  /* ── Cleanup ────────────────────────────────────────────────────────── */

  dispose(): void {
    this.passes.forEach((p) => p.dispose());
    this.rtA.dispose();
    this.rtB.dispose();
  }
}

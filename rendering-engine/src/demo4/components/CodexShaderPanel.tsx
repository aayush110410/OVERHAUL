import type { VisualizationMode } from '../../core/types';
import { VISUALIZATION_MODES } from '../defaults';

interface CodexShaderPanelProps {
  mode: VisualizationMode;
  settings: Record<string, number>;
  onModeChange: (mode: VisualizationMode) => void;
  onSettingChange: (setting: string, value: number) => void;
}

const SLIDERS = [
  { key: 'exposure', label: 'HDR exposure', min: 0.6, max: 1.8, step: 0.01 },
  { key: 'bloomStrength', label: 'Bloom', min: 0, max: 1.8, step: 0.01 },
  { key: 'contrast', label: 'Contrast', min: 0.8, max: 1.9, step: 0.01 },
  { key: 'sharpness', label: 'Sharpness', min: 0, max: 1.2, step: 0.01 },
  { key: 'glowIntensity', label: 'Ambient glow', min: 0, max: 1.2, step: 0.01 },
  { key: 'fogFar', label: 'Fog depth', min: 5000, max: 150000, step: 500 },
];

export function CodexShaderPanel({
  mode,
  settings,
  onModeChange,
  onSettingChange,
}: CodexShaderPanelProps) {
  return (
    <section className="demo4-card">
      <div className="demo4-section-head">
        <span>Visualization</span>
        <span>{mode}</span>
      </div>

      <div className="demo4-mode-grid">
        {VISUALIZATION_MODES.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className={`demo4-mode-btn ${preset.id === mode ? 'is-active' : ''}`}
            onClick={() => onModeChange(preset.id)}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="demo4-slider-grid">
        {SLIDERS.map((slider) => (
          <label key={slider.key} className="demo4-slider">
            <span>{slider.label}</span>
            <input
              type="range"
              min={slider.min}
              max={slider.max}
              step={slider.step}
              value={settings[slider.key] ?? slider.min}
              onChange={(event) => onSettingChange(slider.key, Number(event.target.value))}
            />
            <strong>{(settings[slider.key] ?? 0).toFixed(slider.key === 'fogFar' ? 0 : 2)}</strong>
          </label>
        ))}
      </div>
    </section>
  );
}

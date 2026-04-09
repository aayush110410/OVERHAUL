import type { Demo4LayerState } from '../types';

interface CodexLayerPanelProps {
  layers: Demo4LayerState[];
  onToggleLoaded: (id: string) => void;
  onToggleVisible: (id: string) => void;
  onOpacityChange: (id: string, opacity: number) => void;
}

export function CodexLayerPanel({
  layers,
  onToggleLoaded,
  onToggleVisible,
  onOpacityChange,
}: CodexLayerPanelProps) {
  return (
    <section className="demo4-card demo4-layer-panel">
      <div className="demo4-section-head">
        <span>Layer control</span>
        <span>{layers.filter((layer) => layer.loaded).length}/{layers.length} active</span>
      </div>

      {layers.map((layer) => (
        <div key={layer.id} className="demo4-layer-row">
          <div>
            <div className="demo4-layer-name">{layer.name}</div>
            <div className="demo4-layer-meta">
              <span>{layer.loaded ? 'Loaded' : 'Unloaded'}</span>
              <span>{layer.objectCount.toLocaleString()} objects</span>
              <span>{layer.updateFrequencyHz.toFixed(2)} Hz</span>
            </div>
          </div>

          <div className="demo4-layer-actions">
            <button type="button" className="demo4-chip" onClick={() => onToggleLoaded(layer.id)}>
              {layer.loaded ? 'Unload' : 'Load'}
            </button>
            <button
              type="button"
              className={`demo4-chip ${layer.visible ? 'is-active' : ''}`}
              disabled={!layer.loaded}
              onClick={() => onToggleVisible(layer.id)}
            >
              {layer.visible ? 'Hide' : 'Show'}
            </button>
          </div>

          <input
            aria-label={`${layer.name} opacity`}
            type="range"
            min={20}
            max={100}
            value={Math.round(layer.opacity * 100)}
            disabled={!layer.loaded}
            onChange={(event) => onOpacityChange(layer.id, Number(event.target.value) / 100)}
          />
        </div>
      ))}
    </section>
  );
}

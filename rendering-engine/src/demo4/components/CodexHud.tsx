import type { PerformanceMetrics } from '../../core/types';

interface CodexHudProps {
  focusName: string;
  coordinates: { latitude: number; longitude: number; altitude: number } | null;
  metrics: PerformanceMetrics;
  activeDatasets: number;
  connectionState: 'connecting' | 'live' | 'offline';
}

export function CodexHud({
  focusName,
  coordinates,
  metrics,
  activeDatasets,
  connectionState,
}: CodexHudProps) {
  return (
    <div className="demo4-hud">
      <div className="demo4-hud-row">
        <span>Focus</span>
        <strong>{focusName}</strong>
      </div>
      <div className="demo4-hud-row">
        <span>Coordinates</span>
        <strong>
          {coordinates
            ? `${coordinates.latitude.toFixed(3)}, ${coordinates.longitude.toFixed(3)}`
            : '--'}
        </strong>
      </div>
      <div className="demo4-hud-row">
        <span>Camera altitude</span>
        <strong>{coordinates ? `${Math.round(coordinates.altitude).toLocaleString()} m` : '--'}</strong>
      </div>
      <div className="demo4-hud-row">
        <span>Active datasets</span>
        <strong>{activeDatasets}</strong>
      </div>
      <div className="demo4-hud-row">
        <span>Renderer</span>
        <strong>{Math.round(metrics.fps)} FPS</strong>
      </div>
      <div className="demo4-hud-row">
        <span>Connection</span>
        <strong className={`demo4-status-${connectionState}`}>{connectionState}</strong>
      </div>
    </div>
  );
}

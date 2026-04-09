import type { Demo4SimulationReport, Demo4TimelineState } from '../types';

interface CodexTimelineProps {
  timeline: Demo4TimelineState[];
  currentIndex: number;
  report: Demo4SimulationReport | null;
  onChange: (index: number) => void;
}

export function CodexTimeline({ timeline, currentIndex, report, onChange }: CodexTimelineProps) {
  if (timeline.length === 0) return null;

  const current = timeline[currentIndex];
  return (
    <section className="demo4-card demo4-timeline">
      <div className="demo4-section-head">
        <span>Timeline</span>
        <span>{current.label}</span>
      </div>

      <input
        aria-label="Simulation timeline"
        type="range"
        min={0}
        max={timeline.length - 1}
        step={1}
        value={currentIndex}
        onChange={(event) => onChange(Number(event.target.value))}
      />

      <div className="demo4-timeline-labels">
        {timeline.map((step, index) => (
          <button
            key={step.label}
            type="button"
            className={`demo4-step ${index === currentIndex ? 'is-current' : ''}`}
            onClick={() => onChange(index)}
          >
            {step.label}
          </button>
        ))}
      </div>

      <div className="demo4-metric-grid">
        {Object.entries(current.metrics).map(([key, value]) => (
          <div key={key} className="demo4-metric-card">
            <span>{key.replace(/_/g, ' ')}</span>
            <strong>{value.toFixed(1)}</strong>
          </div>
        ))}
      </div>

      {report && (
        <div className="demo4-report-bar">
          <span>Traffic +{report.traffic_improvement_pct.toFixed(1)}%</span>
          <span>Travel time -{report.travel_time_delta_pct.toFixed(1)}%</span>
          <span>Congestion -{report.congestion_delta_pct.toFixed(1)}%</span>
        </div>
      )}
    </section>
  );
}

import React from 'react';

export default function MetricsPanel() {
  return (
    <div className="metrics-panel">
      <div className="metric-card">
        <h3>AQI LEVEL</h3>
        <p className="metric-value warning">184 // POOR</p>
      </div>
      <div className="metric-card">
        <h3>TRAFFIC CONGESTION</h3>
        <p className="metric-value critical">88% // SEVERE</p>
      </div>
      <div className="metric-card">
        <h3>GRID LOAD</h3>
        <p className="metric-value good">42% // STABLE</p>
      </div>
      <style>{`
        .metric-card {
          background: rgba(10, 20, 30, 0.4);
          border: 1px solid rgba(0, 150, 255, 0.2);
          padding: 15px;
          margin-bottom: 12px;
          border-radius: 8px;
        }
        .metric-card h3 {
          margin: 0 0 10px 0;
          font-size: 0.8rem;
          color: rgba(255,255,255,0.6);
        }
        .metric-value {
          margin: 0;
          font-size: 1.5rem;
          font-weight: bold;
        }
        .warning { color: #f59e0b; }
        .critical { color: #ef4444; }
        .good { color: #10b981; }
      `}</style>
    </div>
  );
}

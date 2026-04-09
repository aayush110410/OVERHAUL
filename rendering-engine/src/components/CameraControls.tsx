/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Camera Controls Component
 *
 * Camera mode selector + quick-fly buttons.
 * ────────────────────────────────────────────────────────────────────────── */

import React from 'react';
import { useEngineStore } from '../store/engineStore';
import type { CameraMode } from '../core/types';

const CAMERA_MODES: { id: CameraMode; label: string }[] = [
  { id: 'orbit',     label: 'Orbit' },
  { id: 'pan',       label: 'Pan' },
  { id: 'tilt',      label: 'Tilt' },
  { id: 'free',      label: 'Free' },
  { id: 'cinematic', label: 'Cinematic' },
];

export function CameraControls() {
  const { cameraMode, setCameraMode } = useEngineStore();

  return (
    <div style={panelStyle}>
      <h3 style={{ margin: '0 0 8px', fontSize: 13, letterSpacing: 1, opacity: 0.7 }}>
        CAMERA
      </h3>
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {CAMERA_MODES.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setCameraMode(id)}
            style={{
              ...btnStyle,
              background: id === cameraMode
                ? 'rgba(59,130,246,0.4)'
                : 'rgba(255,255,255,0.06)',
              borderColor: id === cameraMode
                ? 'rgba(59,130,246,0.6)'
                : 'rgba(255,255,255,0.06)',
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 16,
  left: 16,
  background: 'rgba(10,10,20,0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: 10,
  padding: '14px 16px',
  color: '#eee',
  fontFamily: 'system-ui, sans-serif',
  zIndex: 100,
  border: '1px solid rgba(255,255,255,0.08)',
};

const btnStyle: React.CSSProperties = {
  padding: '4px 10px',
  borderRadius: 6,
  border: '1px solid',
  color: '#eee',
  cursor: 'pointer',
  fontSize: 11,
  fontFamily: 'system-ui, sans-serif',
};

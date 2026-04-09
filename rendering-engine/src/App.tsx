/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Rendering Engine App
 * ────────────────────────────────────────────────────────────────────────── */

import { useCallback, useState, useEffect } from 'react';
import { GlobeView } from './components/GlobeView';
import { LayerPanel } from './components/LayerPanel';
import { ShaderControls } from './components/ShaderControls';
import { CameraControls } from './components/CameraControls';
import { PerformanceMonitor } from './components/PerformanceMonitor';
import DemoPage from './pages/DemoPage';
import Demo4CodexPage from './pages/Demo4CodexPage';
import type { Engine } from './core/Engine';

type PageType = 'main' | 'demo' | 'demo-4-codex';

export default function App() {
  const [page, setPage] = useState<PageType>('demo-4-codex');
  const [engine, setEngine] = useState<Engine | null>(null);

  // Listen for hash route changes
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1) || 'demo-4-codex';
      setPage((hash as PageType) || 'demo');
    };
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleEngineReady = useCallback((eng: Engine) => {
    setEngine(eng);
    // Expose engine to console for debugging
    (window as unknown as Record<string, unknown>).overhaulEngine = eng;
  }, []);

  // Route to demo page
  if (page === 'demo') {
    return <DemoPage />;
  }

  if (page === 'demo-4-codex') {
    return <Demo4CodexPage />;
  }

  // Main page
  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <GlobeView
        cesiumToken={import.meta.env.VITE_CESIUM_TOKEN}
        onEngineReady={handleEngineReady}
      />

      {/* ── HUD overlays (rendered only after engine is ready) ───────── */}
      {engine && (
        <>
          <CameraControls />
          <LayerPanel />
          <ShaderControls />
          <PerformanceMonitor />
        </>
      )}

      {/* ── Title watermark ──────────────────────────────────────────── */}
      <div style={titleStyle}>OVERHAUL</div>
    </div>
  );
}

const titleStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: 20,
  left: 20,
  color: 'rgba(255,255,255,0.15)',
  fontSize: 11,
  letterSpacing: 4,
  fontFamily: 'system-ui, sans-serif',
  fontWeight: 600,
  pointerEvents: 'none',
  zIndex: 100,
};

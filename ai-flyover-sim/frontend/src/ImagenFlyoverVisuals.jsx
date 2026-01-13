import React, { useState } from 'react';

const API_BASE = 'http://localhost:8000';

export default function ImagenFlyoverVisuals({ flyoverData, locationName }) {
  const [conceptImage, setConceptImage] = useState(null);
  const [textureImage, setTextureImage] = useState(null);
  const [beforeAfterImages, setBeforeAfterImages] = useState({ before: null, after: null });
  const [loading, setLoading] = useState({
    concept: false,
    texture: false,
    beforeAfter: false
  });
  const [activeTab, setActiveTab] = useState('concept');
  const [imagenAvailable, setImagenAvailable] = useState(null);

  // Check Imagen availability on mount
  React.useEffect(() => {
    fetch(`${API_BASE}/imagen/status`)
      .then(r => r.json())
      .then(data => setImagenAvailable(data.enabled))
      .catch(() => setImagenAvailable(false));
  }, []);

  const generateConcept = async (viewType = 'aerial') => {
    setLoading(prev => ({ ...prev, concept: true }));
    try {
      const resp = await fetch(`${API_BASE}/imagen/flyover/concept`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location_name: locationName || 'Delhi',
          flyover_lanes: flyoverData?.flyover_lanes || 2,
          flyover_width_m: flyoverData?.flyover_width_m || 9.0,
          flyover_height_m: flyoverData?.flyover_height_m || 8.5,
          road_type: flyoverData?.road?.type || 'arterial',
          view_type: viewType
        })
      });
      
      if (resp.ok) {
        const data = await resp.json();
        setConceptImage(data);
      }
    } catch (err) {
      console.error('Failed to generate concept:', err);
    }
    setLoading(prev => ({ ...prev, concept: false }));
  };

  const generateTexture = async (textureType = 'concrete_deck') => {
    setLoading(prev => ({ ...prev, texture: true }));
    try {
      const resp = await fetch(`${API_BASE}/imagen/flyover/texture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texture_type: textureType })
      });
      
      if (resp.ok) {
        const data = await resp.json();
        setTextureImage(data);
      }
    } catch (err) {
      console.error('Failed to generate texture:', err);
    }
    setLoading(prev => ({ ...prev, texture: false }));
  };

  const generateBeforeAfter = async () => {
    setLoading(prev => ({ ...prev, beforeAfter: true }));
    try {
      // Generate both views
      const [beforeResp, afterResp] = await Promise.all([
        fetch(`${API_BASE}/imagen/flyover/before-after`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            location_name: locationName || 'Delhi',
            flyover_lanes: flyoverData?.flyover_lanes || 2,
            view: 'before'
          })
        }),
        fetch(`${API_BASE}/imagen/flyover/before-after`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            location_name: locationName || 'Delhi',
            flyover_lanes: flyoverData?.flyover_lanes || 2,
            view: 'after'
          })
        })
      ]);
      
      const beforeData = beforeResp.ok ? await beforeResp.json() : null;
      const afterData = afterResp.ok ? await afterResp.json() : null;
      
      setBeforeAfterImages({ before: beforeData, after: afterData });
    } catch (err) {
      console.error('Failed to generate before/after:', err);
    }
    setLoading(prev => ({ ...prev, beforeAfter: false }));
  };

  if (imagenAvailable === null) {
    return <div className="imagen-panel loading">Checking Imagen 3 availability...</div>;
  }

  if (!imagenAvailable) {
    return (
      <div className="imagen-panel disabled">
        <div className="imagen-header">
          <span className="imagen-logo">🎨</span>
          <span>Imagen 3 (Nano Banana Pro)</span>
        </div>
        <p className="imagen-status">Not configured. Set IMAGEN_API_KEY to enable AI visualizations.</p>
      </div>
    );
  }

  return (
    <div className="imagen-panel">
      <div className="imagen-header">
        <span className="imagen-logo">🎨</span>
        <span>Imagen 3 - AI Flyover Visuals</span>
        <span className="imagen-badge">Nano Banana Pro</span>
      </div>

      <div className="imagen-tabs">
        <button 
          className={`imagen-tab ${activeTab === 'concept' ? 'active' : ''}`}
          onClick={() => setActiveTab('concept')}
        >
          Concept Render
        </button>
        <button 
          className={`imagen-tab ${activeTab === 'beforeAfter' ? 'active' : ''}`}
          onClick={() => setActiveTab('beforeAfter')}
        >
          Before/After
        </button>
        <button 
          className={`imagen-tab ${activeTab === 'texture' ? 'active' : ''}`}
          onClick={() => setActiveTab('texture')}
        >
          3D Textures
        </button>
      </div>

      <div className="imagen-content">
        {activeTab === 'concept' && (
          <div className="imagen-section">
            <div className="imagen-controls">
              <button 
                onClick={() => generateConcept('aerial')}
                disabled={loading.concept}
                className="imagen-btn primary"
              >
                {loading.concept ? '⏳ Generating...' : '🛩️ Aerial View'}
              </button>
              <button 
                onClick={() => generateConcept('perspective')}
                disabled={loading.concept}
                className="imagen-btn secondary"
              >
                {loading.concept ? '⏳ Generating...' : '📷 Street View'}
              </button>
            </div>
            
            {conceptImage && (
              <div className="imagen-result">
                <img 
                  src={`data:image/png;base64,${conceptImage.image_base64}`} 
                  alt="Flyover Concept"
                  className="imagen-image"
                />
                <div className="imagen-caption">
                  {conceptImage.view_type === 'aerial' ? 'Aerial' : 'Street-level'} concept visualization
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'beforeAfter' && (
          <div className="imagen-section">
            <div className="imagen-controls">
              <button 
                onClick={generateBeforeAfter}
                disabled={loading.beforeAfter}
                className="imagen-btn primary"
              >
                {loading.beforeAfter ? '⏳ Generating...' : '🔄 Generate Comparison'}
              </button>
            </div>
            
            {(beforeAfterImages.before || beforeAfterImages.after) && (
              <div className="imagen-comparison">
                {beforeAfterImages.before && (
                  <div className="imagen-compare-item">
                    <div className="imagen-compare-label before">BEFORE</div>
                    <img 
                      src={`data:image/png;base64,${beforeAfterImages.before.image_base64}`} 
                      alt="Before"
                      className="imagen-image"
                    />
                  </div>
                )}
                {beforeAfterImages.after && (
                  <div className="imagen-compare-item">
                    <div className="imagen-compare-label after">AFTER</div>
                    <img 
                      src={`data:image/png;base64,${beforeAfterImages.after.image_base64}`} 
                      alt="After"
                      className="imagen-image"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'texture' && (
          <div className="imagen-section">
            <div className="imagen-controls">
              <button 
                onClick={() => generateTexture('concrete_deck')}
                disabled={loading.texture}
                className="imagen-btn"
              >
                🛣️ Deck
              </button>
              <button 
                onClick={() => generateTexture('pillar_surface')}
                disabled={loading.texture}
                className="imagen-btn"
              >
                🏛️ Pillar
              </button>
              <button 
                onClick={() => generateTexture('barrier_metal')}
                disabled={loading.texture}
                className="imagen-btn"
              >
                🚧 Barrier
              </button>
            </div>
            
            {textureImage && (
              <div className="imagen-result">
                <img 
                  src={`data:image/png;base64,${textureImage.image_base64}`} 
                  alt="Texture"
                  className="imagen-texture"
                />
                <div className="imagen-caption">
                  {textureImage.texture_type.replace(/_/g, ' ')} texture (tileable)
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

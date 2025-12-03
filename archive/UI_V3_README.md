# OVERHAUL v3 - Modern UI

## 🚀 Quick Start

### Option 1: Automatic Startup (Windows)
```bash
start_v3.bat
```
This will:
1. Start FastAPI backend on port 8000
2. Start HTTP server on port 8080
3. Open browser automatically

### Option 2: Manual Startup

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
myvenv\Scripts\activate

# Start FastAPI
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# Activate virtual environment
myvenv\Scripts\activate

# Start HTTP server
python -m http.server 8080
```

**Browser:**
Open: http://localhost:8080/index_v3.html

---

## ✨ Features

### Visual Design
- 🎨 **Cyberpunk Theme**: Cyan, Magenta, Neon Green color scheme
- 🌌 **Animated Background**: Flowing grid + floating gradient orbs
- 💎 **Glassmorphism**: Blur effects and gradient borders
- ⚡ **Smooth Animations**: 60fps transitions and hover effects

### Interactive Components
- 🗺️ **3D Map Visualization**: MapLibre GL with custom styling
- 📊 **Real-time Charts**: Travel time & AQI trends
- 🎚️ **Custom Sliders**: EV adoption, transit boost, congestion pricing
- 🔘 **Mode Toggle**: Fast (surrogate) vs Deep (full simulation)

### Functionality
- ✅ Full backend integration with FastAPI
- ✅ Real-time traffic analysis
- ✅ AI-generated recommendations
- ✅ Export reports (JSON)
- ✅ Copy manifest to clipboard
- ✅ Keyboard shortcuts (Ctrl+Enter to run)
- ✅ Responsive design

---

## 🔧 Backend Integration

The UI connects to these endpoints:

### POST `/chat`
**Request:**
```json
{
  "prompt": "Analyze 50% EV adoption impact",
  "mode": "fast",
  "scenario": {
    "ev_share_pct": 50,
    "transit_boost_pct": 15,
    "congestion_pricing_rupees": 60
  }
}
```

**Response:**
```json
{
  "summary": "AI-generated analysis...",
  "baseline": {
    "avg_travel_time_min": 25.3,
    "total_vkt": 1200,
    "pm25": 85.2,
    "co2_kg": 450
  },
  "ranked": [...],
  "edges_geojson": {...},
  "manifest": {...},
  "outputs": {
    "impactCards": [...],
    "confidenceLevel": "high"
  }
}
```

---

## 🎮 Usage

1. **Enter Scenario**: Describe your traffic analysis in the prompt box
2. **Select Mode**: 
   - **Fast**: Quick analysis using surrogate models
   - **Deep**: Full SUMO simulation with RAG intelligence
3. **Adjust Sliders**:
   - EV Adoption: 0-100%
   - Transit Boost: 0-50%
   - Congestion Pricing: ₹0-₹200
4. **Run Analysis**: Click button or press Ctrl+Enter
5. **Review Results**:
   - Summary & confidence level
   - KPI stats with deltas
   - Travel time & AQI charts
   - Recommended interventions
   - Live map visualization

---

## 🎨 UI Components

### Stats Cards
- Travel Time (minutes)
- PM₂.₅ Level (µg/m³)
- Vehicle KM
- CO₂ Emissions (kg)

Each shows current value + percentage delta with color coding:
- 🟢 Green = Improvement (negative delta)
- 🔴 Red = Worsening (positive delta)

### Charts
- **Travel Time Trends**: Projected improvement over 12 hours
- **Air Quality Index**: PM₂.₅ reduction timeline

### Recommendations
Top 5 interventions ranked by:
- Time saved (minutes)
- Impact level (High/Medium/Low)
- Cost estimate

---

## 🔥 Performance

- ⚡ 60fps animations
- 🚀 Optimized chart updates
- 💾 Efficient state management
- 🎯 Debounced slider inputs
- 📱 Mobile responsive

---

## 🐛 Troubleshooting

**Backend not connecting:**
- Ensure FastAPI is running: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`
- Check console for CORS errors
- Verify API_BASE URL in code (default: http://127.0.0.1:8000)

**Map not loading:**
- Check MapBox access token in code
- Verify internet connection for tile loading

**Charts empty:**
- Run analysis first to populate data
- Check browser console for errors

---

## 📦 Dependencies

Frontend (loaded from CDN):
- MapLibre GL 3.6.1
- Chart.js 4.4.1
- Outfit & Space Mono fonts

Backend:
- FastAPI
- All dependencies in requirements.txt

---

## 🎯 Keyboard Shortcuts

- `Ctrl + Enter` or `Cmd + Enter`: Run analysis
- Standard browser shortcuts work

---

## 📝 Notes

- First analysis may take 2-3 seconds while backend initializes
- Deep mode takes longer than Fast mode
- Map auto-fits to show corridor edges
- Manifest updates after each successful run
- All timestamps in ISO 8601 format

---

## 🎨 Customization

Colors defined in CSS `:root`:
```css
--primary: #00f5ff;    /* Cyan */
--secondary: #ff00ff;  /* Magenta */
--accent: #00ff88;     /* Neon Green */
--dark-1: #0a0a0f;     /* Background */
```

Animation speeds can be adjusted in `@keyframes` rules.

---

**Built with ❤️ for traffic intelligence**

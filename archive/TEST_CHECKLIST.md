# OVERHAUL v3 - Backend Integration Test Checklist

## ✅ Pre-Launch Checklist

### Backend Setup
- [ ] Virtual environment activated (`myvenv\Scripts\activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] FastAPI running on port 8000
- [ ] Health check passes: http://localhost:8000/health

### Frontend Setup
- [ ] HTTP server running on port 8080
- [ ] Browser opened to http://localhost:8080/index_v3.html
- [ ] No console errors on page load
- [ ] Map tiles loading correctly

---

## 🧪 Functional Tests

### 1. UI Elements Load
- [ ] Navbar displays with logo and status
- [ ] Clock shows current time
- [ ] System status shows "System Online"
- [ ] Map renders with dark style
- [ ] All 4 stat cards visible
- [ ] Both charts initialized
- [ ] Control panel loads
- [ ] All sliders functional

### 2. Fast Mode Analysis
- [ ] Enter prompt: "Analyze 40% EV adoption impact"
- [ ] Mode set to "Fast"
- [ ] EV slider at 40%
- [ ] Click "Run Analysis" button
- [ ] Loading spinner appears
- [ ] System status changes to "Running"
- [ ] Request completes in <5 seconds
- [ ] Summary text updates
- [ ] Stats cards populate with values
- [ ] Delta percentages show (green/red)
- [ ] Charts update with data
- [ ] Recommendations list appears
- [ ] Map shows edges with colors
- [ ] Manifest displays JSON
- [ ] Confidence badge updates

### 3. Deep Mode Analysis
- [ ] Switch mode to "Deep"
- [ ] Enter prompt: "Reduce congestion by 30%"
- [ ] Click "Run Analysis"
- [ ] Loading state persists longer
- [ ] All UI elements update
- [ ] Confidence level shows "High"
- [ ] Deep facts appear (if applicable)

### 4. Slider Interactions
- [ ] Move EV slider → value updates
- [ ] Move Transit slider → value updates
- [ ] Move Pricing slider → value updates
- [ ] Values display correctly (%, ₹)
- [ ] Sliders glow on hover
- [ ] Thumb scales on hover

### 5. Map Interactions
- [ ] Map navigation controls work
- [ ] Zoom in/out functional
- [ ] Pan around map
- [ ] Pitch/bearing adjustable
- [ ] Edges render with gradient colors
- [ ] Colors based on ev_share property
- [ ] Map auto-fits to bounds after analysis

### 6. Chart Updates
- [ ] Travel time chart populates
- [ ] AQI chart populates
- [ ] Both show progressive improvement
- [ ] Tooltips appear on hover
- [ ] Charts animate smoothly
- [ ] No flickering on update

### 7. Recommendations
- [ ] List shows 1-5 items
- [ ] Each has title and description
- [ ] Metrics show (Time, Impact, Cost)
- [ ] Hover effects work
- [ ] Left border animates
- [ ] Items slide in sequentially

### 8. Actions
- [ ] Export Report button works
- [ ] JSON file downloads
- [ ] Filename includes timestamp
- [ ] Copy Manifest button works
- [ ] Shows "✓ Copied!" feedback
- [ ] Clipboard contains JSON

### 9. Keyboard Shortcuts
- [ ] Ctrl+Enter triggers analysis
- [ ] Works from prompt textarea
- [ ] Works from anywhere on page

### 10. Error Handling
- [ ] Stop backend
- [ ] Click Run Analysis
- [ ] Error message appears
- [ ] Shake animation plays
- [ ] Message: "Failed to connect to backend"
- [ ] Status shows error state
- [ ] Restart backend
- [ ] Error clears on success

---

## 🎨 Visual Tests

### Animations
- [ ] Grid background flows
- [ ] Orbs float smoothly
- [ ] Logo shine effect visible
- [ ] Status dot pulses
- [ ] Cards fade in on load
- [ ] Hover lifts cards
- [ ] Button ripple effect
- [ ] Summary box gradient slides

### Styling
- [ ] Colors match theme (cyan/magenta/green)
- [ ] Glassmorphism visible
- [ ] Borders glow on hover
- [ ] Text readable (contrast)
- [ ] Icons display correctly
- [ ] Fonts loaded (Outfit, Space Mono)

### Responsive
- [ ] Resize window to mobile
- [ ] Layout adapts
- [ ] Map resizes
- [ ] Stats stack vertically
- [ ] Control panel accessible
- [ ] No horizontal scroll

---

## 📊 Data Validation

### Backend Response Structure
Check console.log output:
```javascript
{
  summary: "string",
  baseline: {
    avg_travel_time_min: number,
    total_vkt: number,
    pm25: number,
    co2_kg: number
  },
  ranked: [{
    name: string,
    desc: string,
    pred_delta_tt_min: number
  }],
  edges_geojson: {
    type: "FeatureCollection",
    features: [...]
  },
  manifest: {
    run_id: string,
    mode: string,
    ...
  },
  outputs: {
    impactCards: [...],
    confidenceLevel: string
  }
}
```

### UI Updates Correctly
- [ ] Summary = data.summary
- [ ] Stats = data.baseline values
- [ ] Deltas calculated from impactCards
- [ ] Charts use baseline values
- [ ] Recommendations = data.ranked
- [ ] Map uses edges_geojson
- [ ] Manifest = data.manifest

---

## 🔧 Performance Tests

### Load Times
- [ ] Initial page load <2 seconds
- [ ] Map loads <3 seconds
- [ ] Fast mode response <5 seconds
- [ ] Deep mode response <15 seconds

### Animations
- [ ] 60fps smooth (no jank)
- [ ] No lag on slider drag
- [ ] Chart updates instant
- [ ] Hover effects immediate

### Memory
- [ ] No memory leaks
- [ ] Multiple analyses work
- [ ] Browser doesn't slow down

---

## 🐛 Common Issues

### Map not loading
**Solution:** Check MapBox token, verify internet connection

### Stats not updating
**Solution:** Check backend response structure, verify data.baseline exists

### Charts empty
**Solution:** Ensure baseline has numeric values, check chart update logic

### Recommendations missing
**Solution:** Verify data.ranked is array, check backend /chat response

### CORS errors
**Solution:** Ensure FastAPI CORS middleware configured, check API_BASE URL

---

## ✅ Sign-Off

Test Date: _______________
Tester: _______________

- [ ] All functional tests passed
- [ ] All visual tests passed
- [ ] All data validation passed
- [ ] All performance tests passed
- [ ] Ready for production

**Notes:**
______________________________
______________________________
______________________________

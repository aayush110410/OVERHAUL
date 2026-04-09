import React, { useState, useRef, useEffect } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './styles.css';
import { addFlyoverLayer, removeFlyoverLayer } from './FlyoverLayer';
import ImagenFlyoverVisuals from './ImagenFlyoverVisuals';
import Flyover3DViewer from './Flyover3DViewer';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
if (MAPBOX_TOKEN) {
  mapboxgl.accessToken = MAPBOX_TOKEN;
}

const API_BASE = 'http://localhost:8001';

export default function FlyoverSim() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const originMarker = useRef(null);
  const destMarker = useRef(null);

  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [originCoords, setOriginCoords] = useState(null);
  const [destCoords, setDestCoords] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [gettingLocation, setGettingLocation] = useState(false);
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [mapStatus, setMapStatus] = useState('INITIALIZING');
  const [pinMode, setPinMode] = useState(null);
  
  // 3D Viewer state
  const [show3DViewer, setShow3DViewer] = useState(false);
  const [flyoverData, setFlyoverData] = useState(null);

  const addLog = (msg, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, { time, msg, type }]);
  };

  const createMarkerElement = (type) => {
    const el = document.createElement('div');
    el.className = 'pin-marker ' + type + '-marker';
    el.innerHTML = '<div class="pin-icon"><svg width="30" height="40" viewBox="0 0 30 40"><path d="M15 0C6.716 0 0 6.716 0 15c0 8.284 15 25 15 25s15-16.716 15-25C30 6.716 23.284 0 15 0z" fill="' + (type === 'origin' ? '#00ff88' : '#ff4d00') + '"/><circle cx="15" cy="15" r="6" fill="white"/></svg></div><div class="pin-label">' + (type === 'origin' ? 'START' : 'END') + '</div>';
    return el;
  };

  const updateOriginMarker = (coords) => {
    if (originMarker.current) originMarker.current.remove();
    const el = createMarkerElement('origin');
    originMarker.current = new mapboxgl.Marker({ element: el, draggable: true })
      .setLngLat(coords)
      .addTo(map.current);
    originMarker.current.on('dragend', () => {
      const lngLat = originMarker.current.getLngLat();
      setOriginCoords([lngLat.lng, lngLat.lat]);
      setOrigin(lngLat.lat.toFixed(6) + ', ' + lngLat.lng.toFixed(6));
      addLog('Origin moved to: ' + lngLat.lat.toFixed(4) + ', ' + lngLat.lng.toFixed(4), 'info');
    });
  };

  const updateDestMarker = (coords) => {
    if (destMarker.current) destMarker.current.remove();
    const el = createMarkerElement('destination');
    destMarker.current = new mapboxgl.Marker({ element: el, draggable: true })
      .setLngLat(coords)
      .addTo(map.current);
    destMarker.current.on('dragend', () => {
      const lngLat = destMarker.current.getLngLat();
      setDestCoords([lngLat.lng, lngLat.lat]);
      setDestination(lngLat.lat.toFixed(6) + ', ' + lngLat.lng.toFixed(6));
      addLog('Destination moved to: ' + lngLat.lat.toFixed(4) + ', ' + lngLat.lng.toFixed(4), 'info');
    });
  };

  useEffect(() => {
    if (map.current) return;
    if (!mapContainer.current) return;

    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/streets-v11',
        center: [77.2090, 28.6139],
        zoom: 12,
        pitch: 60,
        bearing: -17.6
      });

      map.current.addControl(new mapboxgl.NavigationControl(), 'bottom-right');

      map.current.on('load', () => {
        setMapStatus('READY');
        addLog('Map ready - Use pin buttons to set points', 'system');
        map.current.addLayer({
          id: '3d-buildings',
          source: 'composite',
          'source-layer': 'building',
          type: 'fill-extrusion',
          minzoom: 14,
          paint: {
            'fill-extrusion-color': '#aaaaaa',
            'fill-extrusion-height': ['get', 'height'],
            'fill-extrusion-base': ['get', 'min_height'],
            'fill-extrusion-opacity': 0.7
          }
        });
      });

      map.current.on('error', (e) => {
        console.error('Map error:', e);
        setMapStatus('ERROR');
      });

    } catch (err) {
      console.error('Failed to create map:', err);
      setMapStatus('ERROR');
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!map.current) return;
    
    const handleClick = (e) => {
      const { lng, lat } = e.lngLat;
      if (pinMode === 'origin') {
        setOriginCoords([lng, lat]);
        setOrigin(lat.toFixed(6) + ', ' + lng.toFixed(6));
        updateOriginMarker([lng, lat]);
        addLog('Origin set: ' + lat.toFixed(4) + ', ' + lng.toFixed(4), 'success');
        setPinMode(null);
      } else if (pinMode === 'destination') {
        setDestCoords([lng, lat]);
        setDestination(lat.toFixed(6) + ', ' + lng.toFixed(6));
        updateDestMarker([lng, lat]);
        addLog('Destination set: ' + lat.toFixed(4) + ', ' + lng.toFixed(4), 'success');
        setPinMode(null);
      }
    };

    map.current.on('click', handleClick);
    return () => {
      if (map.current) map.current.off('click', handleClick);
    };
  }, [pinMode]);

  const getUserLocation = () => {
    if (!navigator.geolocation) {
      addLog('Geolocation not supported', 'error');
      return;
    }
    setGettingLocation(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        map.current.flyTo({ center: [longitude, latitude], zoom: 15, duration: 1500 });
        setOriginCoords([longitude, latitude]);
        setOrigin(latitude.toFixed(6) + ', ' + longitude.toFixed(6));
        updateOriginMarker([longitude, latitude]);
        addLog('Location found: ' + latitude.toFixed(4) + ', ' + longitude.toFixed(4), 'success');
        setGettingLocation(false);
      },
      () => {
        addLog('Location access denied', 'error');
        setGettingLocation(false);
      }
    );
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const response = await fetch(
        'https://api.mapbox.com/geocoding/v5/mapbox.places/' + encodeURIComponent(searchQuery) + '.json?access_token=' + mapboxgl.accessToken + '&country=IN'
      );
      const data = await response.json();
      if (data.features && data.features.length > 0) {
        const [lng, lat] = data.features[0].center;
        map.current.flyTo({ center: [lng, lat], zoom: 15, duration: 1500 });
        addLog('Found: ' + data.features[0].place_name, 'success');
      } else {
        addLog('Location not found', 'error');
      }
    } catch (error) {
      addLog('Search failed', 'error');
    }
  };

  const clearMap = () => {
    removeFlyoverLayer(map.current);
    const layers = ['flyover-corridor', 'flyover-path', 'pillars', 'pillar-labels', 'route-line', 'ai-pillars', 'existing-bridges', 'junctions', 'railways', 'street-route'];
    layers.forEach(id => { if (map.current.getLayer(id)) map.current.removeLayer(id); });
    const sources = ['flyover-corridor', 'flyover-path', 'flyover-route', 'pillars', 'route', 'ai-pillars', 'existing-bridges', 'junctions', 'railways', 'street-route'];
    sources.forEach(id => { if (map.current.getSource(id)) map.current.removeSource(id); });
  };

  const geocodeLocation = async (location) => {
    const coordMatch = location.match(/^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$/);
    if (coordMatch) {
      return [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])];
    }
    const response = await fetch(
      'https://api.mapbox.com/geocoding/v5/mapbox.places/' + encodeURIComponent(location) + '.json?access_token=' + mapboxgl.accessToken + '&country=IN'
    );
    const data = await response.json();
    if (data.features && data.features.length > 0) {
      return data.features[0].center;
    }
    throw new Error('Location not found: ' + location);
  };

  // Generate a smooth bezier curve for the flyover path
  const generateFlyoverPath = (start, end, numPoints = 50) => {
    const path = [];
    
    // Calculate direct distance
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Create control points for a smooth S-curve (like real flyovers)
    // Flyovers often have gentle curves for vehicle safety
    const midX = (start[0] + end[0]) / 2;
    const midY = (start[1] + end[1]) / 2;
    
    // Perpendicular offset for curve (varies based on distance)
    const perpX = -dy * 0.15; // 15% offset perpendicular to direct line
    const perpY = dx * 0.15;
    
    // Control points for cubic bezier
    const cp1 = [start[0] + dx * 0.25 + perpX * 0.5, start[1] + dy * 0.25 + perpY * 0.5];
    const cp2 = [start[0] + dx * 0.75 - perpX * 0.5, start[1] + dy * 0.75 - perpY * 0.5];
    
    // Generate bezier curve points
    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;
      const t2 = t * t;
      const t3 = t2 * t;
      const mt = 1 - t;
      const mt2 = mt * mt;
      const mt3 = mt2 * mt;
      
      // Cubic bezier formula
      const x = mt3 * start[0] + 3 * mt2 * t * cp1[0] + 3 * mt * t2 * cp2[0] + t3 * end[0];
      const y = mt3 * start[1] + 3 * mt2 * t * cp1[1] + 3 * mt * t2 * cp2[1] + t3 * end[1];
      
      path.push([x, y]);
    }
    
    return path;
  };

  // Calculate haversine distance in km
  const haversineDistance = (coord1, coord2) => {
    const R = 6371; // Earth radius in km
    const dLat = (coord2[1] - coord1[1]) * Math.PI / 180;
    const dLon = (coord2[0] - coord1[0]) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(coord1[1] * Math.PI / 180) * Math.cos(coord2[1] * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Build an obstacle-aware flyover path that bends around dense obstacles (buildings, junctions, rail)
  const buildObstacleAwarePath = (start, end, infrastructure) => {
    if (!infrastructure) return generateFlyoverPath(start, end, 70);

    // Collect obstacle points with radii (meters)
    const obstacles = [];
    (infrastructure.buildings || []).forEach(function(b) {
      if (b.center) obstacles.push({ coords: b.center, radius: 28 });
    });
    (infrastructure.junctions || []).forEach(function(j) {
      if (j.coords) obstacles.push({ coords: j.coords, radius: 25 });
    });
    (infrastructure.railways || []).forEach(function(r) {
      if (Array.isArray(r.coords) && r.coords.length > 1 && Array.isArray(r.coords[0])) {
        r.coords.forEach(function(rc) { obstacles.push({ coords: rc, radius: 30 }); });
      } else if (r.coords) {
        obstacles.push({ coords: r.coords, radius: 30 });
      }
    });

    if (obstacles.length === 0) return generateFlyoverPath(start, end, 70);

    // Bounding box for grid search
    const allLons = [start[0], end[0]].concat(obstacles.map(o => o.coords[0]));
    const allLats = [start[1], end[1]].concat(obstacles.map(o => o.coords[1]));
    const minLon = Math.min.apply(null, allLons) - 0.0015;
    const maxLon = Math.max.apply(null, allLons) + 0.0015;
    const minLat = Math.min.apply(null, allLats) - 0.0015;
    const maxLat = Math.max.apply(null, allLats) + 0.0015;

    const gridN = 42; // resolution of the grid
    const dLon = (maxLon - minLon) / gridN;
    const dLat = (maxLat - minLat) / gridN;

    const toCell = (pt) => {
      return [
        Math.max(0, Math.min(gridN - 1, Math.round((pt[0] - minLon) / dLon))),
        Math.max(0, Math.min(gridN - 1, Math.round((pt[1] - minLat) / dLat)))
      ];
    };

    const toCoord = (cell) => {
      return [
        minLon + cell[0] * dLon,
        minLat + cell[1] * dLat
      ];
    };

    const startCell = toCell(start);
    const endCell = toCell(end);

    const obstacleCost = (lon, lat) => {
      let cost = 1;
      for (const obs of obstacles) {
        const distM = haversineDistance([lon, lat], obs.coords) * 1000;
        if (distM < obs.radius) return 1e6; // block
        if (distM < obs.radius + 40) {
          cost += (obs.radius + 40 - distM) * 0.4; // soft avoidance
        }
      }
      return cost;
    };

    // A* on small grid
    const key = (x, y) => x + ',' + y;
    const open = new Map();
    const gScore = new Map();
    const fScore = new Map();
    const cameFrom = new Map();

    const h = (x, y) => {
      const dx = x - endCell[0];
      const dy = y - endCell[1];
      return Math.sqrt(dx * dx + dy * dy);
    };

    const startKey = key(startCell[0], startCell[1]);
    gScore.set(startKey, 0);
    fScore.set(startKey, h(startCell[0], startCell[1]));
    open.set(startKey, { x: startCell[0], y: startCell[1], f: fScore.get(startKey) });

    const dirs = [
      [1, 0], [-1, 0], [0, 1], [0, -1],
      [1, 1], [1, -1], [-1, 1], [-1, -1]
    ];

    while (open.size > 0) {
      // Pick node with smallest f
      let currentKey = null;
      let current = null;
      for (const [k, v] of open.entries()) {
        if (!current || v.f < current.f) {
          current = v; currentKey = k;
        }
      }

      if (current.x === endCell[0] && current.y === endCell[1]) {
        // Reconstruct path
        const cells = [];
        let ck = currentKey;
        while (ck) {
          const [cx, cy] = ck.split(',').map(Number);
          cells.push([cx, cy]);
          ck = cameFrom.get(ck);
        }
        cells.reverse();
        // Convert to coords and smooth
        let coords = cells.map(toCoord);
        const smooth = (path) => {
          if (path.length < 3) return path;
          const res = [path[0]];
          for (let i = 0; i < path.length - 1; i++) {
            const p0 = path[i];
            const p1 = path[i + 1];
            res.push([
              0.75 * p0[0] + 0.25 * p1[0],
              0.75 * p0[1] + 0.25 * p1[1]
            ]);
            res.push([
              0.25 * p0[0] + 0.75 * p1[0],
              0.25 * p0[1] + 0.75 * p1[1]
            ]);
          }
          res.push(path[path.length - 1]);
          return res;
        };
        coords = smooth(coords);
        coords = smooth(coords);
        return coords;
      }

      open.delete(currentKey);

      for (const [dx, dy] of dirs) {
        const nx = current.x + dx;
        const ny = current.y + dy;
        if (nx < 0 || ny < 0 || nx >= gridN || ny >= gridN) continue;

        const nKey = key(nx, ny);
        const coord = toCoord([nx, ny]);
        const moveCost = (dx === 0 || dy === 0) ? 1 : Math.SQRT2;
        const tentativeG = gScore.get(currentKey) + moveCost * obstacleCost(coord[0], coord[1]);

        if (tentativeG >= (gScore.get(nKey) || Infinity)) continue;

        cameFrom.set(nKey, currentKey);
        gScore.set(nKey, tentativeG);
        const f = tentativeG + h(nx, ny);
        fScore.set(nKey, f);
        open.set(nKey, { x: nx, y: ny, f });
      }
    }

    // Fallback if path not found
    return generateFlyoverPath(start, end, 70);
  };

  // Sample Mapbox 3D building layer to estimate local height/density for path points
  const sampleBuildingBoost = (coord) => {
    if (!map.current) return { boost: 0, count: 0, avgHeight: 0 };
    const p = map.current.project({ lng: coord[0], lat: coord[1] });
    const radiusPx = 28;
    const features = map.current.queryRenderedFeatures(
      [{ x: p.x - radiusPx, y: p.y - radiusPx }, { x: p.x + radiusPx, y: p.y + radiusPx }],
      { layers: ['3d-buildings'] }
    );
    if (!features || features.length === 0) return { boost: 0, count: 0, avgHeight: 0 };
    const sum = features.reduce((acc, f) => acc + (parseFloat(f.properties?.height) || 0), 0);
    const avg = sum / features.length;
    // Boost height modestly based on average height and density
    const boost = Math.min(12, (avg / 10) + features.length * 0.5);
    return { boost, count: features.length, avgHeight: avg };
  };

  const buildPathHeights = (flyoverPath, baseHeight) => {
    return flyoverPath.map((pt, idx) => {
      // Sample every 2nd point to reduce queries; interpolate for skipped ones
      if (idx % 2 === 0) {
        const { boost } = sampleBuildingBoost(pt);
        return baseHeight + boost;
      }
      // For odd indices, average neighbors if available
      const prev = Math.max(0, idx - 1);
      const next = Math.min(flyoverPath.length - 1, idx + 1);
      const prevHeight = baseHeight + sampleBuildingBoost(flyoverPath[prev]).boost;
      const nextHeight = baseHeight + sampleBuildingBoost(flyoverPath[next]).boost;
      return (prevHeight + nextHeight) / 2;
    });
  };

  const runSimulation = async () => {
    if (!origin.trim() || !destination.trim()) {
      addLog('Please set origin and destination', 'error');
      return;
    }

    setLoading(true);
    setStats(null);
    clearMap();
    addLog('🏗️ L-DRAGO Architect AI initializing...', 'system');

    try {
      let startCoords = originCoords;
      let endCoords = destCoords;

      if (!startCoords) {
        addLog('Geocoding origin...', 'info');
        startCoords = await geocodeLocation(origin);
        setOriginCoords(startCoords);
        updateOriginMarker(startCoords);
      }

      if (!endCoords) {
        addLog('Geocoding destination...', 'info');
        endCoords = await geocodeLocation(destination);
        setDestCoords(endCoords);
        updateDestMarker(endCoords);
      }

      addLog('📍 Start: [' + startCoords[1].toFixed(4) + ', ' + startCoords[0].toFixed(4) + ']', 'info');
      addLog('📍 End: [' + endCoords[1].toFixed(4) + ', ' + endCoords[0].toFixed(4) + ']', 'info');

      // Calculate DIRECT flyover distance (not street route)
      const directDistanceKm = haversineDistance(startCoords, endCoords);
      addLog('📏 Direct distance: ' + directDistanceKm.toFixed(2) + ' km', 'info');

      // Get street route for COMPARISON only (to show how much better flyover is)
      addLog('🛣️ Analyzing existing street route for comparison...', 'system');
      let streetDistanceKm = directDistanceKm * 1.4; // Default estimate
      let streetDurationMin = directDistanceKm * 3; // Rough estimate
      
      try {
        const routeRes = await fetch(
          'https://api.mapbox.com/directions/v5/mapbox/driving/' + startCoords[0] + ',' + startCoords[1] + ';' + endCoords[0] + ',' + endCoords[1] + '?geometries=geojson&overview=full&access_token=' + mapboxgl.accessToken
        );
        const routeData = await routeRes.json();
        if (routeData.routes && routeData.routes.length) {
          streetDistanceKm = routeData.routes[0].distance / 1000;
          streetDurationMin = routeData.routes[0].duration / 60;
          
          // Draw the old street route in gray (for comparison)
          map.current.addSource('street-route', {
            type: 'geojson',
            data: { type: 'Feature', geometry: routeData.routes[0].geometry }
          });
          map.current.addLayer({
            id: 'street-route',
            type: 'line',
            source: 'street-route',
            paint: { 'line-color': '#444444', 'line-width': 4, 'line-dasharray': [2, 2], 'line-opacity': 0.6 }
          });
          addLog('📊 Current street route: ' + streetDistanceKm.toFixed(2) + ' km, ' + Math.round(streetDurationMin) + ' min', 'info');
        }
      } catch (e) {
        addLog('Could not fetch street route for comparison', 'info');
      }

      // Generate PRELIMINARY flyover path (smooth arc) before obstacle-aware refinement
      addLog('✨ Drafting preliminary flyover curve...', 'system');
      const provisionalPath = generateFlyoverPath(startCoords, endCoords, 70);

      // Query AI for infrastructure analysis along the CUSTOM path
      addLog('🔍 Scanning flyover corridor for obstacles...', 'system');
      
      let aiData = null;
      let infrastructure = null;
      
      try {
        const aiRes = await fetch(API_BASE + '/plan-coords', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            start_lat: startCoords[1],
            start_lon: startCoords[0],
            end_lat: endCoords[1],
            end_lon: endCoords[0],
            route_coords: provisionalPath  // Send our draft custom path for infra scan
          })
        });
        
        if (aiRes.ok) {
          aiData = await aiRes.json();
          
          if (aiData.architect_notes) {
            aiData.architect_notes.forEach(function(note) {
              if (note.includes('⚠️')) {
                addLog(note, 'error');
              } else if (note.includes('✓') || note.includes('Design:')) {
                addLog(note, 'success');
              } else {
                addLog(note, 'system');
              }
            });
          }
          
          infrastructure = aiData.infrastructure;
        }
      } catch (aiErr) {
        addLog('⚠️ Using default analysis (backend unavailable)', 'info');
      }

      // Build obstacle-aware flyover path using infrastructure insights
      addLog('🧠 Bending flyover around obstacles (buildings, junctions, rail)...', 'system');
      const flyoverPath = buildObstacleAwarePath(startCoords, endCoords, infrastructure);

      // Calculate flyover distance along the refined curve
      let flyoverDistanceKm = 0;
      for (let i = 1; i < flyoverPath.length; i++) {
        flyoverDistanceKm += haversineDistance(flyoverPath[i-1], flyoverPath[i]);
      }

      // Show distance savings
      const distanceSaved = streetDistanceKm - flyoverDistanceKm;
      const distanceSavedPercent = Math.max(0, (distanceSaved / streetDistanceKm * 100).toFixed(0));
      addLog('🛫 Flyover path: ' + flyoverDistanceKm.toFixed(2) + ' km (direct over obstacles)', 'success');
      addLog('💡 Distance saved: ' + distanceSaved.toFixed(2) + ' km (' + distanceSavedPercent + '% shorter)', 'success');

      // Calculate smart pillar positions along the CUSTOM flyover path
      addLog('🏛️ Computing optimal pillar positions...', 'system');
      
      const pillarSpacingM = 30; // 30 meters between pillars
      const pillarSpacingDeg = pillarSpacingM / 111000; // Approximate degrees
      const smartPillars = [];
      let accumulatedDist = 0;
      let skippedForBuildings = 0;
      
      // Start anchor
      smartPillars.push({
        coords: flyoverPath[0],
        type: 'anchor',
        id: 1,
        reason: 'Ramp start - connects to ground level'
      });
      
      for (let i = 1; i < flyoverPath.length; i++) {
        const prev = flyoverPath[i - 1];
        const curr = flyoverPath[i];
        const segDist = Math.sqrt(Math.pow(curr[0] - prev[0], 2) + Math.pow(curr[1] - prev[1], 2));
        accumulatedDist += segDist;
        
        if (accumulatedDist >= pillarSpacingDeg) {
          // Check if near infrastructure (junction, railway, etc.)
          let pillarType = 'standard';
          let reason = 'Standard support pillar';
          let placePillar = true;
          
          // Avoid placing pillars inside buildings
          if (infrastructure && infrastructure.buildings) {
            for (const b of infrastructure.buildings) {
              const c = b.center || (Array.isArray(b.coords) && b.coords[0]);
              if (!c) continue;
              const bDist = haversineDistance(curr, c) * 1000; // m
              if (bDist < 25) {
                placePillar = false; // span over building
                skippedForBuildings += 1;
                reason = 'Spanning over building - no pillar here';
                break;
              }
            }
          }
          
          if (infrastructure) {
            // Check for nearby junctions
            for (const junction of (infrastructure.junctions || [])) {
              const jDist = Math.sqrt(
                Math.pow(curr[0] - junction.coords[0], 2) + 
                Math.pow(curr[1] - junction.coords[1], 2)
              );
              if (jDist < 0.0003) { // ~30m
                pillarType = 'elevated';
                reason = 'Elevated span over junction - no pillar here';
                break;
              }
            }
            
            // Check for railways
            for (const rail of (infrastructure.railways || [])) {
              if (rail.coords && Array.isArray(rail.coords[0])) {
                for (const rc of rail.coords) {
                  const rDist = Math.sqrt(Math.pow(curr[0] - rc[0], 2) + Math.pow(curr[1] - rc[1], 2));
                  if (rDist < 0.0004) {
                    pillarType = 'special';
                    reason = 'Railway crossing - increased clearance 7m+';
                    break;
                  }
                }
              }
            }
          }
          
          if (placePillar && pillarType !== 'elevated') {
            smartPillars.push({
              coords: curr,
              type: pillarType,
              id: smartPillars.length + 1,
              reason: reason
            });
          }
          accumulatedDist = 0;
        }
      }
      
      // End anchor
      smartPillars.push({
        coords: flyoverPath[flyoverPath.length - 1],
        type: 'anchor',
        id: smartPillars.length + 1,
        reason: 'Ramp end - connects to ground level'
      });
      
      addLog('✓ Designed ' + smartPillars.length + ' support pillars', 'success');
      if (skippedForBuildings > 0) {
        addLog('🏢 Spanning over ' + skippedForBuildings + ' building zones (no pillars placed there)', 'info');
      }

      // Draw the CUSTOM flyover path (bright green - the new direct route)
      map.current.addSource('flyover-route', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates: flyoverPath } }
      });

      map.current.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'flyover-route',
        paint: { 'line-color': '#88aa44', 'line-width': 3, 'line-opacity': 0.4 }
      });

      // Draw 3D flyover corridor
      const corridorWidth = 0.00018; // Wider for visibility
      const corridorPolygons = [];

      for (let i = 0; i < flyoverPath.length - 1; i++) {
        const p1 = flyoverPath[i];
        const p2 = flyoverPath[i + 1];
        const angle = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
        const perpAngle = angle + Math.PI / 2;
        const dx = Math.cos(perpAngle) * corridorWidth;
        const dy = Math.sin(perpAngle) * corridorWidth;

        corridorPolygons.push([[
          [p1[0] - dx, p1[1] - dy],
          [p1[0] + dx, p1[1] + dy],
          [p2[0] + dx, p2[1] + dy],
          [p2[0] - dx, p2[1] - dy],
          [p1[0] - dx, p1[1] - dy]
        ]]);
      }

      map.current.addSource('flyover-corridor', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'MultiPolygon', coordinates: corridorPolygons } }
      });

      const flyoverHeight = (aiData && aiData.flyover_height_m) || 12;
      // Skip the 2D corridor extrusion since we have proper 3D now

      // Build per-point heights based on Mapbox 3D buildings to lift deck over dense areas
      const pathHeights = buildPathHeights(flyoverPath, flyoverHeight);

      // Add Three.js deck & pillars for richer 3D visualization
      addFlyoverLayer(map.current, {
        flyover_path: flyoverPath,
        pathHeights,
        flyover_width_m: (aiData && aiData.flyover_width_m) || 12,
        flyover_height_m: flyoverHeight,
        pillars: smartPillars.map(p => p.coords)
      });

      // Draw pillars with different colors by type
      const pillarFeatures = smartPillars.map(function(pillar) {
        return {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: pillar.coords },
          properties: {
            id: pillar.id,
            type: pillar.type,
            reason: pillar.reason
          }
        };
      });

      map.current.addSource('ai-pillars', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: pillarFeatures }
      });

      map.current.addLayer({
        id: 'pillars',
        type: 'circle',
        source: 'ai-pillars',
        paint: {
          'circle-radius': 4,
          'circle-color': [
            'match',
            ['get', 'type'],
            'anchor', '#4a90a4',
            'special', '#8b5a8b',
            '#6b6b6b'
          ],
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff',
          'circle-opacity': 0.6
        }
      });

      // Pillar labels - smaller and subtler
      map.current.addLayer({
        id: 'pillar-labels',
        type: 'symbol',
        source: 'ai-pillars',
        layout: {
          'text-field': ['get', 'id'],
          'text-size': 8,
          'text-offset': [0, -1.2]
        },
        paint: {
          'text-color': '#555555',
          'text-halo-color': '#ffffff',
          'text-halo-width': 0.8,
          'text-opacity': 0.7
        }
      });

      // Draw infrastructure overlays
      if (infrastructure) {
        if (infrastructure.existing_bridges && infrastructure.existing_bridges.length > 0) {
          const bridgeFeatures = infrastructure.existing_bridges
            .filter(b => b.coords && b.coords.length > 1)
            .map(bridge => ({
              type: 'Feature',
              geometry: { type: 'LineString', coordinates: bridge.coords },
              properties: { name: bridge.name }
            }));
          
          if (bridgeFeatures.length > 0) {
            map.current.addSource('existing-bridges', {
              type: 'geojson',
              data: { type: 'FeatureCollection', features: bridgeFeatures }
            });
            map.current.addLayer({
              id: 'existing-bridges',
              type: 'line',
              source: 'existing-bridges',
              paint: { 'line-color': '#00d4ff', 'line-width': 4, 'line-dasharray': [2, 2] }
            });
            addLog('🌉 Found ' + bridgeFeatures.length + ' existing bridges (blue)', 'info');
          }
        }

        if (infrastructure.junctions && infrastructure.junctions.length > 0) {
          const junctionFeatures = infrastructure.junctions.map(j => ({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: j.coords },
            properties: { type: j.type }
          }));
          
          map.current.addSource('junctions', {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: junctionFeatures }
          });
          map.current.addLayer({
            id: 'junctions',
            type: 'circle',
            source: 'junctions',
            paint: {
              'circle-radius': 15,
              'circle-color': 'transparent',
              'circle-stroke-width': 3,
              'circle-stroke-color': '#ffff00'
            }
          });
          addLog('⚠️ ' + junctionFeatures.length + ' junctions (flyover spans over)', 'info');
        }

        if (infrastructure.railways && infrastructure.railways.length > 0) {
          const railFeatures = infrastructure.railways
            .filter(r => r.coords && Array.isArray(r.coords) && r.coords.length > 1 && Array.isArray(r.coords[0]))
            .map(rail => ({
              type: 'Feature',
              geometry: { type: 'LineString', coordinates: rail.coords },
              properties: { name: rail.name }
            }));
          
          if (railFeatures.length > 0) {
            map.current.addSource('railways', {
              type: 'geojson',
              data: { type: 'FeatureCollection', features: railFeatures }
            });
            map.current.addLayer({
              id: 'railways',
              type: 'line',
              source: 'railways',
              paint: { 'line-color': '#ff00ff', 'line-width': 3 }
            });
            addLog('🚂 ' + railFeatures.length + ' railway lines (magenta)', 'info');
          }
        }
      }

      // Fit map to show both old route and new flyover
      const bounds = new mapboxgl.LngLatBounds();
      flyoverPath.forEach(c => bounds.extend(c));
      map.current.fitBounds(bounds, { padding: 100, duration: 1500 });

      // Calculate stats
      const flyoverSpeedKmh = 60; // Flyovers allow higher speed
      const streetSpeedKmh = 25; // Congested street speed
      const flyoverTimeMin = (flyoverDistanceKm / flyoverSpeedKmh) * 60;
      const timeSaved = Math.round(streetDurationMin - flyoverTimeMin);
      const timeSavedPercent = Math.round((timeSaved / streetDurationMin) * 100);

      const anchorPillars = smartPillars.filter(p => p.type === 'anchor').length;
      const specialPillars = smartPillars.filter(p => p.type === 'special').length;

      setStats({
        lanes: flyoverDistanceKm > 1.5 ? 4 : 2,
        width: flyoverDistanceKm > 1.5 ? 14 : 9,
        height: flyoverHeight,
        pillars: smartPillars.length,
        anchorPillars: anchorPillars,
        specialPillars: specialPillars,
        pillarSpacing: pillarSpacingM,
        flyoverDistance: flyoverDistanceKm.toFixed(2),
        streetDistance: streetDistanceKm.toFixed(2),
        distanceSaved: distanceSaved.toFixed(2),
        distanceSavedPercent: distanceSavedPercent,
        streetTime: Math.round(streetDurationMin),
        flyoverTime: Math.round(flyoverTimeMin),
        timeSaved: timeSaved,
        timeSavedPercent: timeSavedPercent,
        existingBridges: infrastructure ? (infrastructure.existing_bridges || []).length : 0,
        junctions: infrastructure ? (infrastructure.junctions || []).length : 0,
        railways: infrastructure ? (infrastructure.railways || []).length : 0
      });

      // Store flyover data for 3D viewer
      setFlyoverData({
        flyover_path: flyoverPath,
        pillars: smartPillars.map(p => p.coords),
        flyover_width_m: flyoverDistanceKm > 1.5 ? 14 : 9,
        flyover_height_m: flyoverHeight,
        flyover_lanes: flyoverDistanceKm > 1.5 ? 4 : 2,
        pillar_spacing_m: pillarSpacingM,
      });

      setMapStatus('VISUALIZED');
      addLog('✅ Custom flyover design complete!', 'success');
      addLog('⏱️ Time saved: ' + timeSaved + ' min (' + timeSavedPercent + '% faster!)', 'success');
      addLog('📏 Distance saved: ' + distanceSaved.toFixed(1) + ' km (' + distanceSavedPercent + '% shorter!)', 'success');

    } catch (error) {
      console.error('Simulation error:', error);
      addLog('❌ Error: ' + error.message, 'error');
      setMapStatus('ERROR');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flyover-page">
      <nav className="flyover-navbar">
        <a href="/" className="flyover-logo">L-DRAGO</a>
        <div className="flyover-nav-center">
          <span className="flyover-nav-dot"></span>
          <span>AI FLYOVER SIMULATOR</span>
          <span className="flyover-nav-tag">v3.0</span>
        </div>
        <div className="nav-status">STATUS: {mapStatus}</div>
      </nav>

      <main className="flyover-main">
        <section className="flyover-map-section">
          <div className="flyover-map-card">
            <div className="flyover-card-header">
              <span>TERRAIN ANALYSIS</span>
              <span className={'flyover-tag ' + (mapStatus === 'ERROR' ? 'error' : mapStatus === 'VISUALIZED' ? 'success' : '')}>
                {pinMode ? 'CLICK MAP TO SET ' + pinMode.toUpperCase() : mapStatus}
              </span>
            </div>
            <div className="flyover-map-wrapper">
              <div ref={mapContainer} className={'flyover-map ' + (pinMode ? 'pin-mode' : '')} />
              
              <div className="map-search-bar">
                <input
                  type="text"
                  placeholder="Search location..."
                  value={searchQuery}
                  onChange={function(e) { setSearchQuery(e.target.value); }}
                  onKeyPress={function(e) { if (e.key === 'Enter') handleSearch(); }}
                />
                <button onClick={handleSearch}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#CCFF00" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="M21 21l-4.35-4.35"/>
                  </svg>
                </button>
              </div>

              <button 
                className="my-location-btn" 
                onClick={getUserLocation}
                disabled={gettingLocation}
                title="Get my location"
              >
                {gettingLocation ? (
                  <div className="loc-spinner" />
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>
                  </svg>
                )}
              </button>
            </div>
          </div>
        </section>

        <aside className="flyover-control-panel">
          <div className="flyover-input-section">
            <div className="flyover-card-header">
              <span>ROUTE PARAMETERS</span>
            </div>
            <div className="flyover-inputs">
              <div className="flyover-input-group">
                <label>ORIGIN (START POINT)</label>
                <div className="input-with-btn">
                  <input
                    type="text"
                    placeholder="Click pin or type location..."
                    value={origin}
                    onChange={function(e) { setOrigin(e.target.value); setOriginCoords(null); }}
                  />
                  <button 
                    className={'pin-btn ' + (pinMode === 'origin' ? 'active' : '')}
                    onClick={function() { setPinMode(pinMode === 'origin' ? null : 'origin'); }}
                    title="Drop pin on map"
                  >
                    📍
                  </button>
                </div>
              </div>

              <div className="flyover-input-group">
                <label>DESTINATION (END POINT)</label>
                <div className="input-with-btn">
                  <input
                    type="text"
                    placeholder="Click pin or type location..."
                    value={destination}
                    onChange={function(e) { setDestination(e.target.value); setDestCoords(null); }}
                  />
                  <button 
                    className={'pin-btn ' + (pinMode === 'destination' ? 'active' : '')}
                    onClick={function() { setPinMode(pinMode === 'destination' ? null : 'destination'); }}
                    title="Drop pin on map"
                  >
                    📍
                  </button>
                </div>
              </div>

              <button 
                className="simulate-btn" 
                onClick={runSimulation}
                disabled={loading || !origin || !destination}
              >
                {loading ? <div className="btn-spinner" /> : 'GENERATE AI FLYOVER'}
              </button>
            </div>
          </div>

          <div className="flyover-log-section">
            <div className="flyover-card-header">
              <span>AI ANALYSIS LOG</span>
              <span className="flyover-tag">{logs.length} ENTRIES</span>
            </div>
            <div className="flyover-log-content">
              {logs.length === 0 ? (
                <div className="log-empty">Drop pins to set start/end points...</div>
              ) : (
                logs.map(function(log, i) {
                  return (
                    <div key={i} className={'log-entry log-' + log.type}>
                      <span className="log-time">{log.time}</span>
                      <span className="log-msg">{log.msg}</span>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {stats && (
            <div className="flyover-stats-section">
              <div className="flyover-card-header">
                <span>🏗️ CUSTOM FLYOVER DESIGN</span>
                <span className="flyover-tag success">DIRECT ROUTE</span>
              </div>
              
              <div className="stats-grid">
                <div className="stat-item highlight">
                  <span className="stat-value">{stats.lanes}</span>
                  <span className="stat-label">FLYOVER LANES</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value">{stats.width}m</span>
                  <span className="stat-label">DECK WIDTH</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{stats.height}m</span>
                  <span className="stat-label">CLEARANCE</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{stats.pillars}</span>
                  <span className="stat-label">TOTAL PILLARS</span>
                </div>
              </div>

              <div className="stats-divider">📏 DISTANCE COMPARISON</div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#444'}}>{stats.streetDistance || stats.distance} km</span>
                  <span className="stat-label">STREET ROUTE</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#CCFF00'}}>{stats.flyoverDistance || stats.distance} km</span>
                  <span className="stat-label">FLYOVER (DIRECT)</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#00ff88'}}>-{stats.distanceSaved || '0'} km</span>
                  <span className="stat-label">DISTANCE SAVED</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#00ff88'}}>{stats.distanceSavedPercent || 0}% ↓</span>
                  <span className="stat-label">SHORTER</span>
                </div>
              </div>

              <div className="stats-divider">⏱️ TIME COMPARISON</div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#ff6b6b'}}>{stats.streetTime || stats.currentTime} min</span>
                  <span className="stat-label">STREET TIME</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#CCFF00'}}>{stats.flyoverTime || stats.newTime} min</span>
                  <span className="stat-label">FLYOVER TIME</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#00ff88'}}>-{stats.timeSaved} min</span>
                  <span className="stat-label">TIME SAVED</span>
                </div>
                <div className="stat-item highlight">
                  <span className="stat-value" style={{color: '#00ff88'}}>{stats.timeSavedPercent || 0}% ↓</span>
                  <span className="stat-label">FASTER</span>
                </div>
              </div>

              <div className="stats-divider">🔍 INFRASTRUCTURE SCAN</div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#00d4ff'}}>{stats.existingBridges || 0}</span>
                  <span className="stat-label">EXISTING BRIDGES</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#ffff00'}}>{stats.junctions || 0}</span>
                  <span className="stat-label">JUNCTIONS SPANNED</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#ff00ff'}}>{stats.railways || 0}</span>
                  <span className="stat-label">RAILWAY CROSSINGS</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value" style={{color: '#ff00ff'}}>{stats.specialPillars || 0}</span>
                  <span className="stat-label">SPECIAL PILLARS</span>
                </div>
              </div>

              <div className="pillar-legend">
                <span><span className="legend-dot anchor"></span> Anchor</span>
                <span><span className="legend-dot standard"></span> Standard</span>
                <span><span className="legend-dot" style={{backgroundColor: '#ff00ff'}}></span> Railway</span>
              </div>

              <div className="route-legend" style={{marginTop: '10px', fontSize: '11px', color: '#888'}}>
                <span style={{color: '#444'}}>━━</span> Old Street Route &nbsp;
                <span style={{color: '#CCFF00'}}>━━</span> New Flyover
              </div>

              <div className="eco-badge">
                DIRECT ROUTE OVER OBSTACLES | ARCHITECT OPTIMIZED
              </div>

              {/* 3D Explorer Button */}
              <button 
                className="explore-3d-btn"
                onClick={() => setShow3DViewer(true)}
                style={{
                  width: '100%',
                  padding: '14px 20px',
                  marginTop: '16px',
                  background: 'linear-gradient(135deg, #CCFF00 0%, #88cc00 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#0a0a0a',
                  fontSize: '14px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  transition: 'all 0.3s',
                  boxShadow: '0 4px 20px rgba(204, 255, 0, 0.3)',
                }}
              >
                🏗️ EXPLORE IN 3D
                <span style={{
                  background: '#0a0a0a',
                  color: '#CCFF00',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '10px',
                }}>
                  HIGH-FIDELITY
                </span>
              </button>
            </div>
          )}

          {/* Imagen 3 AI Visualizations */}
          {stats && (
            <ImagenFlyoverVisuals 
              flyoverData={{
                flyover_lanes: stats.lanes,
                flyover_width_m: parseFloat(stats.width),
                flyover_height_m: parseFloat(stats.height),
                road: { type: 'arterial' }
              }}
              locationName={origin || 'Delhi'}
            />
          )}
        </aside>
      </main>

      {/* 3D Flyover Viewer Modal */}
      {show3DViewer && flyoverData && (
        <Flyover3DViewer 
          flyoverData={flyoverData}
          locationName={origin || 'Delhi Flyover'}
          onClose={() => setShow3DViewer(false)}
        />
      )}
    </div>
  );
}

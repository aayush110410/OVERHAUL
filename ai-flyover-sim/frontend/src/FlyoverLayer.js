/**
 * HIGH-QUALITY 3D FLYOVER LAYER FOR MAPBOX
 * 
 * Creates photorealistic flyover models with:
 * - Smooth curved decks (using ExtrudeGeometry for continuity)
 * - Precision-fix for coordinate jitter
 * - Detailed barriers and railings
 * - Realistic pillars
 * - Matches the "Nano Banana Pro" high-fidelity mode
 */

import * as THREE from 'three'
import mapboxgl from 'mapbox-gl'

const LAYER_ID = 'flyover-3d'

let _layer = null
let _scene = null
let _renderer = null
let _camera = null
let _map = null

// Quality configuration
const CONFIG = {
  deckThickness: 1.2,
  barrierHeight: 1.1,
  railRadius: 0.06,
  pillarRadius: 1.0,
  pillarSpacing: 35,
  maxSegments: 2500,
  colors: {
    deckTop: 0x2a2a2a,      // Dark asphalt
    deckSide: 0x909090,     // Concrete gray
    barrier: 0xcccccc,      // Light concrete barriers
    pillar: 0x808080,       // Concrete pillar
    rail: 0xeeeeee,
  }
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n))
}

function createAsphaltTexture() {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  // Base
  ctx.fillStyle = '#2a2a2a'
  ctx.fillRect(0, 0, size, size)

  // Speckle noise
  const img = ctx.getImageData(0, 0, size, size)
  for (let i = 0; i < img.data.length; i += 4) {
    const r = Math.random()
    const v = 30 + Math.floor(r * 45)
    img.data[i + 0] = v
    img.data[i + 1] = v
    img.data[i + 2] = v
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)

  // Faint streaks
  ctx.globalAlpha = 0.10
  ctx.strokeStyle = '#000000'
  for (let y = 0; y < size; y += 6) {
    ctx.beginPath()
    ctx.moveTo(0, y + Math.random() * 2)
    ctx.lineTo(size, y + Math.random() * 2)
    ctx.stroke()
  }
  ctx.globalAlpha = 1

  const tex = new THREE.CanvasTexture(canvas)
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.RepeatWrapping
  tex.anisotropy = 8
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

function buildSweptDeckGeometry(curve, segments, widthM, thicknessM, topZOffset = 0) {
  const up = new THREE.Vector3(0, 0, 1)
  const hw = widthM / 2

  // 4 verts per segment point: LT, RT, LB, RB
  const positions = new Float32Array((segments + 1) * 4 * 3)
  const indices = []

  const p = new THREE.Vector3()
  const t = new THREE.Vector3()
  const lateral = new THREE.Vector3()
  const left = new THREE.Vector3()
  const right = new THREE.Vector3()

  for (let i = 0; i <= segments; i++) {
    const u = i / segments
    curve.getPointAt(u, p)
    curve.getTangentAt(u, t)

    // Fixed up-vector frame to avoid twist artifacts
    lateral.copy(up).cross(t)
    if (lateral.lengthSq() < 1e-8) {
      // Tangent is ~parallel to up; pick a stable fallback
      lateral.set(1, 0, 0)
    } else {
      lateral.normalize()
    }

    left.copy(p).addScaledVector(lateral, hw)
    right.copy(p).addScaledVector(lateral, -hw)

    const topZ = p.z + topZOffset
    const botZ = topZ - thicknessM

    const base = i * 12

    // LT
    positions[base + 0] = left.x
    positions[base + 1] = left.y
    positions[base + 2] = topZ

    // RT
    positions[base + 3] = right.x
    positions[base + 4] = right.y
    positions[base + 5] = topZ

    // LB
    positions[base + 6] = left.x
    positions[base + 7] = left.y
    positions[base + 8] = botZ

    // RB
    positions[base + 9] = right.x
    positions[base + 10] = right.y
    positions[base + 11] = botZ
  }

  for (let i = 0; i < segments; i++) {
    const a = i * 4
    const b = (i + 1) * 4

    // Top (LT, RT, RT_next, LT_next)
    indices.push(a + 0, a + 1, b + 1)
    indices.push(a + 0, b + 1, b + 0)

    // Bottom (LB, LB_next, RB_next, RB)
    indices.push(a + 2, b + 2, b + 3)
    indices.push(a + 2, b + 3, a + 3)

    // Left wall (LB, LT, LT_next, LB_next)
    indices.push(a + 2, a + 0, b + 0)
    indices.push(a + 2, b + 0, b + 2)

    // Right wall (RB, RB_next, RT_next, RT)
    indices.push(a + 3, b + 3, b + 1)
    indices.push(a + 3, b + 1, a + 1)
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  return geo
}

function buildRailCurve(curve, segments, lateralOffsetM, verticalOffsetM) {
  const up = new THREE.Vector3(0, 0, 1)
  const pts = []
  const p = new THREE.Vector3()
  const t = new THREE.Vector3()
  const lateral = new THREE.Vector3()

  for (let i = 0; i <= segments; i++) {
    const u = i / segments
    curve.getPointAt(u, p)
    curve.getTangentAt(u, t)
    lateral.copy(up).cross(t)
    if (lateral.lengthSq() < 1e-8) lateral.set(1, 0, 0)
    else lateral.normalize()
    pts.push(new THREE.Vector3(
      p.x + lateral.x * lateralOffsetM,
      p.y + lateral.y * lateralOffsetM,
      p.z + verticalOffsetM
    ))
  }
  return new THREE.CatmullRomCurve3(pts)
}

export function removeFlyoverLayer(map) {
  if (map && map.getLayer && map.getLayer(LAYER_ID)) {
    map.removeLayer(LAYER_ID)
  }
  if (_renderer) {
    _renderer.dispose()
    _renderer = null
  }
  if (_scene) {
    _scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose()
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose())
        } else {
          obj.material.dispose()
        }
      }
    })
  }
  _scene = null
  _camera = null
  _layer = null
  _map = null
}

export function addFlyoverLayer(map, flyover) {
  if (!flyover || !flyover.flyover_path || flyover.flyover_path.length < 2) {
    console.error('Invalid flyover data:', flyover)
    return
  }

  removeFlyoverLayer(map)
  _map = map

  const path = flyover.flyover_path
  const pathHeights = flyover.pathHeights || []
  const deckWidthM = flyover.flyover_width_m || 14
  const deckThicknessM = CONFIG.deckThickness
  const defaultHeight = flyover.flyover_height_m || 12

  _layer = {
    id: LAYER_ID,
    type: 'custom',
    renderingMode: '3d',

    onAdd: function (mapInstance, gl) {
      _scene = new THREE.Scene()
      _camera = new THREE.Camera()

      // Lighting setup
      const ambient = new THREE.AmbientLight(0xffffff, 0.6)
      _scene.add(ambient)
      
      const sun = new THREE.DirectionalLight(0xffffff, 1.5)
      sun.position.set(20, 50, 20)
      _scene.add(sun)
      
      const fill = new THREE.DirectionalLight(0xaaccff, 0.5)
      fill.position.set(-20, 10, -10)
      _scene.add(fill)

      _renderer = new THREE.WebGLRenderer({
        canvas: mapInstance.getCanvas(),
        context: gl,
        antialias: true,
        alpha: true
      })
      _renderer.autoClear = false
      _renderer.outputColorSpace = THREE.SRGBColorSpace
      _renderer.toneMapping = THREE.ACESFilmicToneMapping
      _renderer.toneMappingExposure = 1.15
      _renderer.physicallyCorrectLights = true

      // --- COORDINATE SYSTEM SETUP (PRECISION FIX) ---
      const startLngLat = { lng: path[0][0], lat: path[0][1] }
      // Center the model at the first point's Mercator coordinate
      const centerCoord = mapboxgl.MercatorCoordinate.fromLngLat(startLngLat, defaultHeight)
      const center = new THREE.Vector3(centerCoord.x, centerCoord.y, centerCoord.z)
      
      const scale = centerCoord.meterInMercatorCoordinateUnits()
      
      // Convert all points to LOCAL METERS relative to center
      const pointsLocal = path.map((coord, i) => {
        const h = pathHeights[i] !== undefined ? pathHeights[i] : defaultHeight
        const mc = mapboxgl.MercatorCoordinate.fromLngLat({ lng: coord[0], lat: coord[1] }, h)
        return new THREE.Vector3(
          (mc.x - center.x) / scale,
          (mc.y - center.y) / scale,
          (mc.z - center.z) / scale
        )
      })

      // Create Smooth Curve
      const curve = new THREE.CatmullRomCurve3(pointsLocal, false, 'centripetal', 0.5)
      const curveLength = curve.getLength()
      // Use high resolution but cap to avoid huge geometries / main-thread stalls
      // ~1 segment per 2m feels smooth without freezing on multi-km routes.
      const segments = clamp(Math.ceil(curveLength / 2), 250, CONFIG.maxSegments)
      const smoothCurve = new THREE.CatmullRomCurve3(curve.getPoints(segments), false, 'centripetal', 0.5)

      // --- BUILD GEOMETRY ---
      const group = new THREE.Group()
      
      // Place the group in Mercator world space
      group.position.copy(center)
      group.scale.set(scale, scale, scale)
      _scene.add(group)

      // 1) DECK (Swept mesh with fixed up-vector; prevents twist/chunk artifacts)
      const deckGeo = buildSweptDeckGeometry(smoothCurve, segments, deckWidthM, deckThicknessM, 0)
      const deckMat = new THREE.MeshStandardMaterial({
        color: CONFIG.colors.deckSide,
        roughness: 0.85,
        metalness: 0.02,
        flatShading: false,
        side: THREE.DoubleSide,
      })
      const deckMesh = new THREE.Mesh(deckGeo, deckMat)
      deckMesh.frustumCulled = false
      group.add(deckMesh)

      // 2) ROAD SURFACE (separate ribbon on top, textured)
      const asphalt = createAsphaltTexture()
      if (asphalt) {
        asphalt.repeat.set(Math.max(1, curveLength / 40), 2)
      }
      const roadWidth = Math.max(2, deckWidthM - 0.9)
      const roadGeo = buildSweptDeckGeometry(smoothCurve, segments, roadWidth, 0.08, 0.02)
      const roadMat = new THREE.MeshStandardMaterial({
        color: CONFIG.colors.deckTop,
        map: asphalt || null,
        roughness: 0.95,
        metalness: 0.0,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
        side: THREE.DoubleSide,
      })
      const roadMesh = new THREE.Mesh(roadGeo, roadMat)
      roadMesh.frustumCulled = false
      group.add(roadMesh)

      // 3) RAILS (tube rails on both edges)
      const railOffset = deckWidthM / 2 - 0.25
      const railSegments = Math.max(160, Math.floor(segments / 3))
      const leftRail = buildRailCurve(smoothCurve, railSegments, railOffset, 0.95)
      const rightRail = buildRailCurve(smoothCurve, railSegments, -railOffset, 0.95)
      const railGeo = new THREE.TubeGeometry(leftRail, railSegments, CONFIG.railRadius, 10, false)
      const railGeo2 = new THREE.TubeGeometry(rightRail, railSegments, CONFIG.railRadius, 10, false)
      const railMat = new THREE.MeshStandardMaterial({
        color: CONFIG.colors.rail,
        roughness: 0.35,
        metalness: 0.25,
      })
      const rail1 = new THREE.Mesh(railGeo, railMat)
      const rail2 = new THREE.Mesh(railGeo2, railMat)
      rail1.frustumCulled = false
      rail2.frustumCulled = false
      group.add(rail1)
      group.add(rail2)

      // 3. BARRIERS & PILLARS
      // Simple loop over points for pillars
      const pillarMat = new THREE.MeshStandardMaterial({ color: CONFIG.colors.pillar })
      const spacing = CONFIG.pillarSpacing
      let dist = 0
      const curvePoints = smoothCurve.getPoints(segments)
      
      for (let i = 1; i < curvePoints.length; i++) {
        dist += curvePoints[i].distanceTo(curvePoints[i-1])
        if (dist >= spacing) {
          dist = 0
          const p = curvePoints[i]
          
          // Determine pillar height (down to ground)
          // Ground Z (local) = (0 altitude - center.z) / scale
          // NOTE: Mapbox Z might assume flat plane at 0 if no terrain data access in custom layer
          // We assume pillars go down ~defaultHeight meters
          
           // Approx ground Z in this local meter space (no terrain sampling available here)
           const groundZLocal = (0 - center.z) / scale
           const pillarHeight = (p.z - deckThicknessM) - groundZLocal
          if (pillarHeight > 2) {
             const geo = new THREE.CylinderGeometry(CONFIG.pillarRadius, CONFIG.pillarRadius, pillarHeight, 16)
             geo.rotateX(Math.PI / 2) // Z-up
             const mesh = new THREE.Mesh(geo, pillarMat)
             // Position: deck bottom minus half pillar height
             mesh.position.set(p.x, p.y, p.z - deckThicknessM - pillarHeight/2)
             mesh.frustumCulled = false
             group.add(mesh)
          }
        }
      }

      console.log('✅ Flyover Pro Layer Built: Single continuous mesh, no chunks.')
    },

    render: function (gl, matrix) {
      if (!_scene || !_camera || !_renderer) return
      _camera.projectionMatrix = new THREE.Matrix4().fromArray(matrix)
      _renderer.resetState()
      _renderer.render(_scene, _camera)
      _map.triggerRepaint()
    }
  }

  map.addLayer(_layer)
}

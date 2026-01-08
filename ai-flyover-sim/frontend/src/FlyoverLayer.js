import * as THREE from 'three'
import mapboxgl from 'mapbox-gl'

const LAYER_ID = 'flyover-3d'

let _layer = null
let _scene = null
let _renderer = null
let _camera = null
let _map = null

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

function buildDeckRibbonGeometry(points, width, thickness) {
  const n = points.length
  if (n < 2) return null

  const up = new THREE.Vector3(0, 0, 1)
  const hw = width / 2
  const ht = thickness / 2

  const positions = []
  const indices = []

  for (let i = 0; i < n; i++) {
    const curr = points[i]
    const prev = points[Math.max(0, i - 1)]
    const next = points[Math.min(n - 1, i + 1)]

    const tangent = new THREE.Vector3().subVectors(next, prev)
    if (tangent.lengthSq() < 1e-18) tangent.set(1, 0, 0)
    tangent.normalize()

    // Keep the deck horizontal: right is tangent x globalUp.
    const right = new THREE.Vector3().crossVectors(tangent, up)
    if (right.lengthSq() < 1e-18) right.set(1, 0, 0)
    right.normalize()

    // upAxis is right x tangent.
    const upAxis = new THREE.Vector3().crossVectors(right, tangent)
    if (upAxis.lengthSq() < 1e-18) upAxis.copy(up)
    upAxis.normalize()

    const bl = new THREE.Vector3().copy(curr).addScaledVector(right, -hw).addScaledVector(upAxis, -ht)
    const br = new THREE.Vector3().copy(curr).addScaledVector(right, hw).addScaledVector(upAxis, -ht)
    const tr = new THREE.Vector3().copy(curr).addScaledVector(right, hw).addScaledVector(upAxis, ht)
    const tl = new THREE.Vector3().copy(curr).addScaledVector(right, -hw).addScaledVector(upAxis, ht)

    positions.push(
      bl.x, bl.y, bl.z,
      br.x, br.y, br.z,
      tr.x, tr.y, tr.z,
      tl.x, tl.y, tl.z
    )
  }

  for (let i = 0; i < n - 1; i++) {
    const b = i * 4
    // Top
    indices.push(b + 3, b + 2, b + 6)
    indices.push(b + 3, b + 6, b + 7)
    // Bottom
    indices.push(b + 0, b + 4, b + 5)
    indices.push(b + 0, b + 5, b + 1)
    // Left
    indices.push(b + 0, b + 3, b + 7)
    indices.push(b + 0, b + 7, b + 4)
    // Right
    indices.push(b + 1, b + 5, b + 6)
    indices.push(b + 1, b + 6, b + 2)
  }

  // Caps
  indices.push(0, 1, 2)
  indices.push(0, 2, 3)
  const last = (n - 1) * 4
  indices.push(last, last + 2, last + 1)
  indices.push(last, last + 3, last + 2)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

function sampleSmoothPoints(points3D, segments) {
  const curve = new THREE.CatmullRomCurve3(points3D, false, 'catmullrom', 0.5)
  return curve.getSpacedPoints(segments)
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
  const deckWidthM = (flyover.flyover_width_m || 14) * 2.6 // large structure
  const deckThicknessM = 3.0
  const defaultHeight = flyover.flyover_height_m || 18

  _layer = {
    id: LAYER_ID,
    type: 'custom',
    renderingMode: '3d',

    onAdd: function (mapInstance, gl) {
      _scene = new THREE.Scene()
      _camera = new THREE.Camera()

      // Simple lighting
      const light = new THREE.DirectionalLight(0xffffff, 1.2)
      light.position.set(0, 0, 1)
      _scene.add(light)

      const ambient = new THREE.AmbientLight(0xffffff, 0.6)
      _scene.add(ambient)

      _renderer = new THREE.WebGLRenderer({
        canvas: mapInstance.getCanvas(),
        context: gl,
        antialias: true
      })
      _renderer.autoClear = false
      _renderer.setClearColor(0x000000, 0)

      // Scale factor
      const refCoord = mapboxgl.MercatorCoordinate.fromLngLat({ lng: path[0][0], lat: path[0][1] }, 0)
      const scale = refCoord.meterInMercatorCoordinateUnits()

      // Convert path to THREE.js Vector3 points
      const points3D = path.map((coord, i) => {
        const h = pathHeights[i] || defaultHeight
        const mc = mapboxgl.MercatorCoordinate.fromLngLat({ lng: coord[0], lat: coord[1] }, h)
        return new THREE.Vector3(mc.x, mc.y, mc.z)
      })

      const segments = Math.max(260, path.length * 40)
      const smooth = sampleSmoothPoints(points3D, segments)

      // Deck dimensions (in mercator units)
      const deckWidth = deckWidthM * scale
      const deckThickness = deckThicknessM * scale

      // Materials
      const deckMaterial = new THREE.MeshStandardMaterial({
        color: 0x7a7a7a,
        roughness: 0.9,
        metalness: 0.05,
        side: THREE.DoubleSide
      })
      const pillarMaterial = new THREE.MeshStandardMaterial({
        color: 0x5f5f5f,
        roughness: 0.95,
        metalness: 0.05
      })

      // ========== DECK: horizontal thick rectangle (smooth) ==========
      const deckGeometry = buildDeckRibbonGeometry(smooth, deckWidth, deckThickness)
      if (deckGeometry) {
        _scene.add(new THREE.Mesh(deckGeometry, deckMaterial))
      }

      // ========== CYLINDRICAL PILLARS: placed along the smooth path ==========
      const pillarRadius = 1.3 * scale
      const pillarSpacingM = 32

      let accMeters = 0
      let pillarCount = 0

      for (let i = 1; i < smooth.length; i++) {
        const p0 = smooth[i - 1]
        const p1 = smooth[i]
        const segMeters = p0.distanceTo(p1) / scale
        accMeters += segMeters
        if (accMeters < pillarSpacingM) continue
        accMeters = 0

        const deckBottomZ = p1.z - deckThickness / 2
        const pillarHeight = Math.max(0, deckBottomZ)
        if (pillarHeight <= 0) continue

        const cylinderGeom = new THREE.CylinderGeometry(pillarRadius, pillarRadius * 1.08, pillarHeight, 20)
        cylinderGeom.rotateX(Math.PI / 2)

        const pillar = new THREE.Mesh(cylinderGeom, pillarMaterial)
        pillar.position.set(p1.x, p1.y, pillarHeight / 2)
        _scene.add(pillar)
        pillarCount++
      }

      console.log(`✓ Flyover deck ready (segments: ${segments}), pillars: ${pillarCount}`)
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

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

export function addFlyoverLayer(map, flyover) {
  if (!flyover || !flyover.flyover_path || flyover.flyover_path.length < 2) {
    console.error('Invalid flyover data:', flyover)
    return
  }

  removeFlyoverLayer(map)
  _map = map

  const path = flyover.flyover_path
  const pathHeights = flyover.pathHeights || []
  const deckWidthM = (flyover.flyover_width_m || 14) * 1.5  // Wider deck
  const defaultHeight = flyover.flyover_height_m || 15

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

      // Scale factor
      const refCoord = mapboxgl.MercatorCoordinate.fromLngLat({ lng: path[0][0], lat: path[0][1] }, 0)
      const scale = refCoord.meterInMercatorCoordinateUnits()

      // Convert path to THREE.js Vector3 points
      const points3D = path.map((coord, i) => {
        const h = pathHeights[i] || defaultHeight
        const mc = mapboxgl.MercatorCoordinate.fromLngLat({ lng: coord[0], lat: coord[1] }, h)
        return new THREE.Vector3(mc.x, mc.y, mc.z)
      })

      // Create smooth curve
      const curve = new THREE.CatmullRomCurve3(points3D, false, 'catmullrom', 0.5)

      // Deck dimensions
      const deckWidth = deckWidthM * scale
      const deckThickness = 1.0 * scale

      // Materials - simple gray concrete
      const deckMaterial = new THREE.MeshLambertMaterial({ 
        color: 0x808080,
        side: THREE.DoubleSide
      })
      
      const pillarMaterial = new THREE.MeshLambertMaterial({ 
        color: 0x606060 
      })

      // ========== DECK using TubeGeometry (smooth) ==========
      // Create a flat rectangular cross-section shape
      const deckShape = new THREE.Shape()
      const hw = deckWidth / 2
      const ht = deckThickness / 2
      deckShape.moveTo(-hw, -ht)
      deckShape.lineTo(hw, -ht)
      deckShape.lineTo(hw, ht)
      deckShape.lineTo(-hw, ht)
      deckShape.closePath()

      // Extrude along curve
      const segments = Math.max(100, path.length * 20)
      const deckGeometry = new THREE.ExtrudeGeometry(deckShape, {
        steps: segments,
        bevelEnabled: false,
        extrudePath: curve
      })

      const deckMesh = new THREE.Mesh(deckGeometry, deckMaterial)
      _scene.add(deckMesh)

      // ========== CYLINDRICAL PILLARS ==========
      const pillarRadius = 0.8 * scale
      const pillarSpacingM = 35  // Every 35 meters

      // Generate pillar positions along path
      const pillarCoords = []
      let distAccum = 0

      for (let i = 1; i < path.length; i++) {
        const p1 = path[i - 1]
        const p2 = path[i]
        const segDist = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 111000

        distAccum += segDist

        while (distAccum >= pillarSpacingM) {
          const t = 1 - (distAccum - pillarSpacingM) / segDist
          const lng = p1[0] + t * (p2[0] - p1[0])
          const lat = p1[1] + t * (p2[1] - p1[1])
          const h = pathHeights[i] || defaultHeight
          pillarCoords.push({ lng, lat, h })
          distAccum -= pillarSpacingM
        }
      }

      // Add backend-provided pillars
      ;(flyover.pillars || []).forEach(coord => {
        let h = defaultHeight
        for (let i = 0; i < path.length; i++) {
          const d = Math.hypot(coord[0] - path[i][0], coord[1] - path[i][1])
          if (d < 0.001) h = pathHeights[i] || defaultHeight
        }
        pillarCoords.push({ lng: coord[0], lat: coord[1], h })
      })

      // Create each cylindrical pillar
      pillarCoords.forEach(({ lng, lat, h }) => {
        const mc = mapboxgl.MercatorCoordinate.fromLngLat({ lng, lat }, 0)
        const pillarHeight = h * scale

        // Cylinder: radiusTop, radiusBottom, height, radialSegments
        const cylinderGeom = new THREE.CylinderGeometry(
          pillarRadius, 
          pillarRadius * 1.1,  // Slightly wider at base
          pillarHeight, 
          16  // Smooth cylinder
        )

        // Rotate to stand upright (cylinder default is Y-up, we need Z-up)
        cylinderGeom.rotateX(Math.PI / 2)

        const pillar = new THREE.Mesh(cylinderGeom, pillarMaterial)
        // Position at ground level, cylinder center is at half height
        pillar.position.set(mc.x, mc.y, -pillarHeight / 2)
        _scene.add(pillar)
      })

      console.log(`✓ Simple flyover: ${path.length} path points, ${pillarCoords.length} pillars`)
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

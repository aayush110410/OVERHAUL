/**
 * HIGH-FIDELITY 3D FLYOVER LAYER
 * 
 * Creates photorealistic 3D flyover models with:
 * - Smooth curved deck geometry with proper thickness
 * - Detailed pillars with caps and bases
 * - Safety barriers on both sides
 * - Street lighting poles
 * - Lane markings
 * - PBR materials with realistic shading
 * - Interactive camera controls
 * - Hover effects and animations
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const LAYER_ID = 'flyover-3d-hifi'

// Global state
let _map = null
let _scene = null
let _camera = null
let _renderer = null
let _controls = null
let _animationId = null
let _flyoverGroup = null
let _raycaster = null
let _mouse = null
let _hoveredObject = null

// Configuration
const CONFIG = {
  // Deck specifications
  deckThickness: 0.8,      // meters
  deckEdgeThickness: 0.15, // Edge lip
  curveSegments: 100,      // Smoothness
  
  // Pillar specifications  
  pillarRadius: 0.8,       // meters
  pillarCapHeight: 0.4,
  pillarBaseRadius: 1.2,
  pillarBaseHeight: 0.3,
  
  // Barrier specifications
  barrierHeight: 1.1,      // meters
  barrierThickness: 0.15,
  barrierPostSpacing: 3,   // meters
  
  // Lighting poles
  poleHeight: 8,           // meters
  poleSpacing: 30,         // meters
  
  // Materials
  concreteColor: 0x8a8a8a,
  deckTopColor: 0x3a3a3a,
  barrierColor: 0xcccccc,
  poleColor: 0x555555,
  lightColor: 0xffffee,
  
  // Lighting
  ambientIntensity: 0.4,
  sunIntensity: 1.2,
  sunPosition: [100, 200, 100],
}

/**
 * Convert lat/lon to Mapbox Mercator coordinates
 */
function lngLatToMercator(lng, lat) {
  const mercatorScale = 1 / Math.cos(lat * Math.PI / 180)
  const worldSize = 512 * Math.pow(2, _map?.getZoom() || 15)
  
  const x = (180 + lng) / 360 * worldSize
  const y = (180 - (180 / Math.PI * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)))) / 360 * worldSize
  
  return new THREE.Vector3(x, 0, y)
}

/**
 * Convert flyover path to 3D world coordinates
 */
function pathToWorldCoords(path, heightM) {
  if (!_map) return []
  
  const center = _map.getCenter()
  const centerMerc = lngLatToMercator(center.lng, center.lat)
  const metersPerPixel = 0.075 * Math.pow(2, 15 - _map.getZoom())
  
  return path.map(([lng, lat]) => {
    const merc = lngLatToMercator(lng, lat)
    return new THREE.Vector3(
      (merc.x - centerMerc.x) / metersPerPixel,
      heightM / metersPerPixel,
      (merc.z - centerMerc.z) / metersPerPixel
    )
  })
}

/**
 * Create smooth Catmull-Rom spline from path points
 */
function createSmoothPath(points, segments = CONFIG.curveSegments) {
  if (points.length < 2) return points
  
  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal', 0.5)
  return curve.getPoints(segments)
}

/**
 * Create high-quality deck geometry with proper cross-section
 */
function createDeckGeometry(path, width, thickness = CONFIG.deckThickness) {
  const smoothPath = createSmoothPath(path)
  const n = smoothPath.length
  if (n < 2) return null

  const hw = width / 2
  const ht = thickness / 2
  const edgeH = CONFIG.deckEdgeThickness

  const positions = []
  const normals = []
  const uvs = []
  const indices = []

  // Generate cross-section at each point
  for (let i = 0; i < n; i++) {
    const curr = smoothPath[i]
    const prev = smoothPath[Math.max(0, i - 1)]
    const next = smoothPath[Math.min(n - 1, i + 1)]

    // Calculate tangent and perpendicular
    const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
    const up = new THREE.Vector3(0, 1, 0)
    const right = new THREE.Vector3().crossVectors(tangent, up).normalize()

    // Cross-section points (more detailed profile)
    // Bottom left corner
    const bl = new THREE.Vector3().copy(curr).addScaledVector(right, -hw).addScaledVector(up, -ht)
    // Bottom right corner
    const br = new THREE.Vector3().copy(curr).addScaledVector(right, hw).addScaledVector(up, -ht)
    // Top right outer edge
    const tro = new THREE.Vector3().copy(curr).addScaledVector(right, hw).addScaledVector(up, ht)
    // Top right inner (for edge lip)
    const tri = new THREE.Vector3().copy(curr).addScaledVector(right, hw - edgeH).addScaledVector(up, ht + edgeH)
    // Top left inner (for edge lip)
    const tli = new THREE.Vector3().copy(curr).addScaledVector(right, -(hw - edgeH)).addScaledVector(up, ht + edgeH)
    // Top left outer edge
    const tlo = new THREE.Vector3().copy(curr).addScaledVector(right, -hw).addScaledVector(up, ht)

    // UV coordinate along length
    const u = i / (n - 1)

    // Add vertices (6 per cross-section for detailed profile)
    const verts = [bl, br, tro, tri, tli, tlo]
    const uvCoords = [0, 1, 1, 0.9, 0.1, 0]
    
    verts.forEach((v, vi) => {
      positions.push(v.x, v.y, v.z)
      uvs.push(u * 10, uvCoords[vi]) // Repeat texture
    })

    // Calculate normals
    normals.push(0, -1, 0) // bl - bottom
    normals.push(0, -1, 0) // br - bottom
    normals.push(1, 0, 0)  // tro - right side
    normals.push(0, 1, 0)  // tri - top
    normals.push(0, 1, 0)  // tli - top
    normals.push(-1, 0, 0) // tlo - left side
  }

  // Create faces
  for (let i = 0; i < n - 1; i++) {
    const base = i * 6

    // Bottom face
    indices.push(base, base + 6, base + 7, base, base + 7, base + 1)
    // Right side
    indices.push(base + 1, base + 7, base + 8, base + 1, base + 8, base + 2)
    // Top right edge
    indices.push(base + 2, base + 8, base + 9, base + 2, base + 9, base + 3)
    // Top surface
    indices.push(base + 3, base + 9, base + 10, base + 3, base + 10, base + 4)
    // Top left edge
    indices.push(base + 4, base + 10, base + 11, base + 4, base + 11, base + 5)
    // Left side
    indices.push(base + 5, base + 11, base + 6, base + 5, base + 6, base + 0)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  
  return geometry
}

/**
 * Create detailed pillar with cap and base
 */
function createPillarGeometry(height, radius = CONFIG.pillarRadius) {
  const group = new THREE.Group()
  
  // Main column
  const columnGeo = new THREE.CylinderGeometry(radius, radius * 1.05, height - CONFIG.pillarCapHeight - CONFIG.pillarBaseHeight, 24)
  const column = new THREE.Mesh(columnGeo)
  column.position.y = (height - CONFIG.pillarCapHeight - CONFIG.pillarBaseHeight) / 2 + CONFIG.pillarBaseHeight
  group.add(column)
  
  // Capital (top cap)
  const capGeo = new THREE.CylinderGeometry(radius * 1.3, radius, CONFIG.pillarCapHeight, 24)
  const cap = new THREE.Mesh(capGeo)
  cap.position.y = height - CONFIG.pillarCapHeight / 2
  group.add(cap)
  
  // Base
  const baseGeo = new THREE.CylinderGeometry(radius * 1.05, CONFIG.pillarBaseRadius, CONFIG.pillarBaseHeight, 24)
  const base = new THREE.Mesh(baseGeo)
  base.position.y = CONFIG.pillarBaseHeight / 2
  group.add(base)
  
  // Footing
  const footGeo = new THREE.BoxGeometry(CONFIG.pillarBaseRadius * 2.2, 0.2, CONFIG.pillarBaseRadius * 2.2)
  const foot = new THREE.Mesh(footGeo)
  foot.position.y = 0.1
  group.add(foot)
  
  return group
}

/**
 * Create safety barrier along the deck edge
 */
function createBarrierGeometry(path, width, side = 'left') {
  const smoothPath = createSmoothPath(path)
  const n = smoothPath.length
  if (n < 2) return null

  const group = new THREE.Group()
  const hw = width / 2
  const offset = side === 'left' ? -hw + 0.1 : hw - 0.1

  // Continuous barrier rail
  const railPoints = []
  for (let i = 0; i < n; i++) {
    const curr = smoothPath[i]
    const prev = smoothPath[Math.max(0, i - 1)]
    const next = smoothPath[Math.min(n - 1, i + 1)]
    
    const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
    const up = new THREE.Vector3(0, 1, 0)
    const right = new THREE.Vector3().crossVectors(tangent, up).normalize()
    
    const point = new THREE.Vector3().copy(curr)
      .addScaledVector(right, offset)
      .addScaledVector(up, CONFIG.deckThickness / 2 + CONFIG.barrierHeight / 2)
    
    railPoints.push(point)
  }

  // Create rail as tube
  const railCurve = new THREE.CatmullRomCurve3(railPoints)
  const railGeo = new THREE.TubeGeometry(railCurve, n * 2, CONFIG.barrierThickness / 2, 8, false)
  const rail = new THREE.Mesh(railGeo)
  group.add(rail)

  // Add vertical posts
  const pathLength = calculatePathLength(smoothPath)
  const numPosts = Math.floor(pathLength / CONFIG.barrierPostSpacing)
  
  for (let i = 0; i <= numPosts; i++) {
    const t = i / numPosts
    const idx = Math.floor(t * (n - 1))
    const point = railPoints[Math.min(idx, railPoints.length - 1)]
    
    const postGeo = new THREE.BoxGeometry(0.08, CONFIG.barrierHeight, 0.08)
    const post = new THREE.Mesh(postGeo)
    post.position.copy(point)
    post.position.y -= CONFIG.barrierHeight / 4
    group.add(post)
  }

  return group
}

/**
 * Create street lighting poles
 */
function createLightingPoles(path, width, spacing = CONFIG.poleSpacing) {
  const smoothPath = createSmoothPath(path, 50)
  const n = smoothPath.length
  const group = new THREE.Group()
  
  const pathLength = calculatePathLength(smoothPath)
  const numPoles = Math.floor(pathLength / spacing)
  const hw = width / 2

  for (let i = 0; i <= numPoles; i++) {
    const t = i / numPoles
    const idx = Math.min(Math.floor(t * (n - 1)), n - 1)
    const curr = smoothPath[idx]
    const prev = smoothPath[Math.max(0, idx - 1)]
    const next = smoothPath[Math.min(n - 1, idx + 1)]
    
    const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
    const up = new THREE.Vector3(0, 1, 0)
    const right = new THREE.Vector3().crossVectors(tangent, up).normalize()

    // Alternate sides
    const side = i % 2 === 0 ? 1 : -1
    const offset = (hw - 0.5) * side

    // Pole
    const poleGeo = new THREE.CylinderGeometry(0.08, 0.12, CONFIG.poleHeight, 8)
    const pole = new THREE.Mesh(poleGeo)
    pole.position.copy(curr)
      .addScaledVector(right, offset)
      .addScaledVector(up, CONFIG.deckThickness / 2 + CONFIG.poleHeight / 2)
    group.add(pole)

    // Lamp arm
    const armGeo = new THREE.CylinderGeometry(0.04, 0.04, 1.5, 6)
    const arm = new THREE.Mesh(armGeo)
    arm.rotation.z = Math.PI / 2 * -side
    arm.position.copy(curr)
      .addScaledVector(right, offset - 0.6 * side)
      .addScaledVector(up, CONFIG.deckThickness / 2 + CONFIG.poleHeight - 0.3)
    group.add(arm)

    // Light fixture
    const lightGeo = new THREE.SphereGeometry(0.15, 16, 16)
    const light = new THREE.Mesh(lightGeo)
    light.position.copy(curr)
      .addScaledVector(right, offset - 1.2 * side)
      .addScaledVector(up, CONFIG.deckThickness / 2 + CONFIG.poleHeight - 0.5)
    light.userData.isLight = true
    group.add(light)

    // Actual point light
    const pointLight = new THREE.PointLight(CONFIG.lightColor, 0.5, 20)
    pointLight.position.copy(light.position)
    group.add(pointLight)
  }

  return group
}

/**
 * Create lane markings on deck surface
 */
function createLaneMarkings(path, width, lanes) {
  const smoothPath = createSmoothPath(path, 200)
  const n = smoothPath.length
  const group = new THREE.Group()
  
  const laneWidth = width / lanes
  const markingWidth = 0.15
  const dashLength = 3
  const gapLength = 6

  // Center line (if even lanes) or lane dividers
  for (let lane = 1; lane < lanes; lane++) {
    const offset = -width / 2 + lane * laneWidth
    const isCenter = lanes % 2 === 0 && lane === lanes / 2

    // Create dashed line
    let dashStart = 0
    const pathLength = calculatePathLength(smoothPath)
    
    while (dashStart < pathLength) {
      const dashEnd = Math.min(dashStart + dashLength, pathLength)
      const startT = dashStart / pathLength
      const endT = dashEnd / pathLength
      
      const startIdx = Math.floor(startT * (n - 1))
      const endIdx = Math.floor(endT * (n - 1))
      
      if (endIdx > startIdx) {
        const markingPoints = []
        for (let i = startIdx; i <= endIdx; i++) {
          const curr = smoothPath[i]
          const prev = smoothPath[Math.max(0, i - 1)]
          const next = smoothPath[Math.min(n - 1, i + 1)]
          
          const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
          const up = new THREE.Vector3(0, 1, 0)
          const right = new THREE.Vector3().crossVectors(tangent, up).normalize()
          
          const point = new THREE.Vector3().copy(curr)
            .addScaledVector(right, offset)
            .addScaledVector(up, CONFIG.deckThickness / 2 + 0.01)
          
          markingPoints.push(point)
        }

        if (markingPoints.length >= 2) {
          const curve = new THREE.CatmullRomCurve3(markingPoints)
          const geo = new THREE.TubeGeometry(curve, markingPoints.length, markingWidth / 2, 4, false)
          const marking = new THREE.Mesh(geo)
          marking.userData.isMarking = true
          group.add(marking)
        }
      }
      
      dashStart += dashLength + gapLength
    }
  }

  return group
}

/**
 * Calculate total path length
 */
function calculatePathLength(points) {
  let length = 0
  for (let i = 1; i < points.length; i++) {
    length += points[i].distanceTo(points[i - 1])
  }
  return length
}

/**
 * Create PBR materials
 */
function createMaterials(textureBase64 = null) {
  const materials = {}

  // Concrete texture (procedural or from Imagen)
  let concreteTexture = null
  if (textureBase64) {
    const img = new Image()
    img.src = `data:image/png;base64,${textureBase64}`
    concreteTexture = new THREE.Texture(img)
    concreteTexture.wrapS = THREE.RepeatWrapping
    concreteTexture.wrapT = THREE.RepeatWrapping
    concreteTexture.repeat.set(4, 4)
    img.onload = () => { concreteTexture.needsUpdate = true }
  }

  // Deck material (top surface - darker asphalt)
  materials.deckTop = new THREE.MeshStandardMaterial({
    color: CONFIG.deckTopColor,
    roughness: 0.9,
    metalness: 0.0,
    map: concreteTexture,
  })

  // Deck sides/bottom (concrete)
  materials.concrete = new THREE.MeshStandardMaterial({
    color: CONFIG.concreteColor,
    roughness: 0.8,
    metalness: 0.0,
  })

  // Pillar material
  materials.pillar = new THREE.MeshStandardMaterial({
    color: 0xdcdcdc,
    roughness: 0.7,
    metalness: 0.0,
  })

  // Barrier material (metal)
  materials.barrier = new THREE.MeshStandardMaterial({
    color: CONFIG.barrierColor,
    roughness: 0.4,
    metalness: 0.6,
  })

  // Pole material
  materials.pole = new THREE.MeshStandardMaterial({
    color: CONFIG.poleColor,
    roughness: 0.5,
    metalness: 0.3,
  })

  // Light material (emissive)
  materials.light = new THREE.MeshStandardMaterial({
    color: CONFIG.lightColor,
    emissive: CONFIG.lightColor,
    emissiveIntensity: 0.8,
    roughness: 0.2,
    metalness: 0.0,
  })

  // Lane marking material
  materials.marking = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.0,
    emissive: 0xffffff,
    emissiveIntensity: 0.1,
  })

  return materials
}

/**
 * Setup lighting for the scene
 */
function setupLighting(scene) {
  // Ambient light
  const ambient = new THREE.AmbientLight(0xffffff, CONFIG.ambientIntensity)
  scene.add(ambient)

  // Main sun light with shadows
  const sun = new THREE.DirectionalLight(0xffffff, CONFIG.sunIntensity)
  sun.position.set(...CONFIG.sunPosition)
  sun.castShadow = true
  sun.shadow.mapSize.width = 2048
  sun.shadow.mapSize.height = 2048
  sun.shadow.camera.near = 10
  sun.shadow.camera.far = 500
  sun.shadow.camera.left = -100
  sun.shadow.camera.right = 100
  sun.shadow.camera.top = 100
  sun.shadow.camera.bottom = -100
  scene.add(sun)

  // Fill light (softer, from opposite side)
  const fill = new THREE.DirectionalLight(0x8888ff, 0.3)
  fill.position.set(-50, 100, -50)
  scene.add(fill)

  // Hemisphere light for sky/ground color variation
  const hemi = new THREE.HemisphereLight(0x87ceeb, 0x3a3a2a, 0.3)
  scene.add(hemi)
}

/**
 * Create the complete flyover 3D model
 */
function createFlyoverModel(flyoverData, materials) {
  const group = new THREE.Group()
  
  const { flyover_path, pillars, flyover_width_m, flyover_height_m, flyover_lanes } = flyoverData
  
  // Convert path to world coordinates
  const worldPath = pathToWorldCoords(flyover_path, flyover_height_m)
  if (worldPath.length < 2) return group

  const metersPerPixel = 0.075 * Math.pow(2, 15 - _map.getZoom())
  const scaledWidth = flyover_width_m / metersPerPixel
  const scaledHeight = flyover_height_m / metersPerPixel

  // 1. Create deck
  const deckGeo = createDeckGeometry(worldPath, scaledWidth, CONFIG.deckThickness / metersPerPixel)
  if (deckGeo) {
    const deck = new THREE.Mesh(deckGeo, materials.deckTop)
    deck.castShadow = true
    deck.receiveShadow = true
    deck.userData.type = 'deck'
    group.add(deck)
  }

  // 2. Create pillars
  pillars.forEach(([lng, lat], idx) => {
    const pos = pathToWorldCoords([[lng, lat]], 0)[0]
    if (!pos) return

    const pillarGroup = createPillarGeometry(scaledHeight, CONFIG.pillarRadius / metersPerPixel)
    pillarGroup.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = materials.pillar
        child.castShadow = true
        child.receiveShadow = true
      }
    })
    pillarGroup.position.copy(pos)
    pillarGroup.userData.type = 'pillar'
    pillarGroup.userData.pillarIndex = idx
    group.add(pillarGroup)
  })

  // 3. Create barriers
  const leftBarrier = createBarrierGeometry(worldPath, scaledWidth, 'left')
  if (leftBarrier) {
    leftBarrier.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = materials.barrier
        child.castShadow = true
      }
    })
    group.add(leftBarrier)
  }

  const rightBarrier = createBarrierGeometry(worldPath, scaledWidth, 'right')
  if (rightBarrier) {
    rightBarrier.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = materials.barrier
        child.castShadow = true
      }
    })
    group.add(rightBarrier)
  }

  // 4. Create lighting poles
  const poles = createLightingPoles(worldPath, scaledWidth, CONFIG.poleSpacing / metersPerPixel)
  if (poles) {
    poles.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        if (child.userData.isLight) {
          child.material = materials.light
        } else {
          child.material = materials.pole
          child.castShadow = true
        }
      }
    })
    group.add(poles)
  }

  // 5. Create lane markings
  const markings = createLaneMarkings(worldPath, scaledWidth, flyover_lanes)
  if (markings) {
    markings.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = materials.marking
      }
    })
    group.add(markings)
  }

  return group
}

/**
 * Initialize the 3D scene
 */
function initScene(container) {
  // Scene
  _scene = new THREE.Scene()
  _scene.background = null // Transparent for map overlay
  _scene.fog = new THREE.Fog(0x87ceeb, 500, 2000)

  // Camera
  _camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 10000)
  _camera.position.set(0, 100, 200)
  _camera.lookAt(0, 0, 0)

  // Renderer with antialiasing
  _renderer = new THREE.WebGLRenderer({ 
    antialias: true, 
    alpha: true,
    powerPreference: 'high-performance'
  })
  _renderer.setSize(container.clientWidth, container.clientHeight)
  _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  _renderer.shadowMap.enabled = true
  _renderer.shadowMap.type = THREE.PCFSoftShadowMap
  _renderer.toneMapping = THREE.ACESFilmicToneMapping
  _renderer.toneMappingExposure = 1.0
  _renderer.outputColorSpace = THREE.SRGBColorSpace
  container.appendChild(_renderer.domElement)

  // Orbit controls for exploration
  _controls = new OrbitControls(_camera, _renderer.domElement)
  _controls.enableDamping = true
  _controls.dampingFactor = 0.05
  _controls.minDistance = 20
  _controls.maxDistance = 1000
  _controls.maxPolarAngle = Math.PI / 2 - 0.1
  _controls.target.set(0, 0, 0)

  // Raycaster for hover effects
  _raycaster = new THREE.Raycaster()
  _mouse = new THREE.Vector2()

  // Setup lighting
  setupLighting(_scene)

  // Ground plane for shadows
  const groundGeo = new THREE.PlaneGeometry(2000, 2000)
  const groundMat = new THREE.ShadowMaterial({ opacity: 0.3 })
  const ground = new THREE.Mesh(groundGeo, groundMat)
  ground.rotation.x = -Math.PI / 2
  ground.receiveShadow = true
  _scene.add(ground)

  // Event listeners
  container.addEventListener('mousemove', onMouseMove)
  container.addEventListener('resize', onResize)

  return { scene: _scene, camera: _camera, renderer: _renderer, controls: _controls }
}

/**
 * Mouse move handler for hover effects
 */
function onMouseMove(event) {
  if (!_renderer) return
  
  const rect = _renderer.domElement.getBoundingClientRect()
  _mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  _mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
}

/**
 * Resize handler
 */
function onResize() {
  if (!_camera || !_renderer) return
  
  const container = _renderer.domElement.parentElement
  _camera.aspect = container.clientWidth / container.clientHeight
  _camera.updateProjectionMatrix()
  _renderer.setSize(container.clientWidth, container.clientHeight)
}

/**
 * Animation loop
 */
function animate() {
  _animationId = requestAnimationFrame(animate)

  if (_controls) {
    _controls.update()
  }

  // Hover detection
  if (_raycaster && _mouse && _flyoverGroup) {
    _raycaster.setFromCamera(_mouse, _camera)
    const intersects = _raycaster.intersectObjects(_flyoverGroup.children, true)

    // Reset previous hover
    if (_hoveredObject) {
      if (_hoveredObject.material && _hoveredObject.material.emissive) {
        _hoveredObject.material.emissiveIntensity = _hoveredObject.userData.originalEmissive || 0
      }
      _hoveredObject = null
    }

    // Highlight new hover
    if (intersects.length > 0) {
      const obj = intersects[0].object
      if (obj.material && obj.material.emissive && !obj.userData.isLight && !obj.userData.isMarking) {
        obj.userData.originalEmissive = obj.material.emissiveIntensity
        obj.material.emissiveIntensity = 0.3
        obj.material.emissive.setHex(0x00ff00)
        _hoveredObject = obj
      }
    }
  }

  if (_renderer && _scene && _camera) {
    _renderer.render(_scene, _camera)
  }
}

/**
 * PUBLIC: Add high-fidelity flyover layer
 */
export function addFlyover3DLayer(map, flyoverData, textureBase64 = null) {
  _map = map

  // Create container
  const mapContainer = map.getContainer()
  let container = mapContainer.querySelector('.flyover-3d-container')
  
  if (!container) {
    container = document.createElement('div')
    container.className = 'flyover-3d-container'
    container.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: auto;
      z-index: 100;
    `
    mapContainer.appendChild(container)
  }

  // Initialize scene
  initScene(container)

  // Create materials
  const materials = createMaterials(textureBase64)

  // Create flyover model
  _flyoverGroup = createFlyoverModel(flyoverData, materials)
  _scene.add(_flyoverGroup)

  // Center camera on flyover
  const box = new THREE.Box3().setFromObject(_flyoverGroup)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  
  _controls.target.copy(center)
  _camera.position.set(center.x + size.x, center.y + size.y * 2, center.z + size.z * 1.5)

  // Start animation
  animate()

  return { scene: _scene, camera: _camera, renderer: _renderer, controls: _controls }
}

/**
 * PUBLIC: Remove flyover layer
 */
export function removeFlyover3DLayer() {
  if (_animationId) {
    cancelAnimationFrame(_animationId)
    _animationId = null
  }

  if (_renderer) {
    _renderer.dispose()
    _renderer.domElement.remove()
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
    _scene = null
  }

  _camera = null
  _controls = null
  _flyoverGroup = null
  _map = null

  const container = document.querySelector('.flyover-3d-container')
  if (container) container.remove()
}

/**
 * PUBLIC: Update camera view
 */
export function setCameraView(position, target) {
  if (_camera && _controls) {
    _camera.position.set(position.x, position.y, position.z)
    _controls.target.set(target.x, target.y, target.z)
  }
}

/**
 * PUBLIC: Get current scene for external manipulation
 */
export function getScene() {
  return { scene: _scene, camera: _camera, renderer: _renderer, controls: _controls }
}

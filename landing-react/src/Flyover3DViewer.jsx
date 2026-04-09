/**
 * 3D FLYOVER VIEWER COMPONENT
 * 
 * Full-screen immersive 3D viewer for exploring flyover models
 * with high-quality graphics and interactive controls.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001'

// Configuration for ultra-high quality rendering
const CONFIG = {
  // Quality settings
  antialias: true,
  pixelRatio: Math.min(window.devicePixelRatio, 2),
  shadowMapSize: 4096,
  
  // Deck specifications
  deckThickness: 0.8,
  deckWidth: 9,
  
  // Materials
  colors: {
    concrete: 0x8a8a8a,
    asphalt: 0x555555,
    barrier: 0xd0d0d0,
    pillar: 0xdcdcdc,
    marking: 0xffffff,
    light: 0xffffee,
    sky: 0x87ceeb,
  },
  
  // Bloom effect
  bloom: {
    strength: 0.3,
    radius: 0.5,
    threshold: 0.8,
  }
}

export default function Flyover3DViewer({ flyoverData, locationName, onClose }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const composerRef = useRef(null)
  const animationRef = useRef(null)
  const flyoverGroupRef = useRef(null)

  const [isLoading, setIsLoading] = useState(true)
  const [viewMode, setViewMode] = useState('orbit') // 'orbit', 'firstPerson', 'topDown'
  const [showWireframe, setShowWireframe] = useState(false)
  const [timeOfDay, setTimeOfDay] = useState('day') // 'day', 'sunset', 'night'
  const [quality, setQuality] = useState('high') // 'low', 'medium', 'high', 'ultra'
  const [stats, setStats] = useState({ vertices: 0, faces: 0, fps: 0 })

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || !flyoverData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(CONFIG.colors.sky)
    scene.fog = new THREE.FogExp2(CONFIG.colors.sky, 0.0008)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000)
    camera.position.set(50, 30, 80)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: CONFIG.antialias,
      powerPreference: 'high-performance',
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(CONFIG.pixelRatio)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    renderer.outputColorSpace = THREE.SRGBColorSpace
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Post-processing
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    
    // Bloom for lights
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      CONFIG.bloom.strength,
      CONFIG.bloom.radius,
      CONFIG.bloom.threshold
    )
    composer.addPass(bloomPass)
    
    // Anti-aliasing
    const smaaPass = new SMAAPass(width, height)
    composer.addPass(smaaPass)
    
    // Crucial for tone mapping and sRGB output when using EffectComposer in modern three.js
    const outputPass = new OutputPass()
    composer.addPass(outputPass)
    
    composerRef.current = composer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 5
    controls.maxDistance = 500
    controls.maxPolarAngle = Math.PI / 2 - 0.05
    controls.target.set(0, 5, 0)
    controlsRef.current = controls

    // Setup scene
    setupLighting(scene, timeOfDay)
    setupEnvironment(scene)
    
    // Build flyover
    const flyoverGroup = buildFlyover(flyoverData, scene)
    flyoverGroupRef.current = flyoverGroup
    scene.add(flyoverGroup)

    // Center camera on flyover
    const box = new THREE.Box3().setFromObject(flyoverGroup)
    const center = box.getCenter(new THREE.Vector3())
    controls.target.copy(center)
    camera.position.set(center.x + 50, center.y + 30, center.z + 80)

    // Count geometry
    let verts = 0, faces = 0
    flyoverGroup.traverse((obj) => {
      if (obj.geometry) {
        verts += obj.geometry.attributes.position?.count || 0
        faces += obj.geometry.index ? obj.geometry.index.count / 3 : verts / 3
      }
    })
    setStats(s => ({ ...s, vertices: verts, faces: Math.floor(faces) }))

    setIsLoading(false)

    // Animation loop
    let frameCount = 0
    let lastTime = performance.now()
    
    function animate() {
      animationRef.current = requestAnimationFrame(animate)
      
      controls.update()
      composer.render()
      
      // FPS counter
      frameCount++
      const now = performance.now()
      if (now - lastTime >= 1000) {
        setStats(s => ({ ...s, fps: frameCount }))
        frameCount = 0
        lastTime = now
      }
    }
    animate()

    // Resize handler
    function handleResize() {
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      composer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      renderer.dispose()
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose()
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose())
          else obj.material.dispose()
        }
      })
      container.removeChild(renderer.domElement)
    }
  }, [flyoverData])

  // Update lighting when time of day changes
  useEffect(() => {
    if (!sceneRef.current) return
    updateLighting(sceneRef.current, timeOfDay)
  }, [timeOfDay])

  // Toggle wireframe
  useEffect(() => {
    if (!flyoverGroupRef.current) return
    flyoverGroupRef.current.traverse((obj) => {
      if (obj.material) {
        obj.material.wireframe = showWireframe
      }
    })
  }, [showWireframe])

  // Camera presets
  const setCameraPreset = useCallback((preset) => {
    if (!cameraRef.current || !controlsRef.current || !flyoverGroupRef.current) return
    
    const box = new THREE.Box3().setFromObject(flyoverGroupRef.current)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    
    const camera = cameraRef.current
    const controls = controlsRef.current
    
    switch (preset) {
      case 'orbit':
        camera.position.set(center.x + size.x * 0.8, center.y + size.y * 2, center.z + size.z)
        controls.target.copy(center)
        break
      case 'topDown':
        camera.position.set(center.x, center.y + Math.max(size.x, size.z) * 1.5, center.z)
        controls.target.copy(center)
        break
      case 'firstPerson':
        camera.position.set(center.x - size.x / 2, center.y + 2, center.z)
        controls.target.set(center.x + size.x / 2, center.y + 2, center.z)
        break
      case 'side':
        camera.position.set(center.x, center.y + size.y * 0.5, center.z + size.z * 2)
        controls.target.copy(center)
        break
    }
    setViewMode(preset)
  }, [])

  return (
    <div className="flyover-3d-viewer">
      {/* Loading overlay */}
      {isLoading && (
        <div className="viewer-loading">
          <div className="loading-spinner" />
          <p>Building 3D Model...</p>
        </div>
      )}

      {/* 3D Canvas */}
      <div ref={containerRef} className="viewer-canvas" />

      {/* Controls Panel */}
      <div className="viewer-controls">
        <div className="controls-header">
          <span className="viewer-title">🏗️ 3D FLYOVER EXPLORER</span>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="controls-section">
          <label>Camera View</label>
          <div className="btn-group">
            <button 
              className={viewMode === 'orbit' ? 'active' : ''} 
              onClick={() => setCameraPreset('orbit')}
            >
              🔄 Orbit
            </button>
            <button 
              className={viewMode === 'topDown' ? 'active' : ''} 
              onClick={() => setCameraPreset('topDown')}
            >
              ⬇️ Top
            </button>
            <button 
              className={viewMode === 'firstPerson' ? 'active' : ''} 
              onClick={() => setCameraPreset('firstPerson')}
            >
              👁️ Drive
            </button>
            <button 
              className={viewMode === 'side' ? 'active' : ''} 
              onClick={() => setCameraPreset('side')}
            >
              ↔️ Side
            </button>
          </div>
        </div>

        <div className="controls-section">
          <label>Time of Day</label>
          <div className="btn-group">
            <button 
              className={timeOfDay === 'day' ? 'active' : ''} 
              onClick={() => setTimeOfDay('day')}
            >
              ☀️ Day
            </button>
            <button 
              className={timeOfDay === 'sunset' ? 'active' : ''} 
              onClick={() => setTimeOfDay('sunset')}
            >
              🌅 Sunset
            </button>
            <button 
              className={timeOfDay === 'night' ? 'active' : ''} 
              onClick={() => setTimeOfDay('night')}
            >
              🌙 Night
            </button>
          </div>
        </div>

        <div className="controls-section">
          <label>Display</label>
          <div className="toggle-row">
            <span>Wireframe</span>
            <button 
              className={`toggle ${showWireframe ? 'on' : ''}`}
              onClick={() => setShowWireframe(!showWireframe)}
            >
              {showWireframe ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>

        <div className="controls-section stats">
          <div className="stat-row">
            <span>Vertices</span>
            <span>{stats.vertices.toLocaleString()}</span>
          </div>
          <div className="stat-row">
            <span>Triangles</span>
            <span>{stats.faces.toLocaleString()}</span>
          </div>
          <div className="stat-row">
            <span>FPS</span>
            <span className={stats.fps < 30 ? 'warning' : 'good'}>{stats.fps}</span>
          </div>
        </div>

        <div className="controls-section help">
          <p>🖱️ <strong>Left drag</strong> - Rotate</p>
          <p>🖱️ <strong>Right drag</strong> - Pan</p>
          <p>🖱️ <strong>Scroll</strong> - Zoom</p>
        </div>
      </div>

      {/* Location badge */}
      <div className="location-badge">
        📍 {locationName || 'Flyover Model'}
      </div>
    </div>
  )
}

// ====== HELPER FUNCTIONS ======

function setupLighting(scene, timeOfDay) {
  // Clear existing lights
  scene.children = scene.children.filter(c => !(c instanceof THREE.Light))
  
  const settings = getLightingSettings(timeOfDay)
  
  // Ambient
  scene.add(new THREE.AmbientLight(settings.ambient.color, settings.ambient.intensity))
  
  // Sun/Moon
  const sun = new THREE.DirectionalLight(settings.sun.color, settings.sun.intensity)
  sun.position.set(...settings.sun.position)
  sun.castShadow = true
  sun.shadow.mapSize.width = CONFIG.shadowMapSize
  sun.shadow.mapSize.height = CONFIG.shadowMapSize
  sun.shadow.camera.near = 1
  sun.shadow.camera.far = 500
  sun.shadow.camera.left = -150
  sun.shadow.camera.right = 150
  sun.shadow.camera.top = 150
  sun.shadow.camera.bottom = -150
  sun.shadow.bias = -0.0001
  scene.add(sun)
  
  // Hemisphere
  scene.add(new THREE.HemisphereLight(settings.hemi.sky, settings.hemi.ground, settings.hemi.intensity))
}

function updateLighting(scene, timeOfDay) {
  const settings = getLightingSettings(timeOfDay)
  
  // Update background
  scene.background = new THREE.Color(settings.sky)
  scene.fog = new THREE.FogExp2(settings.sky, settings.fogDensity)
  
  // Update lights
  scene.traverse((obj) => {
    if (obj instanceof THREE.AmbientLight) {
      obj.color.setHex(settings.ambient.color)
      obj.intensity = settings.ambient.intensity
    }
    if (obj instanceof THREE.DirectionalLight) {
      obj.color.setHex(settings.sun.color)
      obj.intensity = settings.sun.intensity
      obj.position.set(...settings.sun.position)
    }
    if (obj instanceof THREE.HemisphereLight) {
      obj.color.setHex(settings.hemi.sky)
      obj.groundColor.setHex(settings.hemi.ground)
      obj.intensity = settings.hemi.intensity
    }
  })
}

function getLightingSettings(timeOfDay) {
  switch (timeOfDay) {
    case 'sunset':
      return {
        sky: 0xff7744,
        fogDensity: 0.001,
        ambient: { color: 0xff8866, intensity: 1.5 },
        sun: { color: 0xff6633, intensity: 3.5, position: [100, 50, 100] },
        hemi: { sky: 0xff8855, ground: 0x553322, intensity: 1.0 },
      }
    case 'night':
      return {
        sky: 0x0a0a1a,
        fogDensity: 0.002,
        ambient: { color: 0x222244, intensity: 1.0 },
        sun: { color: 0x6688aa, intensity: 1.0, position: [-50, 80, 50] },
        hemi: { sky: 0x112233, ground: 0x080808, intensity: 0.5 },
      }
    default: // day
      return {
        sky: 0x87ceeb,
        fogDensity: 0.0008,
        ambient: { color: 0xffffff, intensity: 1.5 },
        sun: { color: 0xffffff, intensity: 3.5, position: [150, 300, 150] },
        hemi: { sky: 0x87ceeb, ground: 0x556655, intensity: 1.5 },
      }
  }
}

function setupEnvironment(scene) {
  // Ground plane
  const groundGeo = new THREE.PlaneGeometry(2000, 2000, 100, 100)
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x819a78,
    roughness: 0.9,
    metalness: 0,
  })
  const ground = new THREE.Mesh(groundGeo, groundMat)
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.1
  ground.receiveShadow = true
  scene.add(ground)

  // Road under flyover
  const roadGeo = new THREE.PlaneGeometry(20, 500)
  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x222222,
    roughness: 0.8,
    metalness: 0,
  })
  const road = new THREE.Mesh(roadGeo, roadMat)
  road.rotation.x = -Math.PI / 2
  road.position.y = 0.05
  road.receiveShadow = true
  scene.add(road)
}

function buildFlyover(data, scene) {
  const group = new THREE.Group()
  
  const { flyover_path, pillars, flyover_width_m, flyover_height_m, flyover_lanes } = data
  
  // Scale factor (convert meters to scene units)
  const scale = 1
  const width = (flyover_width_m || 9) * scale
  const height = (flyover_height_m || 8.5) * scale
  const lanes = flyover_lanes || 2

  // Convert path coordinates to local 3D space
  const pathPoints = flyover_path.map(([lng, lat], i) => {
    // Simple linear mapping for demo - in production use proper projection
    const baseX = flyover_path[0][0]
    const baseY = flyover_path[0][1]
    const x = (lng - baseX) * 111000 * scale // Approx meters per degree
    const z = (lat - baseY) * 111000 * scale
    return new THREE.Vector3(x, height, z)
  })

  // Create smooth curve
  const curve = new THREE.CatmullRomCurve3(pathPoints, false, 'centripetal', 0.5)
  // Determine number of segments based on total length
  const length = curve.getLength();
  const segments = Math.max(200, Math.floor(length / 2)); // 1 point every 2 meters approx
  const smoothPoints = curve.getPoints(segments)

  // === DECK ===
  // Increased thickness to 1.5 to make it look beefier/cleaner
  const deckGroup = createDeck(smoothPoints, width, 1.5)
  group.add(deckGroup)

  // === PILLARS ===
  const pillarMat = new THREE.MeshStandardMaterial({
    color: 0xdcdcdc,
    roughness: 0.7,
    metalness: 0,
  })
  
  pillars.forEach(([lng, lat], i) => {
    const baseX = flyover_path[0][0]
    const baseY = flyover_path[0][1]
    const x = (lng - baseX) * 111000 * scale
    const z = (lat - baseY) * 111000 * scale
    
    // Increased pillar radius to 1.6 for a much more robust look
    const pillar = createPillar(height, 1.6, pillarMat)
    pillar.position.set(x, 0, z)
    group.add(pillar)
  })

  // === BARRIERS ===
  const barriers = createBarriers(smoothPoints, width, height)
  group.add(barriers)

  // === LANE MARKINGS ===
  const markings = createMarkings(smoothPoints, width, lanes, height)
  group.add(markings)

  // === STREET LIGHTS ===
  const lights = createStreetLights(smoothPoints, width, height)
  group.add(lights)

  return group
}

function createDeck(points, width, thickness) {
  const group = new THREE.Group()
  
  // Create deck cross-section shape
  const hw = width / 2
  const ht = thickness / 2
  
  const shape = new THREE.Shape()
  shape.moveTo(-hw, -ht)
  shape.lineTo(hw, -ht)
  shape.lineTo(hw, ht)
  shape.lineTo(hw - 0.2, ht + 0.1) // Edge lip
  shape.lineTo(-hw + 0.2, ht + 0.1)
  shape.lineTo(-hw, ht)
  shape.closePath()

  // Create path curve
  const curve = new THREE.CatmullRomCurve3(points)
  
  // Extrude along path
  const extrudeSettings = {
    steps: points.length * 2,
    bevelEnabled: false,
    extrudePath: curve,
  }
  
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings)
  
  // Materials
  const topMat = new THREE.MeshStandardMaterial({
    color: 0x555555,
    roughness: 0.9,
    metalness: 0,
  })
  
  const deck = new THREE.Mesh(geometry, topMat)
  deck.castShadow = true
  deck.receiveShadow = true
  group.add(deck)
  
  return group
}

function createPillar(height, radius, material) {
  const group = new THREE.Group()
  
  // Main column
  const columnGeo = new THREE.CylinderGeometry(radius, radius * 1.05, height - 0.7, 24)
  const column = new THREE.Mesh(columnGeo, material)
  column.position.y = height / 2 - 0.2
  column.castShadow = true
  group.add(column)
  
  // Capital
  const capGeo = new THREE.CylinderGeometry(radius * 1.4, radius, 0.5, 24)
  const cap = new THREE.Mesh(capGeo, material)
  cap.position.y = height - 0.25
  cap.castShadow = true
  group.add(cap)
  
  // Base
  const baseGeo = new THREE.CylinderGeometry(radius * 1.05, radius * 1.3, 0.4, 24)
  const base = new THREE.Mesh(baseGeo, material)
  base.position.y = 0.2
  base.castShadow = true
  group.add(base)
  
  // Footing
  const footGeo = new THREE.BoxGeometry(radius * 3, 0.2, radius * 3)
  const foot = new THREE.Mesh(footGeo, material)
  foot.position.y = 0.1
  foot.receiveShadow = true
  group.add(foot)
  
  return group
}

function createBarriers(points, width, deckHeight) {
  const group = new THREE.Group()
  const hw = width / 2 - 0.2
  const barrierHeight = 1.1
  
  const barrierMat = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    roughness: 0.4,
    metalness: 0.6,
  })

  // Create barrier on each side
  ;['left', 'right'].forEach(side => {
    const offset = side === 'left' ? -hw : hw
    
    const railPoints = points.map(p => new THREE.Vector3(
      p.x + (side === 'left' ? -hw : hw) * 0.1,
      p.y + barrierHeight / 2,
      p.z
    ))
    
    // Create offset points properly
    for (let i = 0; i < points.length; i++) {
      const prev = points[Math.max(0, i - 1)]
      const next = points[Math.min(points.length - 1, i + 1)]
      const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
      const right = new THREE.Vector3().crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize()
      
      railPoints[i] = new THREE.Vector3()
        .copy(points[i])
        .addScaledVector(right, offset)
      railPoints[i].y += barrierHeight / 2
    }
    
    const railCurve = new THREE.CatmullRomCurve3(railPoints)
    const railGeo = new THREE.TubeGeometry(railCurve, points.length * 2, 0.08, 8, false)
    const rail = new THREE.Mesh(railGeo, barrierMat)
    rail.castShadow = true
    group.add(rail)
    
    // Posts
    for (let i = 0; i < points.length; i += 5) {
      const postGeo = new THREE.BoxGeometry(0.08, barrierHeight, 0.08)
      const post = new THREE.Mesh(postGeo, barrierMat)
      post.position.copy(railPoints[i])
      post.position.y -= barrierHeight / 4
      post.castShadow = true
      group.add(post)
    }
  })
  
  return group
}

function createMarkings(points, width, lanes, deckHeight) {
  const group = new THREE.Group()
  
  const markingMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0,
    emissive: 0xffffff,
    emissiveIntensity: 0.1,
  })
  
  const laneWidth = width / lanes
  
  // Lane dividers
  for (let lane = 1; lane < lanes; lane++) {
    const offset = -width / 2 + lane * laneWidth
    
    // Dashed line
    for (let i = 0; i < points.length - 5; i += 10) {
      const segmentPoints = []
      for (let j = i; j < Math.min(i + 5, points.length); j++) {
        const prev = points[Math.max(0, j - 1)]
        const next = points[Math.min(points.length - 1, j + 1)]
        const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
        const right = new THREE.Vector3().crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize()
        
        const p = new THREE.Vector3().copy(points[j]).addScaledVector(right, offset)
        p.y += 0.42 // Just above deck
        segmentPoints.push(p)
      }
      
      if (segmentPoints.length >= 2) {
        const curve = new THREE.CatmullRomCurve3(segmentPoints)
        const geo = new THREE.TubeGeometry(curve, segmentPoints.length * 2, 0.08, 4, false)
        const marking = new THREE.Mesh(geo, markingMat)
        group.add(marking)
      }
    }
  }
  
  return group
}

function createStreetLights(points, width, deckHeight) {
  const group = new THREE.Group()
  const hw = width / 2 - 0.5
  const poleHeight = 6
  
  const poleMat = new THREE.MeshStandardMaterial({
    color: 0x555555,
    roughness: 0.5,
    metalness: 0.3,
  })
  
  const lightMat = new THREE.MeshStandardMaterial({
    color: 0xffffee,
    emissive: 0xffffee,
    emissiveIntensity: 0.8,
    roughness: 0.2,
    metalness: 0,
  })

  // Add poles every ~30 meters (about every 10 points)
  for (let i = 0; i < points.length; i += 15) {
    const side = (i / 15) % 2 === 0 ? 1 : -1
    const prev = points[Math.max(0, i - 1)]
    const curr = points[i]
    const next = points[Math.min(points.length - 1, i + 1)]
    
    const tangent = new THREE.Vector3().subVectors(next, prev).normalize()
    const right = new THREE.Vector3().crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize()
    
    const basePos = new THREE.Vector3().copy(curr).addScaledVector(right, hw * side)
    basePos.y += 0.4 // Deck surface
    
    // Pole
    const poleGeo = new THREE.CylinderGeometry(0.08, 0.12, poleHeight, 8)
    const pole = new THREE.Mesh(poleGeo, poleMat)
    pole.position.copy(basePos)
    pole.position.y += poleHeight / 2
    pole.castShadow = true
    group.add(pole)
    
    // Arm
    const armGeo = new THREE.CylinderGeometry(0.04, 0.04, 1.5, 6)
    const arm = new THREE.Mesh(armGeo, poleMat)
    arm.rotation.z = Math.PI / 2 * -side
    arm.position.copy(basePos)
    arm.position.y += poleHeight - 0.3
    arm.position.x -= 0.6 * side * right.x
    arm.position.z -= 0.6 * side * right.z
    group.add(arm)
    
    // Light fixture
    const lightGeo = new THREE.SphereGeometry(0.15, 16, 16)
    const light = new THREE.Mesh(lightGeo, lightMat)
    light.position.copy(basePos)
    light.position.y += poleHeight - 0.5
    light.position.x -= 1.2 * side * right.x
    light.position.z -= 1.2 * side * right.z
    group.add(light)
    
    // Point light
    const pointLight = new THREE.PointLight(0xffffee, 0.5, 30)
    pointLight.position.copy(light.position)
    group.add(pointLight)
  }
  
  return group
}

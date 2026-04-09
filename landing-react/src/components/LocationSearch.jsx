import { useState, useCallback, useRef } from 'react'
import { API_BASE } from '../api/config'

export default function LocationSearch({ map, onLocationSelect }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef(null)

  const doSearch = useCallback(async (q) => {
    if (!q || q.length < 3) {
      setResults([])
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/geocode?query=${encodeURIComponent(q)}&limit=5`)
      if (!res.ok) throw new Error('Geocode failed')
      const data = await res.json()
      setResults(data.results || [])
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  function handleInput(e) {
    const val = e.target.value
    setQuery(val)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(val), 400)
  }

  function selectResult(r) {
    const lat = parseFloat(r.lat)
    const lon = parseFloat(r.lon)
    if (isNaN(lat) || isNaN(lon)) return

    setQuery(r.display_name || r.name || '')
    setResults([])

    // Fly to location on map
    if (map) {
      map.flyTo({ center: [lon, lat], zoom: 14, duration: 1500 })
    }

    onLocationSelect?.({ lat, lon, name: r.display_name || r.name })
  }

  return (
    <div className="ov-location-search">
      <div className="ov-location-input-wrap">
        <span className="ov-location-icon">📍</span>
        <input
          type="text"
          className="ov-location-input"
          placeholder="Search location (e.g., Connaught Place, Delhi)"
          value={query}
          onChange={handleInput}
        />
        {loading && <span className="demo-spinner ov-location-spinner" />}
      </div>

      {results.length > 0 && (
        <div className="ov-location-results">
          {results.map((r, i) => (
            <button
              key={i}
              className="ov-location-result"
              onClick={() => selectResult(r)}
            >
              <span className="ov-location-result-name">
                {r.display_name || r.name || `${r.lat}, ${r.lon}`}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

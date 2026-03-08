export function getApiBase() {
  const explicit = (import.meta.env.VITE_API_BASE || '').trim()
  if (explicit) {
    console.log('[API Config] Using explicit VITE_API_BASE:', explicit)
    return explicit
  }

  // Local dev default
  if (typeof window !== 'undefined') {
    const host = window.location.hostname
    if (host === 'localhost' || host === '127.0.0.1') {
      console.log('[API Config] Local dev mode, using localhost:8000')
      return 'http://localhost:8000'
    }
  }

  // Production default (Render)
  console.log('[API Config] Production mode, using Render backend')
  return 'https://overhaul-1.onrender.com'
}

export const API_BASE = getApiBase()
console.log('[API Config] Final API_BASE:', API_BASE)

export async function apiFetchJson(path, options = {}) {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`
  const resp = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  })

  if (!resp.ok) {
    let detail = ''
    try {
      const data = await resp.json()
      detail = data?.detail ? String(data.detail) : JSON.stringify(data)
    } catch {
      try {
        detail = await resp.text()
      } catch {
        detail = ''
      }
    }
    const msg = detail ? `${resp.status} ${resp.statusText}: ${detail}` : `${resp.status} ${resp.statusText}`
    throw new Error(msg)
  }
  return await resp.json()
}

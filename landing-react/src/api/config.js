export function getApiBase() {
  const explicit = (import.meta.env.VITE_API_BASE || '').trim()
  if (explicit) return explicit

  // Local dev default
  if (typeof window !== 'undefined') {
    const host = window.location.hostname
    if (host === 'localhost' || host === '127.0.0.1') {
      return 'http://localhost:8000'
    }
  }

  // Production default (Render)
  return 'https://overhaul-1.onrender.com'
}

export const API_BASE = getApiBase()

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

import { useState } from 'react'
import { useAppStore } from './store'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function PromptBox() {
  const { prompt, setPrompt, setFlyover, loading, setLoading, setError } = useAppStore()
  const [localPrompt, setLocalPrompt] = useState(prompt || 'Build flyover at Connaught Place, Delhi')

  const submit = async () => {
    if (!localPrompt.trim()) return
    setLoading(true)
    setError(null)
    setFlyover(null)
    try {
      const resp = await fetch(`${API_BASE}/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: localPrompt }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.detail || `Request failed: ${resp.status}`)
      }
      const data = await resp.json()
      console.log('Flyover data received:', data)
      setFlyover(data)
      setPrompt(localPrompt)
    } catch (err) {
      console.error('Plan error:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="prompt-box">
      <textarea
        value={localPrompt}
        onChange={(e) => setLocalPrompt(e.target.value)}
        placeholder="e.g., Build flyover at Connaught Place, Delhi"
        disabled={loading}
      />
      <button onClick={submit} disabled={loading || !localPrompt.trim()}>
        {loading ? 'Planning...' : 'Plan Flyover'}
      </button>
    </div>
  )
}

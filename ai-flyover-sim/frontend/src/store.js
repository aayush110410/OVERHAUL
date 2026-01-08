import { create } from 'zustand'

export const useAppStore = create((set) => ({
  prompt: '',
  flyover: null,
  loading: false,
  error: null,
  setPrompt: (p) => set({ prompt: p }),
  setFlyover: (f) => set({ flyover: f }),
  setLoading: (v) => set({ loading: v }),
  setError: (e) => set({ error: e }),
}))

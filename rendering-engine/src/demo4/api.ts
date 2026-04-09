import type {
  Demo4BootstrapResponse,
  Demo4SimulationRequest,
  Demo4SimulationReport,
  Demo4OrchestrationResponse,
  Demo4WorldSnapshot,
} from './types';

export class Demo4ApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly timeout = 25000,
  ) {}

  async health(): Promise<Record<string, unknown>> {
    return this.request('/health');
  }

  async bootstrap(params?: {
    focusName?: string;
    focus?: { latitude: number; longitude: number; altitude?: number };
  }): Promise<Demo4BootstrapResponse> {
    const query = new URLSearchParams();
    if (params?.focusName) query.set('focus_name', params.focusName);
    if (params?.focus) {
      query.set('lat', String(params.focus.latitude));
      query.set('lon', String(params.focus.longitude));
      query.set('altitude', String(params.focus.altitude ?? 12000));
    }
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return this.request(`/api/demo4/bootstrap${suffix}`);
  }

  async simulate(request: Demo4SimulationRequest): Promise<Demo4SimulationReport> {
    return this.request('/api/demo4/simulate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async orchestrate(request: Demo4SimulationRequest): Promise<Demo4OrchestrationResponse> {
    return this.request('/api/demo4/orchestrate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  connectToStream(onSnapshot: (snapshot: Demo4WorldSnapshot) => void, onStatus?: (status: 'live' | 'offline') => void): WebSocket {
    const websocketUrl = this.baseUrl.replace(/^http/, 'ws') + '/ws/demo4/stream';
    const socket = new WebSocket(websocketUrl);

    socket.addEventListener('open', () => {
      onStatus?.('live');
      socket.send('subscribe');
    });

    socket.addEventListener('message', (event) => {
      const payload = JSON.parse(event.data) as { type: string; payload: Demo4WorldSnapshot };
      if (payload.type === 'world_snapshot') {
        onSnapshot(payload.payload);
      }
    });

    socket.addEventListener('close', () => onStatus?.('offline'));
    socket.addEventListener('error', () => onStatus?.('offline'));
    return socket;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        ...init,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      return await response.json();
    } finally {
      window.clearTimeout(timeoutId);
    }
  }
}

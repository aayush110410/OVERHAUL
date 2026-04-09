/**
 * OVERHAUL API Client
 * Connects rendering engine to backend engines, AI models, and live data feeds
 */

export interface ChatV2Request {
  prompt: string;
  city?: string;
  mode?: "full" | "fast";
}

export interface ChatV2Response {
  summary: string;
  outputs: {
    tldr: string;
    confidenceLevel: string;
    impactCards: Array<{ metric: string; value: string }>;
    domains: Record<string, any>;
    engineRecommendations: string[];
    engineWarnings: string[];
    critique?: string;
    logs: string[];
    brainInsights: {
      orchestrator: string;
      models_used: Record<string, string>;
      agent_trace: string[];
      duration_seconds: number;
    };
  };
  viz_data: {
    geojson?: any;
    nodes?: Array<{ id: string; name: string; coords: [number, number] }>;
    edges?: Array<{ id: string; from: string; to: string }>;
    heatmap?: any;
    markers?: any;
    center?: [number, number];
    zoom?: number;
  };
  locations: Array<{ name: string; coords: [number, number] }>;
  parsed_intent: Record<string, any>;
  manifest: {
    run_id: string;
    mode: string;
    prompt: string;
    runtime_s: number;
  };
}

export interface SimulateRequest {
  scenario: Record<string, any>;
  interventions?: Array<{
    type: "flyover" | "new_road" | "lane_expansion" | "closure" | "pricing" | "signal";
    params: Record<string, any>;
  }>;
}

export interface SimulateResponse {
  result: {
    avg_speed: number;
    total_time: number;
    bottleneck_edges: string[];
    congestion_ratio: number;
    path_edge_ids: string[];
    aqi_impact: number;
    co2_emissions: number;
  };
  manifest: {
    run_id: string;
    timestamp: number;
  };
  geojson: any;
}

export interface LiveAQIResponse {
  location: string;
  aqi: number;
  pm25: number;
  pm10: number;
  no2: number;
  o3: number;
  timestamp: string;
}

export interface LiveRouteResponse {
  origin: [number, number];
  destination: [number, number];
  duration: number;
  distance: number;
  congestion: number;
  aqi_along_route: number[];
}

export interface ScenarioTemplate {
  id: string;
  name: string;
  description: string;
  params: Record<string, any>;
}

class OverhaulAPIClient {
  private baseURL: string;
  private timeout: number;

  constructor(baseURL = "http://localhost:8000", timeout = 30000) {
    this.baseURL = baseURL;
    this.timeout = timeout;
  }

  public setBaseURL(url: string) {
    this.baseURL = url;
  }

  /**
   * LDRAGO v2 Chat Endpoint
   * Full cognitive pipeline: Parse → Locate → Plan → Research → Reason → Synthesize
   * Returns viz_data with GeoJSON, nodes, edges for globe rendering
   */
  async chatV2(request: ChatV2Request): Promise<ChatV2Response> {
    return this.post<ChatV2Response>("/chat/v2", {
      prompt: request.prompt,
      city: request.city || "delhi",
      mode: request.mode || "full",
    });
  }

  /**
   * Fast chat mode (reduced reasoning)
   */
  async chatV2Fast(prompt: string): Promise<ChatV2Response> {
    return this.chatV2({ prompt, mode: "fast" });
  }

  /**
   * Simulate endpoint
   * Runs traffic simulation with interventions
   */
  async simulate(request: SimulateRequest): Promise<SimulateResponse> {
    return this.post<SimulateResponse>("/simulate", request);
  }

  /**
   * Temporal simulation
   * Multi-step prediction: current → 1yr → 5yr
   */
  async simulateTemporal(
    prompt: string,
    steps = 4,
    stepDays = 90
  ): Promise<any> {
    return this.post("/simulate/temporal", {
      prompt,
      city: "delhi",
      steps,
      step_days: stepDays,
    });
  }

  /**
   * Live AQI data
   */
  async getLiveAQI(location: string): Promise<LiveAQIResponse> {
    return this.get<LiveAQIResponse>("/live/aqi", { location });
  }

  /**
   * Live route data with traffic and AQI
   */
  async getLiveRoute(
    origin: [number, number],
    destination: [number, number]
  ): Promise<LiveRouteResponse> {
    return this.get<LiveRouteResponse>("/live/route", {
      origin: origin.join(","),
      destination: destination.join(","),
    });
  }

  /**
   * Compare scenarios
   */
  async compareScenarios(
    baseline: Record<string, any>,
    proposed: Record<string, any>
  ): Promise<any> {
    return this.post("/scenarios/compare", { baseline, proposed });
  }

  /**
   * Get scenario templates
   */
  async getScenarioTemplates(): Promise<ScenarioTemplate[]> {
    return this.get<ScenarioTemplate[]>("/scenarios/templates");
  }

  /**
   * Apply template to generate scenario
   */
  async applyScenarioTemplate(templateId: string): Promise<any> {
    return this.post(`/scenarios/templates/${templateId}`, {});
  }

  /**
   * Geocode address
   */
  async geocode(address: string): Promise<{ lat: number; lon: number }> {
    return this.get("/geocode", { location: address });
  }

  /**
   * Reverse geocode coordinates
   */
  async reverseGeocode(lat: number, lon: number): Promise<{ address: string }> {
    return this.get("/reverse-geocode", { lat, lon });
  }

  /**
   * Health check
   */
  async health(): Promise<{ status: string; llm_available: boolean }> {
    return this.get("/health");
  }

  /**
   * LDRAGo status
   */
  async ldragonStatus(): Promise<any> {
    return this.get("/ldrago/status");
  }

  // ────────────────────────────────────────────────────────────────
  // Private HTTP methods
  // ────────────────────────────────────────────────────────────────

  private async get<T>(
    endpoint: string,
    params?: Record<string, any>
  ): Promise<T> {
    const url = new URL(`${this.baseURL}${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        url.searchParams.append(k, String(v));
      });
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url.toString(), {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async post<T>(
    endpoint: string,
    data: Record<string, any>
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `HTTP ${response.status}: ${errorText || response.statusText}`
        );
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

// Singleton instance
let instance: OverhaulAPIClient | null = null;

export function initializeAPIClient(baseURL?: string): OverhaulAPIClient {
  if (!instance) {
    instance = new OverhaulAPIClient(baseURL);
  } else if (baseURL) {
    instance.setBaseURL(baseURL);
  }
  return instance;
}

export function getAPIClient(): OverhaulAPIClient {
  if (!instance) {
    instance = new OverhaulAPIClient();
  }
  return instance;
}

export default OverhaulAPIClient;

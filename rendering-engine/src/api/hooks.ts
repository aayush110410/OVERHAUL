/**
 * React Hooks for OVERHAUL API Integration
 * Handles async data fetching, caching, and error handling
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { getAPIClient, type ChatV2Response, type SimulateResponse } from "./client";

export interface UseApiOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  retries?: number;
  retryDelay?: number;
}

/**
 * Hook for async API calls with loading + error states
 */
export function useApi<T>(
  apiCall: () => Promise<T>,
  options?: UseApiOptions<T>
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const retriesRef = useRef(0);
  const maxRetries = options?.retries || 2;
  const retryDelay = options?.retryDelay || 1000;

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiCall();
      setData(result);
      options?.onSuccess?.(result);
      retriesRef.current = 0;
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));

      if (retriesRef.current < maxRetries) {
        retriesRef.current++;
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
        return execute();
      }

      setError(error);
      options?.onError?.(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiCall, options, maxRetries, retryDelay]);

  return { data, loading, error, execute };
}

/**
 * Hook for LDRAGO v2 Chat
 */
export function useLDRAGoChat(options?: UseApiOptions<ChatV2Response>) {
  const [messages, setMessages] = useState<
    Array<{ role: "user" | "assistant"; content: string; timestamp: number }>
  >([]);

  const chat = useCallback(
    async (prompt: string, mode: "full" | "fast" = "full") => {
      const client = getAPIClient();

      try {
        // Add user message
        setMessages((prev) => [
          ...prev,
          { role: "user", content: prompt, timestamp: Date.now() },
        ]);

        // Call API
        const response = await client.chatV2({ prompt, mode });

        // Extract text response
        const assistantText = response.summary || response.outputs?.tldr || "";

        // Add assistant message
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: assistantText, timestamp: Date.now() },
        ]);

        options?.onSuccess?.(response);
        return response;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        options?.onError?.(error);
        throw error;
      }
    },
    [options]
  );

  return { messages, chat, setMessages };
}

/**
 * Hook for Simulation
 */
export function useSimulation(options?: UseApiOptions<SimulateResponse>) {
  const [results, setResults] = useState<SimulateResponse | null>(null);
  const [history, setHistory] = useState<SimulateResponse[]>([]);

  const simulate = useCallback(
    async (scenario: Record<string, any>, interventions?: any[]) => {
      const client = getAPIClient();

      try {
        const response = await client.simulate({
          scenario,
          interventions: interventions || [],
        });

        setResults(response);
        setHistory((prev) => [response, ...prev].slice(0, 10)); // Keep last 10

        options?.onSuccess?.(response);
        return response;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        options?.onError?.(error);
        throw error;
      }
    },
    [options]
  );

  return { results, history, simulate, setResults };
}

/**
 * Hook for Live AQI Data
 */
export function useLiveAQI(location: string) {
  const [aqi, setAqi] = useState<{
    value: number;
    pm25: number;
    pm10: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!location) return;

    setLoading(true);
    const client = getAPIClient();

    client
      .getLiveAQI(location)
      .then((data) => {
        setAqi({
          value: data.aqi,
          pm25: data.pm25,
          pm10: data.pm10,
        });
      })
      .catch(() => {
        // Fallback to dummy data
        setAqi({
          value: Math.floor(Math.random() * 150) + 50,
          pm25: Math.random() * 50 + 20,
          pm10: Math.random() * 80 + 40,
        });
      })
      .finally(() => setLoading(false));

    // Refresh every 2 minutes
    const interval = setInterval(() => {
      client
        .getLiveAQI(location)
        .then((data) => {
          setAqi({
            value: data.aqi,
            pm25: data.pm25,
            pm10: data.pm10,
          });
        })
        .catch(() => {
          /* silent */
        });
    }, 120000);

    return () => clearInterval(interval);
  }, [location]);

  return { aqi, loading };
}

/**
 * Hook for Scenario Templates
 */
export function useScenarioTemplates() {
  const [templates, setTemplates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const client = getAPIClient();

    client
      .getScenarioTemplates()
      .then(setTemplates)
      .catch(() => {
        // Fallback templates
        setTemplates([
          {
            id: "baseline",
            name: "Baseline (2024)",
            description: "Current traffic conditions",
          },
          {
            id: "ev-50",
            name: "50% EV Adoption",
            description: "Electric vehicle penetration",
          },
          {
            id: "flyover-nnh",
            name: "Noida-NH Flyover",
            description: "New infrastructure",
          },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  return { templates, loading };
}

/**
 * Hook for LDRAGo System Status
 */
export function useLDRAGoStatus() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const client = getAPIClient();

    client
      .ldragonStatus()
      .then(setStatus)
      .catch(() => {
        setStatus({
          pipeline: "ldrago_v2",
          agents: ["parser", "planner", "researcher", "reasoner", "synthesizer"],
          modes: ["full", "fast"],
        });
      })
      .finally(() => setLoading(false));

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      client.ldragonStatus().then(setStatus).catch(() => {});
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  return { status, loading };
}

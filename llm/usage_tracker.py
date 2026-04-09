"""Token usage tracking for LLM calls."""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta

@dataclass
class UsageRecord:
    """Record of token usage for a single API call."""
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    cost_estimate: float = 0.0

@dataclass
class UsageStats:
    """Aggregate usage statistics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_cost: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)

    def add_record(self, record: UsageRecord):
        """Add a usage record to stats."""
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_calls += 1
        self.total_cost += record.cost_estimate

        if record.model in self.model_usage:
            self.model_usage[record.model] += record.total_tokens
        else:
            self.model_usage[record.model] = record.total_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def get_usage_percentage(self, limit: int) -> float:
        """Get percentage of usage relative to a token limit."""
        if limit <= 0:
            return 0.0
        return min(100.0, (self.total_tokens / limit) * 100)

class TokenUsageTracker:
    """Tracks token usage across all LLM calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        self._stats = UsageStats()

    def record_usage(self, record: UsageRecord):
        """Record token usage for an API call."""
        with self._lock:
            self._records.append(record)
            self._stats.add_record(record)

            # Keep only last 1000 records to prevent memory issues
            if len(self._records) > 1000:
                self._records = self._records[-1000:]

    def get_stats(self) -> UsageStats:
        """Get current usage statistics."""
        with self._lock:
            # Return a copy of stats
            return UsageStats(
                total_input_tokens=self._stats.total_input_tokens,
                total_output_tokens=self._stats.total_output_tokens,
                total_calls=self._stats.total_calls,
                total_cost=self._stats.total_cost,
                model_usage=self._stats.model_usage.copy()
            )

    def get_recent_usage(self, hours: int = 24) -> UsageStats:
        """Get usage stats for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_stats = UsageStats()

        with self._lock:
            for record in self._records:
                if record.timestamp >= cutoff:
                    recent_stats.add_record(record)

        return recent_stats

    def display_usage_bar(self, limit: int = 1000000):
        """Display a simple text-based usage bar."""
        stats = self.get_stats()
        percentage = stats.get_usage_percentage(limit)

        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f"\n📊 Token Usage Tracker")
        print(f"[{bar}] {percentage:.1f}%")
        print(f"  Total Tokens: {stats.total_tokens:,}")
        print(f"  Input: {stats.total_input_tokens:,} | Output: {stats.total_output_tokens:,}")
        print(f"  Calls: {stats.total_calls} | Est. Cost: ${stats.total_cost:.4f}")
        print(f"  Models: {', '.join([f'{k}({v:,})' for k, v in stats.model_usage.items()])}")

# Global usage tracker instance
_usage_tracker = TokenUsageTracker()

def get_usage_tracker() -> TokenUsageTracker:
    """Get the global usage tracker instance."""
    return _usage_tracker

def record_qwen_usage(input_tokens: int, output_tokens: int):
    """Record usage for Qwen model."""
    record = UsageRecord(
        model="qwen/qwen3-4b:free",
        provider="OpenRouter",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_estimate=(input_tokens * 0.0000001) + (output_tokens * 0.0000003)  # Approximate costs
    )
    _usage_tracker.record_usage(record)

def record_gemini_usage(input_tokens: int, output_tokens: int):
    """Record usage for Gemini model."""
    record = UsageRecord(
        model="gemini-3.1-pro-preview",
        provider="Google AI",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_estimate=(input_tokens * 0.0000005) + (output_tokens * 0.0000015)  # Approximate costs
    )
    _usage_tracker.record_usage(record)

def display_usage(limit: int = 1000000):
    """Display current usage statistics with a progress bar."""
    _usage_tracker.display_usage_bar(limit)
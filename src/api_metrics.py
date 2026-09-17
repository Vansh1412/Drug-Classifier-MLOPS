"""
src/api_metrics.py — Thread-safe in-memory API metrics collector for MedMLOps-X.

Separates operational API metrics (requests, status codes, latencies, uptime)
from offline ML model evaluation metrics.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


class APIMetricsCollector:
    """Thread-safe collector for real-time operational API metrics."""

    def __init__(self, max_latency_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_samples = max_latency_samples
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to their initial state."""
        with self._lock:
            self._start_time = time.time()
            self._start_timestamp = datetime.now(timezone.utc).isoformat()
            self._total_requests = 0
            self._status_counts: dict[int, int] = defaultdict(int)
            self._endpoint_counts: dict[str, int] = defaultdict(int)
            self._endpoint_latencies: dict[str, list[float]] = defaultdict(list)
            self._prediction_requests = 0
            self._explanation_requests = 0

    def record_request(
        self, endpoint: str, status_code: int, duration_ms: float
    ) -> None:
        """Record an incoming API request with its status code and execution duration.

        Args:
            endpoint: API path (e.g. '/predict', '/explain').
            status_code: HTTP response status code.
            duration_ms: Execution duration in milliseconds.
        """
        with self._lock:
            self._total_requests += 1
            self._status_counts[status_code] += 1
            self._endpoint_counts[endpoint] += 1

            if endpoint == "/predict":
                self._prediction_requests += 1
            elif endpoint == "/explain":
                self._explanation_requests += 1

            samples = self._endpoint_latencies[endpoint]
            samples.append(duration_ms)
            if len(samples) > self._max_samples:
                del samples[: len(samples) - self._max_samples]

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of operational metrics.

        Returns:
            Dictionary containing request totals, status code summaries,
            endpoint counts, and latency statistics.
        """
        with self._lock:
            uptime_seconds = round(time.time() - self._start_time, 2)
            successful_requests = sum(
                count
                for code, count in self._status_counts.items()
                if 200 <= code < 300
            )
            client_errors = sum(
                count
                for code, count in self._status_counts.items()
                if 400 <= code < 500
            )
            server_errors = sum(
                count for code, count in self._status_counts.items() if code >= 500
            )

            latency_stats: dict[str, dict[str, float]] = {}
            for ep, samples in self._endpoint_latencies.items():
                if samples:
                    sorted_samples = sorted(samples)
                    avg_lat = round(sum(samples) / len(samples), 2)
                    p95_idx = int(len(sorted_samples) * 0.95)
                    p95_lat = round(
                        sorted_samples[min(p95_idx, len(sorted_samples) - 1)], 2
                    )
                    latency_stats[ep] = {
                        "avg_latency_ms": avg_lat,
                        "p95_latency_ms": p95_lat,
                        "sample_count": len(samples),
                    }

            return {
                "service": "MedMLOps-X API",
                "uptime_seconds": uptime_seconds,
                "start_timestamp": self._start_timestamp,
                "total_requests": self._total_requests,
                "successful_requests": successful_requests,
                "client_errors_4xx": client_errors,
                "server_errors_5xx": server_errors,
                "prediction_requests": self._prediction_requests,
                "explanation_requests": self._explanation_requests,
                "requests_by_status": dict(self._status_counts),
                "requests_by_endpoint": dict(self._endpoint_counts),
                "latency_ms": latency_stats,
            }


# Global singleton instance for application use
metrics_collector = APIMetricsCollector()

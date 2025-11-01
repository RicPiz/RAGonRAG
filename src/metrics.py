"""
Performance metrics tracking and monitoring for the RAG system.
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path
from datetime import datetime, timedelta

try:
    from .config import get_config
    from .logger import get_logger
except ImportError:
    try:
        from config import get_config
        from logger import get_logger
    except ImportError:
        def get_config(): return None
        def get_logger(name): return None

logger = get_logger("metrics")


@dataclass
class OperationMetric:
    """Represents a single operation metric."""
    operation: str
    duration_ms: float
    timestamp: float
    status: str = "success"  # success, error, timeout
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "status": self.status,
            "metadata": self.metadata,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for an operation."""
    operation: str
    count: int
    total_duration_ms: float
    mean_duration_ms: float
    median_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    success_rate: float
    error_count: int
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "count": self.count,
            "total_duration_ms": self.total_duration_ms,
            "mean_duration_ms": self.mean_duration_ms,
            "median_duration_ms": self.median_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "last_updated": self.last_updated,
            "last_updated_datetime": datetime.fromtimestamp(self.last_updated).isoformat()
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.cached_aggregates: Dict[str, AggregatedMetrics] = {}
        self.cache_ttl = 60.0  # Cache aggregates for 60 seconds
        
        # Track system-level metrics
        self.system_metrics = {
            "start_time": time.time(),
            "total_queries": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Metrics collector initialized", max_history=max_history)
    
    def record_operation(self, 
                        operation: str, 
                        duration_ms: float, 
                        status: str = "success",
                        **metadata) -> None:
        """Record a single operation metric.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            status: Operation status (success, error, timeout)
            **metadata: Additional metadata
        """
        metric = OperationMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
            status=status,
            metadata=metadata
        )
        
        self.metrics[operation].append(metric)
        
        # Invalidate cached aggregate for this operation
        if operation in self.cached_aggregates:
            del self.cached_aggregates[operation]
        
        # Update system metrics
        if operation.startswith("query"):
            self.system_metrics["total_queries"] += 1
        
        if status == "error":
            self.system_metrics["total_errors"] += 1
        
        logger.debug("Operation metric recorded", 
                    operation=operation, duration_ms=duration_ms, status=status)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.system_metrics["cache_hits"] += 1
        logger.debug("Cache hit recorded", cache_type=cache_type)
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.system_metrics["cache_misses"] += 1
        logger.debug("Cache miss recorded", cache_type=cache_type)
    
    def get_operation_metrics(self, operation: str, use_cache: bool = True) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics for an operation.
        
        Args:
            operation: Operation name
            use_cache: Whether to use cached aggregates
            
        Returns:
            AggregatedMetrics or None if no data
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        # Check cache first
        if use_cache and operation in self.cached_aggregates:
            cached = self.cached_aggregates[operation]
            if time.time() - cached.last_updated < self.cache_ttl:
                return cached
        
        # Calculate aggregates
        metrics_list = list(self.metrics[operation])
        durations = [m.duration_ms for m in metrics_list]
        success_count = sum(1 for m in metrics_list if m.status == "success")
        error_count = sum(1 for m in metrics_list if m.status == "error")
        
        aggregated = AggregatedMetrics(
            operation=operation,
            count=len(metrics_list),
            total_duration_ms=sum(durations),
            mean_duration_ms=statistics.mean(durations),
            median_duration_ms=statistics.median(durations),
            p95_duration_ms=self._percentile(durations, 95),
            p99_duration_ms=self._percentile(durations, 99),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            success_rate=success_count / len(metrics_list) if metrics_list else 0.0,
            error_count=error_count,
            last_updated=time.time()
        )
        
        # Cache the result
        self.cached_aggregates[operation] = aggregated
        
        return aggregated
    
    def get_all_metrics(self) -> Dict[str, AggregatedMetrics]:
        """Get aggregated metrics for all operations."""
        all_metrics = {}
        for operation in self.metrics.keys():
            metrics = self.get_operation_metrics(operation)
            if metrics:
                all_metrics[operation] = metrics
        return all_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        uptime_seconds = time.time() - self.system_metrics["start_time"]
        cache_total = self.system_metrics["cache_hits"] + self.system_metrics["cache_misses"]
        cache_hit_rate = (self.system_metrics["cache_hits"] / cache_total) if cache_total > 0 else 0.0
        
        return {
            **self.system_metrics,
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": (self.system_metrics["total_errors"] / 
                          max(1, self.system_metrics["total_queries"])),
            "operations_tracked": len(self.metrics),
            "total_metrics_recorded": sum(len(deq) for deq in self.metrics.values())
        }
    
    def get_recent_metrics(self, operation: str, minutes: int = 60) -> List[OperationMetric]:
        """Get metrics from the last N minutes.
        
        Args:
            operation: Operation name
            minutes: Number of minutes to look back
            
        Returns:
            List of recent metrics
        """
        if operation not in self.metrics:
            return []
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            metric for metric in self.metrics[operation] 
            if metric.timestamp > cutoff_time
        ]
        
        return recent_metrics
    
    def clear_metrics(self, operation: Optional[str] = None) -> None:
        """Clear metrics for an operation or all operations.
        
        Args:
            operation: Specific operation to clear, or None for all
        """
        if operation:
            if operation in self.metrics:
                self.metrics[operation].clear()
                if operation in self.cached_aggregates:
                    del self.cached_aggregates[operation]
                logger.info("Metrics cleared", operation=operation)
        else:
            self.metrics.clear()
            self.cached_aggregates.clear()
            self.system_metrics = {
                "start_time": time.time(),
                "total_queries": 0,
                "total_errors": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            logger.info("All metrics cleared")
    
    def export_metrics(self, file_path: Optional[str] = None) -> str:
        """Export all metrics to JSON file.
        
        Args:
            file_path: Output file path, or None for default
            
        Returns:
            Path to exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"metrics_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": time.time(),
            "export_datetime": datetime.now().isoformat(),
            "system_metrics": self.get_system_metrics(),
            "operation_metrics": {
                op: metrics.to_dict() 
                for op, metrics in self.get_all_metrics().items()
            },
            "raw_metrics": {
                operation: [metric.to_dict() for metric in metrics_list]
                for operation, metrics_list in self.metrics.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info("Metrics exported", file_path=file_path)
        return file_path
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


class PerformanceMonitor:
    """Context manager for monitoring operation performance."""
    
    def __init__(self, 
                 operation: str, 
                 collector: Optional[MetricsCollector] = None,
                 **metadata):
        self.operation = operation
        self.collector = collector or get_metrics_collector()
        self.metadata = metadata
        self.start_time: Optional[float] = None
        self.status = "success"
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.debug("Performance monitoring started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            
            if exc_type is not None:
                self.status = "error"
                self.metadata["error_type"] = exc_type.__name__
                self.metadata["error_message"] = str(exc_val)
            
            self.collector.record_operation(
                self.operation,
                duration_ms,
                self.status,
                **self.metadata
            )
            
            logger.debug("Performance monitoring completed",
                        operation=self.operation,
                        duration_ms=duration_ms,
                        status=self.status)
    
    def mark_error(self, error_type: str, error_message: str) -> None:
        """Mark the operation as having an error."""
        self.status = "error"
        self.metadata["error_type"] = error_type
        self.metadata["error_message"] = error_message


class AlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or get_metrics_collector()
        self.thresholds = {
            "error_rate": 0.05,  # 5% error rate threshold
            "response_time_ms": 5000,  # 5 second response time threshold
            "cache_hit_rate": 0.8  # 80% cache hit rate threshold
        }
        self.alert_history = deque(maxlen=100)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions and return any alerts."""
        alerts = []
        
        # Check system-level metrics
        system_metrics = self.collector.get_system_metrics()
        
        # Error rate alert
        if system_metrics["error_rate"] > self.thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate {system_metrics['error_rate']:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}",
                "value": system_metrics["error_rate"],
                "threshold": self.thresholds["error_rate"]
            })
        
        # Cache hit rate alert
        if system_metrics["cache_hit_rate"] < self.thresholds["cache_hit_rate"]:
            alerts.append({
                "type": "low_cache_hit_rate",
                "severity": "info",
                "message": f"Cache hit rate {system_metrics['cache_hit_rate']:.2%} below threshold {self.thresholds['cache_hit_rate']:.2%}",
                "value": system_metrics["cache_hit_rate"],
                "threshold": self.thresholds["cache_hit_rate"]
            })
        
        # Check operation-specific metrics
        for operation, metrics in self.collector.get_all_metrics().items():
            if metrics.p95_duration_ms > self.thresholds["response_time_ms"]:
                alerts.append({
                    "type": "high_response_time",
                    "severity": "warning",
                    "operation": operation,
                    "message": f"95th percentile response time {metrics.p95_duration_ms:.0f}ms exceeds threshold",
                    "value": metrics.p95_duration_ms,
                    "threshold": self.thresholds["response_time_ms"]
                })
        
        # Store alerts in history
        for alert in alerts:
            alert["timestamp"] = time.time()
            self.alert_history.append(alert)
            # Create a copy and extract message to avoid conflict with logger parameter
            alert_copy = alert.copy()
            alert_msg = alert_copy.pop("message", "Performance alert triggered")
            logger.warning(alert_msg, **alert_copy)
        
        return alerts


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_alert_manager: Optional[AlertManager] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def monitor_performance(operation: str, **metadata) -> PerformanceMonitor:
    """Create a performance monitor context manager.
    
    Args:
        operation: Name of the operation to monitor
        **metadata: Additional metadata to record
        
    Returns:
        PerformanceMonitor context manager
        
    Usage:
        with monitor_performance("query_processing", query_type="embedding"):
            # Your operation here
            pass
    """
    return PerformanceMonitor(operation, **metadata)


def record_metric(operation: str, duration_ms: float, status: str = "success", **metadata) -> None:
    """Record a performance metric.
    
    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        status: Operation status
        **metadata: Additional metadata
    """
    collector = get_metrics_collector()
    collector.record_operation(operation, duration_ms, status, **metadata)
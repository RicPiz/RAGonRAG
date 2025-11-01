"""
Tests for metrics and performance monitoring.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.metrics import (
    MetricsCollector, PerformanceMonitor, AlertManager,
    OperationMetric, AggregatedMetrics,
    get_metrics_collector, monitor_performance, record_metric
)


class TestOperationMetric:
    """Test OperationMetric data class."""
    
    def test_operation_metric_creation(self):
        """Test creating an operation metric."""
        timestamp = time.time()
        metric = OperationMetric(
            operation="test_op",
            duration_ms=123.45,
            timestamp=timestamp,
            status="success",
            metadata={"key": "value"}
        )
        
        assert metric.operation == "test_op"
        assert metric.duration_ms == 123.45
        assert metric.timestamp == timestamp
        assert metric.status == "success"
        assert metric.metadata == {"key": "value"}
    
    def test_operation_metric_to_dict(self):
        """Test converting metric to dictionary."""
        timestamp = time.time()
        metric = OperationMetric(
            operation="test_op",
            duration_ms=123.45,
            timestamp=timestamp
        )
        
        result = metric.to_dict()
        
        assert result["operation"] == "test_op"
        assert result["duration_ms"] == 123.45
        assert result["timestamp"] == timestamp
        assert "datetime" in result


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_record_operation_basic(self):
        """Test basic operation recording."""
        collector = MetricsCollector(max_history=100)
        
        collector.record_operation("test_op", 123.45, "success", key="value")
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics.operation == "test_op"
        assert metrics.count == 1
        assert metrics.mean_duration_ms == 123.45
    
    def test_record_multiple_operations(self):
        """Test recording multiple operations."""
        collector = MetricsCollector(max_history=100)
        
        durations = [100, 200, 150, 300, 250]
        for duration in durations:
            collector.record_operation("test_op", duration, "success")
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics.count == 5
        assert metrics.mean_duration_ms == sum(durations) / len(durations)
        assert metrics.min_duration_ms == min(durations)
        assert metrics.max_duration_ms == max(durations)
    
    def test_record_operations_with_errors(self):
        """Test recording operations with different statuses."""
        collector = MetricsCollector(max_history=100)
        
        collector.record_operation("test_op", 100, "success")
        collector.record_operation("test_op", 200, "error")
        collector.record_operation("test_op", 150, "success")
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics.count == 3
        assert metrics.success_rate == 2/3  # 2 successes out of 3
        assert metrics.error_count == 1
    
    def test_cache_hit_miss_tracking(self):
        """Test cache hit/miss tracking."""
        collector = MetricsCollector(max_history=100)
        
        collector.record_cache_hit("embedding")
        collector.record_cache_hit("query")
        collector.record_cache_miss("embedding")
        
        system_metrics = collector.get_system_metrics()
        assert system_metrics["cache_hits"] == 2
        assert system_metrics["cache_misses"] == 1
        assert system_metrics["cache_hit_rate"] == 2/3
    
    def test_memory_limit(self):
        """Test metrics memory limit enforcement."""
        collector = MetricsCollector(max_history=3)
        
        # Add more items than the limit
        for i in range(5):
            collector.record_operation("test_op", i * 100, "success")
        
        # Should only keep the last 3
        assert len(collector.metrics["test_op"]) == 3
        
        # Should contain the last 3 items (200, 300, 400)
        durations = [m.duration_ms for m in collector.metrics["test_op"]]
        assert durations == [200.0, 300.0, 400.0]
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics within time window."""
        collector = MetricsCollector(max_history=100)
        
        # Add some metrics
        current_time = time.time()
        collector.record_operation("test_op", 100, "success")
        
        # Mock older metric
        old_metric = OperationMetric(
            operation="test_op",
            duration_ms=200,
            timestamp=current_time - 3600,  # 1 hour ago
            status="success"
        )
        collector.metrics["test_op"].appendleft(old_metric)
        
        # Get recent metrics (last 30 minutes)
        recent = collector.get_recent_metrics("test_op", minutes=30)
        
        # Should only include the recent metric
        assert len(recent) == 1
        assert recent[0].duration_ms == 100
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        collector = MetricsCollector(max_history=100)
        
        collector.record_operation("test_op1", 100, "success")
        collector.record_operation("test_op2", 200, "success")
        
        # Clear specific operation
        collector.clear_metrics("test_op1")
        assert collector.get_operation_metrics("test_op1") is None
        assert collector.get_operation_metrics("test_op2") is not None
        
        # Clear all operations
        collector.clear_metrics()
        assert collector.get_operation_metrics("test_op2") is None
    
    def test_export_metrics(self, temp_dir):
        """Test exporting metrics to JSON."""
        collector = MetricsCollector(max_history=100)
        
        collector.record_operation("test_op", 123.45, "success", key="value")
        
        export_path = str(temp_dir / "test_export.json")
        result_path = collector.export_metrics(export_path)
        
        assert result_path == export_path
        
        # Verify file was created and has content
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert "system_metrics" in data
        assert "operation_metrics" in data
        assert "raw_metrics" in data
        assert "test_op" in data["operation_metrics"]


class TestPerformanceMonitor:
    """Test performance monitoring context manager."""
    
    def test_performance_monitor_success(self):
        """Test successful operation monitoring."""
        collector = MetricsCollector(max_history=100)
        
        with PerformanceMonitor("test_op", collector, key="value"):
            time.sleep(0.01)  # Simulate work
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics.count == 1
        assert metrics.success_rate == 1.0
        assert metrics.duration_ms > 0
    
    def test_performance_monitor_with_exception(self):
        """Test monitoring operation that raises exception."""
        collector = MetricsCollector(max_history=100)
        
        with pytest.raises(ValueError):
            with PerformanceMonitor("test_op", collector) as monitor:
                raise ValueError("Test error")
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics.count == 1
        assert metrics.success_rate == 0.0
        assert metrics.error_count == 1
    
    def test_performance_monitor_mark_error(self):
        """Test manually marking operation as error."""
        collector = MetricsCollector(max_history=100)
        
        with PerformanceMonitor("test_op", collector) as monitor:
            monitor.mark_error("CustomError", "Something went wrong")
        
        metrics = collector.get_operation_metrics("test_op")
        assert metrics.success_rate == 0.0
        assert metrics.error_count == 1


class TestAlertManager:
    """Test performance alerting functionality."""
    
    def test_high_error_rate_alert(self):
        """Test alert for high error rate."""
        collector = MetricsCollector(max_history=100)
        alert_manager = AlertManager(collector)
        
        # Set threshold lower than actual error rate
        alert_manager.thresholds["error_rate"] = 0.1  # 10%
        
        # Record operations with 50% error rate
        for i in range(10):
            status = "error" if i < 5 else "success"
            collector.record_operation("test_op", 100, status)
        
        alerts = alert_manager.check_alerts()
        
        # Should trigger high error rate alert
        error_rate_alerts = [a for a in alerts if a["type"] == "high_error_rate"]
        assert len(error_rate_alerts) == 1
        assert error_rate_alerts[0]["severity"] == "warning"
    
    def test_low_cache_hit_rate_alert(self):
        """Test alert for low cache hit rate."""
        collector = MetricsCollector(max_history=100)
        alert_manager = AlertManager(collector)
        
        # Set threshold higher than actual hit rate
        alert_manager.thresholds["cache_hit_rate"] = 0.9  # 90%
        
        # Record cache operations with 50% hit rate
        for i in range(10):
            if i < 5:
                collector.record_cache_hit("test")
            else:
                collector.record_cache_miss("test")
        
        alerts = alert_manager.check_alerts()
        
        # Should trigger low cache hit rate alert
        cache_alerts = [a for a in alerts if a["type"] == "low_cache_hit_rate"]
        assert len(cache_alerts) == 1
        assert cache_alerts[0]["severity"] == "info"
    
    def test_high_response_time_alert(self):
        """Test alert for high response times."""
        collector = MetricsCollector(max_history=100)
        alert_manager = AlertManager(collector)
        
        # Set threshold lower than actual response times
        alert_manager.thresholds["response_time_ms"] = 100  # 100ms
        
        # Record operations with high response times
        for i in range(10):
            collector.record_operation("slow_op", 200, "success")  # 200ms each
        
        alerts = alert_manager.check_alerts()
        
        # Should trigger high response time alert
        response_time_alerts = [a for a in alerts if a["type"] == "high_response_time"]
        assert len(response_time_alerts) == 1
        assert response_time_alerts[0]["operation"] == "slow_op"
    
    def test_no_alerts_when_under_thresholds(self):
        """Test no alerts when metrics are within thresholds."""
        collector = MetricsCollector(max_history=100)
        alert_manager = AlertManager(collector)
        
        # Record good metrics
        collector.record_operation("good_op", 50, "success")  # Fast operation
        collector.record_cache_hit("test")  # Good cache hit rate
        
        alerts = alert_manager.check_alerts()
        
        # Should not trigger any alerts
        assert len(alerts) == 0


class TestGlobalFunctions:
    """Test global metrics functions."""
    
    def test_get_metrics_collector_singleton(self):
        """Test global metrics collector singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_monitor_performance_function(self):
        """Test global monitor_performance function."""
        with monitor_performance("test_op", key="value"):
            time.sleep(0.01)
        
        collector = get_metrics_collector()
        metrics = collector.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics.count >= 1
    
    def test_record_metric_function(self):
        """Test global record_metric function."""
        record_metric("direct_op", 123.45, "success", key="value")
        
        collector = get_metrics_collector()
        metrics = collector.get_operation_metrics("direct_op")
        assert metrics is not None
        assert metrics.mean_duration_ms == 123.45
"""
Comprehensive logging system for the RAG application with structured logging,
performance monitoring, and advanced configuration options.
"""

import logging
import logging.handlers
import json
import sys
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

# Ensure src directory is in path for absolute imports
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import get_config, LoggingConfig
except ImportError:
    from config import get_config, LoggingConfig


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add any extra fields from the record
        if hasattr(record, 'context'):
            log_entry.update(record.context)

        return json.dumps(log_entry, default=str)


class PerformanceFormatter(logging.Formatter):
    """Formatter for performance logging."""

    def format(self, record):
        """Format performance log record."""
        if hasattr(record, 'duration_ms'):
            duration = f"{record.duration_ms:.2f}ms"
        else:
            duration = "N/A"

        base_msg = f"[{record.levelname}] {record.name}: {record.getMessage()}"
        if hasattr(record, 'context') and record.context:
            context_str = " | ".join(f"{k}={v}" for k, v in record.context.items())
            return f"{base_msg} ({duration}) | {context_str}"
        return f"{base_msg} ({duration})"


class LogAnalyzer:
    """Analyze and aggregate log data."""

    def __init__(self, max_entries: int = 1000):
        self.entries = deque(maxlen=max_entries)
        self.lock = threading.Lock()
        self.stats = defaultdict(lambda: defaultdict(int))

    def add_entry(self, record: logging.LogRecord):
        """Add a log entry for analysis."""
        with self.lock:
            entry = {
                'timestamp': record.created,
                'level': record.levelno,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module
            }
            self.entries.append(entry)

            # Update statistics
            self.stats[record.name][record.levelname] += 1
            self.stats['global'][record.levelname] += 1

    def get_recent_entries(self, count: int = 50) -> List[Dict]:
        """Get recent log entries."""
        with self.lock:
            return list(self.entries)[-count:]

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get logging statistics."""
        with self.lock:
            return dict(self.stats)

    def clear(self):
        """Clear all log data."""
        with self.lock:
            self.entries.clear()
            self.stats.clear()


class RAGLogger:
    """Enhanced logger for the RAG system with multiple format support."""

    def __init__(self, name: str = "rag_system", format_type: str = "standard"):
        """
        Initialize logger with specified format.

        Args:
            name: Logger name
            format_type: Format type ('standard', 'json', 'performance')
        """
        self.name = name
        self.format_type = format_type
        self._logger: Optional[logging.Logger] = None
        self._analyzer = None
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with advanced configuration."""
        config = get_config().logging

        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(getattr(logging, config.level.upper()))

        # Clear existing handlers
        self._logger.handlers.clear()

        # Create formatter based on format type
        if self.format_type == "json":
            formatter = JSONFormatter()
        elif self.format_type == "performance":
            formatter = PerformanceFormatter()
        else:
            formatter = logging.Formatter(config.format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler if enabled
        if config.log_to_file:
            log_file = Path(config.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            if self.format_type == "json":
                # Use regular file handler for JSON to avoid rotation complexity
                file_handler = logging.FileHandler(log_file)
            else:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=config.max_file_size,
                    backupCount=config.backup_count
                )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        # Initialize analyzer if this is the main logger
        if self.name == "rag_system":
            self._analyzer = LogAnalyzer()

    def enable_analyzer(self, max_entries: int = 1000):
        """Enable log analysis for this logger."""
        self._analyzer = LogAnalyzer(max_entries)

    def get_analyzer(self) -> Optional[LogAnalyzer]:
        """Get the log analyzer instance."""
        return self._analyzer
    
    def _make_record(self, level: str, message: str, exception: Optional[Exception] = None, **kwargs) -> logging.LogRecord:
        """Create a log record with enhanced context and sensitive data masking."""
        record = logging.LogRecord(
            name=self.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=exception
        )

        # Mask sensitive data in kwargs
        masked_kwargs = self._mask_sensitive_data(kwargs)

        # Add context
        if masked_kwargs:
            if self.format_type == "json":
                record.context = masked_kwargs
            else:
                context_str = " | ".join(f"{k}={v}" for k, v in masked_kwargs.items())
                record.msg = f"{message} | {context_str}"

        return record

    def _mask_sensitive_data(self, data: dict) -> dict:
        """Mask sensitive information like API keys in log data."""
        import re
        masked = {}
        sensitive_keys = ['api_key', 'apikey', 'token', 'password', 'secret', 'authorization']

        for key, value in data.items():
            key_lower = key.lower()
            # Check if key is sensitive
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                # Mask the value
                if isinstance(value, str) and len(value) > 8:
                    masked[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    masked[key] = "***MASKED***"
            elif isinstance(value, str):
                # Also mask values that look like API keys (sk-... or similar patterns)
                if re.match(r'^(sk-|pk-|Bearer\s+)', value):
                    masked[key] = f"{value[:7]}...{value[-4:]}" if len(value) > 11 else "***MASKED***"
                else:
                    masked[key] = value
            else:
                masked[key] = value

        return masked

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        record = self._make_record("debug", message, **kwargs)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        record = self._make_record("info", message, **kwargs)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        record = self._make_record("warning", message, **kwargs)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with optional exception and context."""
        record = self._make_record("error", message, exception, **kwargs)
        if exception:
            record.exc_info = (type(exception), exception, exception.__traceback__)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message with optional exception and context."""
        record = self._make_record("critical", message, exception, **kwargs)
        if exception:
            record.exc_info = (type(exception), exception, exception.__traceback__)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)

    def log_performance(self, operation: str, duration_ms: float, **context) -> None:
        """Log performance metrics."""
        record = self._make_record("info", f"Performance: {operation}", duration_ms=duration_ms, **context)
        self._logger.handle(record)
        if self._analyzer:
            self._analyzer.add_entry(record)


# Global logger instances
_loggers: Dict[str, RAGLogger] = {}
_global_analyzer: Optional[LogAnalyzer] = None


def get_logger(name: str = "rag_system", format_type: str = "standard") -> RAGLogger:
    """Get or create a logger instance."""
    key = f"{name}:{format_type}"
    if key not in _loggers:
        _loggers[key] = RAGLogger(name, format_type)

        # Enable analyzer for main logger
        if name == "rag_system" and format_type == "standard":
            _loggers[key].enable_analyzer()

    return _loggers[key]


def get_global_analyzer() -> Optional[LogAnalyzer]:
    """Get the global log analyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        main_logger = get_logger("rag_system")
        _global_analyzer = main_logger.get_analyzer()
    return _global_analyzer


def log_performance(func_name: str, duration_ms: float, **context) -> None:
    """Log performance metrics."""
    logger = get_logger("performance", "performance")
    logger.log_performance(func_name, duration_ms, **context)


# Performance profiling utilities
class PerformanceProfiler:
    """Comprehensive performance profiling for the RAG system."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_operations = {}
        self.lock = threading.Lock()

    def start_operation(self, operation_name: str, **context) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{threading.get_ident()}_{time.time_ns()}"
        start_time = time.perf_counter()

        with self.lock:
            self.current_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'context': context
            }

        return operation_id

    def end_operation(self, operation_id: str) -> float:
        """End timing an operation and record metrics."""
        end_time = time.perf_counter()

        with self.lock:
            if operation_id not in self.current_operations:
                logger.warning("Operation not found", operation_id=operation_id)
                return 0.0

            operation_data = self.current_operations.pop(operation_id)
            duration_ms = (end_time - operation_data['start_time']) * 1000

            # Record metrics
            self.metrics[operation_data['name']].append({
                'duration_ms': duration_ms,
                'timestamp': end_time,
                'context': operation_data['context']
            })

            # Log performance
            log_performance(operation_data['name'], duration_ms, **operation_data['context'])

            return duration_ms

    def get_operation_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance statistics for operations."""
        with self.lock:
            if operation_name:
                durations = [m['duration_ms'] for m in self.metrics[operation_name]]
                if not durations:
                    return {'count': 0}

                return {
                    'count': len(durations),
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99),
                    'min': np.min(durations),
                    'max': np.max(durations)
                }
            else:
                # Return stats for all operations
                return {name: self.get_operation_stats(name) for name in self.metrics.keys()}

    def get_recent_operations(self, limit: int = 10) -> List[Dict]:
        """Get recent operation records."""
        with self.lock:
            all_operations = []
            for operation_name, records in self.metrics.items():
                for record in records[-limit:]:  # Get last N records per operation
                    all_operations.append({
                        'operation': operation_name,
                        'duration_ms': record['duration_ms'],
                        'timestamp': record['timestamp'],
                        'context': record['context']
                    })

            # Sort by timestamp and return most recent
            all_operations.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_operations[:limit]

    def clear_metrics(self, operation_name: str = None) -> None:
        """Clear performance metrics."""
        with self.lock:
            if operation_name:
                self.metrics[operation_name].clear()
            else:
                self.metrics.clear()

    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        with self.lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.get_operation_stats(),
                'recent_operations': self.get_recent_operations(50)
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


class profile_operation:
    """Context manager for profiling operations."""

    def __init__(self, operation_name: str, **context):
        self.operation_name = operation_name
        self.context = context
        self.operation_id = None
        self.profiler = get_profiler()

    def __enter__(self):
        self.operation_id = self.profiler.start_operation(self.operation_name, **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            duration = self.profiler.end_operation(self.operation_id)
            if exc_type:
                # Log exception in context
                self.profiler.current_operations[self.operation_id]['context']['exception'] = str(exc_val)


def configure_logging(log_level: str = "INFO",
                     json_format: bool = False,
                     enable_analyzer: bool = True) -> None:
    """Configure global logging settings."""
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Configure main logger
    main_logger = get_logger("rag_system", "json" if json_format else "standard")
    if enable_analyzer and not main_logger.get_analyzer():
        main_logger.enable_analyzer()


def get_log_stats() -> Dict[str, Any]:
    """Get comprehensive logging statistics."""
    analyzer = get_global_analyzer()
    if not analyzer:
        return {"error": "Log analyzer not enabled"}

    return {
        "stats": analyzer.get_stats(),
        "recent_entries": analyzer.get_recent_entries(20)
    }


def profile_function(operation_name: Optional[str] = None):
    """Decorator for profiling function execution."""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            operation_id = profiler.start_operation(name, function=func.__name__)

            try:
                result = func(*args, **kwargs)
                profiler.end_operation(operation_id)
                return result
            except Exception as e:
                # Record exception in operation context
                if operation_id in profiler.current_operations:
                    profiler.current_operations[operation_id]['context']['exception'] = str(e)
                profiler.end_operation(operation_id)
                raise

        return wrapper
    return decorator


def create_performance_report() -> Dict[str, Any]:
    """Create a comprehensive performance report."""
    profiler = get_profiler()
    analyzer = get_global_analyzer()

    report = {
        "timestamp": datetime.now().isoformat(),
        "performance_stats": profiler.get_operation_stats(),
        "recent_operations": profiler.get_recent_operations(20),
    }

    if analyzer:
        report["log_stats"] = analyzer.get_stats()
        report["recent_logs"] = analyzer.get_recent_entries(20)

    return report


def save_performance_report(filepath: str) -> None:
    """Save performance report to file."""
    report = create_performance_report()

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Performance report saved", filepath=filepath)
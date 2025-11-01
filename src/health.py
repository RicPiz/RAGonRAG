"""
Health check and system diagnostics for the RAG system.

This module provides comprehensive health monitoring and diagnostic capabilities
for the RAG system, including system status, performance metrics, and component health checks.
"""

import os
import time
import psutil
from typing import Dict, Any, List
from pathlib import Path

try:
    from .config import get_config
    from .logger import get_logger, get_profiler, get_log_stats
    from .cache import get_embedding_cache, get_query_cache, get_chunk_cache
    from .vector_db import get_vector_db_manager
    from .async_utils import get_async_executor
    from .metrics import get_metrics_collector
except ImportError:
    # Fallback for direct imports
    try:
        from config import get_config
        from logger import get_logger, get_profiler, get_log_stats
        from cache import get_embedding_cache, get_query_cache, get_chunk_cache
        from vector_db import get_vector_db_manager
        from async_utils import get_async_executor
        from metrics import get_metrics_collector
    except ImportError:
        # If absolute imports also fail, use dummy implementations
        def get_config(): return None
        def get_logger(name): return None
        def get_profiler(): return None
        def get_log_stats(): return {}
        def get_embedding_cache(): return None
        def get_query_cache(): return None
        def get_chunk_cache(): return None
        def get_vector_db_manager(): return None
        def get_async_executor(): return None
        def get_metrics_collector(): return None

logger = get_logger("health")


class HealthChecker:
    """Comprehensive health checker for RAG system components."""

    def __init__(self):
        self.config = get_config()
        self.logger = logger

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(Path.cwd()))

            return {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent,
                    "status": "healthy" if memory.percent < 90 else "warning"
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": disk.percent,
                    "status": "healthy" if disk.percent < 90 else "warning"
                },
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            self.logger.error("Failed to check system resources", exception=e)
            return {"error": f"Resource check failed: {e}"}

    def check_components(self) -> Dict[str, Any]:
        """Check health of individual RAG system components."""
        components = {}

        # Check caches
        try:
            embedding_cache = get_embedding_cache()
            components["embedding_cache"] = {
                "status": "healthy",
                "info": f"Cache size: {len(embedding_cache._memory_cache)}"
            }
        except Exception as e:
            components["embedding_cache"] = {
                "status": "error",
                "error": str(e)
            }

        try:
            query_cache = get_query_cache()
            components["query_cache"] = {
                "status": "healthy",
                "info": f"Cache size: {len(query_cache._memory_cache)}"
            }
        except Exception as e:
            components["query_cache"] = {
                "status": "error",
                "error": str(e)
            }

        # Check vector database manager
        try:
            db_manager = get_vector_db_manager()
            stats = db_manager.get_all_stats()
            components["vector_db_manager"] = {
                "status": "healthy",
                "databases": len(stats),
                "stats": stats
            }
        except Exception as e:
            components["vector_db_manager"] = {
                "status": "error",
                "error": str(e)
            }

        # Check async executor
        try:
            executor = get_async_executor()
            components["async_executor"] = {
                "status": "healthy",
                "max_workers": executor._max_workers
            }
        except Exception as e:
            components["async_executor"] = {
                "status": "error",
                "error": str(e)
            }

        # Check metrics collector
        try:
            metrics = get_metrics_collector()
            system_metrics = metrics.get_system_metrics()
            components["metrics_collector"] = {
                "status": "healthy",
                "metrics_count": len(system_metrics)
            }
        except Exception as e:
            components["metrics_collector"] = {
                "status": "error",
                "error": str(e)
            }

        return components

    def check_data_integrity(self) -> Dict[str, Any]:
        """Check data integrity and availability."""
        integrity = {}

        # Check data directory
        data_dir = Path(self.config.data_dir)
        if data_dir.exists():
            md_files = list(data_dir.glob("*.md"))
            integrity["data_directory"] = {
                "status": "healthy",
                "documents": len(md_files),
                "path": str(data_dir)
            }
        else:
            integrity["data_directory"] = {
                "status": "error",
                "error": f"Data directory not found: {data_dir}"
            }

        # Check cache directories
        cache_dir = Path(self.config.cache.cache_dir)
        if cache_dir.exists():
            cache_files = list(cache_dir.rglob("*"))
            integrity["cache_directory"] = {
                "status": "healthy",
                "files": len(cache_files),
                "path": str(cache_dir)
            }
        else:
            integrity["cache_directory"] = {
                "status": "warning",
                "info": f"Cache directory not found: {cache_dir}"
            }

        # Check results directory
        results_dir = Path(self.config.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True, exist_ok=True)

        integrity["results_directory"] = {
            "status": "healthy",
            "path": str(results_dir)
        }

        return integrity

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        try:
            profiler = get_profiler()
            stats = profiler.get_operation_stats()
            return {
                "status": "healthy",
                "operation_stats": stats,
                "recent_operations": profiler.get_recent_operations(10)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Performance check failed: {e}"
            }

    def get_log_summary(self) -> Dict[str, Any]:
        """Get logging activity summary."""
        try:
            log_stats = get_log_stats()
            return {
                "status": "healthy",
                "log_stats": log_stats
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Log summary failed: {e}"
            }

    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        start_time = time.time()

        health_report = {
            "timestamp": time.time(),
            "system_resources": self.check_system_resources(),
            "components": self.check_components(),
            "data_integrity": self.check_data_integrity(),
            "performance": self.get_performance_summary(),
            "logging": self.get_log_summary()
        }

        # Overall status determination
        statuses = []
        for section in ["system_resources", "components", "data_integrity", "performance", "logging"]:
            if section in health_report:
                if isinstance(health_report[section], dict):
                    if "status" in health_report[section]:
                        statuses.append(health_report[section]["status"])
                    elif "memory" in health_report[section]:  # System resources
                        statuses.extend([health_report[section]["memory"]["status"],
                                       health_report[section]["disk"]["status"]])
                    else:  # Components section
                        for component in health_report[section].values():
                            if isinstance(component, dict) and "status" in component:
                                statuses.append(component["status"])

        # Determine overall status
        if "error" in statuses:
            overall_status = "error"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        health_report["overall_status"] = overall_status
        health_report["check_duration_ms"] = (time.time() - start_time) * 1000

        return health_report


# Global health checker instance
_health_checker: HealthChecker = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def health_check() -> Dict[str, Any]:
    """Quick health check of the RAG system."""
    checker = get_health_checker()
    return checker.comprehensive_health_check()


def system_status() -> Dict[str, Any]:
    """Get basic system status."""
    checker = get_health_checker()

    return {
        "status": "online",
        "timestamp": time.time(),
        "version": "1.0.0",
        "resources": checker.check_system_resources(),
        "components": checker.check_components()
    }


def quick_diagnostics() -> Dict[str, Any]:
    """Run quick diagnostics and return actionable insights."""
    checker = get_health_checker()
    health = checker.comprehensive_health_check()

    diagnostics = {
        "issues": [],
        "recommendations": [],
        "performance_insights": []
    }

    # Check for issues
    if health["overall_status"] == "error":
        diagnostics["issues"].append("System has critical errors - check logs for details")

    # Check memory usage
    memory = health["system_resources"]["memory"]
    if memory["used_percent"] > 90:
        diagnostics["issues"].append("High memory usage detected")
        diagnostics["recommendations"].append("Consider increasing system memory or optimizing cache sizes")

    # Check component health
    for component_name, component_info in health["components"].items():
        if component_info.get("status") == "error":
            diagnostics["issues"].append(f"Component {component_name} has errors")
            diagnostics["recommendations"].append(f"Check {component_name} configuration and logs")

    # Performance insights
    if "performance" in health and "operation_stats" in health["performance"]:
        stats = health["performance"]["operation_stats"]
        for operation, metrics in stats.items():
            if isinstance(metrics, dict) and metrics.get("count", 0) > 0:
                mean_time = metrics.get("mean", 0)
                if mean_time > 5000:  # 5 seconds
                    diagnostics["performance_insights"].append(
                        f"Operation '{operation}' is slow ({mean_time:.1f}ms avg)"
                    )

    return diagnostics


# Web Framework Integration Utilities
def create_fastapi_health_endpoint():
    """Create a FastAPI-compatible health check endpoint.

    Returns:
        A function that can be used as a FastAPI endpoint

    Example:
        ```python
        from fastapi import FastAPI
        from rag_system.health import create_fastapi_health_endpoint

        app = FastAPI()
        health_endpoint = create_fastapi_health_endpoint()

        @app.get("/health")
        async def health():
            return await health_endpoint()
        ```
    """
    async def health_endpoint() -> Dict[str, Any]:
        """FastAPI health check endpoint."""
        try:
            health_data = health_check()
            status_code = 200 if health_data["overall_status"] == "healthy" else 503
            return health_data
        except Exception as e:
            return {
                "overall_status": "error",
                "error": f"Health check failed: {e}",
                "timestamp": time.time()
            }

    return health_endpoint


def create_flask_health_endpoint():
    """Create a Flask-compatible health check endpoint.

    Returns:
        A function that can be used as a Flask route

    Example:
        ```python
        from flask import Flask, jsonify
        from rag_system.health import create_flask_health_endpoint

        app = Flask(__name__)
        health_endpoint = create_flask_health_endpoint()

        @app.route("/health")
        def health():
            return jsonify(health_endpoint())
        ```
    """
    def health_endpoint() -> Dict[str, Any]:
        """Flask health check endpoint."""
        try:
            return health_check()
        except Exception as e:
            return {
                "overall_status": "error",
                "error": f"Health check failed: {e}",
                "timestamp": time.time()
            }

    return health_endpoint


def create_streamlit_health_page():
    """Create a Streamlit-compatible health dashboard.

    Returns:
        A function that can be used as a Streamlit page

    Example:
        ```python
        import streamlit as st
        from rag_system.health import create_streamlit_health_page

        health_page = create_streamlit_health_page()

        if st.sidebar.button("Health Check"):
            health_page()
        ```
    """
    def health_page():
        """Streamlit health dashboard."""
        st.header("🩺 System Health Dashboard")

        try:
            # Get health data
            health_data = health_check()
            diagnostics = quick_diagnostics()

            # Overall status
            status = health_data["overall_status"]
            if status == "healthy":
                st.success("✅ System is healthy")
            elif status == "warning":
                st.warning("⚠️ System has warnings")
            else:
                st.error("❌ System has errors")

            # System resources
            st.subheader("💻 System Resources")
            resources = health_data["system_resources"]

            col1, col2 = st.columns(2)
            with col1:
                mem = resources["memory"]
                st.metric("Memory Usage", f"{mem['used_percent']:.1f}%")
                st.metric("Available Memory", f"{mem['available_gb']:.1f} GB")

            with col2:
                disk = resources["disk"]
                st.metric("Disk Usage", f"{disk['used_percent']:.1f}%")
                st.metric("Free Disk", f"{disk['free_gb']:.1f} GB")

            # Components
            st.subheader("🔧 Component Status")
            components = health_data["components"]

            for component_name, component_info in components.items():
                if component_info.get("status") == "healthy":
                    st.success(f"✅ {component_name}")
                elif component_info.get("status") == "warning":
                    st.warning(f"⚠️ {component_name}")
                else:
                    st.error(f"❌ {component_name}")

            # Issues and recommendations
            if diagnostics["issues"]:
                st.subheader("🚨 Issues Detected")
                for issue in diagnostics["issues"]:
                    st.error(f"• {issue}")

            if diagnostics["recommendations"]:
                st.subheader("💡 Recommendations")
                for rec in diagnostics["recommendations"]:
                    st.info(f"• {rec}")

            if diagnostics["performance_insights"]:
                st.subheader("⚡ Performance Insights")
                for insight in diagnostics["performance_insights"]:
                    st.warning(f"• {insight}")

            # Raw data
            with st.expander("🔍 Detailed Health Data"):
                st.json(health_data)

        except Exception as e:
            st.error(f"Failed to load health data: {e}")

    return health_page


def create_prometheus_metrics():
    """Create Prometheus-compatible metrics for monitoring.

    Returns:
        A dictionary of metric names and their values

    Example:
        ```python
        from rag_system.health import create_prometheus_metrics

        # For Prometheus integration
        metrics = create_prometheus_metrics()
        # Use with prometheus_client or similar
        ```
    """
    try:
        health_data = health_check()

        metrics = {
            "rag_system_health_status": 1 if health_data["overall_status"] == "healthy" else 0,
            "rag_system_memory_percent": health_data["system_resources"]["memory"]["used_percent"],
            "rag_system_disk_percent": health_data["system_resources"]["disk"]["used_percent"],
            "rag_system_cpu_percent": health_data["system_resources"]["cpu_percent"],
        }

        # Component health metrics
        for component_name, component_info in health_data["components"].items():
            status_value = 1 if component_info.get("status") == "healthy" else 0
            metrics[f"rag_system_component_{component_name}_healthy"] = status_value

        return metrics

    except Exception as e:
        return {
            "rag_system_health_status": 0,
            "rag_system_health_check_error": 1
        }


# CLI Health Check Utility
def cli_health_check():
    """Command-line health check utility.

    Example:
        ```bash
        python -c "from rag_system.health import cli_health_check; cli_health_check()"
        ```
    """
    try:
        print("🔍 RAGonRAG Health Check")
        print("=" * 50)

        health_data = health_check()

        # Overall status
        status = health_data["overall_status"]
        if status == "healthy":
            print("✅ Overall Status: HEALTHY")
        elif status == "warning":
            print("⚠️  Overall Status: WARNING")
        else:
            print("❌ Overall Status: ERROR")

        # System resources
        print("\n💻 System Resources:")
        resources = health_data["system_resources"]
        memory = resources["memory"]
        disk = resources["disk"]
        print(f"  Memory: {memory['used_percent']:.1f}% used ({memory['available_gb']:.1f} GB available)")
        print(f"  Disk:   {disk['used_percent']:.1f}% used ({disk['free_gb']:.1f} GB free)")
        print(f"  CPU:    {resources['cpu_percent']:.1f}%")

        # Components
        print("\n🔧 Components:")
        for component_name, component_info in health_data["components"].items():
            status_icon = "✅" if component_info.get("status") == "healthy" else "❌"
            print(f"  {status_icon} {component_name}")

        # Quick diagnostics
        diagnostics = quick_diagnostics()
        if diagnostics["issues"]:
            print("\n🚨 Issues:")
            for issue in diagnostics["issues"]:
                print(f"  • {issue}")

        if diagnostics["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in diagnostics["recommendations"]:
                print(f"  • {rec}")

        print(f"\n⏱️  Check completed in {health_data.get('check_duration_ms', 0):.1f}ms")

    except Exception as e:
        print(f"❌ Health check failed: {e}")


if __name__ == "__main__":
    cli_health_check()

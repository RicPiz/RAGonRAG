#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly.
"""

import sys
import os

def test_imports():
    """Test all the key imports to ensure they work."""
    print("Testing imports...")

    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    tests = [
        ("rag_system", ["create_rag_system", "EnhancedRAGSystem"]),
        ("config", ["get_config", "RAGConfig"]),
        ("validation", ["RAGError", "InputValidator"]),
        ("logger", ["get_logger"]),
        ("cache", ["get_embedding_cache"]),
        ("chunking", ["get_chunker"]),
        ("vector_db", ["get_vector_db_manager"]),
        ("async_utils", ["AsyncHTTPClient"]),
        ("metrics", ["get_metrics_collector"]),
        ("health", ["health_check"]),
        ("similarity_utils", ["cosine_similarity"]),
        ("evaluation", ["evaluate_dataset"]),
    ]

    results = []

    for module_name, functions in tests:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}: Module imported successfully")

            for func_name in functions:
                if hasattr(module, func_name):
                    print(f"   ✅ {func_name} found")
                else:
                    print(f"   ❌ {func_name} not found")
                    results.append(f"{module_name}.{func_name} missing")

        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
            results.append(f"{module_name} import failed: {e}")
        except Exception as e:
            print(f"⚠️  {module_name}: Other error - {e}")

    print("\n" + "="*50)

    if results:
        print("❌ Some imports failed:")
        for result in results:
            print(f"   {result}")
        return False
    else:
        print("✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)


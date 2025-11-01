"""
RAGonRAG - Advanced Retrieval Augmented Generation System

Features:
- Intelligent Question Answering: Ask questions with hybrid retrieval and generation
- Performance Analytics: View evaluation metrics and system performance
- Tunable Retrieval: Adjust semantic vs keyword search balance
- Comprehensive Evaluation: Ragas-based RAG metrics with triplet visualization

Technology:
- FAISS vector database for semantic search
- Hybrid retrieval (semantic + BM25/TF-IDF)
- OpenAI GPT-4 for answer generation
- Ragas evaluation framework
- Real-time performance monitoring
"""

from __future__ import annotations

import json
import os
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Load environment variables from .env file before other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import numpy as np

# Import core dependencies
try:
    # Try relative imports first (for module execution)
    from .rag_system import create_rag_system, EnhancedRAGSystem
    from .evaluation import evaluate_dataset
    from .config import get_config, RAGConfig, get_openai_api_key
    from .validation import RAGError, ErrorHandler
    from .metrics import get_metrics_collector, get_alert_manager
    from .cache import clear_all_caches
except ImportError:
    # Fall back to absolute imports (for direct execution)
    try:
        from rag_system import create_rag_system, EnhancedRAGSystem
        from evaluation import evaluate_dataset
        from config import get_config, RAGConfig, get_openai_api_key
        from validation import RAGError, ErrorHandler
        from metrics import get_metrics_collector, get_alert_manager
        from cache import clear_all_caches
    except ImportError:
        # If absolute imports also fail, use dummy implementations
        def create_rag_system(*args, **kwargs): return None
        class EnhancedRAGSystem: pass
        def evaluate_dataset(*args, **kwargs): return {}
        def get_config(): return None
        class RAGConfig: pass
        def get_openai_api_key(): return None
        class RAGError(Exception): pass
        class ErrorHandler: pass
        def get_metrics_collector(): return None
        def get_alert_manager(): return None
        def clear_all_caches(): pass


# Ensure src directory is in path for absolute imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
QUESTIONS_DIR = os.path.join(ROOT, "questions")
ANSWERS_DIR = os.path.join(ROOT, "answers")


def _ensure_dirs() -> None:
    """Ensure all required directories exist and warn if source directories are missing."""
    # Create results and cache directories
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Get cache directory from config and ensure it exists
    try:
        config = get_config()
        os.makedirs(config.cache.cache_dir, exist_ok=True)
    except:
        pass

    # Check if source directories exist and warn user
    if not os.path.exists(DATA_DIR):
        st.warning(f"⚠️ Data directory not found: {DATA_DIR}. Please add documents to index.")

    if not os.path.exists(QUESTIONS_DIR):
        st.info(f"ℹ️ Questions directory not found: {QUESTIONS_DIR}. Evaluation features may be unavailable.")

    if not os.path.exists(ANSWERS_DIR):
        st.info(f"ℹ️ Answers directory not found: {ANSWERS_DIR}. Evaluation features may be unavailable.")


@st.cache_resource(show_spinner=False, hash_funcs={dict: lambda x: json.dumps(x, sort_keys=True)})
def _load_rag_system(
    config: Dict[str, Any],
    data_dir: str,
    cache_settings: str,
    retrieval_settings: str,
    force_rebuild: bool = False
) -> EnhancedRAGSystem:
    """Build and cache the enhanced RAG system with expanded cache key."""
    try:
        # Validate API key is available before trying to build
        api_key = get_openai_api_key()
        if not api_key:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.info("Create a .env file in the project root with: OPENAI_API_KEY=your-key-here")
            st.stop()

        # Use nest_asyncio to allow nested event loops in Streamlit
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        # Create RAG system with configuration
        # Check if there's already a running loop
        try:
            loop = asyncio.get_running_loop()
            # If we get here, a loop is already running (Streamlit context)
            # We need to create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(create_rag_system(
                    data_dir=DATA_DIR,
                    api_key=api_key,
                    build_index=True
                )))
                system = future.result()
        except RuntimeError:
            # No loop is running, we can use asyncio.run directly
            system = asyncio.run(create_rag_system(
                data_dir=DATA_DIR,
                api_key=api_key,
                build_index=True
            ))
        return system
    except RAGError as e:
        # Handle validation errors with user-friendly messages
        error_response = ErrorHandler.create_safe_error_response(e, hide_details=True)
        st.error(f"❌ Configuration Error: {error_response['error']}")
        if os.getenv("DEBUG", "").lower() == "true":
            st.json(error_response)
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        if os.getenv("DEBUG", "").lower() == "true":
            st.exception(e)
        st.stop()


def page_query():
    """RAG query interface with tunable hybrid retrieval."""
    # Main header is handled by logo, no text header needed here
    
    # Sidebar configuration
    st.sidebar.header("Query Configuration")
    top_k = st.sidebar.slider("Top-K chunks", min_value=1, max_value=20, value=10)
    
    hybrid_alpha = st.sidebar.slider(
        "Hybrid Retrieval Balance", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="⚖️ **Retrieval Strategy Balance**\n\n" +
             "• **0.0**: Pure keyword search (BM25/TF-IDF) - exact term matching\n" +
             "• **0.5**: Balanced hybrid - combines semantic understanding with keywords\n" +
             "• **1.0**: Pure semantic search - meaning-based retrieval\n\n" +
             "**Recommendation**: Use 0.3-0.7 for most queries, lower for technical terms, higher for conceptual questions."
    )
    
    use_cache = st.sidebar.checkbox("Use caching", value=True, 
                                   help="Enable caching for faster repeated queries")
    
    # Display system status
    config = get_config()
    st.sidebar.info(f"""**System Status**
    - Embedding Model: {config.embedding.model}
    - Generation Model: {config.generation.model}
    - Cache Enabled: {config.cache.enable_disk_cache}
    """)
    
    # Load RAG system with expanded cache key
    system_config = {
        "embedding_model": config.embedding.model,
        "generation_model": config.generation.model
    }
    cache_settings = json.dumps({
        "enable_disk_cache": config.cache.enable_disk_cache,
        "cache_ttl": config.cache.cache_ttl
    }, sort_keys=True)
    retrieval_settings = json.dumps({
        "faiss_index_type": config.retrieval.faiss_index_type,
        "hybrid_alpha": config.retrieval.hybrid_alpha
    }, sort_keys=True)

    rag_system = _load_rag_system(system_config, DATA_DIR, cache_settings, retrieval_settings)
    
    # Query input
    query = st.text_area(
        "Ask a question", 
        height=100, 
        placeholder="Type your question here (minimum 3 characters)...",
        help="Ask any question about the documents in the knowledge base"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        submit_button = st.button("🚀 Submit Query", type="primary")
    
    with col2:
        if st.button("🧹 Clear Cache"):
            clear_all_caches()
            st.success("Cache cleared successfully!")
    
    if submit_button and query.strip():
        try:
            # Validate query
            if len(query.strip()) < 3:
                st.error("Query must be at least 3 characters long.")
                return
            
            # Execute query with progress indicator
            with st.spinner("🔄 Processing your query..."):
                start_time = time.time()

                # Run async query with hybrid_alpha, handling nested event loops
                try:
                    loop = asyncio.get_running_loop()
                    # Running loop exists, use thread executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(rag_system.query(
                            query,
                            top_k=top_k,
                            use_cache=use_cache,
                            hybrid_alpha=hybrid_alpha
                        )))
                        result = future.result()
                except RuntimeError:
                    # No running loop, use asyncio.run directly
                    result = asyncio.run(rag_system.query(
                        query,
                        top_k=top_k,
                        use_cache=use_cache,
                        hybrid_alpha=hybrid_alpha
                    ))

                total_time = time.time() - start_time
            
            # Display results
            st.success(f"✅ Query completed in {total_time:.2f} seconds")
            
            # Answer section
            st.subheader("📝 Answer")
            st.markdown(f"**{result['answer']}**")
            
            # Retrieved context
            st.subheader("📚 Retrieved Context")
            
            for i, chunk_info in enumerate(result['chunks'], 1):
                with st.expander(
                    f"📄 Context {i}: {chunk_info['doc_id']} - {chunk_info['title']} (Score: {chunk_info['similarity_score']:.3f})"
                ):
                    st.markdown(chunk_info['content'])
                    st.caption(f"Source: {chunk_info['source_path']} | Position: {chunk_info['position']}")
            
            # Performance metrics
            st.subheader("⚡ Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Retrieval Time", 
                    f"{result['metadata']['retrieval_time_ms']:.0f} ms",
                    help="Time spent retrieving relevant chunks"
                )
            
            with col2:
                st.metric(
                    "Generation Time", 
                    f"{result['metadata']['generation_time_ms']:.0f} ms",
                    help="Time spent generating the answer"
                )
            
            with col3:
                st.metric(
                    "Total Time", 
                    f"{result['metadata']['total_time_ms']:.0f} ms",
                    help="Total query processing time"
                )
            
            # Additional metadata
            with st.expander("🔧 Technical Details"):
                st.json({
                    "chunks_retrieved": result['metadata']['num_chunks_retrieved'],
                    "embedding_model": result['metadata']['embedding_model'],
                    "generation_model": result['metadata']['generation_model'],
                    "tokens_used": result['metadata'].get('total_tokens', 'N/A'),
                    "finish_reason": result['metadata'].get('finish_reason', 'N/A')
                })
                
        except RAGError as e:
            error_response = ErrorHandler.create_safe_error_response(e, hide_details=True)
            st.error(f"❌ {error_response['error']}")
        except Exception as e:
            error_response = ErrorHandler.create_safe_error_response(e, hide_details=True)
            st.error(f"❌ {error_response['message']}")

            # Only show details if DEBUG mode is enabled
            if os.getenv("DEBUG", "").lower() == "true":
                with st.expander("🐛 Error Details (Debug Mode)"):
                    st.json(error_response)
    
    elif submit_button:
        st.warning("⚠️ Please enter a query before submitting.")


def page_performance():
    """Performance dashboard focused on RAG evaluation metrics."""
    st.header("📊 Performance Analytics")
    st.write("Monitor system performance and evaluate RAG quality using comprehensive metrics.")
    
    # Dataset evaluation section
    st.subheader("🧪 RAG Quality Evaluation")
    st.write("Comprehensive evaluation using Ragas metrics against predefined question-answer pairs. This evaluation is independent of user queries above and focuses on measuring RAG system quality.")
    
    col1, col2 = st.columns(2)
    with col1:
        eval_top_k = st.slider("Evaluation Top-K chunks", 1, 15, 10, key="eval_topk")
    with col2:
        eval_hybrid_alpha = st.slider("Evaluation Hybrid Alpha", 0.0, 1.0, 0.5, step=0.1, key="eval_alpha")
    
    if st.button("🧪 Run Quality Evaluation"):
        with st.spinner("Running comprehensive RAG evaluation..."):
            try:
                out = evaluate_dataset(
                    DATA_DIR,
                    QUESTIONS_DIR,
                    ANSWERS_DIR,
                    RESULTS_DIR,
                    top_k=eval_top_k,
                    retriever_type="hybrid",
                    hybrid_alpha=eval_hybrid_alpha,
                    embedding_model="text-embedding-3-small",
                    use_ragas=True,
                    use_token_metrics=False
                )
                st.success("✅ Evaluation completed!")
                
                # Display Ragas metrics
                if "ragas_metrics" in out["summary"]:
                    st.subheader("📈 Ragas Quality Metrics")
                    ragas_metrics = out["summary"]["ragas_metrics"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Answer Relevancy", f"{ragas_metrics.get('answer_relevancy', 0):.3f}",
                                help="How relevant the generated answer is to the question")
                        st.metric("Context Precision", f"{ragas_metrics.get('context_precision', 0):.3f}",
                                help="How relevant the retrieved context is to the question")
                    
                    with col2:
                        st.metric("Faithfulness", f"{ragas_metrics.get('faithfulness', 0):.3f}",
                                help="How well the answer is grounded in retrieved context")
                        st.metric("Context Recall", f"{ragas_metrics.get('context_recall', 0):.3f}",
                                help="How well context covers ground truth information")
                    
                    with col3:
                        st.metric("Answer Correctness", f"{ragas_metrics.get('answer_correctness', 0):.3f}",
                                help="Factual accuracy compared to reference answer")
                        st.metric("Answer Similarity", f"{ragas_metrics.get('answer_similarity', 0):.3f}",
                                help="Semantic similarity to reference answer")
                
                # Show evaluation summary
                st.json(out["summary"])
                st.caption(f"📁 Detailed results saved to: {out['results_json']}")
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
    
    # Load and display existing evaluation results
    eval_results_path = os.path.join(RESULTS_DIR, "eval_results.json")
    if os.path.exists(eval_results_path):
        st.subheader("📋 Previous Evaluation Results")
        
        # Show summary of existing results
        try:
            with open(eval_results_path, 'r') as f:
                existing_results = json.load(f)
            
            if "summary" in existing_results:
                summary = existing_results["summary"]
                st.info(f"""
                **Last Evaluation Summary:**
                - Samples: {summary.get('num_samples', 0)}
                - Average Response Time: {summary.get('latency', {}).get('total_ms_mean', 0):.0f}ms
                - Ragas Answer Relevancy: {summary.get('ragas_metrics', {}).get('answer_relevancy', 'N/A')}
                - Ragas Faithfulness: {summary.get('ragas_metrics', {}).get('faithfulness', 'N/A')}
                """)
                
        except Exception as e:
            st.warning(f"Could not load previous results: {str(e)}")
        
        st.subheader("🔍 Question-Answer Triplets Analysis")
        st.write("Analyze individual questions, expected answers, and generated responses with their quality metrics.")
        
        try:
            with open(eval_results_path, 'r') as f:
                results = json.load(f)
            
            if "samples" in results and results["samples"]:
                # Create triplet visualization
                sample_idx = st.selectbox(
                    "Select sample to analyze:",
                    range(len(results["samples"])),
                    format_func=lambda x: f"Q{x+1}: {results['samples'][x]['question'][:60]}..."
                )
                
                if sample_idx is not None:
                    sample = results["samples"][sample_idx]
                    
                    st.write("### Query-Expected-Actual Triplet")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**❓ Question:**")
                        st.info(sample["question"])
                        
                        st.write("**✅ Expected Answer:**")
                        st.success(sample["reference"])
                    
                    with col2:
                        st.write("**🤖 Generated Answer:**")
                        st.warning(sample["prediction"])
                        
                        st.write("**📊 Individual Ragas Metrics:**")
                        if "ragas_answer_relevancy" in sample:
                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric("Answer Relevancy", f"{sample.get('ragas_answer_relevancy', 0):.3f}")
                                st.metric("Answer Correctness", f"{sample.get('ragas_answer_correctness', 0):.3f}")
                                st.metric("Context Precision", f"{sample.get('ragas_context_precision', 0):.3f}")
                            with metrics_col2:
                                st.metric("Faithfulness", f"{sample.get('ragas_faithfulness', 0):.3f}")
                                st.metric("Answer Similarity", f"{sample.get('ragas_answer_similarity', 0):.3f}")
                                st.metric("Context Recall", f"{sample.get('ragas_context_recall', 0):.3f}")
                        else:
                            st.info("No individual Ragas metrics available. Run a new evaluation to see per-sample metrics.")
                        
                        # Performance metrics
                        st.write("**⚡ Performance:**")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("Retrieval", f"{sample.get('retrieval_ms', 0):.0f} ms")
                        with perf_col2:
                            st.metric("Generation", f"{sample.get('generation_ms', 0):.0f} ms")
                        with perf_col3:
                            st.metric("Total", f"{sample.get('total_ms', 0):.0f} ms")
                            
                        # Additional info
                        st.write("**📋 Additional Info:**")
                        st.caption(f"Sample ID: {sample.get('id', 'N/A')}")
                        st.caption(f"Context Chunks: {sample.get('context_chunks', 0)}")
                        
            else:
                st.info("No evaluation results found. Run an evaluation first.")
                
        except Exception as e:
            st.error(f"❌ Failed to load results: {str(e)}")
    else:
        st.info("No evaluation results available. Run a quality evaluation first to see triplet analysis.")


def main():
    st.set_page_config(
        page_title="RAGonRAG", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Logo and main description - centered and larger
    logo_path = os.path.join(ROOT, "RAGonRAG_Logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=500)
        else:
            st.title("🤖 RAGonRAG")
    
    st.markdown("""
    **RAGonRAG** is a Retrieval Augmented Generation (RAG) system that combines semantic understanding 
    with traditional keyword search. It is based on transcripts from the DeepLearning.AI RAG course lectures.
    Ask questions about these documents and get intelligent answers powered 
    by hybrid retrieval, FAISS vector database, and GPT-4. Monitor system performance with comprehensive 
    Ragas-based evaluation metrics.
    """)
    
    st.markdown("---")
    
    # Sidebar navigation - centered logo
    sidebar_logo_path = os.path.join(ROOT, "RAG_Sidebar_Logo.png")
    if os.path.exists(sidebar_logo_path):
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.image(sidebar_logo_path, width=150)
    else:
        st.sidebar.markdown("<h1 style='text-align: center;'>RAGonRAG</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["🔍 Query Interface", "📊 Performance Analytics"],
        index=0
    )
    
    # System status indicator
    api_key_present = bool(get_openai_api_key())
    config = get_config()
    
    status_color = "🟢" if api_key_present else "🔴"
    st.sidebar.markdown(f"""
    ### System Status
    {status_color} **API Status:** {'Connected' if api_key_present else 'No API Key'}

    **Configuration:**
    - 🧠 Generation: {config.generation.model}
    - 📝 Embeddings: {config.embedding.model}
    - 💾 Caching: {'Enabled' if config.cache.enable_disk_cache else 'Disabled'}
    """)

    # Rebuild index button
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Index Management")

    if st.sidebar.button("Rebuild Index", help="Force rebuild the FAISS index from data directory"):
        if not api_key_present:
            st.sidebar.error("API key required to rebuild index")
        else:
            with st.sidebar.status("Rebuilding index...", expanded=True) as status:
                st.write("Clearing caches...")
                clear_all_caches()

                st.write("Removing old index files...")
                # Remove old FAISS index files
                import glob
                for pattern in ["*.faiss", "*.json", "faiss_store/*"]:
                    for file in glob.glob(os.path.join(ROOT, pattern)):
                        try:
                            os.remove(file)
                            st.write(f"Removed {os.path.basename(file)}")
                        except:
                            pass

                st.write("Building new index...")
                try:
                    # Clear cached RAG system
                    if 'rag_system' in st.session_state:
                        del st.session_state['rag_system']

                    # Force rebuild by calling with build_index=True
                    _load_rag_system(force_rebuild=True)

                    status.update(label="✅ Index rebuilt successfully!", state="complete", expanded=False)
                    st.sidebar.success("Index rebuilt! Please rerun your queries.")
                except Exception as e:
                    status.update(label="❌ Index rebuild failed", state="error", expanded=True)
                    st.sidebar.error(f"Failed to rebuild index: {str(e)}")

    st.sidebar.markdown("---")
    
    # Version info
    st.sidebar.caption("""
    **RAGonRAG Features**
    
    ✅ Hybrid Semantic + Keyword Search \n
    ✅ FAISS Vector Database  \n
    ✅ Ragas Quality Evaluation \n
    ✅ Tunable Retrieval Balance \n
    ✅ Real-time Performance Analytics \n
    ✅ Question-Answer Triplet Analysis
    """)
    
    # Main content
    if not api_key_present:
        st.error("""
        ⚠️ **OpenAI API Key Required**
        
        Please set up your OpenAI API key:
        
        **Option 1: Create .env file (recommended)**
        ```bash
        cp .env.example .env
        # Edit .env file and add: OPENAI_API_KEY=your_actual_key_here
        ```
        
        **Option 2: Environment variable**
        ```bash
        export OPENAI_API_KEY="your_api_key_here"
        ```
        """)
        st.stop()
    
    _ensure_dirs()
    
    if page == "🔍 Query Interface":
        page_query()
    else:
        page_performance()


if __name__ == "__main__":
    main()
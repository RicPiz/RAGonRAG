"""
Evaluation utilities for the RAG system using Ragas library.

Features:
- Load numbered questions/answers from markdown files.
- Execute RAG pipeline on the questions and record latencies.
- Compute comprehensive RAG evaluation using Ragas metrics.
- Optional token-level metrics using scikit-learn for backwards compatibility.
- Persist per-sample results and aggregate metrics to the results directory.

Notes on metrics:
- Ragas metrics: answer_relevancy, faithfulness, context_precision, context_recall, answer_correctness
- Token metrics: binary bag-of-words via sklearn CountVectorizer (optional).
- Latency statistics: mean and percentiles via numpy.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import core dependencies
try:
    from .rag_system import create_rag_system
    from .logger import get_logger
except ImportError:
    try:
        from rag_system import create_rag_system
        from logger import get_logger
    except ImportError:
        def create_rag_system(*args, **kwargs): return None
        def get_logger(name): return None

logger = get_logger("evaluation")


@dataclass
class QAPair:
    """Represents a single question and its reference answer."""

    id: str
    question: str
    reference: str


def _parse_numbered_items(md_text: str) -> List[str]:
    """Parse a markdown file containing numbered items like '1. ...', '2. ...'.

    Args:
        md_text: Raw markdown content.

    Returns:
        List of item strings, preserving order.
    """
    # Split on lines that begin with a number followed by a dot and space.
    lines = md_text.splitlines()
    items: List[str] = []
    current: List[str] = []
    pat = re.compile(r"^\s*\d+\.\s+")
    for line in lines:
        if pat.match(line):
            # Flush previous item
            if current:
                items.append("\n".join(current).strip())
                current = []
            # Start new item with the line content after the number.
            current.append(pat.sub("", line).strip())
        else:
            current.append(line)
    if current:
        items.append("\n".join(current).strip())
    # Filter out any accidental empties
    return [x for x in items if x]


def load_qa_pairs(questions_dir: str, answers_dir: str) -> List[QAPair]:
    """Load questions and reference answers from directories with Module_* files.

    Assumes files are named like 'Module_1_Questions.md' and 'Module_1_Answers.md'.

    Args:
        questions_dir: Directory containing question files.
        answers_dir: Directory containing answer files.

    Returns:
        List of QAPair objects.
    """
    q_files = sorted([f for f in os.listdir(questions_dir) if f.endswith("_Questions.md")])
    qa_pairs: List[QAPair] = []
    for qf in q_files:
        module_prefix = qf.replace("_Questions.md", "")
        af = f"{module_prefix}_Answers.md"
        q_path = os.path.join(questions_dir, qf)
        a_path = os.path.join(answers_dir, af)
        if not os.path.exists(a_path):
            # Skip if the matching answer file is missing
            continue
        with open(q_path, "r", encoding="utf-8") as fq:
            q_items = _parse_numbered_items(fq.read())
        with open(a_path, "r", encoding="utf-8") as fa:
            a_items = _parse_numbered_items(fa.read())
        for i, (q, a) in enumerate(zip(q_items, a_items), start=1):
            qa_pairs.append(QAPair(id=f"{module_prefix}_{i}", question=q, reference=a))
    return qa_pairs


def _token_metrics_per_pair(vectorizer: CountVectorizer, reference: str, prediction: str) -> Tuple[float, float, float, float]:
    """Compute token-level accuracy, precision, recall, F1 using sklearn metrics.

    The vectors are binary token presence for the union tokens in this pair.

    Args:
        vectorizer: A CountVectorizer fit over reference+prediction corpora (binary=True).
        reference: Ground-truth answer text.
        prediction: Model answer text.

    Returns:
        Tuple (accuracy, precision, recall, f1)
    """
    X = vectorizer.transform([reference, prediction]).toarray()
    y_true = X[0]
    y_pred = X[1]
    # Restrict to union of tokens present in either ref or pred for this pair.
    mask = (y_true + y_pred) > 0
    if not mask.any():
        # No tokens; define metrics as zeros to avoid division by zero.
        return 0.0, 0.0, 0.0, 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    acc = float(accuracy_score(yt, yp))
    prec = float(precision_score(yt, yp, zero_division=0))
    rec = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))
    return acc, prec, rec, f1


def evaluate_dataset(
    data_dir: str,
    questions_dir: str,
    answers_dir: str,
    results_dir: str,
    *,
    top_k: int = 5,
    retriever_type: str = "hybrid",
    hybrid_alpha: float = 0.5,
    embedding_model: str = "text-embedding-3-small",
    use_token_metrics: bool = False,
    use_ragas: bool = True,
) -> Dict[str, object]:
    """Run evaluation over all Q/A pairs and store results.

    Args:
        data_dir: Directory with source markdown documents.
        questions_dir: Directory with question markdown files.
        answers_dir: Directory with reference answer markdown files.
        results_dir: Directory to write results JSON/CSV.
        top_k: Number of retrieved chunks to use per query.
        use_token_metrics: Whether to compute token-based metrics (default: False).
        use_ragas: Whether to compute Ragas RAG metrics (default: True).

    Returns:
        A dictionary of aggregate metrics and file paths to results.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Build RAG pipeline once
    async def build_system():
        return await create_rag_system(
            data_dir=data_dir,
            build_index=True
        )
    
    system = asyncio.run(build_system())

    qa = load_qa_pairs(questions_dir, answers_dir)
    if not qa:
        raise RuntimeError("No QA pairs found. Ensure questions/ and answers/ are populated.")

    # Execute predictions and capture latencies
    per_sample: List[Dict[str, object]] = []
    predictions: List[str] = []
    references: List[str] = []
    questions: List[str] = []
    contexts: List[List[str]] = []
    total_ms_list: List[float] = []
    retrieval_ms_list: List[float] = []
    generation_ms_list: List[float] = []

    llm_used_count = 0
    
    async def process_questions():
        results = []
        for item in qa:
            t0 = time.perf_counter()
            result = await system.query(item.question, top_k=top_k, hybrid_alpha=hybrid_alpha)
            t1 = time.perf_counter()
            results.append((item, result, (t1 - t0) * 1000.0))
        return results
    
    question_results = asyncio.run(process_questions())
    
    for item, result, duration_ms in question_results:
        pred = str(result.get("answer", ""))
        predictions.append(pred)
        references.append(item.reference)
        questions.append(item.question)
        
        # Extract context chunks for RAG evaluation
        retrieved_chunks = result.get("chunks", [])
        context_texts = [chunk.get("content", "") if isinstance(chunk, dict) else str(chunk) 
                        for chunk in retrieved_chunks]
        contexts.append(context_texts)
        
        metadata = result.get("metadata", {})
        per_sample.append(
            {
                "id": item.id,
                "question": item.question,
                "reference": item.reference,
                "prediction": pred,
                "context_chunks": len(context_texts),
                "retrieval_ms": float(metadata.get("retrieval_time_ms", 0.0)),
                "generation_ms": float(metadata.get("generation_time_ms", 0.0)),
                "total_ms": float(metadata.get("total_time_ms", duration_ms)),
                "llm_used": True,  # New system always uses LLM
                "llm_error": "",
            }
        )
        llm_used_count += 1
        retrieval_ms_list.append(float(metadata.get("retrieval_time_ms", 0.0)))
        generation_ms_list.append(float(metadata.get("generation_time_ms", 0.0)))
        total_ms_list.append(float(metadata.get("total_time_ms", duration_ms)))

    # Optional token-level metrics (backwards compatibility)
    macro_token = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if use_token_metrics:
        # Vectorize tokens across all references and predictions using existing CountVectorizer
        vectorizer = CountVectorizer(binary=True)
        vectorizer.fit(references + predictions)

        # Compute per-sample token metrics using sklearn
        per_sample_metrics: List[Dict[str, float]] = []
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            acc, prec, rec, f1 = _token_metrics_per_pair(vectorizer, ref, pred)
            per_sample_metrics.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
            per_sample[i].update({"token_accuracy": acc, "token_precision": prec, "token_recall": rec, "token_f1": f1})

        # Aggregate token metrics (macro over questions)
        macro_token = {
            "accuracy": float(np.mean([m["accuracy"] for m in per_sample_metrics])) if per_sample_metrics else 0.0,
            "precision": float(np.mean([m["precision"] for m in per_sample_metrics])) if per_sample_metrics else 0.0,
            "recall": float(np.mean([m["recall"] for m in per_sample_metrics])) if per_sample_metrics else 0.0,
            "f1": float(np.mean([m["f1"] for m in per_sample_metrics])) if per_sample_metrics else 0.0,
        }

    # Ragas evaluation metrics
    ragas_summary: Dict[str, float] = {}
    ragas_error: str = ""
    
    if use_ragas:
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            )
            from datasets import Dataset
            
            # Create dataset for Ragas evaluation
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "answer": predictions,
                "contexts": contexts,
                "ground_truth": references,
            })
            
            # Define metrics to evaluate
            metrics = [
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            ]
            
            # Run evaluation
            result = evaluate(eval_dataset, metrics=metrics)

            # Convert result to pandas DataFrame for easier handling
            result_df = result.to_pandas()

            # Compute summary statistics using pandas with NaN handling
            ragas_summary = {
                "answer_relevancy": float(result_df["answer_relevancy"].mean(skipna=True)),
                "faithfulness": float(result_df["faithfulness"].mean(skipna=True)),
                "context_precision": float(result_df["context_precision"].mean(skipna=True)),
                "context_recall": float(result_df["context_recall"].mean(skipna=True)),
                "answer_correctness": float(result_df["answer_correctness"].mean(skipna=True)),
                "answer_similarity": float(result_df["answer_similarity"].mean(skipna=True)),
            }

            # Store per-sample results from the dataframe
            for i, sample in enumerate(per_sample):
                if i < len(result_df):
                    try:
                        # Use itertuples for safer iteration
                        row = result_df.iloc[i]
                        sample.update({
                            "ragas_answer_relevancy": float(row["answer_relevancy"]) if not np.isnan(row["answer_relevancy"]) else 0.0,
                            "ragas_faithfulness": float(row["faithfulness"]) if not np.isnan(row["faithfulness"]) else 0.0,
                            "ragas_context_precision": float(row["context_precision"]) if not np.isnan(row["context_precision"]) else 0.0,
                            "ragas_context_recall": float(row["context_recall"]) if not np.isnan(row["context_recall"]) else 0.0,
                            "ragas_answer_correctness": float(row["answer_correctness"]) if not np.isnan(row["answer_correctness"]) else 0.0,
                            "ragas_answer_similarity": float(row["answer_similarity"]) if not np.isnan(row["answer_similarity"]) else 0.0,
                        })
                    except (IndexError, KeyError, TypeError) as e:
                        # Fallback to summary values if individual values can't be extracted
                        logger.warning(f"Failed to extract individual ragas metrics for sample {i}: {e}")
                        sample.update({
                            "ragas_answer_relevancy": ragas_summary["answer_relevancy"],
                            "ragas_faithfulness": ragas_summary["faithfulness"],
                            "ragas_context_precision": ragas_summary["context_precision"],
                            "ragas_context_recall": ragas_summary["context_recall"],
                            "ragas_answer_correctness": ragas_summary["answer_correctness"],
                            "ragas_answer_similarity": ragas_summary["answer_similarity"],
                        })

            # Ensure lengths align - log mismatches
            if len(contexts) != len(questions) or len(contexts) != len(predictions):
                logger.warning(f"Length mismatch: contexts={len(contexts)}, questions={len(questions)}, predictions={len(predictions)}")
                    
        except Exception as e:
            ragas_error = f"ragas_error: {type(e).__name__}: {e}"
            ragas_summary = {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "answer_similarity": 0.0,
            }

    # Latency stats using numpy
    latency = {
        "retrieval_ms_mean": float(np.mean(retrieval_ms_list)) if retrieval_ms_list else 0.0,
        "generation_ms_mean": float(np.mean(generation_ms_list)) if generation_ms_list else 0.0,
        "total_ms_mean": float(np.mean(total_ms_list)) if total_ms_list else 0.0,
        "total_ms_p50": float(np.percentile(total_ms_list, 50)) if total_ms_list else 0.0,
        "total_ms_p95": float(np.percentile(total_ms_list, 95)) if total_ms_list else 0.0,
    }

    summary = {
        "num_samples": len(per_sample),
        "ragas_metrics": ragas_summary,
        "token_metrics": macro_token if use_token_metrics else {"note": "Token metrics disabled - using Ragas instead"},
        "latency": latency,
        "llm_usage_rate": float(llm_used_count / max(1, len(per_sample))),
        "ragas_error": ragas_error,
    }

    # Persist results
    results_json = os.path.join(results_dir, "eval_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        result_data = {
            "summary": summary,
            "samples": per_sample,
        }
        
        json.dump(
            result_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {"summary": summary, "results_json": results_json}


if __name__ == "__main__":
    # Allow running as a script for quick evaluation.
    ROOT = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(ROOT, "data")
    questions_dir = os.path.join(ROOT, "questions")
    answers_dir = os.path.join(ROOT, "answers")
    results_dir = os.path.join(ROOT, "results")
    out = evaluate_dataset(data_dir, questions_dir, answers_dir, results_dir)
    print(json.dumps(out["summary"], indent=2))

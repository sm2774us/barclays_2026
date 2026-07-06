"""High-Performance Vectorized Hybrid (Dense + Sparse) Retriving Framework.

Designed to isolate domain-specific terminology matches and surface structured
contexts for institutional financing desk platforms.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import torch

# Configure institutional-grade logger
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """Explicitly typed container for unified retrieval payloads."""
    text: str
    source: str
    dense_score: float
    sparse_score: float
    final_rank_score: float


@dataclass(slots=True)
class EvaluationReport:
    """Metrics container for system profiling execution paths."""
    latencies_us: dict[str, float]
    retrieved_records: list[RetrievedChunk]


class NativeVectorizedBM25:
    """Vectorized BM25 Okapi engine for high-precision token matching."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        
        # Tokenize documents and construct baseline maps
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.doc_lengths = np.array([len(doc) for doc in self.tokenized_corpus], dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lengths)) if self.corpus_size > 0 else 1.0
        
        # Generate global vocabulary index mapping
        self.vocab: dict[str, int] = {}
        for doc in self.tokenized_corpus:
            for token in doc:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    
        self.vocab_size = len(self.vocab)
        self.idf: dict[str, float] = {}
        self._compute_idf_map()

    def _compute_idf_map(self) -> None:
        """Computes global inverse document frequencies across the indexed corpus."""
        doc_counts = np.zeros(self.vocab_size, dtype=np.int32)
        for doc in self.tokenized_corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_counts[self.vocab[token]] += 1
                
        for token, idx in self.vocab.items():
            n_t = doc_counts[idx]
            # Formulate standard smooth BM25 variant
            calculated_idf = math.log((self.corpus_size - n_t + 0.5) / (n_t + 0.5) + 1.0)
            self.idf[token] = max(calculated_idf, 1e-4)

    def score_query(self, query_tokens: list[str]) -> np.ndarray:
        """Computes BM25 scores across all indexed corpus documents simultaneously."""
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in query_tokens:
            if token not in self.vocab:
                continue
            
            token_idf = self.idf[token]
            # Construct execution arrays for term frequency tracking
            term_freqs = np.array([doc.count(token) for doc in self.tokenized_corpus], dtype=np.float32)
            
            # Execute standard BM25 scaling formula components
            denominator = term_freqs + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / self.avg_doc_len))
            scores += token_idf * ((term_freqs * (self.k1 + 1.0)) / denominator)
            
        return scores


class ProductionHybridEngine:
    """Engine orchestrating dense representations, sparse mappings, and cross-re-ranking."""

    def __init__(self, chunks: list[str], sources: list[str], embedding_dim: int = 384) -> None:
        self.chunks = chunks
        self.sources = sources
        self.embedding_dim = embedding_dim
        
        # Initialize the sparse engine
        self.sparse_engine = NativeVectorizedBM25(chunks)
        
        # Generate simulated semantic space mappings representing normalized pre-trained bi-encoder states
        torch.manual_seed(42)
        raw_embeddings = torch.randn(len(chunks), embedding_dim)
        self.normalized_embeddings = torch.nn.functional.normalize(raw_embeddings, p=2, dim=1)

    def execute_retrieval(self, query: str, top_dense: int = 5, top_sparse: int = 5, final_top_k: int = 3) -> EvaluationReport:
        """Executes a hybrid search query, merges candidate pools, and re-ranks the results."""
        metrics: dict[str, float] = {}
        query_tokens = query.lower().split()
        
        # --- Path A: Dense Scan ---
        t_start = time.perf_counter_ns()
        simulated_query_vector = torch.randn(1, self.embedding_dim)
        normalized_query = torch.nn.functional.normalize(simulated_query_vector, p=2, dim=1)
        
        # Direct matrix tensor dot product matching
        dense_similarities = (self.normalized_embeddings @ normalized_query.T).squeeze(1).numpy()
        dense_candidate_indices = np.argsort(dense_similarities)[::-1][:top_dense]
        metrics["dense_scan_us"] = (time.perf_counter_ns() - t_start) / 1000.0

        # --- Path B: Sparse Scan ---
        t_start = time.perf_counter_ns()
        sparse_scores = self.sparse_engine.score_query(query_tokens)
        sparse_candidate_indices = np.argsort(sparse_scores)[::-1][:top_sparse]
        metrics["sparse_scan_us"] = (time.perf_counter_ns() - t_start) / 1000.0

        # --- Path C: Deduplication & Cross-Attention Re-Ranking ---
        t_start = time.perf_counter_ns()
        unified_candidates = list(set(dense_candidate_indices) | set(sparse_candidate_indices))
        
        scored_records: list[RetrievedChunk] = []
        for idx in unified_candidates:
            text_content = self.chunks[idx]
            
            # Simulated Cross-Encoder Attention layer
            # Models query-to-context token dependencies using exact string matching and dense similarity scores
            exact_matches = sum(1 for token in query_tokens if token in text_content.lower())
            semantic_factor = float(dense_similarities[idx])
            
            # Cross-encoder proxy scoring logic
            rerank_score = (semantic_factor * 0.4) + (exact_matches * 0.6)
            
            scored_records.append(RetrievedChunk(
                text=text_content,
                source=self.sources[idx],
                dense_score=float(dense_similarities[idx]),
                sparse_score=float(sparse_scores[idx]),
                final_rank_score=rerank_score
            ))
            
        # Final sort by descending cross-encoder scores
        scored_records.sort(key=lambda record: record.final_rank_score, reverse=True)
        final_filtered_output = scored_records[:final_top_k]
        metrics["cross_rerank_us"] = (time.perf_counter_ns() - t_start) / 1000.0
        
        return EvaluationReport(latencies_us=metrics, retrieved_records=final_filtered_output)


def export_performance_charts(report: EvaluationReport) -> None:
    """Generates visualization tracking profile latencies and candidate sorting mechanics."""
    # Chart 1: Latency breakdown by engine execution path
    labels = list(report.latencies_us.keys())
    microseconds = list(report.latencies_us.values())
    
    fig_lat = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=microseconds,
            marker_color=["rgba(54, 162, 235, 0.75)", "rgba(255, 159, 64, 0.75)", "rgba(153, 102, 255, 0.75)"],
            text=[f"{v:.1f} μs" for v in microseconds],
            textposition='auto'
        )
    ])
    fig_lat.update_layout(
        title="Institutional RAG Pipeline Latency Profile by Component",
        xaxis_title="Retrieval Execution Phase",
        yaxis_title="Execution Duration (Microseconds)",
        template="plotly_white",
        width=800,
        height=450
    )
    fig_lat.write_html("rag_latency_profile.html")

    # Chart 2: Structural Breakdown of Candidate Rank Signals
    sources = [f"Chunk {i+1} ({rec.source})" for i, rec in enumerate(report.retrieved_records)]
    dense_signals = [rec.dense_score for rec in report.retrieved_records]
    sparse_signals = [rec.sparse_score for rec in report.retrieved_records]
    final_ranks = [rec.final_rank_score for rec in report.retrieved_records]
    
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Bar(x=sources, y=dense_signals, name='Dense Semantic Path Score', marker_color='rgba(75, 192, 192, 0.7)'))
    fig_sig.add_trace(go.Bar(x=sources, y=sparse_signals, name='Sparse Keyword Path Score', marker_color='rgba(255, 99, 132, 0.7)'))
    fig_sig.add_trace(go.Scatter(x=sources, y=final_ranks, name='Unified Cross-Encoder Output', mode='lines+markers', line=dict(color='purple', width=3)))
    
    fig_sig.update_layout(
        title="Candidate Multi-Path Scoring Convergence Graph",
        xaxis_title="Retrieved Knowledge Components",
        yaxis_title="Normalized Score Signals",
        barmode='group',
        template="plotly_white",
        width=900,
        height=500
    )
    fig_sig.write_html("hybrid_search_comparison.html")
    logger.info("RAG analytical validation reports written to disk successfully.")


if __name__ == "__main__":
    # Simulate an unstructured legal document corpus from a financing desk
    knowledge_corpus = [
        "Section 4. Haircut Policy. For global generic General Collateral (GC) repo agreements, the baseline execution haircut penalty tier is fixed explicitly at 2.00% for on-the-run United States Treasury instruments.",
        "Section 5. Off-the-run adjustments. If a transaction involves an off-the-run sovereign asset, an explicit operational premium addition of 0.50% is added directly to the baseline margin requirement, establishing a total 2.50% target threshold.",
        "Master Securities Loan Agreement (MSLA) Margin Framework. Collateral updates require verification of exact ISIN indicators. If clearing operations fail matching checks, a mandatory holding buffer rate of 5.00% triggers immediately.",
        "Risk Policy Amendment Memo 2026-Q2. Margin calls are distributed via automated execution parameters across corporate bond allocations. High-yield tranches maintain an isolated floor haircut bound of 12.00%."
    ]
    
    metadata_sources = [
        "GMRA_CORE_POLICY_SEC4.pdf",
        "GMRA_OFF_THE_RUN_AMEND.pdf",
        "MSLA_MASTER_TEMPLATE_2024.pdf",
        "RISK_EXEC_MEMO_CURRENT.pdf"
    ]

    # Initialize the hybrid retrieval engine
    engine = ProductionHybridEngine(chunks=knowledge_corpus, sources=metadata_sources)
    
    # Run a sample query targeting specific alphanumeric provisions
    target_query = "What is the haircut penalty tier for off-the-run asset instruments under the current policy?"
    execution_report = engine.execute_retrieval(query=target_query, top_dense=3, top_sparse=3, final_top_k=2)

    # Output verified content components to the terminal
    print("\n" + "="*90)
    print("                      INSTITUTIONAL HYBRID RETRIEVAL ENGINE OUTPUT                     ")
    print("="*90)
    for index, record in enumerate(execution_report.retrieved_records):
        print(f"\n[RANK POSITION {index + 1}] Source Doc: {record.source}")
        print(f" -> Dense Correlation Component  : {record.dense_score:.4f}")
        print(f" -> Sparse Term Vector Component : {record.sparse_score:.4f}")
        print(f" -> Final Cross-Encoder Priority : {record.final_rank_score:.4f}")
        print(f" -> Extracted Context Segment    : \"{record.text}\"")
    print("="*90 + "\n")

    # Export execution profiles
    export_performance_charts(execution_report)
"""P5 — Retrieval-Augmented Generation Copilot for the Financing Desk.

Implements a hybrid sparse/dense retrieval pipeline with reciprocal-rank
fusion, a grounded-generation contract that requires per-claim citation
or an explicit refusal, and an evaluation harness reporting recall@k and
a lexical-overlap faithfulness proxy (a fully rigorous LLM-as-judge
faithfulness score requires a live model call and is described, with the
exact contract the production system uses, in the accompanying
dissertation; this module implements the retrieval, fusion, refusal, and
evaluation machinery that is model-agnostic).

Retrieval combines:

* **BM25** (Robertson & Zaragho, 2009) — the industry-standard sparse
  lexical ranking function, implemented from its closed-form definition
  rather than naive term-overlap counting.
* **TF-IDF cosine similarity** as the dense-retrieval proxy (a drop-in
  interface point for a production sentence-embedding encoder).
* **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, 2009) to
  combine the two rankings without needing to calibrate score scales
  against each other, which is the standard hybrid-retrieval fusion
  technique in production search systems.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclasses.dataclass(frozen=True)
class Document:
    """A single retrievable corpus chunk.

    Attributes:
        doc_id: Unique identifier, used as the citation key.
        text: Chunk text.
        source: Human-readable source filename, shown to the trader.
    """

    doc_id: str
    text: str
    source: str


class BM25:
    """Okapi BM25 sparse lexical ranker (Robertson & Zaragho, 2009).

    BM25 scores a document ``d`` against a query ``q`` as
    ``sum_t IDF(t) * (f(t,d)*(k1+1)) / (f(t,d) + k1*(1-b+b*|d|/avgdl))``
    over query terms ``t``, where ``f(t,d)`` is term frequency, ``|d|``
    document length, ``avgdl`` average document length, and ``k1``, ``b``
    tunable saturation/length-normalization parameters. This is the
    de-facto lexical baseline every production hybrid-retrieval system is
    benchmarked against.
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        """Indexes the corpus.

        Args:
            documents: Corpus to index.
            k1: Term-frequency saturation parameter (typical range 1.2-2.0).
            b: Length-normalization parameter in [0, 1].
        """
        self._docs = documents
        self._k1, self._b = k1, b
        self._tokenized = [self._tokenize(d.text) for d in documents]
        self._doc_lens = np.array([len(t) for t in self._tokenized])
        self._avgdl = float(np.mean(self._doc_lens)) if len(self._docs) else 0.0
        self._df: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._df.update(set(tokens))
        self._n = len(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercases and splits text into alphanumeric tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _idf(self, term: str) -> float:
        """Computes the BM25 inverse document frequency for one term."""
        n_t = self._df.get(term, 0)
        return math.log((self._n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        """Scores every indexed document against ``query``.

        Args:
            query: Raw query string.

        Returns:
            Array of BM25 scores, one per indexed document, in corpus order.
        """
        q_terms = self._tokenize(query)
        scores = np.zeros(self._n)
        for i, tokens in enumerate(self._tokenized):
            tf = Counter(tokens)
            doc_len = self._doc_lens[i]
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * doc_len / max(self._avgdl, 1e-8))
                scores[i] += idf * (freq * (self._k1 + 1)) / denom
        return scores


class HybridRetriever:
    """BM25 + TF-IDF-cosine hybrid retriever with reciprocal-rank fusion."""

    def __init__(self, documents: list[Document]) -> None:
        """Indexes the corpus for both sparse and dense retrieval.

        Args:
            documents: Corpus to index.
        """
        self._docs = documents
        self._bm25 = BM25(documents)
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        self._doc_matrix = self._tfidf.fit_transform([d.text for d in documents]) if documents else None

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[tuple[Document, float]]:
        """Retrieves the top-k documents via reciprocal-rank fusion.

        Reciprocal Rank Fusion combines two independently-scored rankings
        via ``RRF(d) = sum_rankers 1 / (rrf_k + rank_ranker(d))``, which
        is scale-free (does not require normalizing BM25 and cosine
        scores onto a comparable range) and empirically robust across
        heterogeneous retrieval signals.

        Args:
            query: Raw query string.
            top_k: Number of documents to return.
            rrf_k: RRF smoothing constant (60 is the standard default
                from the original Cormack et al. paper).

        Returns:
            A list of ``(document, fused_score)`` tuples, highest-scored first.
        """
        if not self._docs:
            return []
        bm25_scores = self._bm25.score(query)
        bm25_rank = np.argsort(-bm25_scores)

        query_vec = self._tfidf.transform([query])
        dense_scores = cosine_similarity(query_vec, self._doc_matrix).flatten()
        dense_rank = np.argsort(-dense_scores)

        rrf_scores = np.zeros(len(self._docs))
        for rank_position, doc_idx in enumerate(bm25_rank):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank_position + 1)
        for rank_position, doc_idx in enumerate(dense_rank):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank_position + 1)

        ranked_idx = np.argsort(-rrf_scores)[:top_k]
        return [(self._docs[i], float(rrf_scores[i])) for i in ranked_idx if rrf_scores[i] > 0]

    def max_dense_similarity(self, query: str) -> float:
        """Returns the raw top TF-IDF cosine similarity for ``query``.

        Used as a principled, scale-meaningful refusal gate: unlike the
        rank-based RRF fusion score (which is bounded and compresses the
        gap between a strong and a spurious single-term match on a small
        corpus), raw cosine similarity directly measures how much lexical
        content the query and the best-matching chunk actually share.

        Args:
            query: Raw query string.

        Returns:
            The maximum cosine similarity in [0, 1] across the corpus,
            or 0.0 for an empty corpus.
        """
        if not self._docs:
            return 0.0
        query_vec = self._tfidf.transform([query])
        return float(cosine_similarity(query_vec, self._doc_matrix).max())


@dataclasses.dataclass(frozen=True)
class GroundedAnswer:
    """A generation-layer response, contractually grounded or refusing.

    Attributes:
        answer: Natural-language answer text, or an ``INSUFFICIENT_EVIDENCE`` refusal.
        citations: Document IDs supporting the answer.
        confidence: One of ``"high"``, ``"medium"``, ``"low"``.
        is_refusal: Whether the system declined to answer.
    """

    answer: str
    citations: list[str]
    confidence: str
    is_refusal: bool


SYSTEM_PROMPT = (
    "You are a Liquid Financing desk copilot. Answer ONLY using the provided context. "
    "Every factual claim must carry a [doc_id] citation. If the context does not contain "
    "enough information to answer confidently, respond exactly with: "
    "INSUFFICIENT_EVIDENCE: <what is missing>. Never invent fee levels, haircuts, or policy terms."
)


def build_prompt(query: str, retrieved: list[tuple[Document, float]]) -> str:
    """Assembles the final grounded-generation prompt.

    Args:
        query: The trader's natural-language question.
        retrieved: Ranked ``(document, score)`` pairs from :meth:`HybridRetriever.retrieve`.

    Returns:
        The full prompt string to send to the generation model.
    """
    context = "\n".join(f"[{d.doc_id}:{d.source}] {d.text}" for d, _ in retrieved)
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"


def generate_grounded_answer(
    query: str,
    retrieved: list[tuple[Document, float]],
    relevance_floor: float = 0.0,
    max_dense_similarity: float | None = None,
    dense_similarity_floor: float = 0.12,
) -> GroundedAnswer:
    """Reference (non-LLM) generation stand-in enforcing the grounding contract.

    In production this function is replaced by a call to a fine-tuned or
    prompted LLM using :data:`SYSTEM_PROMPT`; this reference
    implementation enforces the *same contract* — cite-or-refuse — using
    extractive synthesis, so the refusal and citation behavior can be
    unit-tested deterministically without a live model dependency.

    Refusal is gated primarily on ``max_dense_similarity`` (raw TF-IDF
    cosine similarity between the query and its best-matching chunk):
    on a small corpus, the rank-based RRF fusion score alone compresses
    the gap between a genuinely relevant match and a spurious
    single-term overlap, whereas raw cosine similarity directly measures
    how much lexical content the query and the corpus actually share.

    Args:
        query: The trader's natural-language question.
        retrieved: Ranked ``(document, score)`` pairs.
        relevance_floor: Minimum fused RRF retrieval score for a document
            to be considered at all (secondary filter).
        max_dense_similarity: Precomputed
            :meth:`HybridRetriever.max_dense_similarity` for this query;
            if ``None``, the dense-similarity gate is skipped (RRF-only mode).
        dense_similarity_floor: Minimum raw cosine similarity required to
            treat the corpus as having sufficient evidence.

    Returns:
        A :class:`GroundedAnswer`.
    """
    insufficient_evidence = (
        max_dense_similarity is not None and max_dense_similarity < dense_similarity_floor
    )
    usable = [(d, s) for d, s in retrieved if s > relevance_floor]
    if not usable or insufficient_evidence:
        return GroundedAnswer(
            answer="INSUFFICIENT_EVIDENCE: no retrieved document met the relevance bar for this query.",
            citations=[], confidence="low", is_refusal=True,
        )
    citations = [d.doc_id for d, _ in usable]
    synthesis = " ".join(f"[{d.doc_id}] {d.text}" for d, _ in usable[:3])
    confidence = "high" if len(usable) >= 2 else "medium"
    return GroundedAnswer(answer=synthesis, citations=citations, confidence=confidence, is_refusal=False)


def recall_at_k(
    retriever: HybridRetriever, labeled_queries: list[tuple[str, set[str]]], k: int = 10
) -> float:
    """Computes mean recall@k over a labeled query/relevant-document set.

    Args:
        retriever: A fitted :class:`HybridRetriever`.
        labeled_queries: List of ``(query, relevant_doc_ids)`` pairs.
        k: Cutoff rank.

    Returns:
        Mean recall@k across all labeled queries.
    """
    recalls = []
    for query, relevant_ids in labeled_queries:
        retrieved_ids = {d.doc_id for d, _ in retriever.retrieve(query, top_k=k)}
        if not relevant_ids:
            continue
        recalls.append(len(retrieved_ids & relevant_ids) / len(relevant_ids))
    return float(np.mean(recalls)) if recalls else 0.0


def faithfulness_proxy(answer: GroundedAnswer, retrieved_texts: list[str]) -> float:
    """Lexical-overlap faithfulness proxy (Jaccard token overlap).

    A production system replaces this with an LLM-as-judge faithfulness
    score (does every claim trace to retrieved context); this proxy is a
    fast, deterministic lower-bound sanity check usable in CI: a
    faithful, extractive answer should have high token overlap with its
    cited context by construction.

    Args:
        answer: The generated answer.
        retrieved_texts: Raw text of the documents that were retrieved for this query.

    Returns:
        Jaccard token overlap in [0, 1] between the answer and the union
        of retrieved context (``1.0`` for a correctly-formatted refusal).
    """
    if answer.is_refusal:
        return 1.0
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer.answer.lower()))
    context_tokens: set[str] = set()
    for t in retrieved_texts:
        context_tokens |= set(re.findall(r"[a-z0-9]+", t.lower()))
    if not answer_tokens:
        return 0.0
    return float(len(answer_tokens & context_tokens) / len(answer_tokens))


def _main() -> None:
    """Command-line entry point: builds the corpus, retrieves, and answers a demo query."""
    docs = [
        Document("D1", "BBB-rated financial corporate bonds carry a 15% base haircut, +5% if ADV < $2mm.", "haircut_policy_v7.txt"),
        Document("D2", "XYZ Corp borrow fee spiked to 210bps ahead of the dividend record date; utilization hit 96%.", "desk_note_20260630.txt"),
        Document("D3", "GC-SOFR spread widened 8bps intraday on quarter-end funding pressure.", "desk_note_20260628.txt"),
    ]
    retriever = HybridRetriever(docs)
    query = "Why did the XYZ Corp borrow fee spike, and what is our BBB haircut policy?"
    retrieved = retriever.retrieve(query, top_k=3)
    answer = generate_grounded_answer(query, retrieved)
    print("Citations:", answer.citations)
    print("Answer:", answer.answer)
    print("Faithfulness proxy:", faithfulness_proxy(answer, [d.text for d, _ in retrieved]))


if __name__ == "__main__":
    _main()

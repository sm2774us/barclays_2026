"""Unit tests for the Liquid Financing take-home reference modules.

Run with: pytest tests/ -q
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from liquid_financing import p1_fee_forecasting as p1
from liquid_financing import p2_margin_optimization as p2
from liquid_financing import p3_anomaly_detection as p3
from liquid_financing import p5_rag_copilot as p5


def test_pinball_loss_symmetric_at_median() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 1.5])
    loss = p1.pinball_loss(y_true, y_pred, quantile=0.5)
    assert loss > 0


def test_pinball_loss_zero_for_perfect_forecast() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert p1.pinball_loss(y, y, quantile=0.3) == 0.0


def test_weighted_mape_excludes_near_zero_fees() -> None:
    y_true = np.array([0.5, 50.0])
    y_pred = np.array([100.0, 55.0])
    weights = np.array([1.0, 1.0])
    mape = p1.weighted_mape(y_true, y_pred, weights, min_fee_bps=5.0)
    assert 0 <= mape < 50  # only the 50bps row counts; large outlier row is excluded


def test_conformalize_widens_interval() -> None:
    residuals = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    lo, hi = p1.conformalize(residuals, np.array([1.0]), np.array([2.0]), alpha=0.2)
    assert lo[0] < 1.0
    assert hi[0] > 2.0


def test_kupiec_pof_test_rejects_extreme_breach_rate() -> None:
    breaches = np.ones(100, dtype=bool)  # 100% breach rate vs 1% target -> should reject
    lr_stat, p_value = p2.kupiec_pof_test(breaches, target_rate=0.01)
    assert p_value < 0.01


def test_kupiec_pof_test_accepts_matching_rate() -> None:
    rng = np.random.default_rng(0)
    breaches = rng.random(1000) < 0.01
    _, p_value = p2.kupiec_pof_test(breaches, target_rate=0.01)
    assert p_value > 0.01  # should not reject a correctly-calibrated model


def test_christoffersen_flags_clustered_breaches() -> None:
    breaches = np.zeros(100, dtype=bool)
    breaches[40:50] = True  # ten consecutive breaches = clustering
    lr_stat, p_value = p2.christoffersen_independence_test(breaches)
    assert lr_stat > 0


def test_basel_traffic_light_zones() -> None:
    assert p2.basel_traffic_light(2, 250) == "GREEN"
    assert p2.basel_traffic_light(7, 250) == "AMBER"
    assert p2.basel_traffic_light(15, 250) == "RED"


def test_fuse_scores_bounded() -> None:
    score = p3.fuse_scores(mahalanobis_z=10.0, text_stress_prob=0.9)
    assert 0.0 <= score <= 1.0


def test_fuse_scores_low_for_calm_regime() -> None:
    score = p3.fuse_scores(mahalanobis_z=0.5, text_stress_prob=0.05)
    assert score < 0.3


def test_bm25_ranks_relevant_doc_first() -> None:
    docs = [
        p5.Document("A", "the haircut policy for BBB corporate bonds is fifteen percent", "a.txt"),
        p5.Document("B", "the weather in London is cloudy today", "b.txt"),
    ]
    bm25 = p5.BM25(docs)
    scores = bm25.score("BBB haircut policy")
    assert scores[0] > scores[1]


def test_hybrid_retriever_returns_ranked_results() -> None:
    docs = [
        p5.Document("A", "specialness fee spiked on the dividend record date", "a.txt"),
        p5.Document("B", "quarterly earnings call transcript summary", "b.txt"),
    ]
    retriever = p5.HybridRetriever(docs)
    results = retriever.retrieve("dividend fee spike", top_k=2)
    assert len(results) >= 1
    assert results[0][0].doc_id == "A"


def test_generate_grounded_answer_refuses_without_evidence() -> None:
    answer = p5.generate_grounded_answer("irrelevant query", [], max_dense_similarity=0.0)
    assert answer.is_refusal
    assert "INSUFFICIENT_EVIDENCE" in answer.answer


def test_generate_grounded_answer_cites_when_grounded() -> None:
    docs = [p5.Document("A", "haircut is fifteen percent for BBB bonds", "a.txt")]
    retrieved = [(docs[0], 0.5)]
    answer = p5.generate_grounded_answer("what is the haircut", retrieved, max_dense_similarity=0.5)
    assert not answer.is_refusal
    assert "A" in answer.citations


def test_faithfulness_proxy_is_one_for_refusal() -> None:
    refusal = p5.GroundedAnswer(answer="INSUFFICIENT_EVIDENCE: x", citations=[], confidence="low", is_refusal=True)
    assert p5.faithfulness_proxy(refusal, []) == 1.0

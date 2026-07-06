"""Production-grade model evaluation framework for quantitative infrastructure.

Implements deterministic numerical entity validation, simulated rubric-driven 
LLM-as-Judge scoring, and automated validation dashboard generation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go

# Configure infrastructure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationRecord:
    """Container tracking validation scores, text outputs, and safety flags."""
    query: str
    numeric_groundedness: float
    rubric_alignment: float
    length_penalty: float
    final_composite_score: float
    hallucination_detected: bool


class DeterministicFactualityScorer:
    """Validates that all numerical metrics match the retrieved source documents."""

    def __init__(self) -> None:
        # Capture numbers, percentages, spreads, and financial notations (e.g., 5.25%, $300M, 45bp)
        self.numeric_pattern = re.compile(r'\b\d+(?:\.\d+)?%?|\b\$\d+(?:M|B)?|\b\d+\s?bp\b')

    def extract_entities(self, text: str) -> set[str]:
        """Extracts unique string representations of numerical components."""
        matches = self.numeric_pattern.findall(text)
        return {m.strip().lower() for m in matches}

    def compute_groundedness(self, generated_text: str, source_context: str) -> tuple[float, bool]:
        """Calculates numerical precision by matching output figures against source data."""
        gen_entities = self.extract_entities(generated_text)
        src_entities = self.extract_entities(source_context)

        if not gen_entities:
            return 1.0, False  # No figures introduced to validate

        # Determine if any numbers generated do not appear in the source context
        unreferenced_entities = gen_entities - src_entities
        hallucination_triggered = len(unreferenced_entities) > 0
        
        # Calculate matching accuracy ratio
        precision_ratio = len(gen_entities & src_entities) / len(gen_entities)
        return precision_ratio, hallucination_triggered


class SemanticRubricJudgeProxy:
    """Simulates a rubric-driven judge model with length bias adjustments."""

    def __init__(self, length_penalty_weight: float = 0.05) -> None:
        self.gamma = length_penalty_weight

    def evaluate_quality(
        self, 
        generated_text: str, 
        reference_text: str, 
        query: str
    ) -> tuple[float, float]:
        """Grades completeness and formats structural length corrections."""
        query_tokens = set(query.lower().split())
        gen_tokens = set(generated_text.lower().split())
        ref_tokens = set(reference_text.lower().split())

        # Determine completeness by checking reference concept coverage
        matched_concepts = ref_tokens & gen_tokens
        base_alignment = len(matched_concepts) / len(ref_tokens) if ref_tokens else 1.0
        
        # Balance score adjustments based on context relevance
        context_factor = len(gen_tokens & query_tokens) / len(query_tokens) if query_tokens else 1.0
        rubric_score = min((base_alignment * 0.7) + (context_factor * 0.3), 1.0)

        # Calculate penalty to counter verbosity biases
        len_gen = len(generated_text)
        len_ref = len(reference_text)
        length_ratio = len_gen / len_ref if len_ref > 0 else 1.0
        
        penalty = self.gamma * np.log(length_ratio + 1.0) if length_ratio > 1.2 else 0.0
        return float(rubric_score), float(penalty)


class QuantitativeEvaluationSuite:
    """Orchestrates system evaluations, validates metrics, and outputs dashboards."""

    def __init__(self) -> None:
        self.factuality_scorer = DeterministicFactualityScorer()
        self.judge_proxy = SemanticRubricJudgeProxy()

    def run_suite(self, validation_dataset: list[dict]) -> list[EvaluationRecord]:
        """Runs the dataset through the joint evaluation pipeline."""
        logger.info("Initializing multi-layered evaluation validation array...")
        results = []

        for item in validation_dataset:
            query = item["query"]
            generated = item["generated_output"]
            source = item["retrieved_context"]
            reference = item["gold_reference"]

            # 1. Run deterministic numerical check
            num_score, hallucinated = self.factuality_scorer.compute_groundedness(generated, source)

            # 2. Run simulated rubric-driven semantic evaluation
            rubric_score, len_penalty = self.judge_proxy.evaluate_quality(generated, reference, query)

            # 3. Calculate final score using numerical data as a strict binary constraint gate
            if hallucinated:
                final_score = 0.0  # Zero tolerance for numeric inaccuracies in trading environments
            else:
                final_score = max((num_score * 0.5 + rubric_score * 0.5) - len_penalty, 0.0)

            results.append(EvaluationRecord(
                query=query,
                numeric_groundedness=num_score,
                rubric_alignment=rubric_score,
                length_penalty=len_penalty,
                final_composite_score=final_score,
                hallucination_detected=hallucinated
            ))

        return results


def export_evaluation_visualizations(records: list[EvaluationRecord]) -> None:
    """Generates structural quality diagnostics and metric correlation dashboards."""
    indices = [f"Case {i+1}" for i in range(len(records))]
    numeric_scores = [r.numeric_groundedness for r in records]
    rubric_scores = [r.rubric_alignment for r in records]
    final_scores = [r.final_composite_score for r in records]
    
    # Chart 1: Multi-Dimensional Metric Evaluation Breakdown
    fig = go.Figure()
    fig.add_trace(go.Bar(x=indices, y=numeric_scores, name="Numeric Groundedness Precision", marker_color="rgba(46, 204, 113, 0.75)"))
    fig.add_trace(go.Bar(x=indices, y=rubric_scores, name="Rubric Semantic Alignment", marker_color="rgba(52, 152, 219, 0.75)"))
    fig.add_trace(go.Scatter(x=indices, y=final_scores, name="Final Composite Pipeline Metric", mode="lines+markers", line=dict(color="crimson", width=3)))
    
    fig.update_layout(
        title="GenAI System Evaluation Profile: Classical Checking vs Judgement Modeling",
        xaxis_title="Validation Reference Cases",
        yaxis_title="Normalized Evaluation Scores",
        barmode="group",
        template="plotly_white",
        width=950,
        height=500
    )
    fig.write_html("model_evaluation_profile.html")
    
    # Chart 2: Cumulative Distribution of Validation Performance
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Box(y=final_scores, name="Pipeline Scores Space", boxpoints="all", jitter=0.3, pointpos=-1.8, marker_color="purple"))
    fig_dist.update_layout(
        title="Final Score Population Distribution (Safety Gated)",
        yaxis_title="Calculated Composite Limits",
        template="plotly_white",
        width=600,
        height=450
    )
    fig_dist.write_html("metric_distribution_profile.html")
    logger.info("Evaluation metrics successfully written to analytical dashboard files.")


if __name__ == "__main__":
    # Define an evaluation dataset based on common repo desk operations
    test_eval_dataset = [
        {
            "query": "What is the specific haircut requirement for alternative clean energy corporate bounds?",
            "retrieved_context": "Section 9. Corporate Allocation Spreads. High-grade green alternative bonds maintain a fixed baseline haircut limit of 6.50%. Speculative tier entities scale to a 14.00% boundary floor.",
            "gold_reference": "Clean energy corporate bonds require a baseline haircut limit of 6.50% for high-grade assets, while speculative tier assets require a 14.00% floor.",
            "generated_output": "Based on Section 9, alternative clean energy corporate bonds maintain a baseline haircut limit of 6.50% for high-grade profiles, scaling up to a 14.00% boundary floor for speculative tier entities."
        },
        {
            "query": "Identify the margin buffer rate required upon counterparty settlement delays.",
            "retrieved_context": "Settlement Exceptions: In the event of a recognized delivery pause, an immediate operational clearing margin buffer of 3.25% must be posted by the initiator.",
            "gold_reference": "A settlement exception requires an immediate operational clearing margin buffer of 3.25%.",
            "generated_output": "If settlement delays happen, the initiator must instantly post an operational clearing margin buffer of 5.75% to address clearance exceptions." 
            # Note: The output contains a simulated numeric hallucination (5.75% instead of 3.25%)
        },
        {
            "query": "Detail the maximum exposure capacity limit assigned to the global sovereign arbitrage basket.",
            "retrieved_context": "Arbitrage Allocations: Sovereign portfolio concentration risk exposure bounds are strictly restricted to a maximum capacity threshold of $300M across all clearings.",
            "gold_reference": "The maximum capacity exposure limit for the sovereign arbitrage basket is capped at $300M.",
            "generated_output": "The allocation limits matrix states that for sovereign arbitrage, exposure bounds are capped at a maximum capacity threshold of $300M across clearings. This policy is explicitly confirmed by regional risk desks to prevent systemic concentration spikes across global portfolios."
            # Note: The output is highly verbose, which will trigger a structural length penalty adjustment
        }
    ]

    # Initialize and execute the evaluation pipeline
    suite = QuantitativeEvaluationSuite()
    evaluation_records = suite.run_suite(test_eval_dataset)

    # Output metric performance metrics directly to the console
    print("\n" + "="*100)
    print("                    INSTITUTIONAL VALIDATION STACK PERFORMANCE REPORT                    ")
    print("="*100)
    for index, record in enumerate(evaluation_records):
        print(f"\n[EVALUATION CASE {index + 1}]")
        print(f" ├── Target Query Context     : {record.query}")
        print(f" ├── Numeric Precision Score  : {record.numeric_groundedness * 100:.2f}%")
        print(f" ├── Semantic Rubric Score    : {record.rubric_alignment * 100:.2f}%")
        print(f" ├── Active Length Penalty    : {record.length_penalty:.4f}")
        print(f" ├── Hallucination Intercepted: {record.hallucination_detected}")
        print(f" └── Final Gated Output Metric: {record.final_composite_score:.4f}")
    print("="*100 + "\n")

    # Save visual performance dashboards
    export_evaluation_visualizations(evaluation_records)
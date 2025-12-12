"""
Evaluation metrics for diagnosis and reconstruction quality.

Implements all evaluation measures required for analysing:

    1. Diagnosis quality (ranking-based SFL)
    2. Reconstruction quality (precision/recall/F1 and similarity)

Following the definitions in:
"Spectrum-based fault diagnosis with partial traces"
and standard SFL literature.
"""

from __future__ import annotations

from typing import List, Tuple, Set, Dict


# ============================================================================
# DIAGNOSIS QUALITY  (Ranking-based evaluation)
# ============================================================================

# -------------------------------
# 1. Basic WE (already correct)
# -------------------------------

def wasted_effort(ranking: List[Tuple[str, float]], faulty_component: str) -> int:
    """
    Standard Wasted Effort (WE):
        Number of components ranked *before* the faulty one.
    """
    for idx, (comp, _) in enumerate(ranking):
        if comp == faulty_component:
            return idx
    return len(ranking)


def normalized_wasted_effort(we: int, num_components: int) -> float:
    """
    NWE = WE / (M - 1), with M = number of ranked components.
    """
    if num_components <= 1:
        return 0.0
    return we / float(num_components - 1)


# -----------------------------------------
# 2. Tie-aware WE (used in research papers)
# -----------------------------------------

def tie_aware_wasted_effort(ranking: List[Tuple[str, float]], faulty: str) -> float:
    """
    Tie-aware WE:
    If several components share the same suspiciousness score,
    we take the *average rank* of all tied items.

    Example:
        Ranking: [A:0.9, B:0.8, C:0.8, D:0.1]
        If faulty=C:
            tied block = {B,C} at positions {1,2}
            average rank = (1+2)/2 = 1.5
            WE = 1.5
    """
    scores = [score for _, score in ranking]

    # First find the faulty score
    faulty_score = None
    for comp, score in ranking:
        if comp == faulty:
            faulty_score = score
            break
    if faulty_score is None:
        return float(len(ranking))

    # Find all items tied with the faulty one
    tied_indices = [i for i, (_, s) in enumerate(ranking) if s == faulty_score]

    # Average rank of tied block
    avg_rank = sum(tied_indices) / len(tied_indices)
    return avg_rank


# ------------------------------------
# 3. Rank position + MRR
# ------------------------------------

def rank_position(ranking: List[Tuple[str, float]], faulty: str) -> float:
    """
    Return the (0-based) position of the faulty component.
    If missing, return len(ranking).
    """
    for i, (comp, _) in enumerate(ranking):
        if comp == faulty:
            return i
    return float(len(ranking))


def mean_reciprocal_rank(ranking: List[Tuple[str, float]], faulty: str) -> float:
    """
    MRR = 1 / (rank position + 1)
    If faulty is missing, MRR = 0.
    """
    pos = rank_position(ranking, faulty)
    if pos >= len(ranking):
        return 0.0
    return 1.0 / (pos + 1)


# ------------------------------
# 4. EXAM Score (classic SFL)
# ------------------------------

def exam_score(ranking: List[Tuple[str, float]], faulty: str) -> float:
    """
    EXAM = fraction of components examined until reaching the faulty one.
    Range: [0, 1].
    """
    pos = rank_position(ranking, faulty)
    return pos / max(1, len(ranking))


# ------------------------------
# 5. Top-k accuracy
# ------------------------------

def top_k_accuracy(ranking: List[Tuple[str, float]], faulty: str, k: int) -> bool:
    """
    True if the faulty component appears in the top-k ranked components.
    """
    top = [comp for comp, _ in ranking[:k]]
    return faulty in top


# ============================================================================
# RECONSTRUCTION QUALITY  (Trace reconstruction quality)
# ============================================================================

def reconstruction_quality(
        predicted_trace: Set[str],
        actual_full_trace: Set[str],
        hidden_components: Set[str],
) -> Tuple[float, float, float]:
    """
    Precision / Recall / F1 for reconstructed hidden components only.
    """
    retrieved_hidden = predicted_trace & hidden_components
    relevant_hidden = actual_full_trace & hidden_components

    tp = len(retrieved_hidden & relevant_hidden)
    fp = len(retrieved_hidden - relevant_hidden)
    fn = len(relevant_hidden - retrieved_hidden)

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


# -------------------------------------------
# Per-test reconstruction metrics (optional)
# -------------------------------------------

def per_test_reconstruction(
        predicted_traces: Dict[int, Set[str]],
        actual_traces: Dict[int, Set[str]],
        hidden: Set[str]
) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute precision/recall/F1 for each test separately.
    """
    results = {}
    for t_idx in predicted_traces:
        results[t_idx] = reconstruction_quality(
            predicted_traces[t_idx],
            actual_traces[t_idx],
            hidden,
        )
    return results


# ---------------------------------------------------
# Aggregated reconstruction (union or intersection)
# ---------------------------------------------------

def aggregate_union(predicted_traces: Dict[int, Set[str]]) -> Set[str]:
    """
    Union of reconstructed traces across all tests.
    """
    agg = set()
    for s in predicted_traces.values():
        agg |= s
    return agg


def aggregate_intersection(predicted_traces: Dict[int, Set[str]]) -> Set[str]:
    """
    Intersection of reconstructed traces across all tests.
    """
    all_sets = list(predicted_traces.values())
    if not all_sets:
        return set()
    inter = all_sets[0].copy()
    for s in all_sets[1:]:
        inter &= s
    return inter


# ----------------------------------------------
# Additional similarity measures for reconstruction
# ----------------------------------------------

def hamming_distance(pred: Set[str], actual: Set[str], universe: Set[str]) -> int:
    """
    Hamming distance between two binary indicator vectors defined by sets.
    Count of components where pred != actual.
    """
    dist = 0
    for c in universe:
        if (c in pred) != (c in actual):
            dist += 1
    return dist


def jaccard_similarity(pred: Set[str], actual: Set[str]) -> float:
    """
    Jaccard(pred, actual) = |pred ∩ actual| / |pred ∪ actual|
    """
    inter = len(pred & actual)
    union = len(pred | actual)
    if union == 0:
        return 1.0
    return inter / union


from typing import Optional, Dict
from .data_models import SpectrumData


class EvaluationSuite:
    """
    High-level helper to compute diagnosis and reconstruction metrics.

    Typical usage:

        suite = EvaluationSuite(
            ranking=ranking,
            faulty_component="module.faulty_func",
            spectrum=spectrum,                 # optional, used for |C| or H
            predicted_traces=predicted_traces, # optional
            actual_traces=actual_traces,       # optional
        )

        diagnosis = suite.diagnosis_metrics(top_k=[1, 3, 5])
        recon = suite.reconstruction_metrics()
        all_metrics = suite.all_metrics(top_k=[1, 3, 5])
    """

    def __init__(
            self,
            ranking: List[Tuple[str, float]],
            faulty_component: str,
            spectrum: Optional[SpectrumData] = None,
            predicted_traces: Optional[Dict[int, Set[str]]] = None,
            actual_traces: Optional[Dict[int, Set[str]]] = None,
            hidden_components: Optional[Set[str]] = None,
    ) -> None:
        self.ranking = ranking
        self.faulty_component = faulty_component
        self.spectrum = spectrum
        self.predicted_traces = predicted_traces
        self.actual_traces = actual_traces

        # Hidden components: explicit argument takes priority.
        if hidden_components is not None:
            self.hidden_components = set(hidden_components)
        elif spectrum is not None:
            self.hidden_components = set(spectrum.H)
        else:
            self.hidden_components = set()

    # ------------------------------------------------------------------ #
    # Diagnosis metrics                                                 #
    # ------------------------------------------------------------------ #

    def diagnosis_metrics(self, top_k: Optional[List[int]] = None) -> Dict[str, object]:
        """
        Compute diagnosis metrics for the current ranking and faulty component.

        Returns a dict with:
            - "WE", "NWE", "Tie-WE", "EXAM", "MRR"
            - "Top-k" : {k: bool, ...} if top_k is given
        """
        M = len(self.ranking)

        we = wasted_effort(self.ranking, self.faulty_component)
        nwe = normalized_wasted_effort(we, M)
        tie_we = tie_aware_wasted_effort(self.ranking, self.faulty_component)
        exam = exam_score(self.ranking, self.faulty_component)
        mrr = mean_reciprocal_rank(self.ranking, self.faulty_component)

        metrics: Dict[str, object] = {
            "WE": we,
            "NWE": nwe,
            "Tie-WE": tie_we,
            "EXAM": exam,
            "MRR": mrr,
        }

        if top_k:
            metrics["Top-k"] = {
                k: top_k_accuracy(self.ranking, self.faulty_component, k)
                for k in top_k
            }

        return metrics

    # ------------------------------------------------------------------ #
    # Reconstruction metrics                                            #
    # ------------------------------------------------------------------ #

    def reconstruction_metrics(self) -> Dict[str, object]:
        """
        Compute reconstruction metrics if predicted/actual traces are given.

        Returns a dict with per-test and aggregated scores:
            - "per_test": {t_idx: (precision, recall, f1), ...}
            - "macro_precision", "macro_recall", "macro_f1"
            - "union_precision", "union_recall", "union_f1"
            - "jaccard_union"
            - "hamming_union" (if a universe can be derived)
        """
        if self.predicted_traces is None or self.actual_traces is None:
            return {}

        hidden = self.hidden_components

        # Per-test metrics
        per_test = per_test_reconstruction(
            predicted_traces=self.predicted_traces,
            actual_traces=self.actual_traces,
            hidden=hidden,
        )

        # Macro-averages across tests
        if per_test:
            precisions = [p for (p, _, _) in per_test.values()]
            recalls = [r for (_, r, _) in per_test.values()]
            f1s = [f for (_, _, f) in per_test.values()]

            macro_precision = sum(precisions) / len(precisions)
            macro_recall = sum(recalls) / len(recalls)
            macro_f1 = sum(f1s) / len(f1s)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        # Union-based evaluation
        pred_union = aggregate_union(self.predicted_traces)
        actual_union = aggregate_union(self.actual_traces)

        union_precision, union_recall, union_f1 = reconstruction_quality(
            predicted_trace=pred_union,
            actual_full_trace=actual_union,
            hidden_components=hidden,
        )

        # Similarity measures on the union
        jacc = jaccard_similarity(pred_union, actual_union)

        universe = set(pred_union) | set(actual_union)
        if universe:
            ham = hamming_distance(pred_union, actual_union, universe)
        else:
            ham = 0

        return {
            "per_test": per_test,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "union_precision": union_precision,
            "union_recall": union_recall,
            "union_f1": union_f1,
            "jaccard_union": jacc,
            "hamming_union": ham,
        }

    # ------------------------------------------------------------------ #
    # Combined view                                                     #
    # ------------------------------------------------------------------ #

    def all_metrics(self, top_k: Optional[List[int]] = None) -> Dict[str, object]:
        """
        Convenience method: return both diagnosis and reconstruction metrics.

        Returns:
            {
              "diagnosis": {...},
              "reconstruction": {...}
            }
        """
        return {
            "diagnosis": self.diagnosis_metrics(top_k=top_k),
            "reconstruction": self.reconstruction_metrics(),
        }

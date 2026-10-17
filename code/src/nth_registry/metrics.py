"""Pair-level, fixed-gate, and registry-query evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def calibrate_max_threshold(
    negative_distances: np.ndarray,
    target_fpr: float,
) -> float:
    """Largest acceptance threshold with empirical FPR no greater than target."""

    negatives = np.asarray(negative_distances, dtype=float)
    if negatives.size == 0:
        raise ValueError("At least one negative distance is required")
    values, counts = np.unique(negatives, return_counts=True)
    empirical_fpr = np.cumsum(counts) / negatives.size
    valid = values[empirical_fpr <= target_fpr]
    if valid.size == 0:
        return float(np.nextafter(values[0], -np.inf))
    return float(valid[-1])


def fixed_gate_score(
    nth_distances: np.ndarray,
    embedding_distances: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Continuous score for a fixed NTH radius.

    Candidates rejected by NTH receive a finite sentinel strictly below every
    passing score.  This score is useful for ranking only.  Generic ROC helpers
    append a terminal threshold below the sentinel and therefore incorrectly
    recover every gate-rejected pair; use :func:`fixed_gate_operating_curve`
    for an operational fixed-gate curve.
    """

    nth = np.asarray(nth_distances)
    emb = np.asarray(embedding_distances, dtype=float)
    score = -emb
    sentinel = float(np.min(score[np.isfinite(score)]) - 1.0) if score.size else -1.0
    return np.where(nth <= radius, score, sentinel)


def fixed_gate_operating_curve(
    labels: np.ndarray,
    nth_distances: np.ndarray,
    embedding_distances: np.ndarray,
    radius: int,
) -> dict[str, object]:
    """Sweep only the second-stage threshold while the NTH gate stays fixed.

    The returned ROC/precision points stop when every *gate-passing* candidate
    has been accepted.  Gate-rejected positives are never made recoverable by
    an artificial terminal threshold.  ``ap`` is the attainable (truncated)
    average precision, i.e. the precision-weighted area from recall zero to the
    fixed gate's maximum recall.
    """

    truth = np.asarray(labels, dtype=bool)
    nth = np.asarray(nth_distances)
    emb = np.asarray(embedding_distances, dtype=float)
    if not (truth.ndim == nth.ndim == emb.ndim == 1):
        raise ValueError("labels and distances must be one-dimensional")
    if not (truth.size == nth.size == emb.size):
        raise ValueError("labels and distances must have equal length")
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("both positive and negative labels are required")

    passed = (nth <= radius) & np.isfinite(emb)
    passing_indices = np.flatnonzero(passed)
    if passing_indices.size == 0:
        empty = np.asarray([0.0])
        return {
            "fpr": empty.copy(),
            "tpr": empty.copy(),
            "precision": empty.copy(),
            "thresholds": np.asarray([np.inf]),
            "ap": 0.0,
            "maximum_recall": 0.0,
            "maximum_fpr": 0.0,
        }

    scores = -emb[passing_indices]
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    ordered_truth = truth[passing_indices][order]
    cumulative_tp = np.cumsum(ordered_truth, dtype=np.int64)
    cumulative_fp = np.cumsum(~ordered_truth, dtype=np.int64)

    # Emit one point after each score tie, matching a <= distance decision.
    group_ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), scores.size - 1]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    precision = np.r_[0.0, tp / np.maximum(tp + fp, 1)]
    thresholds = np.r_[np.inf, scores[group_ends]]
    attainable_ap = float(np.sum(np.diff(tpr) * precision[1:]))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "thresholds": thresholds,
        "ap": attainable_ap,
        "maximum_recall": float(tpr[-1]),
        "maximum_fpr": float(fpr[-1]),
    }


def curve_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
    }


@dataclass(frozen=True)
class RegistrySummary:
    query_count: int
    index_size: int
    radius: int
    true_candidate_recall: float
    any_false_candidate_rate: float
    mean_shortlist: float
    median_shortlist: float
    p95_shortlist: float
    max_shortlist: int


def summarize_registry_candidates(
    distance_matrix: np.ndarray,
    true_indices: np.ndarray,
    radius: int,
) -> RegistrySummary:
    matrix = np.asarray(distance_matrix)
    truth = np.asarray(true_indices, dtype=int)
    accepted = matrix <= radius
    shortlist = accepted.sum(axis=1)
    rows = np.arange(matrix.shape[0])
    true_pass = accepted[rows, truth]
    false_counts = shortlist - true_pass.astype(int)
    return RegistrySummary(
        query_count=int(matrix.shape[0]),
        index_size=int(matrix.shape[1]),
        radius=int(radius),
        true_candidate_recall=float(true_pass.mean()),
        any_false_candidate_rate=float((false_counts > 0).mean()),
        mean_shortlist=float(shortlist.mean()),
        median_shortlist=float(np.median(shortlist)),
        p95_shortlist=float(np.quantile(shortlist, 0.95)),
        max_shortlist=int(shortlist.max(initial=0)),
    )

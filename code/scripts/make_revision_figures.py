#!/usr/bin/env python
"""Generate publication figures directly from revision result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[2]
FIGURES = PROJECT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#2E74B5"
ORANGE = "#E67E22"
GREEN = "#2E8B57"
RED = "#B23A48"
GRAY = "#5B6573"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")
    boxes = [
        (0.25, 2.65, 1.35, 0.85, "Probe image", "#EAF2F8"),
        (1.95, 2.65, 1.45, 0.85, "Versioned 64-bit\nNTH", "#EAF2F8"),
        (3.75, 2.65, 1.55, 0.85, "Exact packed-code\nscan", "#EAF2F8"),
        (5.65, 2.65, 1.55, 0.85, "Candidate\nshortlist", "#FFF3E6"),
        (7.55, 2.65, 1.45, 0.85, "Cached vectors +\nprobe embedding", "#EAF7EF"),
        (9.30, 2.65, 0.95, 0.85, "Policy\ndecision", "#EAF7EF"),
        (3.45, 0.65, 2.45, 0.95, "Append-only off-chain log\n(scores, model, policy, prior head)", "#F4F6F7"),
        (7.00, 0.65, 2.85, 0.95, "Ethereum anchor\nCID hash, NTH, model/policy hashes, log head", "#FDEDEC"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor=GRAY, linewidth=1.1)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.4)
    for start, end in [((1.60, 3.08), (1.95, 3.08)), ((3.40, 3.08), (3.75, 3.08)), ((5.30, 3.08), (5.65, 3.08)), ((7.20, 3.08), (7.55, 3.08)), ((9.00, 3.08), (9.30, 3.08))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": GRAY})
    ax.annotate("", xy=(4.65, 1.60), xytext=(9.75, 2.65), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": GRAY, "connectionstyle": "angle3"})
    ax.annotate("", xy=(7.00, 1.13), xytext=(5.90, 1.13), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": GRAY})
    ax.text(5.25, 2.35, "fixed radius", ha="center", color=GRAY, fontsize=7.5)
    ax.text(8.95, 2.35, "calibration-only threshold", ha="center", color=GRAY, fontsize=7.5)
    ax.text(5.25, 0.28, "The anchor attests to recorded state; it does not verify perceptual correctness.", ha="center", color=RED, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "revised_system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def performance() -> None:
    deep = pd.read_csv(PROJECT / "results" / "deep_frozen_baseline_summary.csv").sort_values("test_recall")
    pair = pd.read_csv(PROJECT / "results" / "two_stage_shufflenet_pair_results.csv").query("split == 'test'")
    registry = pd.read_csv(PROJECT / "results" / "two_stage_shufflenet_registry_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.1))
    labels = deep["display_name"].str.replace(" (ImageNet)", "", regex=False).str.replace(" (LAION-2B)", "", regex=False)
    err = np.vstack([deep["test_recall"] - deep["test_recall_ci_low"], deep["test_recall_ci_high"] - deep["test_recall"]])
    axes[0].barh(labels, deep["test_recall"], xerr=err, color=BLUE, alpha=0.9, capsize=2)
    axes[0].set_xlim(0.80, 1.0); axes[0].set_xlabel("Held-out recall")
    axes[0].set_title("a  Frozen visual encoders")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].plot(pair["nth_radius"], pair["recall"], "o-", color=BLUE, label="pair recall")
    axes[1].plot(registry["nth_radius"], registry["per_query_any_false_match_probability"], "s--", color=RED, label="any false match/query")
    axes[1].set_xlabel("Fixed NTH radius"); axes[1].set_ylim(0, 1.0)
    axes[1].set_title("b  End-to-end outcomes")
    axes[1].grid(alpha=0.25); axes[1].legend(loc="lower right")
    axes[2].plot(registry["nth_radius"], registry["mean_stage1_shortlist"], "o-", color=ORANGE, label="stage-one shortlist")
    axes[2].plot(registry["nth_radius"], registry["mean_final_accepted_candidates"], "s--", color=GREEN, label="final accepted")
    axes[2].set_yscale("log"); axes[2].set_xlabel("Fixed NTH radius")
    axes[2].set_ylabel("Candidates per query (log scale)")
    axes[2].set_title("c  Registry workload")
    axes[2].grid(alpha=0.25, which="both"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "revised_performance_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def hash_tradeoff() -> None:
    data = pd.read_csv(PROJECT / "results" / "hash_registry_tradeoff_curves.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for method, part in data.groupby("method"):
        selected = part[(part["true_candidate_recall"] >= 0.5) & (part["true_candidate_recall"] <= 0.98)]
        ax.plot(selected["mean_shortlist"], selected["true_candidate_recall"], label=method)
    ax.set_xscale("log"); ax.set_xlim(0.5, 1064); ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Mean registry shortlist (log scale)")
    ax.set_ylabel("True-candidate recall")
    ax.grid(alpha=0.25, which="both"); ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "hash_registry_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def stress_scaling() -> None:
    crop = pd.read_csv(PROJECT / "results" / "nth_crop_ratio_results.csv")
    evasion = pd.read_csv(PROJECT / "results" / "nth_bounded_noise_evasion_summary.csv")
    index = pd.read_csv(PROJECT / "results" / "index_scaling.csv")
    gas = pd.read_csv(PROJECT / "results" / "contract_gas_scaling.csv").query("operation == 'registerImage'")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axes[0, 0].plot(crop["crop_fraction_removed"] * 100, crop["recall_at_calibrated_radius"], "o-", color=BLUE)
    axes[0, 0].set(xlabel="Width and height removed (%)", ylabel="Recall at NTH r=4", title="a  Crop sensitivity")
    for attack, part in evasion.groupby("attack"):
        axes[0, 1].plot(part["linf_epsilon_8bit"], part["evasion_rate"], "o-", label=attack.replace("-rademacher", ""))
    axes[0, 1].set(xlabel=r"Perturbation budget $\epsilon$ (8-bit)", ylabel="Evasion rate", title="b  Bounded black-box stress")
    axes[0, 1].legend()
    axes[1, 0].plot(index["index_size"], index["mean_lookup_ms"], "o-", color=ORANGE)
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="Index entries (log)", ylabel="Mean exact-scan time, ms (log)", title="c  Packed-code index scaling")
    axes[1, 1].plot(gas["registry_size_before"] + 1, gas["actual_gas_used"], "o-", color=GREEN)
    axes[1, 1].set_xscale("log"); axes[1, 1].set_ylim(180000, 186000)
    axes[1, 1].set(xlabel="Registry size (log)", ylabel="Gas used", title="d  Registration gas")
    for ax in axes.flat: ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "stress_and_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style(); architecture(); performance(); hash_tradeoff(); stress_scaling()

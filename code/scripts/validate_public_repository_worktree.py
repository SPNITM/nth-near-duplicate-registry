#!/usr/bin/env python
"""Validate an active public-repository checkout without touching its Git metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_PATH = ROOT / "validation" / "PUBLIC_REPOSITORY_VALIDATION.json"
CHECKSUM_PATH = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_signature(text: str) -> dict[str, object]:
    return {
        "citations": sorted({key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}),
        "labels": sorted(re.findall(r"\\label\{([^}]+)\}", text)),
        "references": sorted(re.findall(r"\\ref\{([^}]+)\}", text)),
        "figures": sorted(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)),
        "tables": text.count(r"\begin{table}"),
        "figure_environments": text.count(r"\begin{figure}"),
        "equations": text.count(r"\begin{equation}"),
        "bibliography": re.findall(r"\\bibliography\{([^}]+)\}", text),
    }


def brace_balance(text: str) -> tuple[int, int]:
    depth = 0
    premature = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                premature += 1
                depth = 0
    return depth, premature


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def tracked_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
        and ".pytest_cache" not in path.relative_to(ROOT).parts
        and path != CHECKSUM_PATH
    )


def main() -> int:
    required = [
        "README.md",
        "CITATION.cff",
        "manuscript/main.tex",
        "manuscript/main_highlighted_red.tex",
        "manuscript/ref_revised.bib",
        "results/two_stage_shufflenet_pair_results.csv",
        "results/seed_source_transfer_summary.csv",
        "code/scripts/run_seed_source_transfer.py",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]

    clean = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    marked = (ROOT / "manuscript" / "main_highlighted_red.tex").read_text(encoding="utf-8")
    clean_signature = scientific_signature(clean)
    marked_signature = scientific_signature(marked)

    policy_rows = read_csv(ROOT / "results" / "two_stage_shufflenet_pair_results.csv")
    thresholds = {
        int(row["nth_radius"]): float(row["embedding_threshold"])
        for row in policy_rows
        if row["split"] == "calibration" and int(row["nth_radius"]) in {12, 16}
    }
    transfer_rows = read_csv(ROOT / "results" / "seed_source_transfer_summary.csv")
    transfer_policy_rows = [row for row in transfer_rows if row["policy"].startswith("NTH r=")]
    threshold_mismatches = []
    for row in transfer_policy_rows:
        radius = int(float(row["nth_radius"]))
        expected = thresholds[radius]
        observed = float(row["embedding_threshold_from_main_calibration"])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            threshold_mismatches.append({"dataset": row["dataset"], "radius": radius, "expected": expected, "observed": observed})

    r16 = {row["dataset"]: row for row in transfer_rows if row["policy"] == "NTH r=16 + ShuffleNet"}
    expected_recall_text = {
        "COCO seed subset": f'{float(r16["COCO seed subset"]["recall"]):.4f}',
        "Copydays seed subset": f'{float(r16["Copydays seed subset"]["recall"]):.4f}',
    }
    manuscript_transfer_values_match = all(value in clean and value in marked for value in expected_recall_text.values())

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation_version_match = f'version: {pyproject["project"]["version"]}' in citation
    repository_url_present = "https://github.com/SPNITM/NTH-Near-Duplicate-Registry" in citation

    missing_figures = []
    for reference in sorted(set(clean_signature["figures"]) | set(marked_signature["figures"])):
        if not (ROOT / "manuscript" / reference).resolve().is_file():
            missing_figures.append(reference)

    files = tracked_files()
    forbidden = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx", ".docx", ".pdf", ".zip"}:
            forbidden.append(relative)
        if path.name == ".env":
            forbidden.append(relative)
        if path.suffix.lower() in {".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            forbidden.append(relative)
        if path.suffix.lower() == ".png" and not relative.startswith("figures/"):
            forbidden.append(relative)

    sensitive_patterns = {
        "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "github_token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "jupyter_token": re.compile(r"token=[A-Fa-f0-9]{32,}"),
        "private_hpc_ip": re.compile(r"192\.168\.1\.55"),
        "local_windows_user_path": re.compile(r"C:\\Users\\P24CS004", re.IGNORECASE),
        "local_workspace_path": re.compile(r"D:\\NTH", re.IGNORECASE),
        "hpc_home_path": re.compile("/home/" + "p21cs007", re.IGNORECASE),
    }
    text_suffixes = {".cff", ".csv", ".json", ".md", ".pbs", ".py", ".sol", ".tex", ".txt", ".toml", ".yaml", ".yml", ".mjs"}
    sensitive_findings = []
    for path in files:
        if path.suffix.lower() not in text_suffixes and path.name not in {".gitignore", ".env.example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in sensitive_patterns.items():
            if pattern.search(text):
                sensitive_findings.append({"file": path.relative_to(ROOT).as_posix(), "pattern": label})

    oversized = [
        {"file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
        for path in files
        if path.stat().st_size >= 100 * 1024 * 1024
    ]

    author_actions = {
        "licence_selection_required": "AUTHOR_TO_SELECT" in (ROOT / ".zenodo.json.template").read_text(encoding="utf-8"),
        "repository_doi_required": "DOI-TO-BE-INSERTED-BEFORE-RESUBMISSION" in clean,
        "author_metadata_confirmation_required": True,
    }
    passed = not any((missing, missing_figures, forbidden, sensitive_findings, oversized, threshold_mismatches)) and all(
        (
            clean_signature == marked_signature,
            brace_balance(clean) == (0, 0),
            brace_balance(marked) == (0, 0),
            manuscript_transfer_values_match,
            citation_version_match,
            repository_url_present,
        )
    )
    report = {
        "passed": passed,
        "validation_date": "2026-10-20",
        "missing_required_files": missing,
        "clean_highlighted_scientific_parity": clean_signature == marked_signature,
        "clean_brace_balance": brace_balance(clean),
        "highlighted_brace_balance": brace_balance(marked),
        "missing_figure_references": missing_figures,
        "forbidden_files": sorted(set(forbidden)),
        "sensitive_string_findings": sensitive_findings,
        "files_at_or_above_100_mib": oversized,
        "source_transfer_thresholds_match_final_calibration": not threshold_mismatches,
        "source_transfer_threshold_mismatches": threshold_mismatches,
        "source_transfer_r16_recall": {
            dataset: float(row["recall"]) for dataset, row in r16.items()
        },
        "manuscript_transfer_values_match_results": manuscript_transfer_values_match,
        "citation_version_matches_pyproject": citation_version_match,
        "citation_repository_url_present": repository_url_present,
        "author_actions_remaining": author_actions,
    }
    VALIDATION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    files = tracked_files()
    CHECKSUM_PATH.write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

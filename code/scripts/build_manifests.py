#!/usr/bin/env python
"""Audit supplied files and create deterministic, seed-disjoint manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code" / "src"))

from nth_registry.splits import make_seed_split, seed_id_from_filename


DATASETS = {
    "main": {
        "csv": Path("image_pairs_table.csv"),
        "original_dirs": [Path("NTH_IMAGES"), Path("NTH_IMAGES_EDITED")],
        "transformed_dirs": [Path("NTH_IMAGES_EDITED"), Path("NTH_IMAGES")],
    },
    "coco": {
        "csv": Path("ml part/data set generate/coco_image_pairs_table.csv"),
        "original_dirs": [Path("COCO_DATA"), Path("COCO_DATA_EDITED")],
        "transformed_dirs": [Path("COCO_DATA_EDITED"), Path("COCO_DATA")],
    },
    "copydays": {
        "csv": Path("ml part/data set generate/copy_image_pairs_table.csv"),
        "original_dirs": [Path("COPY_DAYS"), Path("COPY_DAYS_EDITED")],
        "transformed_dirs": [Path("COPY_DAYS_EDITED"), Path("COPY_DAYS")],
    },
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_lookup(root: Path, directories: list[Path]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for relative in directories:
        directory = root / relative
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                lookup.setdefault(path.name.lower(), str(path.resolve()))
    return lookup


def build_seed_lookup(root: Path, directories: list[Path]) -> dict[int, str]:
    """Resolve historical original names by identity, independent of suffix/extension."""
    lookup: dict[int, str] = {}
    for relative in directories:
        directory = root / relative
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            seed = safe_seed(path.name)
            if seed is not None:
                # Prefer an explicit *_orig file when both forms exist.
                if seed not in lookup or "_orig" in path.stem.lower():
                    lookup[seed] = str(path.resolve())
    return lookup


def safe_seed(filename: str) -> int | None:
    try:
        return seed_id_from_filename(filename)
    except ValueError:
        return None


def audit_dataset(name: str, spec: dict, source_root: Path, output: Path) -> dict:
    table_path = source_root / spec["csv"]
    if not table_path.is_file():
        return {
            "dataset": name,
            "status": "not_available",
            "pair_csv": str(table_path.resolve()),
            "reason": "pair CSV is not present under the selected source root",
        }
    frame = pd.read_csv(table_path)
    original_lookup = build_lookup(source_root, spec["original_dirs"])
    original_seed_lookup = build_seed_lookup(source_root, spec["original_dirs"])
    transformed_lookup = build_lookup(source_root, spec["transformed_dirs"])
    frame["seed_id"] = frame["Original Image"].map(safe_seed)
    literal_original = frame["Original Image"].str.lower().map(original_lookup)
    identity_original = frame["seed_id"].map(original_seed_lookup)
    frame["original_path"] = literal_original.fillna(identity_original)
    frame["transformed_path"] = frame["Transformed Image"].str.lower().map(transformed_lookup)
    frame["original_exists"] = frame["original_path"].notna()
    frame["transformed_exists"] = frame["transformed_path"].notna()
    frame["transformed_seed_id"] = frame["Transformed Image"].map(safe_seed)
    frame["usable"] = frame["original_exists"] & frame["transformed_exists"]
    frame.to_csv(output / f"pair_manifest_{name}.csv", index=False)

    positives = frame[frame["Pair-Type"].str.lower().eq("positive")]
    negatives = frame[frame["Pair-Type"].str.lower().eq("negative")]
    summary = {
        "dataset": name,
        "status": "audited",
        "pair_csv": str(table_path.resolve()),
        "rows": int(len(frame)),
        "positive_rows": int(len(positives)),
        "negative_rows": int(len(negatives)),
        "unique_seed_ids": int(frame["seed_id"].nunique(dropna=True)),
        "missing_original_references": int((~frame["original_exists"]).sum()),
        "missing_transformed_references": int((~frame["transformed_exists"]).sum()),
        "missing_positive_transforms": int((~positives["transformed_exists"]).sum()),
        "usable_positive_rows": int(positives["usable"].sum()),
        "usable_negative_rows": int(negatives["usable"].sum()),
    }
    return summary


def key_file_inventory(source_root: Path, output: Path) -> None:
    excluded = {"et cs208.xlsx", "~$h scientific reports comments.docx"}
    rows = []
    for path in source_root.rglob("*"):
        if not path.is_file() or PROJECT in path.parents:
            continue
        if path.name.lower() in excluded:
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(source_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).sort_values("relative_path").to_csv(
        output / "supplied_nonimage_inventory.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "manifests")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Dataset families to audit (default: all available project families).",
    )
    parser.add_argument(
        "--skip-nonimage-inventory",
        action="store_true",
        help="Skip hashing non-image source artefacts during a fast remote audit.",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries = [
        audit_dataset(name, DATASETS[name], source_root, output)
        for name in args.datasets
    ]
    if "main" in args.datasets:
        main_manifest_path = output / "pair_manifest_main.csv"
        if not main_manifest_path.is_file():
            parser.error("The main pair CSV is required when --datasets includes main")
        main_manifest = pd.read_csv(main_manifest_path)
        seed_ids = sorted(main_manifest["seed_id"].dropna().astype(int).unique())
        split = make_seed_split(seed_ids)
        split.to_csv(output / "seed_split_745_160_159.csv", index=False)
        split_map = dict(zip(split["seed_id"], split["split"]))
        main_manifest["original_split"] = main_manifest["seed_id"].map(split_map)
        main_manifest["transformed_split"] = main_manifest["transformed_seed_id"].map(split_map)
        main_manifest["strict_split_usable"] = (
            main_manifest["usable"]
            & main_manifest["original_split"].notna()
            & main_manifest["original_split"].eq(main_manifest["transformed_split"])
        )
        main_manifest.to_csv(main_manifest_path, index=False)
        split_counts = split["split"].value_counts().to_dict()
    else:
        split_counts = {}

    if not args.skip_nonimage_inventory:
        key_file_inventory(source_root, output)
    pd.DataFrame(summaries).to_csv(output / "dataset_audit_summary.csv", index=False)
    unavailable = [item["dataset"] for item in summaries if item["status"] != "audited"]
    unresolved = [
        item["dataset"]
        for item in summaries
        if item["status"] == "audited"
        and (
            item["missing_original_references"]
            or item["missing_transformed_references"]
        )
    ]
    if unavailable or unresolved:
        warning = (
            "Some requested dataset families are unavailable or contain unresolved "
            "file references; retain the documented limited-data scope."
        )
    else:
        warning = "All requested dataset file references resolved."
    payload = {
        "project_seed": 20260830,
        "split_counts": split_counts,
        "datasets": summaries,
        "unavailable_datasets": unavailable,
        "datasets_with_unresolved_references": unresolved,
        "warning": warning,
    }
    (output / "dataset_audit_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

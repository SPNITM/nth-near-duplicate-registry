"""Summarise local and HPC metadata manifests without reading research data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROLE_EXTENSIONS = {
    "images": {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    "notebooks": {".ipynb"},
    "source_code": {".py", ".sol", ".sh", ".ps1", ".m", ".r"},
    "documents": {".docx", ".pdf", ".tex", ".bib", ".md", ".txt"},
    "tabular_data": {".csv", ".tsv", ".xlsx", ".xls"},
    "model_weights": {".pth", ".pt", ".onnx", ".ckpt"},
    "numeric_arrays": {".npy", ".npz", ".pkl", ".pickle"},
    "archives": {".zip", ".gz", ".tgz", ".tar", ".7z"},
}


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def normalise(raw: str) -> str:
    value = raw.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def summarise(path: Path, label: str) -> dict:
    top = defaultdict(lambda: {"files": 0, "bytes": 0, "directories": 0})
    extensions: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    totals = Counter()
    with open_text(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            relative = normalise(row["relative_path"])
            kind = row["type"]
            size = int(float(row["size_bytes"]))
            first = PurePosixPath(relative).parts[0] if relative else "."
            if kind == "f":
                top[first]["files"] += 1
                top[first]["bytes"] += size
                totals["files"] += 1
                totals["bytes"] += size
                suffix = PurePosixPath(relative).suffix.lower() or "[no_ext]"
                extensions[suffix] += 1
                matched = False
                for role, suffixes in ROLE_EXTENSIONS.items():
                    if suffix in suffixes:
                        roles[role] += 1
                        matched = True
                if not matched:
                    roles["other"] += 1
            elif kind == "d":
                top[first]["directories"] += 1
                totals["directories"] += 1
            elif kind == "l":
                totals["symlinks"] += 1
            else:
                totals["other"] += 1
    return {
        "label": label,
        "manifest": str(path.resolve()),
        "totals": dict(totals),
        "top_level": [
            {"name": name, **values} for name, values in sorted(top.items())
        ],
        "file_roles": dict(roles.most_common()),
        "extensions_top_40": [
            {"extension": extension, "count": count}
            for extension, count in extensions.most_common(40)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--hpc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "inventory_version": "2026-08-31",
        "locations": [summarise(args.local, "local_D_NTH"), summarise(args.hpc, "hpc_NTH")],
        "interpretation": {
            "authoritative_main_media": "<HPC_SOURCE_ROOT>/NTH_IMAGES_EDITED",
            "authoritative_revision_workspace": "<LOCAL_SOURCE_ROOT>/Scientific_Reports_Major_Revision",
            "hpc_revision_workspace": "<HPC_PROJECT_ROOT>",
            "historical_roots_are_preserved": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "locations": 2}, indent=2))


if __name__ == "__main__":
    main()

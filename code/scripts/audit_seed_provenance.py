#!/usr/bin/env python
"""Recover seed rename evidence and build a rights/provenance audit register.

The historical notebooks preserve the pre-rename filenames in captured output.
Those filenames are evidence, but they are not treated as proof of a source URL
or licence.  Candidate YouTube locators are therefore explicitly marked
unverified and no image is classified as redistributable without author input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent

RENAME = re.compile(r"Renamed '([^']+)'[^\n]*'((?:Image)\d+\.[^']+)'", re.IGNORECASE)
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def notebook_renames(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: dict[str, str] = {}
    for cell in payload.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            for original, renamed in RENAME.findall(str(text)):
                pairs[renamed] = original
    return pairs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_family(name: str) -> tuple[str, str, str]:
    stem = Path(name).stem
    if YOUTUBE_ID.fullmatch(stem):
        return (
            "filename resembles a YouTube video identifier",
            f"https://www.youtube.com/watch?v={stem}",
            "candidate locator only; derivation and rights are unverified",
        )
    if re.fullmatch(r"\d{4}", stem) and Path(name).suffix.lower() == ".png":
        return "numbered PNG series", "", "source not recoverable from filename alone"
    if re.fullmatch(r"bike_\d+", stem, re.IGNORECASE):
        return "bike BMP series", "", "source not recoverable from filename alone"
    if re.fullmatch(r"cat\.\d+", stem, re.IGNORECASE):
        return "cat JPG series", "", "source not recoverable from filename alone"
    return "other historical filename", "", "source not recoverable from filename alone"


def image_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        exif = {ExifTags.TAGS.get(key, str(key)): value for key, value in image.getexif().items()}
        return {
            "format": image.format or path.suffix.lstrip(".").upper(),
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "exif_datetime_original": str(exif.get("DateTimeOriginal", "")),
            "exif_make": str(exif.get("Make", "")),
            "exif_model": str(exif.get("Model", "")),
            "exif_software": str(exif.get("Software", "")),
            "exif_artist": str(exif.get("Artist", "")),
            "exif_copyright": str(exif.get("Copyright", "")),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-notebook", type=Path, default=WORKSPACE / "ml part" / "data set generate" / "gen_image_pair_editedimage.ipynb")
    parser.add_argument("--coco-notebook", type=Path, default=WORKSPACE / "ml part" / "data set generate" / "COCO_gen_image_pair_editedimage.ipynb")
    parser.add_argument("--seed-dir", type=Path, default=WORKSPACE / "NTH_IMAGES")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data" / "provenance_20260901")
    args = parser.parse_args()

    main_mapping = notebook_renames(args.main_notebook)
    coco_mapping = notebook_renames(args.coco_notebook)
    seed_files = sorted(
        [path for path in args.seed_dir.iterdir() if path.is_file()],
        key=lambda path: int(re.search(r"\d+", path.stem).group()),
    )
    rows = []
    for path in seed_files:
        original = main_mapping.get(path.name, "")
        family, candidate, note = source_family(original) if original else ("unmapped", "", "no historical rename evidence")
        rows.append(
            {
                "seed_id": int(re.search(r"\d+", path.stem).group()),
                "current_filename": path.name,
                "historical_pre_rename_filename": original,
                "source_family_inference": family,
                "candidate_source_locator": candidate,
                "locator_verification_status": "UNVERIFIED" if candidate else "NOT IDENTIFIED",
                "author_confirmed_source_url": "",
                "author_confirmed_creator_or_owner": "",
                "author_confirmed_licence": "",
                "redistribution_status": "HOLD - RIGHTS EVIDENCE REQUIRED",
                "provenance_note": note,
                "sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                **image_metadata(path),
            }
        )
    main_frame = pd.DataFrame(rows)

    coco_rows = []
    for renamed, original in sorted(coco_mapping.items(), key=lambda item: int(re.search(r"\d+", item[0]).group())):
        coco_rows.append(
            {
                "coco_seed_filename": renamed,
                "coco_original_filename": original,
                "coco_image_id": int(re.search(r"(\d+)\.[^.]+$", original).group(1)),
                "official_dataset_home": "https://cocodataset.org/#home",
                "licence_note": "COCO annotations are CC BY 4.0; each underlying Flickr image retains its own licence. Verify image-level rights before redistribution.",
                "redistribution_status": "VERIFY IMAGE-LEVEL FLICKR LICENCE",
            }
        )
    coco_frame = pd.DataFrame(coco_rows)

    copydays_frame = pd.DataFrame(
        [
            {
                "component": "Copydays original references",
                "file_count": 157,
                "source_home": "https://lear.inrialpes.fr/~jegou/data.php.html#copydays",
                "archive_mirror": "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz",
                "archive_sha256": "8B85B7914FD0145BC258BDE3440F6A622243CA91EE085FE2025CE7BDEB0BFB8E",
                "rights_note": "Copyright INRIA; used locally for evaluation and not redistributed.",
            },
            {
                "component": "Copydays strong queries",
                "file_count": 229,
                "source_home": "https://lear.inrialpes.fr/~jegou/data.php.html#copydays",
                "archive_mirror": "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz",
                "archive_sha256": "7852C64AA4A37E8329CE08159FC947C16F09C66737DB0A2479A90FC9AE40760C",
                "rights_note": "Copyright INRIA; used locally for evaluation and not redistributed.",
            },
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_frame.to_csv(args.output_dir / "main_seed_provenance.csv", index=False)
    coco_frame.to_csv(args.output_dir / "coco_seed_provenance.csv", index=False)
    copydays_frame.to_csv(args.output_dir / "copydays_benchmark_provenance.csv", index=False)
    summary = {
        "main_seed_files": len(seed_files),
        "main_notebook_mappings": len(main_mapping),
        "main_mapping_missing": int((main_frame["historical_pre_rename_filename"] == "").sum()),
        "source_family_counts": dict(Counter(main_frame["source_family_inference"])),
        "confirmed_source_urls": int((main_frame["author_confirmed_source_url"] != "").sum()),
        "confirmed_licences": int((main_frame["author_confirmed_licence"] != "").sum()),
        "redistributable_seed_files": int(main_frame["redistribution_status"].str.startswith("OK").sum()),
        "coco_notebook_mappings": len(coco_mapping),
        "copydays_official_originals": 157,
        "copydays_official_strong_queries": 229,
        "conclusion": "All main rename mappings were recovered, but source URLs and licences remain author-controlled evidence fields; main seed images must not be redistributed yet.",
    }
    (args.output_dir / "provenance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

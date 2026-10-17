"""Write a deterministic metadata inventory for a project tree.

The output is a tab-separated manifest with one row per file, directory, or
symbolic link.  File contents are not read, so this remains practical for the
large image corpus stored on the HPC.  Use a ``.gz`` suffix for compressed
output.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Iterator, TextIO


HEADER = "type\tsize_bytes\tmtime_epoch\trelative_path\n"


def _walk(root: Path) -> Iterator[tuple[str, int, float, str]]:
    """Yield sorted, non-following metadata rows below *root*."""

    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            children = sorted(entries, key=lambda entry: entry.name.casefold())
        child_dirs: list[Path] = []
        for entry in children:
            stat = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                kind = "l"
            elif entry.is_dir(follow_symlinks=False):
                kind = "d"
                child_dirs.append(path)
            elif entry.is_file(follow_symlinks=False):
                kind = "f"
            else:
                kind = "o"
            yield kind, stat.st_size, stat.st_mtime, relative
        pending.extend(reversed(child_dirs))


def _open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def build_inventory(root: Path, output: Path) -> dict[str, int]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    counts = {"f": 0, "d": 0, "l": 0, "o": 0}
    with _open_output(output) as handle:
        handle.write(HEADER)
        for kind, size, mtime, relative in _walk(root):
            if "\t" in relative or "\n" in relative or "\r" in relative:
                raise ValueError(f"Path cannot be represented safely in TSV: {relative!r}")
            counts[kind] += 1
            handle.write(f"{kind}\t{size}\t{mtime:.9f}\t{relative}\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Project root to inventory")
    parser.add_argument("output", type=Path, help="Destination .tsv or .tsv.gz")
    args = parser.parse_args()
    counts = build_inventory(args.root, args.output)
    print(
        f"wrote {args.output}: files={counts['f']} dirs={counts['d']} "
        f"symlinks={counts['l']} other={counts['o']}"
    )


if __name__ == "__main__":
    main()

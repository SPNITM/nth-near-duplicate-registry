"""Create syntax-clean copies of the three malformed legacy HPC notebooks.

The source notebooks are historical evidence and are never modified.  The
known, narrowly scoped fixes are applied to copies and recorded in an audit
JSON file so the correction is reviewable.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def fix_after_train(notebook: dict) -> str:
    cell = code_cells(notebook)[17]
    source = "".join(cell.get("source", []))
    if source.count('"""') % 2:
        set_source(cell, source.rstrip() + '\n"""\n')
        return "Closed the intentionally disabled optional EVM helper string."
    return "No change required."


def fix_untitled1(notebook: dict) -> str:
    cell = code_cells(notebook)[0]
    source = "".join(cell.get("source", []))
    if "import os" not in source:
        source = "import os\n\n" + source
    for variable in ("PRUNING_JSON", "PERLABEL_JSON", "ABLATION_JSON"):
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith(variable) and not line.rstrip().endswith(")"):
                lines[index] = line + ")"
        source = "\n".join(lines) + "\n"
    set_source(cell, source)
    return "Added the missing os import and closed three os.path.join calls."


def fix_newfile(notebook: dict) -> str:
    cells = code_cells(notebook)
    target = cells[8]
    target["cell_type"] = "markdown"
    target.pop("execution_count", None)
    target.pop("outputs", None)
    target["source"] = [
        "**Historical repair note.** This cell was an incomplete, indented duplicate "
        "of the training/testing implementation. It has been disabled; the complete "
        "implementation is retained in the following code cell.\n"
    ]
    return "Disabled an incomplete duplicate fragment; the complete next cell is unchanged."


FIXES = {
    "After train.ipynb": fix_after_train,
    "Untitled1.ipynb": fix_untitled1,
    "newfile.ipynb": fix_newfile,
}


def syntax_errors(notebook: dict) -> list[dict]:
    errors: list[dict] = []
    for index, cell in enumerate(code_cells(notebook)):
        source = "".join(cell.get("source", []))
        cleaned = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        if not cleaned.strip():
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as error:
            errors.append(
                {"code_cell": index, "line": error.lineno, "message": error.msg}
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = []
    for source in sorted(args.source.glob("*.ipynb")):
        notebook = json.loads(source.read_text(encoding="utf-8"))
        before = syntax_errors(notebook)
        action = FIXES[source.name](notebook) if source.name in FIXES else "Copied unchanged."
        after = syntax_errors(notebook)
        destination = args.output / source.name
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        audit.append(
            {
                "notebook": source.name,
                "action": action,
                "syntax_errors_before": before,
                "syntax_errors_after": after,
            }
        )
    audit_path = args.output / "notebook_syntax_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    remaining = sum(len(item["syntax_errors_after"]) for item in audit)
    print(f"wrote {len(audit)} notebooks; remaining syntax errors={remaining}")
    if remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

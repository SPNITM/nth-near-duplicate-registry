# Calibrated two-stage near-duplicate image registry

Reproducibility materials for the revised Scientific Reports manuscript
**“Calibrated Two-Stage Near-Duplicate Image Registration with Compact Hashing and Tamper-Evident Anchoring.”**

The project evaluates a two-stage registry workflow: a compact 64-bit Neighborhood Totality Hash (NTH)
generates a shortlist, and a frozen visual encoder verifies candidates. The revision separates pair-level
classification from full-registry query risk, uses calibration-only threshold selection, evaluates modern
encoders and learned controls, and narrows the blockchain claim to tamper-evident anchoring of disclosed
off-chain records.

## Main verified evidence

- Audited private corpus: 91,504 pairs, 1,064 identities and zero unresolved file references.
- Six frozen visual encoders, five projection runs, five learned 64-bit controls and four classical hashes.
- Fixed-gate ROC construction and direct any-false-match-per-query measurement.
- Official Copydays strong evaluation, adaptive NTH attack, crop-mitigation study and exact-scan scaling.
- CPU timing on Intel Core i7-14700 and Intel Xeon Gold 5320 platforms.
- Versioned Solidity commitment contract and tamper-evident off-chain audit-chain implementation.

The strongest frozen encoder outperformed the trained projection; NTH was not uniformly better than classical
hashes; and official-benchmark transfer was substantially weaker than the private-corpus result. These negative
findings are retained in the manuscript.

## Repository layout

| Path | Contents |
|---|---|
| `code/src/nth_registry/` | Versioned NTH, indexing, metrics, split and audit-log implementation |
| `code/scripts/` | Experiment, evaluation, benchmark and figure-generation scripts |
| `code/tests/` | Unit tests |
| `code/hpc/` | Versioned PBS scripts used for the HPC verification runs |
| `configs/` | Deterministic revision configuration |
| `contracts/` | Solidity contract and environment-variable template |
| `data/manifests/` | Sanitised pair manifests, audit summaries and deterministic split |
| `data/provenance_20260901/` | Prepared seed-level provenance evidence registers |
| `results/` | Machine-readable aggregate results and benchmark metadata |
| `figures/` | Publication and diagnostic figures generated from the result tables |
| `manuscript/main.tex` | Clean final revised manuscript |
| `manuscript/main_highlighted_red.tex` | Reviewer-driven additions/rewrites highlighted in red |
| `manuscript/ref_revised.bib` | Curated bibliography |
| `supplementary/` | Supplementary methods source |
| `validation/` | Release and manuscript validation summaries |

## Quick start

Python 3.11 or newer is required.

```bash
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m unittest discover -s code/tests -v
```

The complete experiment order and reporting constraints are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Exact CPU and HPC package versions are recorded in the
two lock files at the repository root.

## Manuscript build

From the `manuscript/` directory, compile either entry file with BibTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The red-highlighted source uses the same bibliography, figures, equations, labels and scientific results as the
clean source.

## Public-data boundary

This repository intentionally excludes the 1,064-image private seed corpus because author-confirmed source URLs,
licences and redistribution permissions are incomplete. It also excludes copyrighted Copydays image files,
credentials, environment files, model caches and raw embedding arrays. Sanitised manifests, checksums, aggregate
results and provenance-evidence registers are included. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)
and [`PRE_UPLOAD_CHECKLIST.md`](PRE_UPLOAD_CHECKLIST.md).

## Citation, DOI and licence

`CITATION.cff` records the public repository URL, and `.zenodo.json.template` is prepared for deposition. The authors
must still confirm the citation metadata, select software and derived-data licences, create the version DOI, and
replace the manuscript DOI placeholder before journal resubmission. Until a licence is selected, no reuse licence is granted. See
[`LICENSE_SELECTION_REQUIRED.md`](LICENSE_SELECTION_REQUIRED.md).

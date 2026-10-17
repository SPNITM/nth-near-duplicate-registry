# Reproducibility protocol

1. Record source-file checksums and run `build_manifests.py`.
2. Do not mix identities across train, calibration, and test sets.
3. Select NTH radius and embedding threshold only on calibration identities.
4. Freeze the selected policy before evaluating the held-out test identities.
5. Report pair-level FPR separately from per-query probability of any false
   acceptance against an index.
6. For a fixed-radius gate, sweep only the embedding threshold; rejected pairs
   remain rejected and maximum recall cannot exceed the NTH pass rate.
7. Report all radii, thresholds, shortlist distributions, latency definitions,
   software versions, hardware, random seeds, and raw prediction files.
8. Train each learned method with five independent seeds and report mean,
   standard deviation, and confidence intervals. Bootstrap intervals over one
   checkpoint are not a substitute for training-run variability.
9. Anchor the model-file hash, policy-file hash, and audit-log head when making
   blockchain auditability claims.
10. Use `requirements-local-cpu-lock.txt` or
    `requirements-hpc-a100-lock.txt` to identify the exact verified package
    versions; the broader requirements files remain compatibility ranges.
11. Do not publish accelerator timing while a shared device is materially
    occupied. Record utilisation and rerun `benchmark_embedding_throughput.py`
    only during an uncontended allocation.

## Verified commands (31 August--1 September 2026)

Local selected-model timing:

```text
python code/scripts/benchmark_embedding_throughput.py --device cpu \
  --platform-label local-workstation-cpu --repetitions 10 \
  --output results/embedding_throughput_local_cpu.csv
```

Additional reviewer-requested evaluations:

```text
python code/scripts/run_official_copydays_benchmark.py
python code/scripts/run_adaptive_hash_evasion.py
python code/scripts/run_multiview_crop_mitigation.py
python code/scripts/audit_seed_provenance.py
```

HPC source and tests are synchronized to the versioned, non-destructive working
directory
`<HPC_PROJECT_ROOT>`.
The authoritative private main corpus remains at
`<HPC_SOURCE_ROOT>/NTH_IMAGES_EDITED`; historical notebooks, weights and data
were not overwritten. The complete-corpus manifest audit resolves all 91,504
pair rows. PBS job `4081.nitm-hpcs-master` ran the selected ShuffleNet model and
fixed-gate registry evaluation on `node001` CPU and finished with exit status 0
in 00:06:02. Its raw embeddings, predictions, log and checksums are synchronized
under `results/hpc_full_main_20260831`; the transfer archive SHA-256 is
`5676781f33f03d68fcee75aed04eda1ab2d27abde4ba86b5a3c9889359d451b5`.

PBS job `4091.nitm-hpcs-master` benchmarked the selected ShuffleNet at batches
1, 8, 32 and 64 on a 16-core Intel Xeon Gold 5320 allocation on `node001`; it
finished with exit status 0. The checked output is stored as
`results/embedding_throughput_hpc_cpu.csv` with metadata, hardware record, PBS
log and SHA-256 manifest. Median latency was 9.78, 1.98, 1.24 and 1.25 ms/image;
throughput peaked at 805.6 images/s for batch 32.

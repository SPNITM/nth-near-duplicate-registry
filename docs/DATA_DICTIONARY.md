# Data dictionary and provenance status

## Pair table

| Column | Meaning |
|---|---|
| Original Image | Seed/original filename |
| Transformed Image | Edited image filename |
| Label | Transformation name; `none` for unrelated pairs |
| Pair-Type | `positive` for same-seed edits, `negative` otherwise |
| File Type | Nominal transformed-image extension |

The historical table contains 91,504 rows: 29,792 positives and 61,712
negatives over 1,064 seed identities. File existence is not assumed. The
manifest builder emits a row-level status and never silently drops missing
references.

## Required provenance fields before public release

- Source URL or acquisition method for every original image
- Copyright holder and redistribution permission
- Dataset/version identifier for COCO and Copydays
- Licence for images, code, and derived manifests
- SHA-256 checksum for every distributed file


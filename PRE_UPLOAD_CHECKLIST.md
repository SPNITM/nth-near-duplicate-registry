# Public repository pre-upload checklist

Complete these author-controlled steps before making the repository public:

- [ ] Select and add an approved software licence as `LICENSE`.
- [ ] Select and document the licence for derived metadata/result tables.
- [ ] Confirm all four authors, author order, affiliations, corresponding-author email and any ORCIDs.
- [ ] Create the GitHub repository and replace `AUTHOR_TO_INSERT_PUBLIC_REPOSITORY_URL` in `CITATION.cff`.
- [ ] Replace `AUTHOR_TO_SELECT` and the repository placeholder in `.zenodo.json.template`.
- [ ] Keep the private seed images and copyrighted Copydays media out of the repository unless redistribution
      rights are documented for each item.
- [ ] Review `data/provenance_20260901/main_seed_provenance.csv`; empty source/licence fields must not be presented
      as verified provenance.
- [ ] Run `python -m unittest discover -s code/tests -v` after cloning the uploaded repository.
- [ ] Create a versioned GitHub release and archive it with Zenodo or another recognised DOI provider.
- [ ] Replace `DOI-TO-BE-INSERTED-BEFORE-RESUBMISSION` in both manuscript files with the version DOI.
- [ ] Compile and visually check the clean and highlighted manuscripts.
- [ ] Obtain all-author approval before the repository URL/DOI is sent to the journal.

Suggested upload sequence:

1. Create an empty GitHub repository without an automatically generated README or licence.
2. Upload the contents of this directory, preserving the folder structure.
3. Add the author-approved licence(s), update the citation/Zenodo metadata, and commit.
4. Verify the GitHub Actions test workflow.
5. Create the release, obtain the DOI, update the manuscripts, and upload the journal files separately.

Journal decision letters, reviewer-response documents and cover letters are intentionally not included here.
They belong in the journal portal package, not the public scientific repository.

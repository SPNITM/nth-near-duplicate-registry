# Public repository release checklist

The repository is public. Complete the remaining author-controlled steps before creating the DOI-bearing release:

- [ ] Select and add an approved software licence as `LICENSE`.
- [ ] Select and document the licence for derived metadata/result tables.
- [ ] Confirm all four authors, author order, affiliations, corresponding-author email and any ORCIDs.
- [x] Create the public GitHub repository and record its URL in `CITATION.cff`.
- [ ] Replace `AUTHOR_TO_SELECT` in `.zenodo.json.template` after the licence decision.
- [ ] Keep the private seed images and copyrighted Copydays media out of the repository unless redistribution
      rights are documented for each item.
- [ ] Review `data/provenance_20260901/main_seed_provenance.csv`; empty source/licence fields must not be presented
      as verified provenance.
- [ ] Push the corrected source-transfer results and run `python -m unittest discover -s code/tests -v` through GitHub Actions.
- [ ] Create a versioned GitHub release and archive it with Zenodo or another recognised DOI provider.
- [ ] Replace `DOI-TO-BE-INSERTED-BEFORE-RESUBMISSION` in both manuscript files with the version DOI.
- [ ] Compile and visually check the clean and highlighted manuscripts.
- [ ] Obtain all-author approval before the repository URL/DOI is sent to the journal.

Remaining release sequence:

1. Push the corrected result tables, manuscript values and metadata updates.
2. Add the author-approved licence(s), confirm citation/Zenodo metadata, and commit.
3. Verify the GitHub Actions test workflow.
4. Create the release, obtain the DOI, update the manuscripts, and upload the journal files separately.

Journal decision letters, reviewer-response documents and cover letters are intentionally not included here.
They belong in the journal portal package, not the public scientific repository.

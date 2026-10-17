import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const project = "<LOCAL_SOURCE_ROOT>/Scientific_Reports_Major_Revision";
const dataDir = `${project}/data/provenance_20260901`;
const outputDir = `${project}/outputs/20260901_completion`;
const previewDir = `${project}/artifacts/final_provenance_workbook_render_20260901`;
const outputPath = `${outputDir}/NTH_Seed_Provenance_and_Final_Submission_Register_20260901.xlsx`;
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

async function csvValues(fileName, sheetName) {
  const csv = (await fs.readFile(`${dataDir}/${fileName}`, "utf8")).replace(/^\uFEFF/, "");
  const imported = await Workbook.fromCSV(csv, { sheetName });
  return imported.worksheets.getItem(sheetName).getUsedRange().values;
}

const mainValues = await csvValues("main_seed_provenance.csv", "Main seed provenance");
const cocoValues = await csvValues("coco_seed_provenance.csv", "COCO subset");
const copydaysValues = await csvValues("copydays_benchmark_provenance.csv", "Official Copydays");
const provenanceSummary = JSON.parse(await fs.readFile(`${dataDir}/provenance_summary.json`, "utf8"));

const workbook = Workbook.create();
const readiness = workbook.worksheets.add("Final readiness");
const main = workbook.worksheets.add("Main seed provenance");
const families = workbook.worksheets.add("Source-family summary");
const coco = workbook.worksheets.add("COCO subset");
const copydays = workbook.worksheets.add("Official Copydays");
const authors = workbook.worksheets.add("Author metadata");
const actions = workbook.worksheets.add("Remaining author actions");

const navy = "#17365D";
const blue = "#2E74B5";
const lightBlue = "#D9EAF7";
const pale = "#F3F6F8";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const gray = "#5B6573";

function titleBlock(sheet, title, subtitle, lastColumn) {
  sheet.showGridLines = false;
  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  sheet.mergeCells(`A2:${lastColumn}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: lightBlue,
    font: { italic: true, color: gray, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 28;
}

function styleTable(sheet, headerRange, bodyRange) {
  sheet.getRange(headerRange).format = {
    fill: blue,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(bodyRange).format = { wrapText: true, verticalAlignment: "top" };
}

titleBlock(
  readiness,
  "NTH provenance and final-submission readiness",
  "Evidence recovered on 1 September 2026; blank author-confirmed fields are intentional and must not be inferred.",
  "H",
);
readiness.getRange("A4:D16").values = [
  ["Control", "Verified value", "Status", "Owner / next action"],
  ["Main seed files", provenanceSummary.main_seed_files, "PASS", "No action"],
  ["Historical rename mappings", provenanceSummary.main_notebook_mappings, "PASS", "No action"],
  ["Missing rename mappings", provenanceSummary.main_mapping_missing, "PASS", "No action"],
  ["Confirmed source URLs", provenanceSummary.confirmed_source_urls, "BLOCKER", "Authors complete columns G-I in Main seed provenance"],
  ["Confirmed image licences", provenanceSummary.confirmed_licences, "BLOCKER", "Authors supply evidence; do not infer from filenames"],
  ["Redistributable main seeds", provenanceSummary.redistributable_seed_files, "BLOCKER", "Keep raw main seed images out of public release"],
  ["COCO mappings recovered", provenanceSummary.coco_notebook_mappings, "PASS WITH RIGHTS CHECK", "Verify per-image Flickr licence before redistribution"],
  ["Official Copydays originals", provenanceSummary.copydays_official_originals, "PASS", "Evaluation only; do not redistribute"],
  ["Official Copydays strong queries", provenanceSummary.copydays_official_strong_queries, "PASS", "Evaluation only; do not redistribute"],
  ["Author declarations", "Recovered from submitted Scientific Reports source", "VERIFY", "All authors confirm text in Author metadata"],
  ["ORCID identifiers", "Not found", "MISSING OPTIONAL METADATA", "Each author supplies ORCID or confirms none"],
  ["Repository DOI/version", "Not created", "BLOCKER FOR PUBLIC RELEASE", "Corresponding author publishes release and inserts DOI/version"],
];
styleTable(readiness, "A4:D4", "A5:D16");
readiness.getRange("A5:A16").format.fill = pale;
readiness.getRange("C5:C16").format = { font: { bold: true }, horizontalAlignment: "center", wrapText: true };
readiness.getRange("C5:C7").format.fill = green;
readiness.getRange("C8:C10").format.fill = red;
readiness.getRange("C11:C14").format.fill = amber;
readiness.getRange("C15:C16").format.fill = red;
readiness.getRange("A4:A16").format.columnWidth = 34;
readiness.getRange("B4:B16").format.columnWidth = 48;
readiness.getRange("C4:C16").format.columnWidth = 28;
readiness.getRange("D4:D16").format.columnWidth = 64;
readiness.freezePanes.freezeRows(4);

titleBlock(
  main,
  "Main seed provenance register (1,064 files)",
  "Recovered pre-rename filename, cryptographic checksum and image metadata. Candidate locators are unverified; author-confirmed URL/owner/licence fields remain blank.",
  "W",
);
main.getRange(`A4:W${3 + mainValues.length}`).values = mainValues;
styleTable(main, "A4:W4", `A5:W${3 + mainValues.length}`);
main.getRange(`G5:I${3 + mainValues.length}`).format.fill = amber;
main.getRange(`J5:J${3 + mainValues.length}`).format.fill = red;
main.getRange(`A4:W${3 + mainValues.length}`).format.font = { size: 9 };
const widths = [9, 18, 27, 31, 43, 22, 29, 29, 25, 31, 45, 66, 16, 12, 12, 12, 10, 22, 20, 20, 22, 20, 20];
for (let index = 0; index < widths.length; index += 1) {
  const column = String.fromCharCode(65 + index);
  main.getRange(`${column}4:${column}${3 + mainValues.length}`).format.columnWidth = widths[index];
}
main.getRange(`A4:W${3 + mainValues.length}`).format.rowHeight = 28;
main.getRange("A4:W4").format.rowHeight = 52;
main.tables.add(`A4:W${3 + mainValues.length}`, true, "MainSeedProvenanceTable");
main.freezePanes.freezeRows(4);
main.freezePanes.freezeColumns(3);

titleBlock(families, "Recovered source-family summary", "Filename-family inference is audit triage, not legal provenance.", "E");
const familyRows = Object.entries(provenanceSummary.source_family_counts);
families.getRange(`A4:E${4 + familyRows.length}`).values = [
  ["Recovered filename family", "Count", "Source URL status", "Licence status", "Required action"],
  ...familyRows.map(([name, count]) => [name, count, "Not author-confirmed", "Not author-confirmed", "Trace and document evidence before any redistribution"]),
];
styleTable(families, "A4:E4", `A5:E${4 + familyRows.length}`);
families.getRange(`C5:D${4 + familyRows.length}`).format.fill = red;
families.getRange(`A4:A${4 + familyRows.length}`).format.columnWidth = 46;
families.getRange(`B4:B${4 + familyRows.length}`).format.columnWidth = 12;
families.getRange(`C4:D${4 + familyRows.length}`).format.columnWidth = 25;
families.getRange(`E4:E${4 + familyRows.length}`).format.columnWidth = 58;
families.freezePanes.freezeRows(4);

titleBlock(coco, "COCO subset rename register", "160 rename mappings recovered; COCO annotation licence does not replace per-image Flickr rights.", "F");
coco.getRange(`A4:F${3 + cocoValues.length}`).values = cocoValues;
styleTable(coco, "A4:F4", `A5:F${3 + cocoValues.length}`);
coco.getRange(`A4:F${3 + cocoValues.length}`).format = { wrapText: true, verticalAlignment: "top", font: { size: 9 } };
coco.getRange(`F5:F${3 + cocoValues.length}`).format.fill = amber;
[18, 42, 18, 36, 78, 36].forEach((width, index) => {
  const column = String.fromCharCode(65 + index);
  coco.getRange(`${column}4:${column}${3 + cocoValues.length}`).format.columnWidth = width;
});
coco.tables.add(`A4:F${3 + cocoValues.length}`, true, "CocoProvenanceTable");
coco.freezePanes.freezeRows(4);

titleBlock(copydays, "Official Copydays benchmark provenance", "Official benchmark archives were used locally only; raw INRIA images are excluded from the public release.", "F");
copydays.getRange(`A4:F${3 + copydaysValues.length}`).values = copydaysValues;
styleTable(copydays, "A4:F4", `A5:F${3 + copydaysValues.length}`);
[34, 14, 58, 67, 68, 58].forEach((width, index) => {
  const column = String.fromCharCode(65 + index);
  copydays.getRange(`${column}4:${column}${3 + copydaysValues.length}`).format.columnWidth = width;
});
copydays.getRange(`A4:F${3 + copydaysValues.length}`).format.rowHeight = 48;
copydays.freezePanes.freezeRows(4);

titleBlock(authors, "Recovered author metadata and declarations", "Recovered verbatim in substance from the previously submitted Scientific Reports LaTeX; every author must verify before resubmission.", "E");
const authorRows = [
  ["Author", "Email", "Affiliation", "Recovered contribution", "ORCID / confirmation"],
  ["Stuti Pandey", "stuti.pandey@jaipur.manipal.edu", "Department of Artificial Intelligence and Machine Learning, Manipal University Jaipur", "Conceptualization; methodology; model development; dataset preparation; writing—original draft", ""],
  ["Akhilendra Pratap Singh", "akhilendra.singh@nitm.ac.in", "Department of Computer Science and Engineering, National Institute of Technology Meghalaya", "Investigation; validation; blockchain integration; writing—review and editing", ""],
  ["Dharmender Singh Kushwaha", "dsk@mnnit.ac.in", "Department of Computer Science and Engineering, Motilal Nehru National Institute of Technology Allahabad", "Supervision; validation; technical guidance", ""],
  ["Ashish Pandey", "ashish.pandey@jaipur.manipal.edu", "Department of Data Science and Engineering, Manipal University Jaipur", "Conceptualization; supervision; system integration", ""],
];
authors.getRange("A4:E8").values = authorRows;
styleTable(authors, "A4:E4", "A5:E8");
authors.getRange("E5:E8").format.fill = amber;
authors.getRange("A4:A8").format.columnWidth = 30;
authors.getRange("B4:B8").format.columnWidth = 40;
authors.getRange("C4:C8").format.columnWidth = 62;
authors.getRange("D4:D8").format.columnWidth = 65;
authors.getRange("E4:E8").format.columnWidth = 30;
authors.getRange("A10:E16").values = [
  ["Declaration", "Recovered text / status", "Author confirmation", "Evidence source", "Notes"],
  ["Funding", "No funds, grants, or other support was received.", "", "Prior submitted main.tex", "Confirm unchanged"],
  ["Competing interests", "The authors declare no competing interests.", "", "Prior submitted main.tex", "Confirm unchanged"],
  ["Ethics and consent to participate", "Not applicable.", "", "Prior submitted main.tex", "Confirm appropriate for the data"],
  ["Consent to publish", "Not applicable.", "", "Prior submitted main.tex", "Confirm appropriate"],
  ["All-author approval", "All authors reviewed and approved the final manuscript.", "", "Prior submitted main.tex", "Obtain explicit approval"],
  ["Corresponding author", "Ashish Pandey", "", "Prior submitted main.tex", "Confirm portal contact"],
];
styleTable(authors, "A10:E10", "A11:E16");
authors.getRange("C11:C16").format.fill = amber;
authors.freezePanes.freezeRows(4);

titleBlock(actions, "Remaining author-controlled actions", "These items cannot be truthfully completed from code, data or historical submission files alone.", "F");
const actionRows = [
  ["A1", "Main seed source URLs", "Complete author-confirmed URL for every seed or document lawful first-party origin", "All authors", "SUBMISSION BLOCKER", "Open"],
  ["A2", "Image rights/licences", "Provide owner, licence and redistribution evidence for every main seed", "All authors", "PUBLIC-RELEASE BLOCKER", "Open"],
  ["A3", "Repository publication", "Publish the prepared versioned archive without restricted raw images", "Corresponding author", "REVIEWER BLOCKER", "Open"],
  ["A4", "DOI and release version", "Insert repository URL, version DOI and concept DOI into manuscript/CITATION.cff", "Corresponding author", "REVIEWER BLOCKER", "Open"],
  ["A5", "Software/metadata licence", "Choose and approve licences; do not infer author legal intent", "All authors", "PUBLIC-RELEASE BLOCKER", "Open"],
  ["A6", "ORCID values", "Supply one per author or confirm omission", "Each author", "METADATA", "Open"],
  ["A7", "Exact portal deadline", "Copy the date displayed in the Scientific Reports submission system", "Corresponding author", "SUBMISSION MANAGEMENT", "Open"],
  ["A8", "Final declaration approval", "Verify funding, interests, ethics, consent and contribution text", "All authors", "SUBMISSION BLOCKER", "Open"],
];
actions.getRange(`A4:F${4 + actionRows.length}`).values = [["ID", "Item", "Exact action", "Owner", "Priority", "Status"], ...actionRows];
styleTable(actions, "A4:F4", `A5:F${4 + actionRows.length}`);
actions.getRange(`A5:F${4 + actionRows.length}`).format.fill = "#FFFFFF";
actions.getRange(`E5:E${4 + actionRows.length}`).format.fill = red;
actions.getRange(`F5:F${4 + actionRows.length}`).format.fill = amber;
[10, 30, 72, 25, 30, 16].forEach((width, index) => {
  const column = String.fromCharCode(65 + index);
  actions.getRange(`${column}4:${column}${4 + actionRows.length}`).format.columnWidth = width;
});
actions.tables.add(`A4:F${4 + actionRows.length}`, true, "RemainingAuthorActionsTable");
actions.freezePanes.freezeRows(4);

const previews = [
  ["Final readiness", 1],
  ["Main seed provenance", 0.35],
  ["Source-family summary", 1],
  ["COCO subset", 0.45],
  ["Official Copydays", 0.8],
  ["Author metadata", 0.8],
  ["Remaining author actions", 0.8],
];
for (const [sheetName, scale] of previews) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale, format: "png" });
  const fileName = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  await fs.writeFile(path.join(previewDir, `${fileName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 5,
  tableMaxCols: 8,
  tableMaxCellChars: 120,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir, sheets: previews.map(([name]) => name), inspection, formulaErrors }, null, 2));

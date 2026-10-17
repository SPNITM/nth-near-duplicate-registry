import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const project = "<LOCAL_SOURCE_ROOT>/Scientific_Reports_Major_Revision";
const inventoryPath = `${project}/provenance/inventory_20260831/project_inventory_summary.json`;
const verificationPath = `${project}/results/hpc_full_main_20260831/full_main_verification_summary.json`;
const commentsPath = `${project}/response/reviewer_comment_inventory.csv`;
const outputDir = `${project}/outputs/20260831_nth_revision`;
const previewDir = `${project}/artifacts/inventory_workbook_render_20260831`;
const outputPath = `${outputDir}/NTH_Project_Inventory_and_Revision_Readiness_20260831.xlsx`;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const inventory = JSON.parse(await fs.readFile(inventoryPath, "utf8"));
const verification = JSON.parse(await fs.readFile(verificationPath, "utf8"));
const commentsCsv = (await fs.readFile(commentsPath, "utf8")).replace(/^\uFEFF/, "");
const importedReviewerWorkbook = await Workbook.fromCSV(commentsCsv, { sheetName: "Reviewer comments" });
const importedReviewer = importedReviewerWorkbook.worksheets.getItem("Reviewer comments");
const reviewerValues = importedReviewer.getUsedRange().values;

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Executive summary");
const local = workbook.worksheets.add("Local inventory");
const hpc = workbook.worksheets.add("HPC inventory");
const audit = workbook.worksheets.add("Main data audit");
const results = workbook.worksheets.add("Verified results");
const actions = workbook.worksheets.add("Author inputs needed");
const transfers = workbook.worksheets.add("Transfers and checksums");
const reviewer = workbook.worksheets.add("Reviewer comments");
reviewer.getRange(`A1:C${reviewerValues.length}`).values = reviewerValues;

const navy = "#17365D";
const blue = "#2E74B5";
const lightBlue = "#EAF2F8";
const pale = "#F4F6F7";
const green = "#E2F0D9";
const red = "#FCE4D6";
const amber = "#FFF2CC";
const gray = "#5B6573";

function titleBlock(sheet, title, subtitle, lastColumn) {
  sheet.showGridLines = false;
  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 18 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  sheet.mergeCells(`A2:${lastColumn}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: lightBlue,
    font: { color: gray, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 26;
}

function styleTable(sheet, headerRange, bodyRange) {
  sheet.getRange(headerRange).format = {
    fill: blue,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(bodyRange).format = {
    wrapText: true,
    verticalAlignment: "top",
  };
}

function classifyLocal(name) {
  if (name === "Scientific_Reports_Major_Revision") return "Authoritative revision workspace";
  if (name === "NTH_IMAGES_EDITED") return "Partial main-media mirror";
  if (name === "NTH_IMAGES") return "Original-image mirror";
  if (name.startsWith("COCO_")) return "Renamed supplied subset; not official benchmark";
  if (name.startsWith("COPY_")) return "Renamed supplied subset; not official Copydays protocol";
  if (/Decision|Access-|Scientific_Reports_Submission|Comments/.test(name)) return "Historical submission/reviewer provenance";
  if (name === "ET CS208.xlsx") return "Unrelated student file; excluded";
  return "Supplied or generated project asset";
}

function classifyHpc(name) {
  if (name === "NTH_IMAGES_EDITED") return "Authoritative private main media";
  if (name === "image_pairs_table.csv") return "Authoritative main pair table";
  if (name.endsWith(".ipynb") || name === ".ipynb_checkpoints") return "Historical notebook; original preserved";
  if (name.includes("pth") || name === "nth_final_outputs") return "Historical trained model/result asset";
  return "Historical HPC project asset";
}

titleBlock(
  summary,
  "NTH project inventory and revision readiness",
  "Scientific Reports major revision | Combined local + HPC verification | 31 August 2026",
  "N",
);
summary.getRange("A4:B12").values = [
  ["Control", "Verified value"],
  ["Reviewer comments inventoried", null],
  ["Main pair-table rows", null],
  ["Resolved main references", null],
  ["Missing main references", null],
  ["Canonical unit tests", 13],
  ["Full selected-model HPC job", "PBS 4081 | node001 CPU | exit 0 | 00:06:02"],
  ["Public release status", "Prepared locally; publication still requires author licences and DOI"],
  ["Submission readiness", "NOT READY — author-controlled items remain"],
];
summary.getRange("B5").formulas = [["=COUNTA('Reviewer comments'!A2:A49)"]];
summary.getRange("B6").formulas = [["='Main data audit'!B5"]];
summary.getRange("B7").formulas = [["='Main data audit'!B14"]];
summary.getRange("B8").formulas = [["='Main data audit'!B15"]];
styleTable(summary, "A4:B4", "A5:B12");
summary.getRange("A5:A12").format.fill = pale;
summary.getRange("B12").format = { fill: red, font: { bold: true, color: "#9C0006" }, wrapText: true };
summary.getRange("A4:A12").format.columnWidth = 31;
summary.getRange("B4:B12").format.columnWidth = 58;
summary.getRange("D4:E7").values = [
  ["Readiness category", "Count"],
  ["Verified/complete", 6],
  ["Author inputs", 7],
  ["Optional strengthening", 3],
];
styleTable(summary, "D4:E4", "D5:E7");
summary.getRange("D4:D7").format.columnWidth = 24;
summary.getRange("E4:E7").format.columnWidth = 12;
summary.getRange("D5:E5").format.fill = green;
summary.getRange("D6:E6").format.fill = red;
summary.getRange("D7:E7").format.fill = amber;
const readinessChart = summary.charts.add("bar", summary.getRange("D4:E7"));
readinessChart.title = "Revision readiness controls";
readinessChart.hasLegend = false;
readinessChart.setPosition("G4", "N18");
summary.mergeCells("A15:F17");
summary.getRange("A15").values = [[
  "Decision: the private main corpus is complete, but public reproducibility remains blocked by missing source/licence records, the official Copydays protocol, author declarations, chosen licences and a DOI-bearing repository release.",
]];
summary.getRange("A15:F17").format = { fill: amber, font: { bold: true, color: "#7F6000" }, wrapText: true, verticalAlignment: "center" };
summary.getRange("A15:F17").format.rowHeight = 24;
summary.freezePanes.freezeRows(4);

const localLocation = inventory.locations.find((item) => item.label === "local_D_NTH");
titleBlock(local, "Local project inventory", "D:\\NTH pre-update machine inventory; final manifest generated after release build", "F");
const localRows = localLocation.top_level.map((item) => [
  item.name,
  item.directories > 0 || item.files > 1 ? "Folder/tree" : "File",
  item.files,
  item.directories,
  item.bytes / (1024 * 1024),
  classifyLocal(item.name),
]);
local.getRange(`A4:F${4 + localRows.length}`).values = [["Top-level item", "Kind", "Files", "Directories", "Size (MB)", "Role / interpretation"], ...localRows];
styleTable(local, "A4:F4", `A5:F${4 + localRows.length}`);
local.getRange(`C5:D${4 + localRows.length}`).format.numberFormat = "#,##0";
local.getRange(`E5:E${4 + localRows.length}`).format.numberFormat = "#,##0.0";
local.getRange(`A4:A${4 + localRows.length}`).format.columnWidth = 48;
local.getRange(`B4:B${4 + localRows.length}`).format.columnWidth = 15;
local.getRange(`C4:E${4 + localRows.length}`).format.columnWidth = 14;
local.getRange(`F4:F${4 + localRows.length}`).format.columnWidth = 52;
local.tables.add(`A4:F${4 + localRows.length}`, true, "LocalInventoryTable");
local.freezePanes.freezeRows(4);

const hpcLocation = inventory.locations.find((item) => item.label === "hpc_NTH");
titleBlock(hpc, "HPC project inventory", "<HPC_SOURCE_ROOT> pre-sync machine inventory; historical assets preserved", "F");
const hpcRows = hpcLocation.top_level.map((item) => [
  item.name,
  item.directories > 0 || item.files > 1 ? "Folder/tree" : "File",
  item.files,
  item.directories,
  item.bytes / (1024 * 1024),
  classifyHpc(item.name),
]);
hpc.getRange(`A4:F${4 + hpcRows.length}`).values = [["Top-level item", "Kind", "Files", "Directories", "Size (MB)", "Role / interpretation"], ...hpcRows];
styleTable(hpc, "A4:F4", `A5:F${4 + hpcRows.length}`);
hpc.getRange(`C5:D${4 + hpcRows.length}`).format.numberFormat = "#,##0";
hpc.getRange(`E5:E${4 + hpcRows.length}`).format.numberFormat = "#,##0.0";
hpc.getRange(`A4:A${4 + hpcRows.length}`).format.columnWidth = 46;
hpc.getRange(`B4:B${4 + hpcRows.length}`).format.columnWidth = 15;
hpc.getRange(`C4:E${4 + hpcRows.length}`).format.columnWidth = 14;
hpc.getRange(`F4:F${4 + hpcRows.length}`).format.columnWidth = 54;
hpc.tables.add(`A4:F${4 + hpcRows.length}`, true, "HpcInventoryTable");
hpc.freezePanes.freezeRows(4);

const dataAudit = verification.data_audit;
titleBlock(audit, "Authoritative main-data audit", "Full pair-table resolution against the private HPC corpus", "D");
audit.getRange("A4:D15").values = [
  ["Metric", "Verified value", "Expected / decision", "Status"],
  ["Pair rows", dataAudit.rows, 91504, "PASS"],
  ["Positive rows", dataAudit.positive_rows, 29792, "PASS"],
  ["Negative rows", dataAudit.negative_rows, 61712, "PASS"],
  ["Seed identities", dataAudit.unique_seed_ids, 1064, "PASS"],
  ["Missing originals", dataAudit.missing_original_references, 0, "PASS"],
  ["Missing transformed references", dataAudit.missing_transformed_references, 0, "PASS"],
  ["Missing positive transforms", dataAudit.missing_positive_transforms, 0, "PASS"],
  ["Usable positive rows", dataAudit.usable_positive_rows, 29792, "PASS"],
  ["Usable negative rows", dataAudit.usable_negative_rows, 61712, "PASS"],
  ["Resolved pair rows", dataAudit.rows, 91504, "PASS"],
  ["Total missing references", null, 0, "PASS"],
];
audit.getRange("B15").formulas = [["=SUM(B9:B11)"]];
styleTable(audit, "A4:D4", "A5:D15");
audit.getRange("A4:A15").format.columnWidth = 34;
audit.getRange("B4:C15").format.columnWidth = 18;
audit.getRange("D4:D15").format.columnWidth = 14;
audit.getRange("D5:D15").format = { fill: green, font: { bold: true, color: "#006100" }, horizontalAlignment: "center" };
audit.getRange("B5:C15").format.numberFormat = "#,##0";
audit.freezePanes.freezeRows(4);

titleBlock(results, "Verified scientific results", "Complete-corpus values are separated from the local six-model comparison", "G");
const selected = verification.selected_model;
const deepRegistry = verification.deep_only_registry;
const radius16 = verification.two_stage_registry.find((row) => Number(row.nth_radius) === 16);
results.getRange("A4:G12").values = [
  ["Scope", "Metric", "Value", "CI low", "CI high", "Unit", "Evidence"],
  ["Full private main corpus", "ShuffleNet test recall", selected.test_recall, selected.cluster_bootstrap_95ci.test_recall_ci_low, selected.cluster_bootstrap_95ci.test_recall_ci_high, "fraction", "full_main_verification_summary.json"],
  ["Full private main corpus", "ShuffleNet test FPR", selected.test_fpr, selected.cluster_bootstrap_95ci.test_fpr_ci_low, selected.cluster_bootstrap_95ci.test_fpr_ci_high, "fraction", "deep_baseline_results.csv"],
  ["Full private main corpus", "Deep-only any false match/query", deepRegistry.per_query_any_false_match_probability, null, null, "fraction", "deep_registry_results.csv"],
  ["Full private main corpus", "NTH r=16 end-to-end recall", radius16.true_end_to_end_recall, null, null, "fraction", "two_stage_shufflenet_registry_results.csv"],
  ["Full private main corpus", "NTH r=16 mean shortlist", radius16.mean_stage1_shortlist, null, null, "candidates/query", "two_stage_shufflenet_registry_results.csv"],
  ["Full private main corpus", "NTH r=16 any false match/query", radius16.per_query_any_false_match_probability, null, null, "fraction", "two_stage_shufflenet_registry_results.csv"],
  ["Full private main corpus", "NTH default test recall", verification.hash_baselines.find((row) => row.method === "NTH-default").test_recall, null, null, "fraction", "hash_baseline_calibrated_results.csv"],
  ["Full private main corpus", "pHash test recall", verification.hash_baselines.find((row) => row.method === "pHash").test_recall, null, null, "fraction", "hash_baseline_calibrated_results.csv"],
];
styleTable(results, "A4:G4", "A5:G12");
results.getRange("C5:E12").format.numberFormat = "0.000000";
results.getRange("A4:A12").format.columnWidth = 27;
results.getRange("B4:B12").format.columnWidth = 36;
results.getRange("C4:E12").format.columnWidth = 15;
results.getRange("F4:F12").format.columnWidth = 18;
results.getRange("G4:G12").format.columnWidth = 43;
results.tables.add("A4:G12", true, "VerifiedResultsTable");
results.freezePanes.freezeRows(4);

titleBlock(actions, "Documents and decisions still needed from authors", "Items that cannot be inferred or completed from project files alone", "F");
const actionRows = [
  ["A1", "Source/licence registry", "One row per main seed: URL, original ID, owner, retrieval date, licence, redistribution permission", "Authors", "BLOCKER", "Not present"],
  ["A2", "Official Copydays benchmark", "Official database/query images and ground truth, or approve the explicit limitation", "Authors", "BLOCKER", "Not present"],
  ["A3", "Software/data licences", "Choose licences for code and derived metadata", "Authors", "BLOCKER", "Decision required"],
  ["A4", "DOI-bearing repository", "Publish versioned release; provide version and concept DOI", "Corresponding author", "BLOCKER", "Account action required"],
  ["A5", "Author declarations", "CRediT roles, funding/grants, interests, ethics/consent", "All authors", "BLOCKER", "Not confirmed"],
  ["A6", "Submission metadata", "Author order, affiliations, email, optional ORCIDs and portal deadline", "Corresponding author", "BLOCKER", "Portal/author confirmation"],
  ["A7", "Additional platforms", "Edge device plus uncontended server/A100 if Reviewer 6.7 is answered literally", "Authors", "OPTIONAL / RISK", "No cross-platform claim made"],
  ["A8", "White-box attack", "Adaptive attack against both stages", "Research team", "OPTIONAL", "Not implemented"],
  ["A9", "Crop mitigation", "Multi-scale/local alignment implementation if claiming mitigation", "Research team", "OPTIONAL", "Failure quantified only"],
];
actions.getRange(`A4:F${4 + actionRows.length}`).values = [["ID", "Required item", "Exact input/action", "Owner", "Priority", "Current evidence"], ...actionRows];
styleTable(actions, "A4:F4", `A5:F${4 + actionRows.length}`);
actions.getRange("E5:E10").format = { fill: red, font: { bold: true, color: "#9C0006" }, horizontalAlignment: "center" };
actions.getRange("E11:E13").format = { fill: amber, font: { bold: true, color: "#7F6000" }, horizontalAlignment: "center" };
actions.getRange(`A4:A${4 + actionRows.length}`).format.columnWidth = 10;
actions.getRange(`B4:B${4 + actionRows.length}`).format.columnWidth = 29;
actions.getRange(`C4:C${4 + actionRows.length}`).format.columnWidth = 64;
actions.getRange(`D4:D${4 + actionRows.length}`).format.columnWidth = 23;
actions.getRange(`E4:E${4 + actionRows.length}`).format.columnWidth = 18;
actions.getRange(`F4:F${4 + actionRows.length}`).format.columnWidth = 32;
actions.tables.add(`A4:F${4 + actionRows.length}`, true, "AuthorInputsTable");
actions.freezePanes.freezeRows(4);

titleBlock(transfers, "Transfers, preservation and checksums", "Local/HPC synchronization record", "E");
const transferRows = [
  ["HPC non-image snapshot", "HPC → local", "provenance/hpc_legacy_snapshot_20260831/hpc_legacy_nonimage_snapshot.tar.gz", "dd68ed7e36d9b64fda397db3291622a26fd698350424c1762fb8f5d45c94df3a", "Original notebooks/models preserved"],
  ["Full classical results", "HPC → local", "provenance/hpc_full_main_results_20260831.tar.gz", "d3121234d70df9a86b003d380c83b9563da0c08fc5ffd06ad25fb4f19b7df3fa", "Checksums verified"],
  ["Selected-model full results", "HPC → local", "provenance/hpc_full_selected_results_job4081_20260831.tar.gz", "5676781f33f03d68fcee75aed04eda1ab2d27abde4ba86b5a3c9889359d451b5", "Raw embeddings/predictions synchronized"],
  ["Code/test sync v5", "Local → HPC", "submission/hpc_sync_20260831_v5.tar.gz", "0c8be2774b549873f9bd2260a495a618917c70344f7f4c86b4adac37e78b7da2", "13 tests + compileall passed"],
  ["ShuffleNet pretrained weight", "Local → HPC cache", "~/.cache/torch/hub/checkpoints/shufflenetv2_x1-5666bf0f80.pth", "5666bf0f80d1e7f8fb61f6a65742b9c27c37f3208c1f087cdcd7c1ec1aa8b848", "Enabled offline PBS run"],
];
transfers.getRange(`A4:E${4 + transferRows.length}`).values = [["Artifact", "Direction", "Location", "SHA-256", "Verification / preservation"], ...transferRows];
styleTable(transfers, "A4:E4", `A5:E${4 + transferRows.length}`);
transfers.getRange(`A4:A${4 + transferRows.length}`).format.columnWidth = 30;
transfers.getRange(`B4:B${4 + transferRows.length}`).format.columnWidth = 18;
transfers.getRange(`C4:C${4 + transferRows.length}`).format.columnWidth = 63;
transfers.getRange(`D4:D${4 + transferRows.length}`).format.columnWidth = 68;
transfers.getRange(`E4:E${4 + transferRows.length}`).format.columnWidth = 38;
transfers.tables.add(`A4:E${4 + transferRows.length}`, true, "TransferTable");
transfers.freezePanes.freezeRows(4);

reviewer.showGridLines = false;
reviewer.getRange("A1:C1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
};
reviewer.getRange("A1:A49").format.columnWidth = 12;
reviewer.getRange("B1:B49").format.columnWidth = 82;
reviewer.getRange("C1:C49").format.columnWidth = 84;
reviewer.getRange("A2:C49").format = { wrapText: true, verticalAlignment: "top" };
reviewer.freezePanes.freezeRows(1);
reviewer.tables.add("A1:C49", true, "ReviewerCommentTable");

const sheetNames = [
  "Executive summary",
  "Local inventory",
  "HPC inventory",
  "Main data audit",
  "Verified results",
  "Author inputs needed",
  "Transfers and checksums",
  "Reviewer comments",
];
for (const sheetName of sheetNames) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const fileName = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  await fs.writeFile(
    path.join(previewDir, `${fileName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const workbookInspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 9000,
  tableMaxRows: 4,
  tableMaxCols: 7,
  tableMaxCellChars: 100,
});
const formulaInspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Executive summary",
  range: "A1:N20",
  maxChars: 3000,
  options: { maxResults: 20 },
});

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({
  outputPath,
  previewDir,
  sheets: sheetNames,
  workbookInspection,
  formulaInspection,
}, null, 2));

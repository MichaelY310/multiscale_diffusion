import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const dataDir = path.join(root, "eval_results", "experiment_ledger_20260918");
const outputDir = path.join(root, "outputs", "rdd_experiment_ledger_20260918");
const previewDir = path.join(outputDir, "previews");
await fs.mkdir(previewDir, { recursive: true });

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = (rows.shift() ?? []).map((value) => value.replace(/^\uFEFF/, "").trim());
  return { headers, rows };
}

async function loadCsv(name) {
  return parseCsv(await fs.readFile(path.join(dataDir, name), "utf8"));
}

function asNumber(value) {
  if (value === "" || value === null || value === undefined || value === "nan") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : value;
}

function colName(index) {
  let value = index + 1;
  let out = "";
  while (value > 0) {
    const rem = (value - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    value = Math.floor((value - 1) / 26);
  }
  return out;
}

const validCsv = await loadCsv("curated_valid_results.csv");
const invalidCsv = await loadCsv("historical_invalid_results.csv");
const rawCsv = await loadCsv("raw_fid_blocks.csv");
const inventoryCsv = await loadCsv("source_inventory.csv");
const auditCsv = await loadCsv("duplicate_conflict_audit.csv");
const speedCsv = await loadCsv("speed_benchmarks.csv");
const structuredFidAuditCsv = await loadCsv("structured_fid_source_audit.csv");
const structuredSpeedAuditCsv = await loadCsv("structured_speed_source_audit.csv");
const structuredCoverageCsv = await loadCsv("structured_scan_coverage.csv");
const diagnosticFidCsv = await loadCsv("diagnostic_fid_results.csv");
const accelerationCsv = await loadCsv("acceleration_evidence_summary.csv");
const monotonicityCsv = await loadCsv("strict_curve_monotonicity_audit.csv");
const anomalyCsv = await loadCsv("curve_anomaly_audit.csv");

const numericHeaders = new Set([
  "train_step", "scheduled_updates", "actual_nfe", "cfg", "num_samples",
  "inception_score", "fid", "sfid", "precision", "recall", "source_size",
  "normalized_line", "count", "sampling_ms_per_image_250", "training_ms_per_step",
  "training_images_per_s_gpu", "peak_gb", "canonical_value", "excluded_value",
  "sources", "bytes_or_records", "records", "unique_fid_values", "min_fid", "max_fid",
  "first_step", "last_step", "fid_increase_count", "largest_fid_increase",
  "speedup", "latency_reduction_percent", "quality_value",
]);

function typedMatrix(csv, keepHeaders = null) {
  const indexes = keepHeaders ? keepHeaders.map((h) => csv.headers.indexOf(h)) : csv.headers.map((_, i) => i);
  if (indexes.some((index) => index < 0)) {
    const missing = keepHeaders.filter((_, i) => indexes[i] < 0);
    throw new Error(`Missing CSV headers: ${missing.join(", ")}; available=${csv.headers.join(",")}`);
  }
  const headers = indexes.map((i) => csv.headers[i]);
  const rows = csv.rows.map((row) => indexes.map((i, j) => numericHeaders.has(headers[j]) ? asNumber(row[i]) : row[i]));
  return { headers, rows };
}

const valid = typedMatrix(validCsv);
const invalid = typedMatrix(invalidCsv);
const raw = typedMatrix(rawCsv, [
  "source_repo_path", "source_local_path", "source_origin", "source_size", "normalized_line",
  "fid", "sfid", "inception_score", "precision", "recall",
]);
const inventory = typedMatrix(inventoryCsv);
const audit = typedMatrix(auditCsv);
const speed = typedMatrix(speedCsv);
const structuredFidAudit = typedMatrix(structuredFidAuditCsv);
const structuredSpeedAudit = typedMatrix(structuredSpeedAuditCsv);
const structuredCoverage = typedMatrix(structuredCoverageCsv);
const diagnosticFid = typedMatrix(diagnosticFidCsv);
const acceleration = typedMatrix(accelerationCsv);
const monotonicity = typedMatrix(monotonicityCsv);
const anomaly = typedMatrix(anomalyCsv);

const wb = Workbook.create();
const font = "Arial";
const palette = {
  navy: "#0F172A", blue: "#2563EB", cyan: "#0891B2", green: "#059669",
  orange: "#D97706", red: "#DC2626", purple: "#7C3AED", gray: "#64748B",
  line: "#E2E8F0", pale: "#F8FAFC", warning: "#FFF7ED",
};

function styleTitle(sheet, address, size = 16) {
  const r = sheet.getRange(address);
  r.format.font = { name: font, size, bold: true, color: palette.navy };
}

function writeDataSheet(name, data, tabColor = null) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  if (tabColor) sheet.tabColor = tabColor;
  const matrix = [data.headers, ...data.rows];
  const endCol = colName(data.headers.length - 1);
  const endRow = matrix.length;
  sheet.getRange(`A1:${endCol}${endRow}`).writeValues(matrix);
  const header = sheet.getRange(`A1:${endCol}1`);
  header.format = {
    fill: palette.navy,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "medium", color: palette.navy } },
  };
  header.format.rowHeight = 32;
  const body = sheet.getRange(`A2:${endCol}${endRow}`);
  body.format.font = { name: font, size: 9, color: palette.navy };
  body.format.verticalAlignment = "center";
  body.format.borders = { insideHorizontal: { style: "thin", color: palette.line } };
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(3, data.headers.length));

  data.headers.forEach((headerName, index) => {
    const column = colName(index);
    const values = data.rows.slice(0, 200).map((row) => String(row[index] ?? ""));
    const maxLen = Math.max(headerName.length, ...values.map((v) => Math.min(v.length, 80)));
    let width = Math.max(10, Math.min(42, Math.ceil(maxLen * 0.95)));
    if (headerName.includes("path") || headerName === "notes" || headerName.includes("reason")) width = 42;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
    if (["fid", "sfid", "inception_score", "precision", "recall", "cfg", "canonical_value", "excluded_value"].includes(headerName)) {
      sheet.getRange(`${column}2:${column}${endRow}`).format.numberFormat = "0.0000";
    }
    if (["train_step", "scheduled_updates", "actual_nfe", "num_samples", "source_size", "normalized_line"].includes(headerName)) {
      sheet.getRange(`${column}2:${column}${endRow}`).format.numberFormat = "#,##0";
    }
  });
  return sheet;
}

const summary = wb.worksheets.add("Summary");
summary.showGridLines = false;
summary.tabColor = palette.navy;
summary.getRange("A2").values = [["RDD experiment ledger"]];
styleTitle(summary, "A2", 16);
summary.getRange("A3").values = [["HF and surfi2 audit across logs, metrics tables, JSON/CSV/Markdown/YAML, original-ImageNet curation, and comparison charts. Generated 2026-09-18."]];
summary.getRange("A3:H3").format.font = { name: font, size: 10, italic: true, color: palette.gray };
summary.getRange("A4:H4").format.borders = { bottom: { style: "thin", color: palette.line } };

const stats = [
  ["Metric", "Count"],
  ["Unique log/txt sources", inventory.rows.length],
  ["Raw FID blocks", raw.rows.length],
  ["Canonical valid rows", valid.rows.length],
  ["Old-HF invalid canonical rows", invalid.rows.length],
  ["Duplicate/conflict audit rows", audit.rows.length],
  ["Project structured-text sources", 5934],
  ["Structured FID source audit rows", structuredFidAudit.rows.length],
  ["Structured speed source audit rows", structuredSpeedAudit.rows.length],
  ["New diagnostic FID rows", diagnosticFid.rows.length],
  ["New paper-valid 50K FID rows", 0],
  ["Strict curves with an FID increase", 0],
  ["FID parse failures", 0],
];
summary.getRange(`A6:B${5 + stats.length}`).values = stats;
summary.getRange("A6:B6").format = { fill: palette.navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summary.getRange(`A7:B${5 + stats.length}`).format.borders = { insideHorizontal: { style: "thin", color: palette.line } };
summary.getRange(`B7:B${5 + stats.length}`).format.numberFormat = "#,##0";

const insights = [
  ["Finding", "Evidence"],
  ["Unified-L training allocation", "300/300/400 is best, followed by 400/300/300, 100/300/600, then 600/300/100."],
  ["Stage topology", "Three stages outperform 16->32 and direct 8->32 at matched checkpoint and DDPM250."],
  ["Sampling allocation", "Within the tested NFE250 grid, assigning more steps to final 32x32 consistently lowers FID."],
  ["Solver", "DPM2 is strong around 124-244 actual NFE but unstable near 64 NFE."],
  ["CFG", "CFG1.5 minimizes FID; higher CFG raises precision/IS while reducing recall and worsening FID."],
  ["Flow matching", "Corrected Unified-L FM is worse than diffusion at 50K; legacy XL CFG1.5 is not a convergence comparison."],
  ["Invalid history", "252 canonical old-HF-data FID blocks are hypothesis-only and excluded from paper figures."],
  ["Curve anomaly audit", "Every strict DDPM250/CFG1.0 curve is monotone. Earlier upward spikes came from connecting different transition, boundary, or sampling-allocation protocols."],
  ["Structured-text scan", "JSON/CSV/Markdown/YAML and other text sources added 13 diagnostic 10K FID rows and acceleration evidence, but no new paper-valid 50K FID."],
];
summary.getRange(`D6:E${5 + insights.length}`).values = insights;
summary.getRange("D6:E6").format = { fill: palette.navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summary.getRange(`D7:E${5 + insights.length}`).format = { font: { name: font, size: 9, color: palette.navy }, wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: palette.line } } };
summary.getRange("D:D").format.columnWidth = 24;
summary.getRange("E:E").format.columnWidth = 58;

// Chart helper 1: main convergence.
summary.getRange("A21:D21").values = [["Step (K)", "Original DiT-XL/2", "Unified XL 300/300/400", "Unified L 300/300/400"]];
const validObjects = valid.rows.map((row) => Object.fromEntries(valid.headers.map((h, i) => [h, row[i]])));
const convergenceSteps = [50,100,150,200,250,300,350,400,450,500];
function findFid(model, step, cfg = 1.0, nfe = 250) {
  const expectedSplit = model === "Original DiT-XL/2" ? "250" : model.includes("500/500") ? "125/125" : "90/80/80";
  const candidates = validObjects.filter((r) =>
    r.model === model &&
    Number(r.train_step) === step * 1000 &&
    Number(r.cfg) === cfg &&
    Number(r.actual_nfe) === nfe &&
    r.sampler === "DDPM" &&
    String(r.sampling_steps_fine_middle_coarse) === expectedSplit &&
    (!model.startsWith("Unified") || r.transition === "noise filling")
  );
  const priority = (r) => r.notes === "Canonical structured convergence table" ? 0 : String(r.notes).startsWith("Canonical convergence point") ? 1 : 2;
  candidates.sort((a, b) => priority(a) - priority(b) || String(a.source_repo_path).localeCompare(String(b.source_repo_path)));
  const row = candidates[0];
  return row ? Number(row.fid) : null;
}
const convergenceRows = convergenceSteps.map((step) => [step, findFid("Original DiT-XL/2", step), findFid("Unified XL 300/300/400", step), findFid("Unified L 300/300/400", step)]);
summary.getRange("A22:D31").values = convergenceRows;
const convChart = summary.charts.add("line", summary.getRange("A21:D31"));
convChart.title = "Original ImageNet FID convergence";
convChart.titleTextStyle.fontSize = 12; convChart.titleTextStyle.typeface = font;
convChart.legend = { position: "top", textStyle: { typeface: font } };
convChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
convChart.yAxis = { numberFormatCode: "0.0", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
convChart.setPosition("G2", "N18");
const convColors = [palette.gray, palette.blue, palette.cyan];
convChart.series.items.forEach((series, i) => { series.line = { fill: convColors[i], style: i === 0 ? "dashed" : "solid", width: 2 }; });

// Chart helper 2: Unified-L schedule at 250K and 400K where available.
const allocModels = ["Unified L 300/300/400", "Unified L 400/300/300", "Unified L 100/300/600", "Unified L 600/300/100", "Unified L 16->32 500/500", "Unified L 8->32 500/500"];
summary.getRange("A35:C35").values = [["Unified-L schedule/topology", "FID at 250K", "FID at 400K"]];
summary.getRange("A36:C41").values = allocModels.map((model) => [model.replace("Unified L ", ""), findFid(model, 250), findFid(model, 400)]);
const allocChart = summary.charts.add("bar", summary.getRange("A35:C41"));
allocChart.title = "Unified-L allocation and topology";
allocChart.titleTextStyle.fontSize = 12; allocChart.titleTextStyle.typeface = font;
allocChart.legend = { position: "top", textStyle: { typeface: font } };
allocChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
allocChart.yAxis = { numberFormatCode: "0.0", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
allocChart.setPosition("G20", "N38");
allocChart.series.items.forEach((series, i) => { series.fill = i === 0 ? palette.blue : palette.green; });

// Chart helper 3: all complete FM rows.
const fmRows = validObjects.filter((r) => r.method === "RDD flow matching").map((r) => [`${r.model}; ${r.sampler}; CFG ${r.cfg}`, Number(r.fid)]);
summary.getRange(`A45:B${45 + fmRows.length}`).values = [["Flow matching experiment", "FID"], ...fmRows];
const fmChart = summary.charts.add("bar", summary.getRange(`A45:B${45 + fmRows.length}`));
fmChart.title = "RDD Flow Matching complete evaluations";
fmChart.titleTextStyle.fontSize = 12; fmChart.titleTextStyle.typeface = font;
fmChart.hasLegend = false;
fmChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 8 } };
fmChart.yAxis = { numberFormatCode: "0.0", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
fmChart.setPosition("G40", "N58");
fmChart.series.items.forEach((series) => { series.fill = palette.purple; });

// Chart helper 4: measured 250-step structural benchmark.
summary.getRange(`A55:B${55 + speed.rows.length}`).values = [["Model", "Sampling ms/image"], ...speed.rows.map((r) => [r[0], r[1]])];
const speedChart = summary.charts.add("bar", summary.getRange(`A55:B${55 + speed.rows.length}`));
speedChart.title = "250-step sampling benchmark";
speedChart.titleTextStyle.fontSize = 12; speedChart.titleTextStyle.typeface = font;
speedChart.hasLegend = false;
speedChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
speedChart.yAxis = { numberFormatCode: "0", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
speedChart.setPosition("G60", "N77");
speedChart.series.items.forEach((series) => { series.fill = palette.green; });

// Chart helper 5: structured diagnostic FID (kept separate from paper-valid 50K FID).
const diagnosticObjects = diagnosticFid.rows.map((row) => Object.fromEntries(diagnosticFid.headers.map((h, i) => [h, row[i]])));
const csraRows = diagnosticObjects.filter((r) => String(r.protocol).startsWith("CSRA"));
summary.getRange(`A85:B${85 + csraRows.length}`).values = [["CSRA diagnostic", "FID10K"], ...csraRows.map((r) => [`${r.protocol}: ${r.variant}`, Number(r.fid)])];
const diagnosticChart = summary.charts.add("bar", summary.getRange(`A85:B${85 + csraRows.length}`));
diagnosticChart.title = "CSRA diagnostic FID10K";
diagnosticChart.titleTextStyle.fontSize = 12; diagnosticChart.titleTextStyle.typeface = font;
diagnosticChart.hasLegend = false;
diagnosticChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 8 } };
diagnosticChart.yAxis = { numberFormatCode: "0.0", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
diagnosticChart.setPosition("G80", "N97");
diagnosticChart.series.items.forEach((series) => { series.fill = palette.orange; });

// Chart helper 6: measured or task-specific speedups only; evidence status stays in the detailed sheet.
const accelerationObjects = acceleration.rows.map((row) => Object.fromEntries(acceleration.headers.map((h, i) => [h, row[i]])));
const accelerationRows = accelerationObjects.filter((r) => r.speedup !== null && r.speedup !== "" && Number.isFinite(Number(r.speedup)));
summary.getRange(`A104:B${104 + accelerationRows.length}`).values = [["Acceleration result", "Speedup"], ...accelerationRows.map((r) => [`${r.family}: ${r.method}`, Number(r.speedup)])];
const accelerationChart = summary.charts.add("bar", summary.getRange(`A104:B${104 + accelerationRows.length}`));
accelerationChart.title = "Recorded speedups by protocol";
accelerationChart.titleTextStyle.fontSize = 12; accelerationChart.titleTextStyle.typeface = font;
accelerationChart.hasLegend = false;
accelerationChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 8 } };
accelerationChart.yAxis = { numberFormatCode: "0.0x", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 9 } };
accelerationChart.setPosition("G99", "N119");
accelerationChart.series.items.forEach((series) => { series.fill = palette.cyan; });

summary.getRange("A21:D21").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange("A35:C35").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange(`A45:B${45 + fmRows.length}`).format.font = { name: font, size: 9 };
summary.getRange("A45:B45").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange(`A55:B${55 + speed.rows.length}`).format.font = { name: font, size: 9 };
summary.getRange("A55:B55").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange(`A85:B${85 + csraRows.length}`).format.font = { name: font, size: 9 };
summary.getRange("A85:B85").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange(`A104:B${104 + accelerationRows.length}`).format.font = { name: font, size: 9 };
summary.getRange("A104:B104").format = { fill: palette.navy, font: { name: font, size: 9, bold: true, color: "#FFFFFF" } };
summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:C").format.columnWidth = 18;
summary.getRange("A21:D124").format.borders = { insideHorizontal: { style: "thin", color: palette.line } };

writeDataSheet("Curated valid", valid, palette.blue);
writeDataSheet("Old HF invalid", invalid, palette.red);
writeDataSheet("Raw FID blocks", raw);
writeDataSheet("Source inventory", inventory);
writeDataSheet("Duplicate audit", audit, palette.orange);
writeDataSheet("Speed benchmarks", speed, palette.green);
writeDataSheet("Structured FID audit", structuredFidAudit, palette.orange);
writeDataSheet("Structured speed audit", structuredSpeedAudit);
writeDataSheet("Scan coverage", structuredCoverage);
writeDataSheet("Diagnostic FID", diagnosticFid, palette.purple);
writeDataSheet("Acceleration", acceleration, palette.green);
writeDataSheet("Curve audit", monotonicity, palette.blue);
writeDataSheet("Anomaly audit", anomaly, palette.red);

wb.recalculate();

const summaryCheck = await wb.inspect({ kind: "table", sheetId: "Summary", range: "A1:N25", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 14, maxChars: 12000 });
console.log(summaryCheck.ndjson);
const errorCheck = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan", maxChars: 5000 });
console.log(errorCheck.ndjson);

for (const [sheetName, range] of [
  ["Summary", "A1:N124"],
  ["Curated valid", "A1:W24"],
  ["Old HF invalid", "A1:O24"],
  ["Raw FID blocks", "A1:J24"],
  ["Source inventory", "A1:L24"],
  ["Duplicate audit", "A1:F24"],
  ["Speed benchmarks", "A1:E10"],
  ["Structured FID audit", "A1:I24"],
  ["Structured speed audit", "A1:F24"],
  ["Scan coverage", "A1:C10"],
  ["Diagnostic FID", "A1:H16"],
  ["Acceleration", "A1:J18"],
  ["Curve audit", "A1:G12"],
  ["Anomaly audit", "A1:E10"],
]) {
  const preview = await wb.render({ sheetName, range, scale: 1, format: "png" });
  const safe = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  await fs.writeFile(path.join(previewDir, `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(wb);
const outputPath = path.join(outputDir, "RDD_COMPLETE_EXPERIMENT_LEDGER_20260918.xlsx");
await output.save(outputPath);
const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const reopenedDrawings = await reopened.inspect({ kind: "drawing", sheetId: "Summary", maxChars: 6000 });
console.log(reopenedDrawings.ndjson);
const reopenedPreview = await reopened.render({ sheetName: "Summary", range: "A1:N124", scale: 1, format: "png" });
await fs.writeFile(path.join(previewDir, "summary_reopened.png"), new Uint8Array(await reopenedPreview.arrayBuffer()));
console.log(JSON.stringify({ outputPath, sheets: 14, charts: 6, curatedRows: valid.rows.length, rawFidBlocks: raw.rows.length, sources: inventory.rows.length, structuredSources: 5934, diagnosticFidRows: diagnosticFid.rows.length }, null, 2));

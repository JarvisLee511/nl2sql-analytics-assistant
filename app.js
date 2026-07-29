import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0/+esm";

// ---------------------------------------------------------------------------
// Warehouse schema (also fed to Claude for free-form NL->SQL)
// ---------------------------------------------------------------------------
const TABLES = [
  "customer", "aws_account", "aws_service", "purchase", "invoice", "payment",
  "support_ticket", "employee", "account_manager_assignment", "usage_log",
  "region", "account_region", "contract",
];

const TABLE_DOCS = {
  customer: "customer_id, company_name, industry, segment (Enterprise/Mid-Market/SMB), signup_date, status (Active/Churned)",
  aws_account: "account_id, account_name, customer_id → customer",
  aws_service: "service_id, service_name, product_code, category, service_tier, current_unit_price",
  purchase: "purchase_id, purchase_date, quantity, unit_price_at_purchase, customer_id, service_id",
  invoice: "invoice_id, purchase_id → purchase, invoice_date, total_amount, payment_status (Paid/Pending/Overdue)",
  payment: "payment_id, invoice_id → invoice, payment_date, payment_method, amount_paid",
  support_ticket: "ticket_id, customer_id, issue_type, priority_level, created_date, resolved_date, status (Open/Resolved)",
  employee: "employee_id, employee_name, department (Sales/Support/Account Management), team, hire_date",
  account_manager_assignment: "assignment_id, customer_id, employee_id → employee, assignment_date",
  usage_log: "usage_id, account_id → aws_account, service_id → aws_service, usage_date, usage_hours, cost",
  region: "region_id, region_name, country, continent",
  account_region: "account_id → aws_account, region_id → region (many-to-many)",
  contract: "contract_id, customer_id, start_date, end_date, committed_amount, discount_pct",
};

const SCHEMA_DDL = Object.entries(TABLE_DOCS)
  .map(([t, cols]) => `${t}(${cols})`).join("\n");

const SYSTEM_PROMPT = `You are a senior data analyst that writes DuckDB SQL.
Given a question about this AWS-reseller analytics warehouse, return ONE valid DuckDB SQL SELECT statement that answers it.

Schema (table(columns), → marks a foreign key):
${SCHEMA_DDL}

Rules:
- Return ONLY the SQL, no prose, no markdown fences.
- SELECT/WITH queries only. Never modify data.
- Always add a sensible LIMIT (<= 50) unless the question implies an aggregate that returns few rows.
- Round money to whole numbers; alias columns with clear names.
- To avoid fan-out when a customer maps to many regions/accounts, filter with IN (subquery) instead of extra JOINs.
- "today" = CURRENT_DATE. The data spans 2022-01 to 2026-06.`;

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const loader = $("loader");
const loaderText = $("loaderText");
function showLoader(t) { loaderText.textContent = t; loader.hidden = false; }
function hideLoader() { loader.hidden = true; }

// ---------------------------------------------------------------------------
// DuckDB-WASM init
// ---------------------------------------------------------------------------
let conn = null;
let CURATED = [];

async function initDuckDB() {
  showLoader("Booting DuckDB-WASM…");
  const bundles = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(bundles);
  const worker = await duckdb.createWorker(bundle.mainWorker);
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  conn = await db.connect();

  const base = new URL("./db/", window.location.href).href;
  let loaded = 0;
  for (const t of TABLES) {
    showLoader(`Loading data… (${++loaded}/${TABLES.length})`);
    const url = `${base}${t}.parquet`;
    await db.registerFileURL(`${t}.parquet`, url, duckdb.DuckDBDataProtocol.HTTP, false);
    await conn.query(`CREATE TABLE ${t} AS SELECT * FROM read_parquet('${t}.parquet')`);
  }
  const totalSQL = "SELECT " + TABLES.map((t) => `(SELECT count(*) FROM ${t})`).join(" + ") + " AS n";
  const r = await conn.query(totalSQL);
  return Number(r.toArray()[0].toJSON().n);
}

async function runSQL(sql) {
  const res = await conn.query(sql);
  const cols = res.schema.fields.map((f) => f.name);
  const temporal = new Set(
    res.schema.fields.filter((f) => /date|timestamp|time/i.test(String(f.type))).map((f) => f.name)
  );
  // raw DECIMAL columns come back as unscaled integers — restore the scale
  const decimals = {};
  res.schema.fields.forEach((f) => {
    if (/decimal/i.test(String(f.type))) decimals[f.name] = f.type.scale ?? 0;
  });
  const rows = res.toArray().map((r) => {
    const o = r.toJSON();
    for (const [name, scale] of Object.entries(decimals)) {
      if (o[name] !== null && o[name] !== undefined && scale > 0) o[name] = Number(o[name]) / 10 ** scale;
    }
    return o;
  });
  return { cols, rows, temporal };
}

// column names that hold dates/timestamps in the current result
let TEMPORAL = new Set();

// ---------------------------------------------------------------------------
// Value formatting
// ---------------------------------------------------------------------------
function norm(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") return Number(v);
  return v;
}
function isNum(v) { return typeof v === "number" && Number.isFinite(v); }
function toDateStr(v) {
  if (v instanceof Date) return v.toISOString().slice(0, 10);
  let n = typeof v === "bigint" ? Number(v) : v;
  if (typeof n !== "number") return String(v);
  if (Math.abs(n) < 1e6) n = n * 86400000; // epoch days -> ms
  const d = new Date(n);
  return isNaN(d.getTime()) ? String(v) : d.toISOString().slice(0, 10);
}
function fmt(v, temporal) {
  if (v === null || v === undefined) return "—";
  if (temporal) return toDateStr(v);
  v = norm(v);
  if (v === null) return "—";
  if (v instanceof Date) return v.toISOString().slice(0, 10);
  if (isNum(v)) {
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  return String(v);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderTable(cols, rows) {
  const numericCol = {};
  cols.forEach((c) => { numericCol[c] = rows.length > 0 && !TEMPORAL.has(c) && isNum(norm(rows[0][c])); });
  const thead = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((row) =>
    `<tr>${cols.map((c) =>
      `<td class="${numericCol[c] ? "num" : ""}">${fmt(row[c], TEMPORAL.has(c))}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("dataTable").innerHTML = thead + tbody;
  $("rowCount").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
}

function pickViz(cols, rows, hint) {
  if (hint && hint !== "auto") return hint;
  if (cols.length < 2 || rows.length === 0 || rows.length > 30) return "table";
  const v1 = norm(rows[0][cols[1]]);
  if (!isNum(v1)) return "table";
  const l0 = norm(rows[0][cols[0]]);
  const dateish = /date|month|day|week|year/i.test(cols[0]) ||
    (typeof l0 === "string" && /^\d{4}-\d{2}/.test(l0));
  return dateish ? "line" : "bar";
}

function renderChart(cols, rows, viz) {
  const panel = $("chartPanel"), grid = document.querySelector(".result-grid");
  if (viz === "table") { panel.hidden = true; grid.classList.remove("has-chart"); return; }
  panel.hidden = false; grid.classList.add("has-chart");
  const labelKey = cols[0], valKey = cols[1];
  const labelTemporal = TEMPORAL.has(labelKey);
  const data = rows.map((r) => ({ label: fmt(r[labelKey], labelTemporal), value: Number(norm(r[valKey])) || 0 }));
  const max = Math.max(...data.map((d) => d.value), 1);

  if (viz === "bar") {
    $("chart").innerHTML = data.map((d) =>
      `<div class="bar-row"><div class="bl" title="${d.label}">${d.label}</div>
       <div class="bt"><div class="bf" style="width:${(d.value / max * 100).toFixed(1)}%"></div></div>
       <div class="bv">${fmt(d.value)}</div></div>`).join("");
  } else { // line
    const W = 600, H = 240, pad = 34;
    const n = data.length;
    const x = (i) => pad + (i * (W - 2 * pad)) / Math.max(n - 1, 1);
    const y = (v) => H - pad - (v / max) * (H - 2 * pad);
    const pts = data.map((d, i) => `${x(i).toFixed(1)},${y(d.value).toFixed(1)}`).join(" ");
    const dots = data.map((d, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(d.value).toFixed(1)}" r="3" fill="#5EEAD4"/>`).join("");
    const xlabs = data.map((d, i) => (n <= 13 || i % 2 === 0)
      ? `<text x="${x(i).toFixed(1)}" y="${H - 10}" font-size="9" fill="#9BA7BD" text-anchor="middle" font-family="monospace">${d.label.slice(5)}</text>` : "").join("");
    $("chart").innerHTML =
      `<svg class="line" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <polyline points="${pts}" fill="none" stroke="#5EEAD4" stroke-width="2"/>
        ${dots}${xlabs}
       </svg>`;
  }
}

// Lightweight non-LLM summary for curated queries (keeps them $0).
function templateSummary(cols, rows, viz) {
  if (rows.length === 0) return "No rows matched.";
  const r0 = rows[0];
  const f0 = fmt(r0[cols[0]], TEMPORAL.has(cols[0]));
  const f1 = fmt(r0[cols[1]], TEMPORAL.has(cols[1]));
  if (viz === "table") {
    return `${rows.length} result${rows.length === 1 ? "" : "s"}. Top row: ${f0} (${cols[1]}: ${f1}).`;
  }
  // bar/line: report the leading data point
  return `${rows.length} ${cols[0]} values. Highest ${cols[1]}: ${f0} at ${f1}.`;
}

function setSummary(text) {
  const el = $("summary");
  if (!text) { el.hidden = true; return; }
  el.hidden = false; el.textContent = text;
}

// ---------------------------------------------------------------------------
// Result orchestration
// ---------------------------------------------------------------------------
function showResult() { $("result").hidden = false; $("result").scrollIntoView({ behavior: "smooth", block: "start" }); }

async function present(question, sql, vizHint, { summary } = {}) {
  $("resultQ").textContent = question;
  $("sqlOut").textContent = sql.trim();
  $("resultStatus").textContent = "running in browser…";
  showResult();
  let out;
  try {
    out = await runSQL(sql);
    TEMPORAL = out.temporal;
  } catch (e) {
    $("resultStatus").innerHTML = `<span class="err">SQL error</span>`;
    $("dataTable").innerHTML = `<tbody><tr><td class="err">${String(e.message || e)}</td></tr></tbody>`;
    $("chartPanel").hidden = true; setSummary(null); $("rowCount").textContent = "";
    return null;
  }
  const viz = pickViz(out.cols, out.rows, vizHint);
  renderTable(out.cols, out.rows);
  renderChart(out.cols, out.rows, viz);
  setSummary(summary || templateSummary(out.cols, out.rows, viz));
  const t0 = performance.now();
  $("resultStatus").textContent = `${out.rows.length} rows · DuckDB-WASM`;
  return out;
}

// ---------------------------------------------------------------------------
// Free-form: Claude generates SQL (Opus) + summary (Haiku), browser-side
// ---------------------------------------------------------------------------
function cleanSQL(text) {
  let s = text.trim().replace(/^```[a-z]*\s*/i, "").replace(/```\s*$/i, "").trim();
  const m = s.match(/\b(WITH|SELECT)\b[\s\S]+/i);
  if (m) s = m[0].trim();
  if (s.endsWith(";")) s = s.slice(0, -1).trim();
  return s;
}
function guardSQL(sql) {
  const low = " " + sql.toLowerCase().replace(/\s+/g, " ") + " ";
  if (!/^\s*(with|select)\b/i.test(sql)) throw new Error("Only SELECT queries are allowed.");
  for (const bad of ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "attach ", "copy ", "pragma ", "truncate ", "replace "]) {
    if (low.includes(" " + bad)) throw new Error(`Blocked keyword: ${bad.trim()}`);
  }
  if (sql.replace(/;\s*$/, "").includes(";")) throw new Error("Only one statement is allowed.");
  return sql;
}
async function claude(key, model, system, user, maxTokens) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model, max_tokens: maxTokens,
      system, messages: [{ role: "user", content: user }],
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Claude API ${res.status}: ${t.slice(0, 200)}`);
  }
  const data = await res.json();
  return (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
}

async function askFreeform(question) {
  const key = $("apiKeyInput").value.trim();
  if (!key) {
    $("byokPanel").hidden = false;
    $("apiKeyInput").focus();
    setSummary("Add your Claude API key above to ask free-form questions, or pick an example below.");
    $("result").hidden = false;
    return;
  }
  $("askBtn").disabled = true;
  try {
    showLoader("Claude Opus is writing SQL…");
    const raw = await claude(key, "claude-opus-4-8", SYSTEM_PROMPT, question, 2000);
    const sql = guardSQL(cleanSQL(raw));
    hideLoader();
    const out = await present(question, sql, "auto", { summary: "Summarizing…" });
    if (out) {
      showLoader("Claude Haiku is summarizing…");
      const sample = JSON.stringify(out.rows.slice(0, 15), (k, v) => typeof v === "bigint" ? Number(v) : v);
      const sys = "You summarize SQL query results in one or two plain, specific sentences for a business reader. No preamble.";
      const summary = await claude(key, "claude-haiku-4-5", sys,
        `Question: ${question}\nColumns: ${out.cols.join(", ")}\nRows (sample): ${sample}`, 200);
      setSummary(summary.trim());
      hideLoader();
    }
  } catch (e) {
    hideLoader();
    $("result").hidden = false;
    $("resultStatus").innerHTML = `<span class="err">error</span>`;
    setSummary("⚠ " + (e.message || e));
  } finally {
    $("askBtn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Wire up UI
// ---------------------------------------------------------------------------
function renderChips() {
  const wrap = $("chips");
  wrap.innerHTML = CURATED.map((q) =>
    `<button class="chip" data-id="${q.id}"><span class="cat">${q.category}</span>${q.en}</button>`).join("");
  wrap.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = CURATED.find((x) => x.id === btn.dataset.id);
      $("questionInput").value = q.en;
      present(q.en, q.sql, q.viz);
    });
  });
}
function renderSchema() {
  $("schemaGrid").innerHTML = Object.entries(TABLE_DOCS).map(([t, cols]) =>
    `<div class="tbl-card"><h4>${t}</h4><p>${cols}</p></div>`).join("");
}

async function main() {
  renderSchema();
  try {
    CURATED = await (await fetch("./queries.json")).json();
  } catch { CURATED = []; }
  renderChips();

  const rows = await initDuckDB();
  $("statRows").textContent = (rows / 1e6).toFixed(2) + "M+";
  hideLoader();

  $("askBtn").addEventListener("click", () => {
    const q = $("questionInput").value.trim();
    if (q) askFreeform(q);
  });
  $("questionInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("askBtn").click();
  });
  $("byokToggle").addEventListener("click", () => {
    $("byokPanel").hidden = !$("byokPanel").hidden;
    if (!$("byokPanel").hidden) $("apiKeyInput").focus();
  });
  $("copySql").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("sqlOut").textContent);
    $("copySql").textContent = "copied!";
    setTimeout(() => ($("copySql").textContent = "copy"), 1200);
  });
}

main().catch((e) => {
  hideLoader();
  document.body.insertAdjacentHTML("beforeend",
    `<div style="position:fixed;bottom:1rem;left:1rem;right:1rem;background:#3b0d0d;color:#fca5a5;padding:1rem;border-radius:10px;font-family:monospace;font-size:13px">Init failed: ${e.message || e}</div>`);
});

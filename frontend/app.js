/* sunseg dashboard — vanilla JS + Plotly (ไม่มีขั้นตอน build)
 *
 * หน้าเว็บแบ่งเป็นสามฉาก: hero (ตัวเลขสรุป + จานสุริยะ) → dashboard (แผงข้อมูล)
 * → about (วิธีการและผลการทดลอง) ทั้งสามฉากดึงจาก API ชุดเดียวกัน
 *
 * โครงสร้าง: ดึงสถานะระบบตอนโหลด -> ปรับ UI ตามส่วนที่พร้อมใช้งาน -> ผูก event
 * ทุกส่วนออกแบบให้ degrade ได้: ถ้าโมเดลใดยังไม่ถูกเทรน แผงนั้นจะบอกวิธีสร้าง
 * แทนที่จะพังทั้งหน้า
 *
 * dashboard ใช้ "ช่วงเวลาเดียว" จากแถบกรอง — GOES X-ray, จานสุริยะใน hero และตัวเลข
 * สรุป วาดจากช่วงเดียวกันเสมอ เพื่อให้เทียบข้ามแผงได้โดยตรง
 *
 * การ์ดภาพ (ซ้าย) กับการ์ดความเข้มแสง (ขวา) ผูกกันแน่นกว่านั้น: ทั้งคู่วาดจาก
 * **คำขอ /api/segment ครั้งเดียวกัน** ใน showFrame() การ์ดขวาไม่ยิง network เอง
 * เพื่อรับประกันว่าเส้นขอบบนภาพกับแท่งกราฟมาจาก mask อันเดียวกันเสมอ
 *
 * สีทั้งหมดอ่านจาก CSS custom property ใน style.css ที่จุดเดียว (ดู THEME ด้านล่าง)
 * เพื่อไม่ให้สีใน Plotly กับสีใน CSS เพี้ยนไปคนละทาง
 */

"use strict";

const state = {
  harps: [],          // HARP ทั้งหมดที่ backend มี — กรอง/เรียงฝั่ง client
  selectedHarp: null,
  health: null,
  info: null,
  frames: [],
  frameIndex: 0,
  layer: "mag",       // เลเยอร์ภาพที่กำลังแสดง — "mag" หรือคีย์ช่อง AIA
  frameRequest: 0,    // นับคำขอเฟรม กันผลที่มาช้ากว่ามาทับผลใหม่
  extraction: null,   // ผล /api/intensity-series ล่าสุด — ใช้ตอนขยับเส้นบอกเฟรม
  extractionRequest: 0,
  goes: null,         // ผลลัพธ์ /api/goes ล่าสุด — ใช้ร่วมกันระหว่าง hero กับ dashboard
  pinnedFlare: null,  // flare ที่ถูกคลิกตรึงไว้ — มีค่าแล้ว hover จะไม่ทับ (null = follow hover)
  chartedFlare: null, // flare ที่กราฟโปรตอนแสดงอยู่จริง ๆ (ตรึงไว้ หรือเลือกอัตโนมัติ)
  diskClasses: new Set(["B", "C", "M", "X"]),  // class ที่แสดงบนจาน — chip ใต้จานเป็นตัวสลับ
  diskHighlight: null, // ฟังก์ชันเลื่อนวงแหวนไฮไลต์บนจานไปยัง flare ที่ระบุ (ตั้งค่าใน renderFlareDisk)
  sort: { key: "n_positive", dir: "desc" },
  isPlaying: false,    // เล่นภาพเคลื่อนไหวแบบ Timelapse หรือไม่
  playTimer: null,

  // แผง confusion matrix — ตรึงอยู่กับ split ของตัวเอง ไม่ผูกกับช่วงเวลาในแถบกรอง
  confusion: null,          // ผล /api/forecast/confusion-matrix ล่าสุด (sweep ทั้งกริด)
  confusionModel: "lstm",   // "lstm" | "baseline"
  confusionSplit: "test",   // "val" | "test"
  confusionIndex: null,     // ตำแหน่งบนกริดที่สไลเดอร์ชี้อยู่ตอนนี้
  confusionCell: null,      // ช่องที่คลิกค้างไว้ ("tp"|"fp"|"fn"|"tn") — null = ไม่มีลิสต์เปิดอยู่
  confusionSamples: null,   // ผล /api/forecast/confusion-matrix/samples ล่าสุด ("loading" ระหว่างรอ)
};

/* ── อ่านสีจาก CSS custom property — single source of truth ─────── */

const css = getComputedStyle(document.documentElement);
const cssVar = (name) => css.getPropertyValue(name).trim();

const THEME = {
  textPrimary: cssVar("--text-primary"),
  textSecondary: cssVar("--text-secondary"),
  textMuted: cssVar("--text-muted"),
  gridline: cssVar("--gridline"),
  baseline: cssVar("--baseline"),
  // --surface-1 เป็นสีโปร่งแสง (การ์ดกระจก) ซึ่ง Plotly เรนเดอร์ได้ไม่ดี
  // จึงมีคู่แฝดแบบทึบไว้ใช้กับ hoverlabel และวงแหวนรอบ marker
  surface1: cssVar("--surface-1-solid"),
  border: cssVar("--border"),
  brand: cssVar("--brand"),
  series1: cssVar("--series-1"),
  series1Wash: cssVar("--series-1-wash"),
  statusCritical: cssVar("--status-critical"),
  seq: [cssVar("--seq-1"), cssVar("--seq-2"), cssVar("--seq-3"), cssVar("--seq-4")],
  // ramp ของช่องพลังงานโปรตอน เรียง >1 → >100 MeV (เข้ม → สว่าง = พลังงานสูงขึ้น)
  pflux: [cssVar("--pflux-1"), cssVar("--pflux-2"), cssVar("--pflux-3"), cssVar("--pflux-4")],
  // สีประจำช่อง AIA — คีย์ตรงกับที่ backend ส่งมาใน layers[].key
  aia: {
    "171": cssVar("--aia-171"),
    "304": cssVar("--aia-304"),
    "1600": cssVar("--aia-1600"),
  },
  fontSans: cssVar("--font-sans"),
  fontMono: cssVar("--font-mono"),
};

/* ── ธีมกราฟร่วม ──────────────────────────────────────────────── */

/** เส้นกริด/เส้นฐานของแกน — ค่าตั้งต้นที่ทุกกราฟใช้ร่วมกัน */
const AXIS_BASE = { gridcolor: THEME.gridline, zerolinecolor: THEME.baseline };

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: THEME.textSecondary, family: THEME.fontSans, size: 11 },
  margin: { l: 52, r: 18, t: 12, b: 40 },

  // Plotly *เขียนค่าที่คำนวณเองกลับลงใน object แกนที่รับไป* (autorange, range, type, …)
  // ถ้าทุกกราฟชี้ไปที่ก้อนเดียวกัน ค่าที่กราฟก่อนหน้าทิ้งไว้จะรั่วไปทับกราฟถัดไป —
  // เคยทำให้ range ที่ตั้งไว้ชัด ๆ ถูก autorange:true ที่ตกค้างจาก GOES X-ray เขี่ย
  // ทิ้งแบบเงียบ ๆ  getter คืนสำเนาใหม่ทุกครั้งที่ถูกอ่าน (รวมถึงตอน spread) จึงกัน
  // ได้ทุกจุดเรียกโดยไม่ต้องไล่แก้ทีละที่
  get xaxis() { return { ...AXIS_BASE }; },
  get yaxis() { return { ...AXIS_BASE }; },
  showlegend: false,
  hoverlabel: {
    bgcolor: THEME.surface1,
    bordercolor: THEME.border,
    font: { family: THEME.fontSans },
  },
};

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/** ระดับฟลักซ์พื้นหลังของดวงอาทิตย์ที่สงบ — ใช้เป็นฐานของ light curve */
const BACKGROUND_FLUX = 1e-8;

/** เส้นแบ่งคลาส GOES — ทำให้อ่านแกน log ได้โดยไม่ต้องแปลงเลขในใจ */
const GOES_LEVELS = [
  { flux: 1e-7, label: "B" }, { flux: 1e-6, label: "C" },
  { flux: 1e-5, label: "M" }, { flux: 1e-4, label: "X" },
];

const SEQ_INDEX = { B: 0, C: 1, M: 2, X: 3 };

/** ศัพท์คู่ของ confusion matrix — สถิติ (TP/FP/FN/TN) + วงการ space weather (hit/…)
 *  role คือ modifier class ที่กำหนดสี ("correct-negative" ไม่มีสีเด่นโดยตั้งใจ —
 *  เป็นช่องที่มีจำนวนมากที่สุดแต่สื่อสาระน้อยที่สุด) */
const CM_CELL_META = {
  tp: { stat: "TP", term: "hit",              role: "hit" },
  fp: { stat: "FP", term: "false alarm",      role: "false-alarm" },
  fn: { stat: "FN", term: "miss",             role: "miss" },
  tn: { stat: "TN", term: "correct negative", role: "correct-negative" },
};

/* ── ตัวช่วย ─────────────────────────────────────────────────── */

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `คำขอล้มเหลว (HTTP ${response.status})`);
  }
  return payload;
}

function el(id) { return document.getElementById(id); }

function showBanner(html) {
  const banner = el("banner");
  banner.innerHTML = html;
  banner.classList.remove("hidden");
}

function formatDateTime(iso) {
  return iso.replace("T", " ").slice(0, 16);
}

/** จัดรูปตัวเลขที่มีช่วงค่ากว้างมาก (SHARP parameters ต่างกันหลาย order of magnitude) */
function formatValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1e5 || magnitude < 1e-3) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(1);
  return value.toFixed(3);
}

const RISK_LABEL = {
  low: "ต่ำ", moderate: "ปานกลาง", elevated: "สูงกว่าเกณฑ์", high: "สูงมาก",
};

/** ขนาดจุดตามลอการิทึมของฟลักซ์ — flare แรงจึงเด่นออกมาชัดเจน (ขั้นต่ำ 8px) */
function markerSize(flux) {
  return 8 + 2.2 * (Math.log10(flux) + 7);
}

function seqColor(goesClass) {
  return THEME.seq[SEQ_INDEX[goesClass[0]] ?? 0];
}

/** นับขึ้นสู่ค่าเป้าหมาย — ใช้กับตัวเลขสรุปใน hero เท่านั้น
 *  ระบบที่ตั้งค่าลดการเคลื่อนไหวไว้จะได้ค่าปลายทางทันทีโดยไม่วิ่ง */
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function countTo(node, target, format) {
  if (REDUCED_MOTION || !Number.isFinite(target)) {
    node.textContent = format(target);
    return;
  }
  const duration = 700;
  const start = performance.now();
  function step(now) {
    const progress = Math.min(1, (now - start) / duration);
    // ease-out cubic — เร็วตอนต้น ช้าตอนใกล้ค่าจริง
    node.textContent = format(target * (1 - Math.pow(1 - progress, 3)));
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── สถานะระบบ ───────────────────────────────────────────────── */

async function loadHealth() {
  const health = await fetchJson("/api/health");
  state.health = health;

  const flags = {
    forecast: health.forecast_model,
    segmentation: health.segmentation_model,
    sequences: health.sequence_store,
  };

  // คลังโปรตอนเป็นข้อมูลภายนอกที่อาจไม่มีในเครื่องนี้ — ซ่อนทั้งแผงดีกว่าโชว์กราฟว่าง
  // (ทั้งการ์ดใหญ่ในแดชบอร์ดและกราฟย่อใต้จานสุริยะใน hero ที่ตอนนี้แสดงโปรตอนเหมือนกัน)
  el("protonCard").classList.toggle("hidden", !health.proton_flux);
  el("heroMinichart").classList.toggle("hidden", !health.proton_flux);
  document.querySelectorAll(".status").forEach((node) => {
    const ready = flags[node.dataset.key];
    node.className = `status status--${ready ? "ok" : "off"}`;
  });

  const missing = [];
  if (!health.sequence_store) {
    missing.push("<code>python backend/scripts/download_metadata.py</code> แล้ว <code>python backend/scripts/build_sequences.py</code>");
  } else if (!health.forecast_model) {
    missing.push("<code>python backend/scripts/train_lstm.py</code>");
  }
  if (!health.segmentation_model && health.n_frames === 0) {
    missing.push("<code>python backend/scripts/download_images.py</code> แล้ว <code>python backend/scripts/build_masks.py</code>");
  }
  if (missing.length) {
    showBanner(`ยังใช้งานได้ไม่ครบทุกส่วน — รันขั้นตอนต่อไปนี้เพื่อเปิดใช้: ${missing.join(" · ")}`);
  }

  return health;
}

/** เติมตัวเลขที่มาจาก config/metrics จริง — footer, ช่วงข้อมูลใน hero, ตารางผลใน about */
async function loadInfo() {
  let info;
  try {
    info = await fetchJson("/api/info");
  } catch {
    return;   // ไม่สำคัญพอที่จะรบกวนผู้ใช้
  }
  state.info = info;

  const tss = info.forecast?.metrics?.test?.tss;
  const dice = info.segmentation?.metrics?.test?.dice;

  const parts = [];
  if (tss !== undefined && tss !== null) parts.push(`LSTM test TSS = ${tss.toFixed(4)}`);
  if (dice !== undefined && dice !== null) parts.push(`U-Net test Dice = ${dice.toFixed(4)}`);
  parts.push(`พยากรณ์ล่วงหน้า ${info.data.horizon_hours} ชม. · เกณฑ์ ≥ ${info.data.positive_class}`);
  el("modelSummary").textContent = parts.join("  ·  ");

  // ตารางผลใน about เขียนค่าจากรายงานไว้เป็นค่าเริ่มต้น — ทับด้วยค่าจริงถ้ามี checkpoint
  if (tss !== undefined && tss !== null) el("lstmTss").textContent = tss.toFixed(3);
  if (dice !== undefined && dice !== null) {
    el("unetDice").textContent = `Dice ${dice.toFixed(3)}`;
  }

  const range = info.data.time_range;
  if (range?.start && range?.end) {
    el("heroRange").textContent = `${range.start.slice(0, 4)}–${range.end.slice(2, 4)}`;
  }
  el("heroRangeMeta").textContent =
    `${info.data.n_sequences.toLocaleString()} sample · ${state.harps.length.toLocaleString()} HARP`;
}

/* ── Confusion matrix ─────────────────────────────────────────
 * แผงนี้ตรึงอยู่กับ split ของตัวเอง ไม่ขยับตามช่วงเวลาในแถบกรอง (ต่างจากการ์ด
 * อื่นทั้งหมดในหน้านี้) จึงโหลดครั้งเดียวตอน init แล้วโหลดใหม่เฉพาะตอนสลับ
 * โมเดล/split เท่านั้น — การนับ TP/FP/TN/FN ทั้งหมดมาจาก backend
 * (sunseg.metrics.threshold_sweep) ฝั่งนี้แค่เปิด index ตามสไลเดอร์แล้วคูณ/หารเลข
 * ที่ backend นับมาให้แล้ว
 */

async function loadConfusionMatrix() {
  const body = el("cmBody");
  body.innerHTML = `<p class="empty">กำลังโหลด…</p>`;
  state.confusionCell = null;
  state.confusionSamples = null;

  try {
    state.confusion = await fetchJson(
      `/api/forecast/confusion-matrix?model=${state.confusionModel}&split=${state.confusionSplit}`
    );
    state.confusionIndex = state.confusion.frozen_index;
    renderConfusionMatrix();
  } catch (error) {
    state.confusion = null;
    body.innerHTML = `<p class="empty">${error.message}</p>`;
  }
}

/** คลิกช่องหนึ่งของตาราง — สลับเปิด/ปิดลิสต์ sample ของช่องนั้น
 *  threshold ที่ส่งไปคือตำแหน่งสไลเดอร์ ณ ตอนคลิก ไม่ใช่ค่าที่ freeze ไว้เสมอไป
 *  เพราะผู้ใช้อาจลากสไลเดอร์ไปสำรวจก่อนแล้วค่อยคลิกดูว่าช่องนั้นมีใครอยู่บ้าง */
async function onConfusionCellClick(cell) {
  if (state.confusionCell === cell) {
    state.confusionCell = null;
    state.confusionSamples = null;
    renderConfusionMatrix();
    return;
  }

  const data = state.confusion;
  const threshold = data.thresholds[state.confusionIndex];

  state.confusionCell = cell;
  state.confusionSamples = "loading";
  renderConfusionMatrix();

  try {
    state.confusionSamples = await fetchJson(
      `/api/forecast/confusion-matrix/samples?model=${data.model}&split=${data.split}` +
      `&threshold=${threshold}&cell=${cell}&limit=200`
    );
  } catch (error) {
    state.confusionSamples = { error: error.message };
  }
  renderConfusionMatrix();
}

/** สร้าง markup ของลิสต์ sample ใต้ตาราง — เรียกเฉพาะตอนมีช่องที่เลือกอยู่ */
function renderCmSamplesHtml() {
  const cell = state.confusionCell;
  const meta = CM_CELL_META[cell];
  const result = state.confusionSamples;

  let body;
  if (result === "loading") {
    body = `<p class="empty">กำลังโหลด…</p>`;
  } else if (result?.error) {
    body = `<p class="empty">${result.error}</p>`;
  } else if (!result.samples.length) {
    body = `<p class="empty">ไม่มี sample ในช่องนี้ ณ threshold ปัจจุบัน</p>`;
  } else {
    body = `
      <div class="table-wrap">
        <table class="cm__sample-table">
          <thead>
            <tr><th>HARP</th><th>NOAA AR</th><th>เวลาออกพยากรณ์</th><th class="num">ความน่าจะเป็น</th></tr>
          </thead>
          <tbody>
            ${result.samples.map((row) => `
              <tr data-harp="${row.harpnum}" data-time="${row.issue_time}" tabindex="0">
                <td>${row.harpnum}</td>
                <td>${row.noaa_ar ?? "—"}</td>
                <td>${formatDateTime(row.issue_time)}</td>
                <td class="num">${(row.probability * 100).toFixed(1)}%</td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>`;
  }

  const count = result && !result.error && result !== "loading"
    ? `<span class="hint">· ${result.total.toLocaleString()} รายการ` +
      `${result.samples.length < result.total ? ` (แสดง ${result.samples.length.toLocaleString()} รายการแรกที่มั่นใจที่สุด)` : ""}</span>`
    : "";

  return `
    <div class="cm__samples">
      <div class="cm__samples-head">
        <h4>${meta.stat} <span class="hint">${meta.term}</span> ${count}</h4>
        <button type="button" class="cm__samples-close" id="cmSamplesClose" aria-label="ปิดรายการ">ปิด ✕</button>
      </div>
      ${body}
    </div>`;
}

/** คลิก sample หนึ่งแถว -> เลือก HARP นั้นใน dashboard และขยับช่วงวันที่ให้ครอบเวลาออก
 *  พยากรณ์ของ sample (±7 วัน) — ถ้าตั้งแค่ HARP อย่างเดียวผู้ใช้จะเจอกราฟว่าง เพราะ
 *  ช่วงวันที่ปริยายของ filterbar (ต.ค. 2014) มักอยู่คนละปีกับ sample ที่ดูอยู่ */
async function jumpToConfusionSample(harpnum, isoTime) {
  const center = new Date(isoTime);
  const spanMs = 7 * 24 * 3600 * 1000;
  el("goesStart").value = new Date(center.getTime() - spanMs).toISOString().slice(0, 10);
  el("goesEnd").value = new Date(center.getTime() + spanMs).toISOString().slice(0, 10);

  await selectHarp(harpnum, { syncRange: false });
  await applyRange();
  el("dashboard").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth" });
}

function renderConfusionMatrix() {
  const data = state.confusion;
  const idx = state.confusionIndex;
  if (!data || idx === null) return;

  const tp = data.tp[idx], fp = data.fp[idx], tn = data.tn[idx], fn = data.fn[idx];
  const total = tp + fp + tn + fn;
  const pct = (n) => (total ? `${(100 * n / total).toFixed(1)}%` : "—");

  // ตัวชี้วัดคำนวณจากจำนวนนับทั้งสี่ค่าด้วยเลขคณิตล้วน — ห้ามนับ TP/FP/TN/FN
  // ซ้ำที่นี่ (ดูเหตุผลใน spec: ฝั่ง JS ของโปรเจคนี้ไม่มี test คุมเลยสักตัว)
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const fpr = fp + tn ? fp / (fp + tn) : 0;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const tssValue = recall - fpr;
  const hss2Num = 2 * (tp * tn - fn * fp);
  const hss2Den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn);
  const hss2 = hss2Den ? hss2Num / hss2Den : 0;

  const threshold = data.thresholds[idx];
  const isFrozen = idx === data.frozen_index;
  const modelLabel = data.model === "lstm" ? "LSTM" : "logistic baseline";
  const signed = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(4)}`;

  el("cmBody").innerHTML = `
    ${data.split === "val" ? `
      <p class="cm__warn"><b>val</b> คือชุดที่ใช้เลือก threshold นี้เอง — ตัวเลขในโหมดนี้ไม่ใช่ผลที่ควรอ้างเป็นผลงาน ดูที่ <b>test</b> แทน</p>
    ` : ""}
    <div class="cm__grid">
      <table class="cm__table" aria-label="confusion matrix ของ ${modelLabel} บนชุด ${data.split}">
        <thead>
          <tr><th></th><th colspan="2" class="cm__group">เกิดขึ้นจริง</th></tr>
          <tr><th></th><th class="num">positive</th><th class="num">negative</th></tr>
        </thead>
        <tbody>
          <tr>
            <th class="cm__rowhead">ทำนาย<br>positive</th>
            ${cmCell("tp", tp, pct(tp))}
            ${cmCell("fp", fp, pct(fp))}
          </tr>
          <tr>
            <th class="cm__rowhead">ทำนาย<br>negative</th>
            ${cmCell("fn", fn, pct(fn))}
            ${cmCell("tn", tn, pct(tn))}
          </tr>
        </tbody>
      </table>

      <div class="cm__side">
        <div class="cm__slider">
          <div class="cm__slider-head">
            <span>threshold = <b>${threshold.toFixed(3)}</b></span>
            <button type="button" class="cm__reset" id="cmReset" ${isFrozen ? "disabled" : ""}>
              กลับไป ${data.frozen_threshold.toFixed(3)}
            </button>
          </div>
          <input type="range" id="cmSlider" min="0" max="${data.thresholds.length - 1}" step="1" value="${idx}"
                 aria-label="threshold ของ ${modelLabel}">
          <div class="cm__slider-note">${isFrozen ? "ค่าที่ freeze ไว้จาก validation (ค่าที่ระบบใช้จริง)" : "ต่างจากค่าที่ freeze ไว้ — ดูเพื่อสำรวจเท่านั้น"}</div>
        </div>

        <dl class="cm__metrics">
          <div><dt>TSS</dt><dd>${signed(tssValue)}</dd></div>
          <div><dt>recall</dt><dd>${recall.toFixed(3)}</dd></div>
          <div><dt>precision</dt><dd>${precision.toFixed(3)}</dd></div>
          <div><dt>HSS2</dt><dd>${signed(hss2)}</dd></div>
        </dl>

        <p class="cm__meta">
          AUC ${data.auc.toFixed(4)} <span class="hint">(ไม่ขึ้นกับ threshold)</span><br>
          ${data.n.toLocaleString()} sample · positive ${data.n_positive.toLocaleString()}
        </p>
      </div>
    </div>
    ${state.confusionCell ? renderCmSamplesHtml() : ""}`;

  // เปลี่ยน threshold แล้วปิดลิสต์ sample ที่เปิดค้างไว้เสมอ — กันไม่ให้แสดงรายการ
  // ของ threshold เก่าทับกับตัวเลขของ threshold ใหม่ และกันไม่ให้ยิง network ทุกครั้ง
  // ที่ลาก (ลิสต์เรียกใหม่เฉพาะตอนคลิกช่องอีกครั้งเท่านั้น)
  el("cmSlider").addEventListener("input", (event) => {
    state.confusionIndex = Number(event.target.value);
    state.confusionCell = null;
    state.confusionSamples = null;
    renderConfusionMatrix();
  });
  el("cmReset")?.addEventListener("click", () => {
    state.confusionIndex = data.frozen_index;
    state.confusionCell = null;
    state.confusionSamples = null;
    renderConfusionMatrix();
  });

  document.querySelectorAll(".cm__cell[data-cell]").forEach((cellEl) => {
    cellEl.addEventListener("click", () => onConfusionCellClick(cellEl.dataset.cell));
    cellEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onConfusionCellClick(cellEl.dataset.cell);
      }
    });
  });

  el("cmSamplesClose")?.addEventListener("click", () => {
    state.confusionCell = null;
    state.confusionSamples = null;
    renderConfusionMatrix();
  });

  document.querySelectorAll(".cm__sample-table tr[data-harp]").forEach((row) => {
    row.addEventListener("click", () => jumpToConfusionSample(Number(row.dataset.harp), row.dataset.time));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        jumpToConfusionSample(Number(row.dataset.harp), row.dataset.time);
      }
    });
  });
}

function cmCell(key, count, pctText) {
  const meta = CM_CELL_META[key];
  const selected = state.confusionCell === key;
  return `
    <td class="cm__cell cm__cell--${meta.role}${selected ? " cm__cell--selected" : ""}"
        data-cell="${key}" tabindex="0" role="button" aria-pressed="${selected}"
        aria-label="${meta.stat} (${meta.term}) ${count.toLocaleString()} sample — คลิกเพื่อดูรายการ">
      <span class="cm__cell-stat">${meta.stat}</span>
      <span class="cm__cell-term">${meta.term}</span>
      <span class="cm__cell-count">${count.toLocaleString()}</span>
      <span class="cm__cell-pct">${pctText}</span>
    </td>`;
}

/* ── รายการ active region ────────────────────────────────────── */

async function loadHarps() {
  const tbody = el("harpList");
  if (!state.health?.sequence_store) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty">ยังไม่มีข้อมูล sequence</td></tr>`;
    return;
  }

  try {
    // ดึงมาครั้งเดียวทั้งชุด แล้วกรอง/เรียงฝั่ง client — การติ๊ก "เฉพาะที่เคยเกิด flare"
    // หรือกดเรียงคอลัมน์จึงตอบสนองทันทีโดยไม่ต้องรอ backend
    state.harps = await fetchJson("/api/harps?limit=3000&only_flaring=false");
    renderHarps();
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty">${error.message}</td></tr>`;
  }
}

/** กรองตามคำค้น + สวิตช์ แล้วเรียงตามคอลัมน์ที่เลือกไว้ */
function visibleHarps() {
  const query = el("harpSearch").value.trim().toLowerCase();
  const onlyFlaring = el("onlyFlaring").checked;

  const rows = state.harps.filter((h) => {
    if (onlyFlaring && !h.n_positive) return false;
    if (!query) return true;
    return String(h.harpnum).includes(query) ||
           (h.noaa_ar && String(h.noaa_ar).includes(query));
  });

  const { key, dir } = state.sort;
  const sign = dir === "asc" ? 1 : -1;
  return rows.sort((a, b) => {
    // HARP ที่ยังไม่มีเลข NOAA ให้ตกไปท้ายตารางเสมอ ไม่ว่าจะเรียงทางไหน
    const left = a[key] ?? -Infinity;
    const right = b[key] ?? -Infinity;
    return left === right ? a.harpnum - b.harpnum : (left < right ? -sign : sign);
  });
}

function renderHarps() {
  const rows = visibleHarps();
  const tbody = el("harpList");

  el("harpCount").textContent =
    `${rows.length.toLocaleString()} จาก ${state.harps.length.toLocaleString()} active region`;

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty">ไม่พบ active region ที่ตรงกับคำค้น</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((h) => `
    <tr data-harp="${h.harpnum}" tabindex="0" class="${h.harpnum === state.selectedHarp ? "selected" : ""}">
      <td>${h.harpnum}</td>
      <td>${h.noaa_ar ?? "—"}</td>
      <td class="num">${h.n_samples.toLocaleString()}</td>
      <td class="num ${h.n_positive ? "flare-count" : ""}">${h.n_positive.toLocaleString()}</td>
    </tr>`).join("");

  tbody.querySelectorAll("tr[data-harp]").forEach((row) => {
    row.addEventListener("click", () => selectHarp(Number(row.dataset.harp), { syncRange: true }));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectHarp(Number(row.dataset.harp), { syncRange: true });
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const next = event.key === "ArrowDown"
          ? row.nextElementSibling : row.previousElementSibling;
        next?.focus();
      }
    });
  });
}

/** ตั้งคอลัมน์ที่ใช้เรียง — กดคอลัมน์เดิมซ้ำคือสลับทิศ */
function setSort(key) {
  const current = state.sort;
  state.sort = key === current.key
    ? { key, dir: current.dir === "desc" ? "asc" : "desc" }
    : { key, dir: key === "harpnum" || key === "noaa_ar" ? "asc" : "desc" };

  document.querySelectorAll(".sort").forEach((button) => {
    if (button.dataset.sort === state.sort.key) {
      button.dataset.active = "";
      button.dataset.dir = state.sort.dir;
    } else {
      delete button.dataset.active;
      delete button.dataset.dir;
    }
  });
  renderHarps();
}

/* ── พยากรณ์ของ AR ที่เลือก ──────────────────────────────────── */

async function selectHarp(harpnum, { syncRange = false } = {}) {
  state.selectedHarp = harpnum;
  renderHarps();

  const panel = el("riskPanel");
  panel.className = "risk";
  panel.innerHTML = `<p class="empty">กำลังคำนวณ…</p>`;

  try {
    const series = await fetchJson(`/api/forecast?harpnum=${harpnum}&limit=2000`);
    renderRisk(series);
    renderAttention(series.latest);
    renderFeatures(series.latest);
    // ปรับช่วงเวลาของทั้งหน้าให้ตรงกับอายุของ AR ที่เพิ่งเลือก แล้ววาดใหม่ทุกแผง
    if (syncRange && series.times.length) {
      el("goesStart").value = series.times[0].slice(0, 10);
      el("goesEnd").value = series.times[series.times.length - 1].slice(0, 10);
      await applyRange();
    }
    // ไม่มี else: ตัวเลขความเสี่ยงใน hero ผูกกับช่วงเวลา ไม่ใช่ AR ที่เลือก
    // การเลือก AR จึงไม่ต้องยิงคำขอนั้นซ้ำอีก
  } catch (error) {
    panel.className = "risk";
    panel.innerHTML = `<p class="empty">${error.message}</p>`;
  }
}

function renderRisk(series) {
  const panel = el("riskPanel");
  const latest = series.latest;
  if (!latest) {
    panel.className = "risk";
    panel.innerHTML = `<p class="empty">ไม่มีข้อมูล</p>`;
    return;
  }

  // สเกลแท่งเทียบกับ 3 เท่าของ threshold เพื่อให้เห็นความต่างชัด — ค่าดิบมักต่ำ
  // เพราะ positive จริงมีเพียง ~2% โมเดลจึงแทบไม่เคยให้ค่าเกิน 0.5
  const scale = Math.max(latest.threshold * 3, 0.05);
  const fillPct = Math.min(100, (latest.probability / scale) * 100);
  const thresholdPct = Math.min(100, (latest.threshold / scale) * 100);

  let outcome = "";
  if (latest.actual_label !== null && latest.actual_label !== undefined) {
    outcome = latest.actual_label === 1
      ? `<span class="outcome outcome--yes">ผลจริง: เกิด flare ≥ ${series.positive_class}</span>`
      : `<span class="outcome outcome--no">ผลจริง: ไม่เกิด flare</span>`;
  }

  panel.className = `risk risk--${latest.risk_level}`;
  panel.innerHTML = `
    <div class="risk__value" id="riskValue">0.0%</div>
    <div class="risk__label">${RISK_LABEL[latest.risk_level] ?? latest.risk_level}</div>
    <div class="risk__bar">
      <div class="risk__fill" style="width:0"></div>
      <div class="risk__threshold" style="left:${thresholdPct}%" title="เกณฑ์ตัดสิน"></div>
    </div>
    <div class="risk__meta">
      HARP ${series.harpnum}${series.noaa_ar ? ` · NOAA ${series.noaa_ar}` : ""}<br>
      ณ ${formatDateTime(latest.issue_time)}<br>
      โอกาสเกิด flare ≥ ${series.positive_class} ใน ${series.horizon_hours} ชม.<br>
      <span class="hint">เกณฑ์ตัดสิน ${(latest.threshold * 100).toFixed(1)}% (ปรับจาก validation)</span>
    </div>
    ${outcome}`;

  countTo(el("riskValue"), latest.probability * 100, (v) => `${v.toFixed(1)}%`);
  // ตั้งความกว้างในเฟรมถัดไปเพื่อให้ transition ของ CSS ได้ทำงาน (0 -> ค่าจริง)
  requestAnimationFrame(() => {
    const fill = panel.querySelector(".risk__fill");
    if (fill) fill.style.width = `${fillPct}%`;
  });
}

function renderAttention(latest) {
  if (!latest?.attention?.length) {
    Plotly.purge("attentionChart");
    return;
  }

  const n = latest.attention.length;
  // แกน x เป็นชั่วโมง "ก่อนเวลาออกพยากรณ์" — 0 คือปัจจุบัน, -23 คือเมื่อวาน
  const hours = latest.attention.map((_, i) => -(n - 1 - i));
  const peak = Math.max(...latest.attention);

  Plotly.react("attentionChart", [{
    x: hours,
    y: latest.attention,
    type: "bar",
    marker: {
      // ไล่สีตามน้ำหนัก: ชั่วโมงที่โมเดลสนใจมากที่สุดจะเป็นสีแบรนด์ ที่เหลือเป็นน้ำเงิน
      color: latest.attention.map((w) => (w >= peak * 0.92 ? THEME.brand : THEME.series1)),
    },
    hovertemplate: "%{x} ชม. ก่อนหน้า<br>น้ำหนัก %{y:.3f}<extra></extra>",
  }], {
    ...PLOT_LAYOUT,
    margin: { l: 46, r: 14, t: 6, b: 34 },
    xaxis: { ...PLOT_LAYOUT.xaxis, title: "ชั่วโมงก่อนเวลาพยากรณ์" },
    yaxis: { ...PLOT_LAYOUT.yaxis, title: "น้ำหนัก" },
    bargap: 0.25,
  }, PLOT_CONFIG);
}

function renderFeatures(latest) {
  const panel = el("featureList");
  if (!latest?.features) {
    panel.innerHTML = `<p class="empty">ไม่มีข้อมูล</p>`;
    return;
  }
  panel.innerHTML = Object.entries(latest.features)
    .map(([name, value]) => `
      <div class="feature">
        <span class="feature__name">${name}</span>
        <span class="feature__value">${formatValue(value)}</span>
      </div>`)
    .join("");
}

/* ── ความเสี่ยงสูงสุดในช่วงเวลาที่เลือก (ตัวเลขในhero) ────────── */

/** เดิมเป็นผลพลอยได้ของการ์ด small multiples ที่ถูกแทนที่ไปแล้ว
 *  แยกออกมาเป็นฟังก์ชันของตัวเองเพราะตัวเลขนี้ผูกกับ *ช่วงเวลา* ไม่ใช่ AR ที่เลือก */
async function loadHeroRisk() {
  if (!state.health?.forecast_model || !state.health?.sequence_store) return;

  const end = el("goesEnd").value;
  if (!end) return;

  try {
    const ranked = await fetchJson(`/api/forecast/at?time=${end}T00:00:00&tolerance_hours=24`);
    updateHeroRisk(ranked[0]);
  } catch {
    // ตัวเลขใน hero เป็นข้อมูลเสริม — ล้มเหลวเงียบ ๆ ดีกว่าขึ้น error คาหน้าแรก
  }
}

/* ── ความเข้มแสงราย AR ตามเวลา แยกตามชั้นบรรยากาศ ───────────── */

/** ลำดับแถวจากบนลงล่าง = จากรากสนามแม่เหล็กที่โฟโตสเฟียร์ ไล่ขึ้นชั้นบรรยากาศ
 *  ทำให้กราฟอ่านเป็นภาพตัดขวางของดวงอาทิตย์ได้ตรง ๆ — แถวบนสุดเป็นสนามแม่เหล็ก (เชิงเส้น)
 *  ที่เหลือเป็น AIA (log) เพราะความเข้มแสงตอน flare พุ่งขึ้นหลายสิบเท่าในไม่กี่นาที
 *  สเกลเชิงเส้นจะทำให้เห็นแค่ยอดพีคแหลม ๆ แล้วรายละเอียดช่วงเงียบจมหายไป */
const EXTRACTION_ROWS = [
  { key: "b_peak", label: "HMI |B|", region: "สนามแม่เหล็กที่โฟโตสเฟียร์", scale: "linear" },
  { key: "171", label: "171 Å", region: "โคโรนา", scale: "log" },
  { key: "304", label: "304 Å", region: "โครโมสเฟียร์", scale: "log" },
  { key: "1600", label: "1600 Å", region: "โฟโตสเฟียร์", scale: "log" },
];

const EXTRACTION_DOMAINS = [
  [0.775, 0.975],
  [0.530, 0.730],
  [0.285, 0.485],
  [0.040, 0.240],
];

/** สีของแต่ละ AR — ใช้ชุดเดียวกันทั้งสามแถว เพื่อให้ตาไล่ AR ดวงเดิมข้ามชั้นได้ */
const AR_COLORS = [
  THEME.series1, THEME.seq[3], THEME.seq[1],
  THEME.brand, THEME.pflux[2], THEME.statusCritical,
];

/** ครึ่งความกว้างของช่วงที่ปุ่ม "กระโดด" ตั้งให้ (วัน)
 *
 *  ต้องกว้างพอให้ได้ **อย่างน้อยสองเฟรมที่ติดกัน** ไม่ใช่แค่เฟรมเดียว — tracker ต้องเห็น
 *  AR ดวงเดิมติดกัน 2 เฟรมจึงยืนยันเป็น track (n_confirm) ถ้ากระโดดไปแล้วได้เฟรมเดียว
 *  กราฟก็ยังว่างอยู่ดี ซึ่งทำให้ปุ่มดูเหมือนกดไม่ติด
 *
 *  วัดรอบเฟรม 2012-06-24 บนคลังหลักที่เก็บภาพทุก 7 วัน:
 *  ±7 วัน → 2 เฟรม 0 track · ±14 → 3 เฟรม 3 track · **±21 → 5 เฟรม 6 track** · ±30 → 7 เฟรม 6 track
 *  ±21 จึงเป็นจุดที่ได้ track เต็มจำนวนโดยไม่คำนวณเกินจำเป็น */
const SNAP_DAYS = 21;

/**
 * แสดงสถานะว่างในพื้นที่กราฟ
 * @param {string} message - ข้อความอธิบาย
 * @param {string|null} nearestFrame - วันที่ YYYY-MM-DD ของเฟรมที่ใกล้ที่สุด (ถ้ามี)
 */
function extractionEmpty(message, nearestFrame = null) {
  Plotly.purge("extractionChart");
  let html = `<p class="empty">${message}</p>`;
  if (nearestFrame) {
    const [year, month, day] = nearestFrame.split("-").map(Number);
    // แยกส่วนแล้วสร้างเป็นเวลาท้องถิ่น — new Date("2012-06-24") ตีความเป็นเที่ยงคืน UTC
    // พอดึงกลับด้วย getDate() ที่เป็นเวลาท้องถิ่น เครื่องที่อยู่ทางตะวันตกของ UTC จะได้วันก่อนหน้า
    const d = new Date(year, month - 1, day);
    const pad = (n) => String(n).padStart(2, "0");
    const fmt = (dt) => `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
    const snapStart = fmt(new Date(d.getTime() - SNAP_DAYS * 86400000));
    const snapEnd = fmt(new Date(d.getTime() + SNAP_DAYS * 86400000));
    html += `
      <button type="button" class="btn btn--sm btn--snap" id="snapToFrameBtn"
              data-start="${snapStart}" data-end="${snapEnd}"
              title="ขยายช่วงเวลาให้รอบเฟรมที่ใกล้ที่สุด (${nearestFrame})">
        📅 กระโดดไปช่วง ${nearestFrame} ±${SNAP_DAYS} วัน
      </button>`;
  }
  el("extractionChart").innerHTML = html;

  // ผูก event listener หลัง inject HTML
  //
  // เรียก applyRange() ตรง ๆ ไม่ใช่ยิง .click() ใส่ปุ่มในแถบกรอง — การอ้าง id ของปุ่ม
  // ทำให้พังเงียบมาแล้วครั้งหนึ่ง (el() คืน null เมื่อไม่เจอ พอต่อ ?.click() ก็ไม่มีอะไร
  // เกิดขึ้นและไม่มี error ให้เห็นด้วย) applyRange เป็นตัวเดียวกับที่ปุ่ม "แสดงข้อมูล" ผูกไว้
  const snapBtn = el("snapToFrameBtn");
  if (snapBtn) {
    snapBtn.addEventListener("click", () => {
      const { start, end } = snapBtn.dataset;
      el("goesStart").value = start;
      el("goesEnd").value = end;
      applyRange();       // โหลดใหม่ทั้ง GOES + เฟรม + กราฟความเข้มแสง
    });
  }
}

/** ดึงความเข้มแสงตามเวลาของทั้งช่วงที่กรองไว้ แล้ววาด
 *
 *  ผูกกับ *ช่วงเวลา* ไม่ใช่เฟรมเดียว จึงเรียกจาก applyRange() คู่กับกราฟอื่น ๆ
 *  ส่วนเฟรมที่การ์ดซ้ายกำลังแสดงอยู่จะถูกทำเครื่องหมายเป็นเส้นแนวตั้ง */
async function loadExtraction() {
  const start = el("goesStart").value;
  const end = el("goesEnd").value;
  if (!start || !end) return;

  const hint = el("extractionHint");
  hint.textContent = "กำลังคำนวณ…";

  const requestId = ++state.extractionRequest;
  try {
    const data = await fetchJson(
      `/api/intensity-series?start=${start}&end=${end}`
      + `&use_ground_truth=${el("useTruth").checked}&top=6`
    );
    if (requestId !== state.extractionRequest) return;

    state.extraction = data;
    renderExtraction(data);
  } catch (error) {
    if (requestId !== state.extractionRequest) return;
    state.extraction = null;
    hint.textContent = "";
    extractionEmpty(error.message);
  }
}

function syncArOptions(tracks) {
  const select = el("extractionArSelect");
  if (!select) return;
  const currentVal = select.value;

  // เรียงลำดับ AR จาก b_peak สูงสุดไปต่ำสุด เพื่อให้ AR ที่แรงสุดอยู่บนสุด
  const sorted = [...tracks].sort((a, b) => {
    const bA = a.b_peak && a.b_peak.length ? Math.max(...a.b_peak) : 0;
    const bB = b.b_peak && b.b_peak.length ? Math.max(...b.b_peak) : 0;
    return bB - bA;
  });

  let options = '<option value="all">ทุก AR (แสดงรวม)</option>';
  sorted.forEach((t) => {
    const bPeakMax = t.b_peak && t.b_peak.length ? Math.max(...t.b_peak) : 0;
    const bStr = bPeakMax > 0 ? ` · ${bPeakMax.toFixed(0)} G` : "";
    const areaKPx = (t.max_area_px / 1000).toFixed(1);
    options += `<option value="${t.track_id}">${t.label}${bStr} (${areaKPx}k px)</option>`;
  });
  select.innerHTML = options;

  // คงค่าที่เลือกไว้ก่อนหน้า หรือเลือก AR ที่ b_peak สูงสุดเป็นค่าเริ่มต้น
  if (tracks.some((t) => String(t.track_id) === currentVal)) {
    select.value = currentVal;
  } else if (sorted.length > 0) {
    select.value = String(sorted[0].track_id); // AR ที่ b_peak สูงสุด
  } else {
    select.value = "all";
  }
}

function renderExtraction(data) {
  const hint = el("extractionHint");
  const tracks = (data.tracks || []).filter((t) => t.n_points > 0);

  if (!tracks.length) {
    hint.textContent = "";
    el("extractionNote").classList.add("hidden");
    extractionEmpty(
      data.note
        || (data.channels?.some((c) => c.available)
          ? "ไม่พบ active region ในช่วงเวลานี้"
          : "ช่วงเวลานี้ยังไม่มีข้อมูลสำหรับ active region"),
      data.nearest_frame || null,  // ส่งเฟรมใกล้สุดให้แสดงปุ่ม snap
    );
    return;
  }

  // อัปเดต dropdown เลือก AR เสมอ (เมื่อโหลดข้อมูลชุดใหม่ รายการ AR จะเปลี่ยนไป)
  syncArOptions(tracks);

  const selectedArId = el("extractionArSelect") ? el("extractionArSelect").value : "all";
  const selectedMetric = el("extractionMetricSelect") ? el("extractionMetricSelect").value : "b_peak";

  el("extractionChart").innerHTML = "";

  // กรองเฉพาะ track ที่เลือก หรือเอาทั้งหมด
  const filteredTracks = selectedArId === "all"
    ? tracks
    : tracks.filter((t) => String(t.track_id) === selectedArId);

  const displayTracks = filteredTracks.length ? filteredTracks : tracks;
  const targetTrack = displayTracks[0];

  // รวบรวมเส้นประเหตุการณ์ Flare ในช่วงเวลานี้ — สีตามระดับความรุนแรงด้วย seqColor()
  // ตัวเดียวกับที่จานสุริยะและกราฟ X-ray ใช้ ไม่ประดิษฐ์ชุดสีใหม่ขึ้นมาอีกชุด (B<C<M<X
  // ไล่เข้มขึ้นตามความแรง) ป้ายกำกับ (ตัวอักษร class) แสดงเฉพาะ M/X เพื่อไม่ให้แน่นเกินไป
  // เมื่อช่วงเวลาที่เลือกยาวและมี flare ระดับ C จำนวนมาก — เส้น C ยังขึ้นแต่ไม่มีป้าย
  const flareShapes = [];
  const flareAnnotations = [];
  if (state.goes && state.goes.events) {
    state.goes.events.forEach((evt) => {
      if (!evt.peak_time || !evt.goes_class) return;
      const color = seqColor(evt.goes_class);
      const major = evt.goes_class[0] === "M" || evt.goes_class[0] === "X";
      flareShapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: evt.peak_time, x1: evt.peak_time, y0: 0, y1: 1,
        line: { color, width: major ? 1.4 : 1, dash: "dash" },
        opacity: major ? 0.85 : 0.4,
      });
      if (major) {
        flareAnnotations.push({
          xref: "x", yref: "paper",
          x: evt.peak_time, y: 0.995,
          text: evt.goes_class,
          textangle: -90,
          showarrow: false,
          font: { size: 9, color, family: THEME.fontMono },
          xanchor: "center", yanchor: "top",
        });
      }
    });
  }

  // กรณี 1: แสดงทุกชั้นบรรยากาศ + สนามแม่เหล็กแบบ subplots ซ้อนกัน (แถวบนสุด = รากสนาม
  // แม่เหล็กที่โฟโตสเฟียร์ เชิงเส้น · ที่เหลือ = AIA ไล่ขึ้นชั้นบรรยากาศ สเกล log)
  if (selectedMetric === "all_aia") {
    const traces = [];
    const annotations = [];

    EXTRACTION_ROWS.forEach((row, index) => {
      const axis = index === 0 ? "" : String(index + 1);
      const isLog = row.scale === "log";

      displayTracks.forEach((track, order) => {
        const raw = row.key === "b_peak" ? (track.b_peak || []) : ((track.series && track.series[row.key]) || []);
        if (!raw.some((v) => v !== null && v > 0)) return;

        // สเกล log ห้ามรับค่า 0 หรือติดลบ — พลาดตรงนี้แล้ว Plotly จะลากเส้นข้ามจุดนั้นแบบ
        // เงียบ ๆ ไม่ error แต่รูปทรงเพี้ยนไปจากที่ควรเป็น จึงกรองเป็น null (=ช่องว่างจริง)
        const values = isLog ? raw.map((v) => (v && v > 0 ? v : null)) : raw;

        traces.push({
          type: "scatter",
          mode: "lines+markers",
          line: { shape: "spline", smoothing: 0.6, width: 2, color: AR_COLORS[order % AR_COLORS.length] },
          marker: { size: 5, color: AR_COLORS[order % AR_COLORS.length] },
          connectgaps: false,
          x: track.times,
          y: values,
          xaxis: "x",
          yaxis: `y${axis}`,
          name: track.label,
          legendgroup: track.label,
          showlegend: index === 0,
          hovertemplate: row.key === "b_peak"
            ? `<b>${track.label}</b> · ${row.label}<br>%{x|%d %b %H:%M}<br>%{y:,.0f} G<extra></extra>`
            : `<b>${track.label}</b> · ${row.label}<br>%{x|%d %b %H:%M}<br>%{y:.1f} DN/s<extra></extra>`,
        });
      });

      annotations.push({
        xref: "paper", yref: `y${axis} domain`,
        x: 0, y: 1, xanchor: "left", yanchor: "top",
        text: `<b>${row.label}</b> · ${row.region}${isLog ? " (log)" : ""}`,
        showarrow: false,
        font: { size: 11, color: THEME.textSecondary, family: THEME.fontSans },
      });
    });

    // ป้ายชื่อรวมบนสุดของกราฟ — บอกว่ากำลังดู AR ดวงไหน (หรือกี่ดวงถ้าเลือก "ทุก AR")
    const titleLabel = displayTracks.length === 1 ? displayTracks[0].label : `${displayTracks.length} AR`;
    annotations.push({
      xref: "paper", yref: "paper",
      x: 0, y: 1.055, xanchor: "left", yanchor: "bottom",
      text: `<b>${titleLabel}</b> · ความเข้มแสงต่อชั้นบรรยากาศตามเวลา`,
      showarrow: false,
      font: { size: 12, color: THEME.textPrimary, family: THEME.fontSans },
    });

    const layout = {
      ...PLOT_LAYOUT,
      margin: { l: 56, r: 14, t: 30, b: 34 },
      showlegend: displayTracks.length > 1,
      legend: { orientation: "h", y: -0.12, x: 0, font: { size: 10 } },
      xaxis: {
        ...PLOT_LAYOUT.xaxis,
        type: "date",
        anchor: `y${EXTRACTION_ROWS.length}`,
        tickfont: { size: 10 },
      },
      annotations: [...annotations, ...flareAnnotations],
      shapes: [...flareShapes, ...frameMarkerShapes()],
    };

    EXTRACTION_DOMAINS.forEach((domain, index) => {
      const axis = index === 0 ? "" : String(index + 1);
      const isLog = EXTRACTION_ROWS[index].scale === "log";
      layout[`yaxis${axis}`] = {
        ...PLOT_LAYOUT.yaxis,
        domain,
        type: isLog ? "log" : "linear",
        rangemode: isLog ? undefined : "tozero",
        dtick: isLog ? 1 : undefined,
        exponentformat: isLog ? "power" : undefined,
        tickfont: { size: 10 },
      };
    });

    Plotly.react("extractionChart", traces, layout, PLOT_CONFIG);
  }
  // กรณี 2: กราฟเดี่ยว Single Metric (เช่น b_peak, 171, 304, 1600, flux) ต่อ AR ID แบบ Image 2
  else {
    const traces = [];
    let yTitle = "ค่าความเข้ม";
    let yUnit = "";
    let metricLabel = "b_peak (max-in-mask)";
    let lineColor = "#8ba2ff"; // Neon periwinkle / cyber blue
    let fillColor = "rgba(138, 162, 255, 0.12)";

    if (selectedMetric === "b_peak") {
      yTitle = "B_peak (Gauss)";
      yUnit = " G";
      metricLabel = "b_peak (max-in-mask)";
      lineColor = "#8ba2ff";
      fillColor = "rgba(138, 162, 255, 0.10)";
    } else if (selectedMetric === "flux") {
      yTitle = "Total Unsigned Flux (Mx)";
      yUnit = " Mx";
      metricLabel = "Total Magnetic Flux";
      lineColor = "#ffb454";
      fillColor = "rgba(255, 180, 84, 0.10)";
    } else if (selectedMetric === "171") {
      yTitle = "AIA 171 Å (DN/s)";
      yUnit = " DN/s";
      metricLabel = "AIA 171 Å · โคโรนา";
      lineColor = THEME.aia["171"];
      fillColor = `rgba(255, 200, 80, 0.08)`;
    } else if (selectedMetric === "304") {
      yTitle = "AIA 304 Å (DN/s)";
      yUnit = " DN/s";
      metricLabel = "AIA 304 Å · โครโมสเฟียร์";
      lineColor = THEME.aia["304"];
      fillColor = `rgba(232, 89, 12, 0.08)`;
    } else if (selectedMetric === "1600") {
      yTitle = "AIA 1600 Å (DN/s)";
      yUnit = " DN/s";
      metricLabel = "AIA 1600 Å · โฟโตสเฟียร์";
      lineColor = THEME.aia["1600"];
      fillColor = `rgba(90, 160, 90, 0.08)`;
    }

    displayTracks.forEach((track, order) => {
      let values = [];
      if (selectedMetric === "b_peak") {
        values = track.b_peak || [];
      } else if (selectedMetric === "flux") {
        values = track.flux || [];
      } else {
        values = (track.series && track.series[selectedMetric]) || [];
      }

      // กรณีที่ไม่มีข้อมูล AIA ให้แสดง b_peak แทนพร้อมหมายเหตุ
      const hasValues = values.some((v) => v !== null && v !== undefined && v > 0);
      if (!hasValues) {
        if (selectedMetric !== "b_peak" && selectedMetric !== "flux" && track.b_peak && track.b_peak.length) {
          values = track.b_peak;
          metricLabel = "b_peak (AIA ไม่มีข้อมูล — แสดง B_peak แทน)";
          lineColor = "#8ba2ff";
        } else {
          return;
        }
      }

      const trackColor = displayTracks.length > 1 ? AR_COLORS[order % AR_COLORS.length] : lineColor;
      const trackFill = displayTracks.length > 1 ? "none" : fillColor;

      // area fill (tozeroy) ใต้เส้น — ทำให้กราฟดูเป็น premium แบบ space weather dashboard
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: track.label,
        x: track.times,
        y: values,
        fill: trackFill,
        fillcolor: trackFill !== "none" ? trackFill : undefined,
        line: { shape: "spline", smoothing: 0.7, width: 2.5, color: trackColor },
        marker: {
          size: 5,
          color: trackColor,
          line: { width: 1.5, color: "rgba(255,255,255,0.5)" },
        },
        connectgaps: false,
        hovertemplate: `<b>${track.label}</b><br>%{x|%d %b %Y %H:%M UT}<br><b>%{y:,.1f}${yUnit}</b><extra></extra>`,
      });
    });

    if (!traces.length) {
      hint.textContent = "";
      extractionEmpty(`ไม่มีข้อมูล ${metricLabel} ใน AR ที่เลือก`);
      return;
    }

    const labelColor = displayTracks.length === 1 ? lineColor : THEME.textSecondary;
    const labelText = displayTracks.length === 1
      ? `<b>${targetTrack.label}</b>`
      : `<b>${displayTracks.length} AR</b>`;

    const headerAnnotations = [
      // ชื่อ metric มุมขวาบน
      {
        xref: "paper", yref: "paper",
        x: 0.99, y: 1.04,
        xanchor: "right", yanchor: "bottom",
        text: `<b>${metricLabel}</b>`,
        showarrow: false,
        font: { size: 11.5, color: THEME.textMuted, family: THEME.fontMono },
      },
      // label AR มุมซ้ายบน
      {
        xref: "paper", yref: "paper",
        x: 0.01, y: 0.96,
        xanchor: "left", yanchor: "top",
        text: labelText,
        showarrow: false,
        font: { size: 13, color: labelColor, family: THEME.fontMono },
      },
    ];

    const layout = {
      ...PLOT_LAYOUT,
      margin: { l: 65, r: 20, t: 30, b: 38 },
      showlegend: displayTracks.length > 1,
      legend: { orientation: "h", y: -0.18, x: 0, font: { size: 10 } },
      xaxis: {
        ...PLOT_LAYOUT.xaxis,
        type: "date",
        tickfont: { size: 10 },
        hoverformat: "%d %b %Y %H:%M UT",
      },
      yaxis: {
        ...PLOT_LAYOUT.yaxis,
        title: { text: yTitle, font: { size: 10, color: THEME.textMuted } },
        tickfont: { size: 10 },
        tickformat: selectedMetric === "flux" ? ".2s" : ",",
        rangemode: "tozero",
      },
      annotations: [...headerAnnotations, ...flareAnnotations],
      shapes: [...flareShapes, ...frameMarkerShapes()],
    };

    Plotly.react("extractionChart", traces, layout, PLOT_CONFIG);
  }

  const source = el("useTruth").checked ? "mask SHARP (Ground Truth)" : "mask U-Net (Segmentation)";
  hint.textContent = `${displayTracks.length} AR · ${data.n_frames} เฟรม · ${source}`;
  hint.title = data.note || "";
  el("extractionNote").textContent = data.note || "";
  el("extractionNote").classList.toggle("hidden", !data.note);
}

/** เส้นแนวตั้งบอกว่าการ์ดซ้ายกำลังแสดงเฟรมไหน — ผูกสองการ์ดเข้าด้วยกันด้วยตา */
function frameMarkerShapes() {
  const stamp = state.frames[state.frameIndex];
  if (!stamp) return [];

  const iso = `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}`
    + `T${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}`;
  return [{
    type: "line", xref: "x", yref: "paper",
    x0: iso, x1: iso, y0: 0, y1: 1,
    line: { color: "rgba(255, 255, 255, 0.85)", width: 1.5, dash: "dot" },
  }];
}

/** ย้ายเส้นแนวตั้งอย่างเดียว ไม่ต้องดึงข้อมูลใหม่ — เรียกตอนเปลี่ยนเฟรม */
function updateFrameMarker() {
  if (!state.extraction || !el("extractionChart").querySelector(".plot-container")) return;
  // เรียก renderExtraction เพื่ออัปเดต marker ให้ถูกต้อง
  renderExtraction(state.extraction);
}

/* ── GOES X-ray light curve + จานสุริยะใน hero ────────────────── */

async function loadGoes() {
  const start = el("goesStart").value;
  const end = el("goesEnd").value;
  if (!start || !end) return;

  try {
    const data = await fetchJson(
      `/api/goes?start=${start}&end=${end}&min_class=${el("goesMinClass").value}`
    );
    state.goes = data;
    // เปลี่ยนช่วงเวลาแล้ว flare ที่เคยตรึง/เคยวาดไว้อาจไม่อยู่ในช่วงใหม่แล้ว
    state.pinnedFlare = null;
    state.chartedFlare = null;
    renderXrayCurve(data);
    renderFlareDisk(data);
    updateHeroFlares(data);
    // เลือก flare ที่แรงที่สุดให้อัตโนมัติ — อัปเดตทั้งแผงโปรตอนในแดชบอร์ดและ
    // กราฟย่อใต้จานสุริยะใน hero ไปพร้อมกัน (ดู loadProton)
    autoSelectProton(data);

    el("rangeNote").textContent =
      `${data.n_events.toLocaleString()} flare ในช่วงนี้ · ` +
      ["B", "C", "M", "X"].map((c) => `${c}=${data.class_counts[c]}`).join("  ");
  } catch (error) {
    Plotly.purge("xrayChart");
    Plotly.purge("diskChart");
    hideDiskTip();
    el("diskFilter").innerHTML = "";
    Plotly.purge("heroProtonChart");
    Plotly.purge("protonChart");
    el("xrayChart").innerHTML = `<p class="empty">${error.message}</p>`;
    el("diskChart").innerHTML = "";
    el("heroProtonChart").innerHTML = "";
    el("protonChart").innerHTML = "";
    el("protonSummary").textContent = "";
    el("heroProtonHint").textContent = "";
    el("rangeNote").textContent = "";
  }
}

/** ลำดับคำขอฟลักซ์ต่อเนื่องล่าสุด — เปลี่ยนช่วงวันที่รัว ๆ แล้วคำตอบมาสลับกัน
 *  จะวาดของเก่าทับของใหม่ (รูปแบบเดียวกับ protonRequest) */
let xrayRequest = 0;

/** ขอบล่างจริงของแกน log — ฟลักซ์เงียบสุด (quiet sun) วัดได้ต่ำถึง ~1e-9 W/m²
 *  ต่ำกว่า BACKGROUND_FLUX (1e-8) ที่ใช้เป็นพื้นของเส้นประกอบจากรายการ flare — ถ้าใช้
 *  ขอบเดิมกับฟลักซ์จริง ช่วงเงียบมาก ๆ จะโดนตัดขาดที่ขอบล่างของกราฟ */
const XRAY_Y_FLOOR = 1e-9;

/** เส้นประกอบจากรายการ flare (เริ่ม-peak-จบ 3 จุดต่อเหตุการณ์) — ใช้เป็นค่าเริ่มต้น
 *  ก่อนฟลักซ์ต่อเนื่องจริงจะโหลดเสร็จ และเป็น fallback ถ้าไม่มีคลัง GOES-15 ในเครื่อง */
function reconstructedXrayCurve(events) {
  const points = [];
  for (const event of events) {
    const start = event.start_time ?? event.peak_time;
    const end = event.end_time ?? event.peak_time;
    points.push(
      { t: start, flux: BACKGROUND_FLUX },
      { t: event.peak_time, flux: event.peak_flux },
      { t: end, flux: BACKGROUND_FLUX },
    );
  }
  points.sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
  return { x: points.map((p) => p.t), y: points.map((p) => p.flux) };
}

function renderXrayCurve(data) {
  if (!data.events.length) {
    Plotly.purge("xrayChart");
    el("xrayChart").innerHTML = `<p class="empty">ไม่พบ flare ในช่วงเวลานี้</p>`;
    el("xrayHint").textContent = "";
    return;
  }

  const reconstructed = reconstructedXrayCurve(data.events);
  const curve = {
    x: reconstructed.x,
    y: reconstructed.y,
    type: "scatter",
    mode: "lines",
    connectgaps: false,
    line: { color: THEME.series1, width: 1.4, shape: "linear" },
    hoverinfo: "skip",
  };

  // จุดยอดของแต่ละเหตุการณ์ — สีบอกคลาส ขนาดบอกความแรง และเป็นตัวที่ hover ได้
  // อยู่คนละ trace จากเส้นฟลักซ์เสมอ ไม่ว่าเส้นข้างล่างจะเป็นแบบประกอบหรือของจริง
  const peaks = {
    x: data.events.map((e) => e.peak_time),
    y: data.events.map((e) => e.peak_flux),
    text: data.events.map((e) => `${e.goes_class}${e.noaa_ar ? ` · AR ${e.noaa_ar}` : ""}`),
    // เก็บดัชนีไว้แทนค่าเดี่ยว ๆ เพราะตัวจัดการคลิกต้องใช้ทั้ง HARP และเวลาที่พีค
    customdata: data.events.map((_, index) => index),
    type: "scatter",
    mode: "markers",
    marker: {
      size: data.events.map((e) => markerSize(e.peak_flux)),
      color: data.events.map((e) => seqColor(e.goes_class)),
      line: { width: 2, color: THEME.surface1 },
    },
    hovertemplate: "%{x|%d %b %Y %H:%M}<br>%{text}<extra></extra>",
  };

  Plotly.react("xrayChart", [curve, peaks], {
    ...PLOT_LAYOUT,
    transition: { duration: 300, easing: "cubic-in-out" },
    yaxis: {
      ...PLOT_LAYOUT.yaxis,
      type: "log",
      title: "flux (W/m²)",
      range: [Math.log10(XRAY_Y_FLOOR), Math.log10(2e-4)],
      tickvals: GOES_LEVELS.map((l) => l.flux),
      ticktext: GOES_LEVELS.map((l) => l.label),
    },
    shapes: GOES_LEVELS.map((level) => ({
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: Math.log10(level.flux), y1: Math.log10(level.flux),
      line: { color: THEME.baseline, width: 1, dash: "dot" },
    })),
  }, PLOT_CONFIG);

  // คลิกจุด peak → ตรึง/ปลดตรึง flare ที่แผงโปรตอนแสดง (เหมือนคลิกบนจานสุริยะ) และ
  // เปิด AR ที่ปะทุถ้าจับคู่ได้
  const node = el("xrayChart");
  node.removeAllListeners?.("plotly_click");
  node.on("plotly_click", (click) => {
    const index = click.points?.[0]?.customdata;
    if (typeof index !== "number") return;   // เส้น light curve ก็ถูกคลิกได้ ต้องกันไว้

    const flare = data.events[index];
    selectFlareEvent(flare);
    if (flare.harpnum) selectHarp(Number(flare.harpnum));
  });

  el("xrayHint").textContent =
    `${data.n_events.toLocaleString()} เหตุการณ์ · ${formatDateTime(data.start)} – ${formatDateTime(data.end)}`;
  // รีเซ็ตข้อความ legend กลับเป็นโหมดเส้นประกอบก่อนเสมอ — เผื่อช่วงที่เลือกใหม่นี้ไม่มี
  // ฟลักซ์ต่อเนื่องจริง (นอกช่วงที่ดาวน์โหลดไว้) จะได้ไม่ค้างข้อความ "GOES-15 จริง"
  // จากการเลือกครั้งก่อน
  el("xrayLegendNote").textContent =
    "สีสว่างขึ้น = แรงขึ้น · เส้นสร้างจากเวลาเริ่ม–peak–สิ้นสุดของแต่ละเหตุการณ์ ไม่ใช่สัญญาณดิบรายนาที";

  upgradeXrayCurveToContinuous(data);
}

/** อัปเกรดเส้นประกอบให้เป็นฟลักซ์ต่อเนื่องจริงรายนาที (GOES-15) แบบ progressive —
 * โหลดเบื้องหลังไม่บล็อกการแสดงผลแรก แล้วสลับเฉพาะเส้นพื้นหลัง (trace 0) เมื่อมาถึง
 * ปล่อยให้เส้นประกอบค้างไว้เงียบๆ ถ้าไม่มีคลัง GOES-15 หรือช่วงนี้ยังไม่ได้ดาวน์โหลด —
 * ผู้ใช้ยังเห็นกราฟได้ปกติ แค่ไม่ได้ความละเอียดรายนาที
 */
async function upgradeXrayCurveToContinuous(data) {
  if (!state.health?.xray_flux) return;
  const request = ++xrayRequest;

  let payload;
  try {
    payload = await fetchJson(`/api/xray?start=${data.start}&end=${data.end}`);
  } catch {
    return;   // เงียบ — เส้นประกอบที่แสดงอยู่แล้วยังใช้งานได้
  }
  if (request !== xrayRequest || !payload.n_samples) return;

  Plotly.restyle("xrayChart", {
    x: [payload.times],
    y: [payload.flux.map((v) => (v === null ? null : Math.max(v, XRAY_Y_FLOOR)))],
    connectgaps: false,
  }, [0]);

  const decimateNote = payload.decimated ? " (ย่อข้อมูลด้วยค่าสูงสุดต่อช่วง — ช่วงยาวเกินไป)" : "";
  el("xrayLegendNote").textContent =
    `สีสว่างขึ้น = แรงขึ้น · เส้นคือฟลักซ์จริงรายนาทีจาก GOES-15${decimateNote} · จุดคือค่า peak ของแต่ละเหตุการณ์`;
}

/* ── ฟลักซ์โปรตอนรอบเวลาที่เกิด flare ─────────────────────────
 *
 * X-ray บอกว่า flare ปะทุแรงแค่ไหน แต่ไม่ได้บอกว่ามีอะไรเดินทางมาถึงโลกหรือเปล่า
 * flare บางดวงตามมาด้วยพายุอนุภาคที่เป็นภัยจริงต่อดาวเทียมและนักบินอวกาศ ส่วนอีก
 * หลายดวงที่แรงพอ ๆ กันกลับเงียบสนิท — ต้องดูอนุกรมโปรตอนรอบเวลานั้นถึงจะแยกออก
 *
 * โหลดทีละเหตุการณ์ตอนผู้ใช้คลิก ไม่ได้ส่งทั้งคลังมาให้ client: อนุกรม 28 ปีที่
 * ความละเอียด 5 นาทีมีเกือบ 3 ล้านตัวอย่างต่อช่อง */

/** มาตราพายุรังสีสุริยะของ NOAA — วัดที่ช่อง >10 MeV หน่วย pfu */
const S_SCALE_LEVELS = [
  { pfu: 1e1, label: "S1" }, { pfu: 1e2, label: "S2" }, { pfu: 1e3, label: "S3" },
  { pfu: 1e4, label: "S4" }, { pfu: 1e5, label: "S5" },
];

/** ช่วงแกน y เริ่มต้น (log10 pfu) — ตรึงไว้เพื่อให้เทียบรูปร่างข้าม flare ได้ทันที
 *  แต่ขยายออกได้ถ้าข้อมูลจริงหลุดกรอบ ดีกว่าปล่อยให้เส้นหายไปนอกจอเงียบ ๆ */
const PROTON_Y_RANGE = [-2, 5];

/** ลำดับคำขอล่าสุด — คลิกรัว ๆ แล้วคำตอบมาสลับกัน จะวาดของเก่าทับของใหม่ */
let protonRequest = 0;

/** ตัวจับเวลา debounce ของการพรีวิวตอน hover — กันยิง request รัวเกินไปตอนกวาดเมาส์
 *  ผ่านจุดที่เกาะกลุ่มกันแน่นบนจาน */
let protonPreviewTimer = null;

/** ฟลักซ์กินช่วงค่าหลายหลัก — toPrecision อย่างเดียวจะได้ "1.5e+3" ซึ่งอ่านยากกว่า */
function formatPfu(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return Math.round(value).toLocaleString("en-US");
  if (value >= 10) return value.toFixed(1);
  return value.toPrecision(2);
}

/** เวลาในข้อมูลเป็น ISO ที่ไม่มีโซน (หมายถึง UTC) — เลื่อนเป็นชั่วโมงแล้วคืนรูปแบบเดิม
 *  ให้ตรงกับค่าที่ Plotly ใช้บนแกน x เป๊ะ ๆ ไม่ให้เส้นกริดเหลื่อมตามโซนของเครื่องผู้ใช้ */
function shiftIsoHours(iso, hours) {
  return new Date(Date.parse(`${iso}Z`) + hours * 3600000).toISOString().slice(0, 19);
}

async function loadProton(event) {
  if (!state.health?.proton_flux || !event?.peak_time) return;

  const request = ++protonRequest;

  const query = new URLSearchParams({ t: event.peak_time });
  if (event.goes_class) query.set("goes_class", event.goes_class);

  el("protonSummary").textContent = "กำลังโหลด…";
  try {
    const data = await fetchJson(`/api/proton?${query}`);
    if (request !== protonRequest) return;    // มีคลิกใหม่แซงมาแล้ว — ทิ้งผลเก่า
    state.chartedFlare = event;
    renderProtonFlux(data, event);
    renderHeroProton(data, event);
    // เลื่อนวงแหวนไฮไลต์บนจานสุริยะไปที่จุดนี้ — ผู้ใช้จะได้เห็นว่ากราฟที่เพิ่งขยับ
    // เป็นของจุดไหน โดยเฉพาะตอน hover ที่ไม่มี tooltip ค้างให้เทียบ
    state.diskHighlight?.(event.peak_time);
  } catch (error) {
    if (request !== protonRequest) return;
    state.chartedFlare = null;
    Plotly.purge("protonChart");
    Plotly.purge("heroProtonChart");
    el("protonChart").innerHTML = `<p class="empty">${error.message}</p>`;
    el("heroProtonChart").innerHTML = `<p class="empty" style="font-size:10px">${error.message}</p>`;
    el("protonSummary").textContent = "";
    el("heroProtonHint").textContent = "";
    state.diskHighlight?.(null);
  }
}

/** ตัวอย่างชั่วคราวตอนเมาส์ชี้ที่จุด flare — debounce กันยิง request ถี่เกินตอนกวาดผ่าน
 *  จุดที่เกาะกลุ่มกัน และไม่ทำอะไรถ้ามี flare ที่ถูกตรึงไว้อยู่แล้ว (คลิกชนะ hover เสมอ) */
function previewFlareEvent(flare) {
  if (!state.health?.proton_flux || state.pinnedFlare) return;
  clearTimeout(protonPreviewTimer);
  protonPreviewTimer = setTimeout(() => loadProton(flare), 90);
}

/** คลิกจุด flare — สลับสถานะ "ตรึงไว้": คลิกจุดใหม่ = ตรึงจุดนั้น, คลิกจุดที่ตรึงอยู่
 *  ซ้ำ = ปลดตรึงแล้วกลับไปแสดง flare ที่แรงที่สุดในช่วงเหมือนตอนเริ่มต้น */
function selectFlareEvent(flare) {
  clearTimeout(protonPreviewTimer);
  const alreadyPinned = state.pinnedFlare?.peak_time === flare.peak_time;
  state.pinnedFlare = alreadyPinned ? null : flare;

  if (state.pinnedFlare) {
    loadProton(flare);
  } else if (state.goes) {
    autoSelectProton(state.goes);
  }
}

/** เลือก flare ที่แรงที่สุดในช่วงให้เอง เพื่อให้แผงมีอะไรแสดงก่อนผู้ใช้คลิกครั้งแรก */
function autoSelectProton(data) {
  if (!state.health?.proton_flux) return;

  const strongest = data.events.reduce(
    (best, event) => (best === null || event.peak_flux > best.peak_flux ? event : best),
    null,
  );
  if (strongest) {
    loadProton(strongest);
    return;
  }
  Plotly.purge("protonChart");
  Plotly.purge("heroProtonChart");
  el("protonChart").innerHTML = `<p class="empty">ไม่พบ flare ในช่วงเวลานี้</p>`;
  el("heroProtonChart").innerHTML = `<p class="empty" style="font-size:10px">ไม่พบ flare ในช่วงเวลานี้</p>`;
  el("protonSummary").textContent = "";
  el("heroProtonHint").textContent = "";
}

/** บรรทัดสรุป: เหตุการณ์ไหน ค่าตอนนั้นเท่าไร และยอดที่ตามมาสูงแค่ไหน */
function protonSummary(data, event) {
  const separator = `<span class="proton__sep">·</span>`;
  const goesClass = event?.goes_class ?? data.goes_class;
  const headline = `${goesClass ? `flare ${goesClass}` : "flare"} ${formatDateTime(data.peak_time)} UT`;
  // บอกด้วยว่ากำลังดู flare ที่ตรึงไว้หรือแค่พรีวิวตาม hover — คลิกซ้ำเพื่อปลด
  const pinTag = state.pinnedFlare?.peak_time === data.peak_time
    ? `${separator}ตรึงไว้ (คลิกซ้ำเพื่อปลด)` : "";

  if (!data.has_data) return `${headline}${separator}ไม่มีข้อมูลโปรตอนช่วงนี้${pinTag}`;

  const parts = [
    headline,
    `&gt;10 MeV ขณะนั้น <b>${formatPfu(data.flux_at_peak)}</b> pfu`,
  ];
  // ค่า ณ เวลาที่ flare พีคอย่างเดียวไม่พอ: อนุภาคมาถึงโลกช้ากว่าแสงเป็นชั่วโมง
  // ตอนนั้นจึงมักยังเป็นระดับพื้นหลังอยู่ ยอดในหน้าต่างหลังคือตัวที่บอกเรื่อง
  if (data.max_after !== null) {
    parts.push(
      `สูงสุดใน +${data.after_hours} ชม. <b>${formatPfu(data.max_after)}</b> pfu` +
      (data.s_scale ? ` (${data.s_scale})` : "")
    );
  }
  if (data.sources.length) parts.push(`ที่มา ${data.sources.join(" + ")}`);
  return parts.join(separator) + pinTag;
}

function renderProtonFlux(data, event) {
  el("protonSummary").innerHTML = protonSummary(data, event);

  if (!data.has_data) {
    Plotly.purge("protonChart");
    el("protonChart").innerHTML =
      `<p class="empty">คลังโปรตอนไม่ครอบคลุมช่วงเวลานี้ — ข้อมูลเริ่มกลางปี 1998</p>`;
    return;
  }

  const traces = data.channels.map((channel, index) => ({
    x: data.times,
    y: channel.values,
    // ค่าใน tooltip ใช้ตัวจัดรูปเดียวกับบรรทัดสรุป เพื่อให้ตัวเลขทั้งแผงอ่านแบบเดียวกัน
    customdata: channel.values.map((v) => formatPfu(v)),
    name: channel.label,
    type: "scatter",
    mode: "lines",
    // null ต้องเป็นช่องว่างจริง ๆ — ลากเส้นข้ามช่วงที่ดาวเทียมไม่ได้วัดคือการแต่งข้อมูล
    connectgaps: false,
    line: { color: THEME.pflux[index], width: 2, shape: "linear" },
    hovertemplate: `<b>%{customdata}</b> pfu<extra>${channel.label}</extra>`,
  }));

  // ยึดกรอบมาตรฐานไว้ แล้วขยายเฉพาะด้านที่ข้อมูลจริงล้นออกไป
  // (ไล่ด้วย reduce ไม่ใช่ Math.min(...array) — หน้าต่างหนึ่งมีหลายพันค่า มากพอที่
  //  การกระจายเป็น argument จะชนเพดานของ engine ได้)
  let [low, high] = PROTON_Y_RANGE;
  for (const channel of data.channels) {
    for (const value of channel.values) {
      if (value === null || value <= 0) continue;
      const exponent = Math.log10(value);
      low = Math.min(low, Math.floor(exponent));
      high = Math.max(high, Math.ceil(exponent));
    }
  }

  // แกนเวลาอ่านเป็น "กี่ชั่วโมงจากจุดพีค" เพราะคำถามคือลำดับเหตุการณ์ ไม่ใช่วันที่
  // ส่วนเวลาจริงยังอยู่ครบใน tooltip
  const tickvals = [];
  const ticktext = [];
  for (let hour = -data.before_hours; hour <= data.after_hours; hour += 12) {
    tickvals.push(shiftIsoHours(data.peak_time, hour));
    ticktext.push(`${hour > 0 ? "+" : ""}${hour} ชม.`);
  }

  const inRange = (pfu) => Math.log10(pfu) >= low && Math.log10(pfu) <= high;

  Plotly.react("protonChart", traces, {
    ...PLOT_LAYOUT,
    margin: { l: 58, r: 34, t: 20, b: 40 },
    // เส้นเดียวไล่ทุกช่องพร้อมกัน — ช่องพลังงานเป็นชุดที่ต้องอ่านเทียบกัน ณ เวลาเดียว
    hovermode: "x unified",
    // สลับ flare แล้วเส้นไหลจากรูปเดิมไปรูปใหม่แทนที่จะตัดภาพทันที — ช่วยให้ตามทัน
    // ว่ากราฟเพิ่งเปลี่ยนไปเพราะ hover/คลิกจุดใหม่ ไม่ใช่แค่ข้อมูลกระตุก
    transition: { duration: 300, easing: "cubic-in-out" },
    xaxis: {
      ...PLOT_LAYOUT.xaxis,
      type: "date",
      // กรอบยืดเลยหน้าต่างข้อมูลไปข้างละ 1.5 ชม. ด้วยสองเหตุผล: ป้าย "-12 ชม." กับ
      // "+60 ชม." จะได้ไม่ไปชนป้ายแกน y และป้าย S-scale ที่ขอบซ้าย/ขวา และ tick
      // สุดท้ายจะได้ไม่ตกขอบ (กริด 5 นาทีปัดลง จุดสุดท้ายจึงมาก่อน +60 ชม.เล็กน้อย)
      range: [
        shiftIsoHours(data.peak_time, -data.before_hours - 1.5),
        shiftIsoHours(data.peak_time, data.after_hours + 1.5),
      ],
      tickvals,
      ticktext,
      // รูปแบบเดียวกับ formatDateTime() ที่ใช้ในบรรทัดสรุปเหนือกราฟ — Plotly ไม่มี
      // locale ไทยมาให้ ชื่อเดือนย่อภาษาอังกฤษจะหลุดธีมของหน้า
      hoverformat: "%Y-%m-%d %H:%M UT",
      fixedrange: true,
    },
    yaxis: {
      ...PLOT_LAYOUT.yaxis,
      type: "log",
      title: "ฟลักซ์ (pfu)",
      range: [low, high],
      dtick: 1,
      exponentformat: "power",
      fixedrange: true,
    },
    shapes: [{
      // เวลาที่ flare พีค — จุดอ้างอิงของทั้งกราฟ
      type: "line", xref: "x", yref: "paper",
      x0: data.peak_time, x1: data.peak_time, y0: 0, y1: 1,
      line: { color: THEME.textSecondary, width: 1.5 },
    }],
    annotations: [
      {
        xref: "x", x: data.peak_time, xanchor: "center",
        yref: "paper", y: 1, yanchor: "bottom",
        text: event?.goes_class ?? data.goes_class ?? "flare",
        showarrow: false,
        font: { size: 10, color: THEME.textPrimary, family: THEME.fontMono },
      },
      // S1–S5 อยู่ที่เส้นกริด decade พอดี (10, 100, … pfu) จึงไม่ต้องวาดเส้นเพิ่ม
      ...S_SCALE_LEVELS.filter((level) => inRange(level.pfu)).map((level) => ({
        xref: "paper", x: 1.008, xanchor: "left",
        yref: "y", y: Math.log10(level.pfu),
        text: level.label, showarrow: false,
        font: { size: 9.5, color: THEME.textMuted, family: THEME.fontMono },
      })),
    ],
  }, PLOT_CONFIG);
}

/** กราฟย่อใต้จานสุริยะใน hero — เวอร์ชันย่อของ renderProtonFlux() ด้านบน
 *
 * แสดงเฉพาะช่อง >10 MeV (ช่องอ้างอิงของ NOAA S-scale) เส้นเดียว ไม่ใช่ทั้งสี่ช่อง:
 * พื้นที่แค่ ~110px สูง ใส่ทั้งสี่เส้น + legend ไม่ได้อยู่แล้ว และเส้นเดียวไม่ต้องมี
 * legend กำกับ (accessibility rule: 1 series ไม่ต้องมี legend) รายละเอียดครบทั้งสี่
 * ช่องยังอยู่ในการ์ด "Proton flux" ที่ dashboard เสมอ — อันนี้เป็นแค่ตัวอย่างย่อ
 */
function renderHeroProton(data, event) {
  const reference = data.channels.find((c) => c.key === "p10") ?? data.channels[1];
  const hasReference = reference.values.some((v) => v !== null);

  if (!data.has_data || !hasReference) {
    Plotly.purge("heroProtonChart");
    el("heroProtonChart").innerHTML =
      `<p class="empty" style="font-size:10px">ไม่มีข้อมูลโปรตอนช่วงนี้</p>`;
    el("heroProtonHint").textContent = "";
    return;
  }

  let [low, high] = PROTON_Y_RANGE;
  for (const value of reference.values) {
    if (value === null || value <= 0) continue;
    const exponent = Math.log10(value);
    low = Math.min(low, Math.floor(exponent));
    high = Math.max(high, Math.ceil(exponent));
  }

  const trace = {
    x: data.times,
    y: reference.values,
    customdata: reference.values.map((v) => formatPfu(v)),
    type: "scatter",
    mode: "lines",
    connectgaps: false,
    line: { color: THEME.pflux[1], width: 1.4, shape: "linear" },
    hovertemplate: `<b>%{customdata}</b> pfu<extra></extra>`,
  };

  // ทุก 24 ชม.แทนที่จะเป็น 12 เหมือนกราฟใหญ่ — พื้นที่แคบกว่ามาก ป้ายชนกันถ้าถี่กว่านี้
  const tickHours = [...new Set([-data.before_hours, 0, 24, 48, data.after_hours])]
    .filter((h) => h >= -data.before_hours && h <= data.after_hours)
    .sort((a, b) => a - b);

  Plotly.react("heroProtonChart", [trace], {
    ...PLOT_LAYOUT,
    margin: { l: 30, r: 6, t: 4, b: 18 },
    transition: { duration: 300, easing: "cubic-in-out" },
    xaxis: {
      ...PLOT_LAYOUT.xaxis,
      type: "date",
      range: [
        shiftIsoHours(data.peak_time, -data.before_hours - 1.5),
        shiftIsoHours(data.peak_time, data.after_hours + 1.5),
      ],
      tickvals: tickHours.map((h) => shiftIsoHours(data.peak_time, h)),
      ticktext: tickHours.map((h) => `${h > 0 ? "+" : ""}${h} ชม.`),
      tickfont: { size: 8, color: THEME.textMuted },
      hoverformat: "%Y-%m-%d %H:%M UT",
    },
    yaxis: {
      ...PLOT_LAYOUT.yaxis,
      type: "log",
      range: [low, high],
      dtick: 2,
      exponentformat: "power",
      tickfont: { size: 8, color: THEME.textMuted },
    },
    shapes: [
      // เส้นประระดับ S-scale เป็นบริบทเบื้องหลัง — ไม่ใส่ป้ายตัวอักษรเพราะพื้นที่แคบ
      // เกินจะอ่านออก ตัวเลขเป๊ะดูได้จากมุมขวาบนหรือการ์ดใหญ่ด้านล่าง
      ...S_SCALE_LEVELS
        .filter((level) => Math.log10(level.pfu) >= low && Math.log10(level.pfu) <= high)
        .map((level) => ({
          type: "line", xref: "paper", x0: 0, x1: 1,
          y0: Math.log10(level.pfu), y1: Math.log10(level.pfu),
          line: { color: THEME.baseline, width: 0.8, dash: "dot" },
        })),
      {
        // เวลาที่ flare พีค — จุดอ้างอิงเดียวกับกราฟใหญ่
        type: "line", xref: "x", yref: "paper",
        x0: data.peak_time, x1: data.peak_time, y0: 0, y1: 1,
        line: { color: THEME.textSecondary, width: 1.2 },
      },
    ],
  }, PLOT_CONFIG);

  const label = event?.goes_class ?? data.goes_class;
  const prefix = label ? `${label} · ` : "";
  // มุมนี้แคบมาก (font 9px) — ใช้คำสั้นสุดแค่ "ตรึง" แทนข้อความเต็มแบบในสรุปด้านล่าง
  const pinTag = state.pinnedFlare?.peak_time === data.peak_time ? " · ตรึง" : "";
  el("heroProtonHint").textContent = (data.s_scale
    ? `${prefix}${data.s_scale} สูงสุด ${formatPfu(data.max_after)} pfu`
    : `${prefix}ต่ำกว่า S1`) + pinTag;
}

/** ฉายพิกัด heliographic ลงจานที่มองเห็นจากโลก (orthographic) */
function projectToDisk(lat, lon) {
  const toRad = Math.PI / 180;
  return {
    x: Math.cos(lat * toRad) * Math.sin(lon * toRad),
    y: Math.sin(lat * toRad),
  };
}

/** เส้นกริด heliographic บนจาน — วาดเป็น trace เส้นบางที่ hover ไม่ได้ */
function diskGraticule() {
  const traces = [];
  const line = { color: THEME.gridline, width: 1 };

  for (const lat of [-60, -30, 0, 30, 60]) {
    const points = [];
    for (let lon = -90; lon <= 90; lon += 5) points.push(projectToDisk(lat, lon));
    traces.push({
      x: points.map((p) => p.x), y: points.map((p) => p.y),
      type: "scatter", mode: "lines", hoverinfo: "skip",
      line: lat === 0 ? { ...line, color: THEME.baseline } : line,
    });
  }
  for (const lon of [-60, -30, 0, 30, 60]) {
    const points = [];
    for (let lat = -90; lat <= 90; lat += 5) points.push(projectToDisk(lat, lon));
    traces.push({
      x: points.map((p) => p.x), y: points.map((p) => p.y),
      type: "scatter", mode: "lines", hoverinfo: "skip",
      line: lon === 0 ? { ...line, color: THEME.baseline } : line,
    });
  }
  return traces;
}

/* ── ตัวกรอง + tooltip ของจานสุริยะ ─────────────────────────── */

const DISK_CLASSES = ["B", "C", "M", "X"];

/** สลับการแสดง class หนึ่งบนจาน แล้ววาดจานใหม่ (ตัวเลขในการ์ดอื่นไม่ขยับตาม —
 *  ตัวกรองนี้เป็นของจานอย่างเดียว ช่วงเวลายังเป็นตัวกำหนดร่วมของทั้งหน้าเหมือนเดิม) */
function toggleDiskClass(goesClass) {
  const shown = state.diskClasses;
  if (shown.has(goesClass)) shown.delete(goesClass); else shown.add(goesClass);
  // ปิดครบทุกปุ่มแล้วจานจะว่างเปล่าโดยไม่มีอะไรบอกว่าทำไม — คงไว้อย่างน้อยหนึ่ง
  if (!shown.size) shown.add(goesClass);
  if (state.goes) renderFlareDisk(state.goes);
}

/** แถบ chip ใต้จาน — ทำหน้าที่สองอย่างพร้อมกัน: บอกสี+จำนวนของแต่ละ class (legend)
 *  และเป็นปุ่มกรองในตัว รวมไว้ปุ่มเดียวกันเพราะคอลัมน์ hero แคบเกินกว่าจะมีสองแถว */
function renderDiskFilter(located) {
  const counts = Object.fromEntries(DISK_CLASSES.map((cls) => [cls, 0]));
  for (const event of located) {
    const key = event.goes_class[0];
    if (key in counts) counts[key] += 1;
  }

  // class ที่ช่วงนี้ไม่มีเลยต้องหลุดจากตัวกรองด้วย ไม่ใช่แค่ไม่แสดงปุ่ม — ไม่งั้นคนที่
  // เคยเลือกไว้แค่ X แล้วเปลี่ยนไปช่วงที่ไม่มี X จะเจอจานว่างโดยไม่มีปุ่มให้กดแก้
  for (const cls of [...state.diskClasses]) {
    if (!counts[cls]) state.diskClasses.delete(cls);
  }
  const present = DISK_CLASSES.filter((cls) => counts[cls] > 0);
  if (!state.diskClasses.size) present.forEach((cls) => state.diskClasses.add(cls));

  const host = el("diskFilter");
  host.innerHTML = "";
  for (const cls of present) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.style.setProperty("--chip-color", THEME.seq[SEQ_INDEX[cls]]);
    button.setAttribute("aria-pressed", String(state.diskClasses.has(cls)));
    button.innerHTML =
      `<i class="chip__dot" aria-hidden="true"></i>${cls}-class ` +
      `<b>${counts[cls].toLocaleString()}</b>`;
    button.addEventListener("click", () => toggleDiskClass(cls));
    host.appendChild(button);
  }
}

/** พิกัดแบบที่ NOAA ใช้ในรายงาน (S12W40) — อ่านคู่กับ catalogue ต้นทางได้ตรง ๆ */
function formatHeliographic(lat, lon) {
  const pad = (value) => String(Math.round(Math.abs(value))).padStart(2, "0");
  return `${lat >= 0 ? "N" : "S"}${pad(lat)}${lon >= 0 ? "W" : "E"}${pad(lon)}`;
}

function hideDiskTip() {
  const tip = el("diskTip");
  tip.classList.remove("is-on");
  tip.setAttribute("aria-hidden", "true");
}

/** tooltip ที่วาดเอง แทน hoverlabel ของ Plotly
 *
 *  hoverlabel ใส่ได้แค่ข้อความหลายบรรทัด แต่สิ่งที่ต้องอ่านคือค่าหลายตัวที่มีหน่วยต่างกัน
 *  — ตารางป้าย/ค่าอ่านได้ทีละแถวโดยไม่ต้องเดาว่าเลขไหนคืออะไร จึงตั้ง hoverinfo:"none"
 *  บน trace (ยังยิง event อยู่ ต่างจาก "skip") แล้ววาง div นี้เองจาก bbox ของจุด
 *
 *  ยึดกับ *จุด* ไม่ใช่ตำแหน่งเมาส์ — ขยับเมาส์เล็กน้อยในจุดเดิมแล้วกล่องจะได้ไม่สั่น */
function showDiskTip(flare, bbox) {
  const tip = el("diskTip");
  const rows = [
    ["เวลาที่พีค", `${formatDateTime(flare.peak_time)} UT`],
    ["flux 1-8 Å", `${flare.peak_flux.toExponential(2)} W/m²`],
    ["ตำแหน่ง", formatHeliographic(flare.lat, flare.lon)],
    ["lat / lon", `${flare.lat.toFixed(1)}° / ${flare.lon.toFixed(1)}°`],
    ["active region", flare.noaa_ar ? `AR ${flare.noaa_ar}` : "—"],
    ["HARP", flare.harpnum ? String(flare.harpnum) : "ไม่มีคู่"],
  ];

  // บอกล่วงหน้าว่าคลิกแล้วจะได้อะไร — ปลายทางของคลิกต่างกันไปตามว่า flare ดวงนี้
  // จับคู่ HARP ได้ไหม และคลังโปรตอนพร้อมใช้หรือยัง
  const actions = [];
  if (state.pinnedFlare?.peak_time === flare.peak_time) {
    actions.push("คลิกซ้ำเพื่อปลดตรึง");
  } else {
    if (state.health?.proton_flux) actions.push("คลิกเพื่อตรึงกราฟโปรตอน");
    if (flare.harpnum) actions.push("เปิด HARP นี้ใน dashboard");
  }

  tip.innerHTML =
    `<div class="disktip__head">` +
    `<span class="disktip__dot" style="background:${seqColor(flare.goes_class)}"></span>` +
    `${flare.goes_class}</div>` +
    `<table><tbody>` +
    rows.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join("") +
    `</tbody></table>` +
    (actions.length ? `<p class="disktip__foot">${actions.join(" · ")}</p>` : "");

  tip.classList.add("is-on");
  tip.setAttribute("aria-hidden", "false");

  // วางไว้ขวาของจุดก่อน แล้วพลิกไปซ้ายเมื่อจะล้นขอบ — จุดที่ขอบตะวันตกของจานอยู่ชิด
  // ขอบกราฟพอดี ถ้าไม่พลิกกล่องจะโดนตัดครึ่ง
  const host = el("diskChart");
  const width = tip.offsetWidth;
  const height = tip.offsetHeight;
  let left = bbox.x1 + 14;
  if (left + width > host.clientWidth) left = bbox.x0 - width - 14;
  const top = (bbox.y0 + bbox.y1) / 2 - height / 2;

  tip.style.left = `${Math.max(0, Math.min(host.clientWidth - width, left))}px`;
  tip.style.top = `${Math.max(0, Math.min(host.clientHeight - height, top))}px`;
}

/** ย้ายวงแหวน hover ทันที (ไม่ใช้ animate) — วงแหวน "จุดที่เลือก" เคลื่อนที่พร้อม
 *  แอนิเมชันเพราะกระโดดไม่บ่อย แต่ hover ต้องตามเมาส์ให้ทันทุกจุดที่กวาดผ่าน */
function moveDiskRing(node, traceIndex, point) {
  Plotly.restyle(node, {
    x: [point ? [point.x] : []],
    y: [point ? [point.y] : []],
  }, [traceIndex]);
}

function renderFlareDisk(data) {
  const located = data.events.filter((e) => e.lat !== null && e.lon !== null);
  renderDiskFilter(located);
  const shown = located.filter((e) => state.diskClasses.has(e.goes_class[0]));

  el("diskHint").textContent = !located.length
    ? "ไม่มี flare ที่ยืนยันพิกัดในช่วงนี้"
    : shown.length === located.length
      ? `${located.length.toLocaleString()} จาก ${data.n_events.toLocaleString()} flare มีพิกัด`
      : `แสดง ${shown.length.toLocaleString()} จาก ${located.length.toLocaleString()} ` +
        `ที่มีพิกัด (ทั้งหมด ${data.n_events.toLocaleString()} flare)`;

  if (!located.length) {
    Plotly.purge("diskChart");
    hideDiskTip();
    el("diskChart").innerHTML =
      `<p class="empty">flare ในช่วงนี้ไม่มีพิกัดยืนยัน — NOAA ระบุตำแหน่งให้ราว 2 ใน 3 ของรายการเท่านั้น</p>`;
    state.diskHighlight = null;
    return;
  }

  const projected = shown.map((e) => projectToDisk(e.lat, e.lon));
  const flares = {
    x: projected.map((p) => p.x),
    y: projected.map((p) => p.y),
    // ดัชนีไม่ใช่ HARP: flare ราวหนึ่งในสามไม่มี HARP คู่กัน แต่ยังดูโปรตอนของมันได้
    customdata: shown.map((_, index) => index),
    type: "scatter",
    mode: "markers",
    marker: {
      size: shown.map((e) => markerSize(e.peak_flux)),
      color: shown.map((e) => seqColor(e.goes_class)),
      line: { width: 2, color: THEME.surface1 },
    },
    // "none" ไม่ใช่ "skip": ปิดกล่อง hover ของ Plotly แต่ยังยิง plotly_hover ให้ tooltip
    // ที่เราวาดเองใช้ (ดู showDiskTip)
    hoverinfo: "none",
  };

  // วงแหวนตามเมาส์ — ตอบสนองทันทีก่อนกราฟโปรตอนจะโหลดเสร็จ ทำให้รู้ว่าชี้โดนจุดไหน
  // แม้ในกลุ่มที่จุดซ้อนกันแน่น
  const hoverRing = {
    x: [], y: [],
    type: "scatter", mode: "markers",
    marker: { size: 30, color: "rgba(0,0,0,0)", line: { width: 1.4, color: THEME.textSecondary } },
    hoverinfo: "skip",
  };

  // วงแหวนไฮไลต์ — เริ่มไม่มีจุด (x/y ว่าง) แล้ว state.diskHighlight ด้านล่างจะย้ายมันไป
  // ยัง flare ที่กราฟโปรตอนกำลังแสดงอยู่ ให้เห็นชัดว่าจุดไหนคือจุดที่ถูกเลือก
  const highlight = {
    x: [], y: [],
    type: "scatter", mode: "markers",
    marker: { size: 22, color: "rgba(0,0,0,0)", line: { width: 2, color: THEME.brand } },
    hoverinfo: "skip",
  };

  const axis = {
    range: [-1.12, 1.12],
    showgrid: false, zeroline: false, showticklabels: false,
    ticks: "", fixedrange: true,
  };

  const traces = [...diskGraticule(), flares, hoverRing, highlight];
  const hoverIndex = traces.length - 2;
  const highlightIndex = traces.length - 1;

  Plotly.react("diskChart", traces, {
    ...PLOT_LAYOUT,
    margin: { l: 8, r: 8, t: 8, b: 8 },
    hovermode: "closest",
    xaxis: { ...axis },
    yaxis: { ...axis, scaleanchor: "x", scaleratio: 1 },
    shapes: [{
      // ขอบจานสุริยะ — ทุกอย่างที่อยู่นอกวงนี้คือด้านที่มองไม่เห็นจากโลก
      type: "circle", xref: "x", yref: "y",
      x0: -1, y0: -1, x1: 1, y1: 1,
      line: { color: "rgba(255, 176, 84, 0.45)", width: 1.5 },
      fillcolor: "rgba(255, 176, 84, 0.05)",
      layer: "below",
    }],
    annotations: [
      { xref: "x", yref: "y", x: -1.03, y: 0, text: "E", showarrow: false,
        font: { size: 10, color: THEME.textMuted }, xanchor: "right" },
      { xref: "x", yref: "y", x: 1.03, y: 0, text: "W", showarrow: false,
        font: { size: 10, color: THEME.textMuted }, xanchor: "left" },
      { xref: "x", yref: "y", x: 0, y: 1.03, text: "N", showarrow: false,
        font: { size: 10, color: THEME.textMuted }, yanchor: "bottom" },
    ],
  }, PLOT_CONFIG);

  // เรียกทุกครั้งที่กราฟโปรตอนเปลี่ยนไปแสดง flare ดวงใหม่ (ดู loadProton) — เลื่อน
  // วงแหวนไปยังจุดนั้นด้วยแอนิเมชันสั้น ๆ แทนการกระโดดทันที ให้ตามสายตาได้ว่าเปลี่ยน
  // จากจุดไหนไปจุดไหน; peakTime ที่หาไม่เจอ (flare ที่ไม่มีพิกัด หรือถูกตัวกรองซ่อนไว้)
  // จะทำให้วงแหวนหายไปแทนที่จะค้างอยู่ผิดตำแหน่ง
  state.diskHighlight = (peakTime) => {
    const match = shown.findIndex((e) => e.peak_time === peakTime);
    const node = el("diskChart");
    if (!node?.data) return;
    const [x, y] = match < 0 ? [[], []] : [[projected[match].x], [projected[match].y]];
    Plotly.animate(node, { data: [{ x, y }], traces: [highlightIndex] }, {
      transition: { duration: 260, easing: "cubic-in-out" },
      frame: { duration: 260, redraw: false },
    });
  };
  // วาดใหม่เพราะเปลี่ยน *ตัวกรอง* (ไม่ใช่เปลี่ยนช่วงเวลา) แล้วกราฟโปรตอนยังแสดง flare
  // ดวงเดิมอยู่ — คืนวงแหวนให้จุดนั้นทันที ไม่ต้องรอให้มีคำขอใหม่มาปลุก
  state.diskHighlight(state.chartedFlare?.peak_time ?? null);

  // ชี้เมาส์ที่จุด flare → วงแหวน + tooltip ทันที และพรีวิวกราฟโปรตอนตามมา (debounce
  // กันยิงถี่), คลิก → ตรึงไว้ (คลิกซ้ำจุดเดิมเพื่อปลด) แล้วถ้าจับคู่กับ HARP ได้ก็กระโดด
  // ลง dashboard ไปดูความเสี่ยงของ AR ดวงนั้นด้วย — hover ไม่แตะ HARP/dashboard เพราะ
  // กวาดเมาส์ผ่านหลายจุดไม่ควรทำให้แผงอื่นกระตุกตาม
  const node = el("diskChart");
  node.removeAllListeners?.("plotly_hover");
  node.removeAllListeners?.("plotly_unhover");
  node.removeAllListeners?.("plotly_click");

  node.on("plotly_hover", (hover) => {
    const point = hover.points?.[0];
    if (typeof point?.customdata !== "number") return;
    moveDiskRing(node, hoverIndex, projected[point.customdata]);
    showDiskTip(shown[point.customdata], point.bbox);
    previewFlareEvent(shown[point.customdata]);
  });
  node.on("plotly_unhover", () => {
    clearTimeout(protonPreviewTimer);
    hideDiskTip();
    moveDiskRing(node, hoverIndex, null);
  });
  node.on("plotly_click", (click) => {
    const point = click.points?.[0];
    if (typeof point?.customdata !== "number") return;   // เส้นกริดบนจานก็ถูกคลิกได้ ต้องกันไว้

    const flare = shown[point.customdata];
    if (!flare.harpnum && !state.health?.proton_flux) return;   // คลิกนี้จะไม่มีผลอะไรเลย

    selectFlareEvent(flare);
    showDiskTip(flare, point.bbox);   // สถานะ "ตรึงไว้" ในกล่องต้องเปลี่ยนตามทันที
    if (flare.harpnum) {
      selectHarp(Number(flare.harpnum));
      el("dashboard").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth" });
    }
  });
}

/* ── ตัวเลขสรุปใน hero ───────────────────────────────────────── */

function updateHeroRisk(topPoint) {
  if (!topPoint) {
    el("heroRisk").textContent = "—";
    el("heroRiskMeta").textContent = "ไม่มีข้อมูลในช่วงนี้";
    return;
  }
  countTo(el("heroRisk"), topPoint.probability * 100, (v) => `${v.toFixed(1)}%`);
  el("heroRiskMeta").textContent =
    `HARP ${topPoint.harpnum}${topPoint.noaa_ar ? ` · NOAA ${topPoint.noaa_ar}` : ""}`;
}

function updateHeroFlares(data) {
  countTo(el("heroFlares"), data.n_events, (v) => Math.round(v).toLocaleString());
  el("heroFlaresMeta").textContent =
    ["B", "C", "M", "X"].map((c) => `${c}:${data.class_counts[c] ?? 0}`).join(" · ");
}

/* ── แผงภาพ magnetogram ──────────────────────────────────────── */

async function loadFrames() {
  if (!state.health?.n_frames) return;

  try {
    state.frames = await fetchJson("/api/frames?limit=5000");
  } catch { return; }
  if (!state.frames.length) return;

  const select = el("frameSelect");
  select.innerHTML = state.frames
    .map((f, i) => `<option value="${i}">${f.slice(0, 4)}-${f.slice(4, 6)}-${f.slice(6, 8)} ${f.slice(9, 11)}:${f.slice(11, 13)}</option>`)
    .join("");

  select.addEventListener("change", () => showFrame(Number(select.value)));
  el("useTruth").addEventListener("change", () => {
    showFrame(state.frameIndex);
    loadExtraction();   // เปลี่ยนที่มาของ mask = ตัวเลขความเข้มแสงเปลี่ยนตาม
  });
  el("layerSelect").addEventListener("change", (event) => {
    setLayer(event.target.value);
  });
  el("framePrev").addEventListener("click", () => showFrame(state.frameIndex - 1));
  el("frameNext").addEventListener("click", () => showFrame(state.frameIndex + 1));

  // ปุ่ม Play/Pause Timelapse
  const playBtn = el("framePlay");
  if (playBtn) {
    playBtn.addEventListener("click", toggleTimelapse);
  }

  // Layer pills event listeners
  document.querySelectorAll(".layer-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      if (pill.disabled) return;
      setLayer(pill.dataset.layer);
    });
  });

  // ยังไม่โหลดภาพที่นี่ — applyRange จะเลือกเฟรมที่ตรงกับช่วงเวลาให้เอง
  // (-1 คือ "ยังไม่เคยโหลด" เพื่อให้ syncFrameToRange ทำงานแม้เฟรมที่เลือกคือ 0)
  state.frameIndex = -1;
}

/** สลับเลเยอร์ภาพและอัปเดต UI ทั้ง select และ pills */
function setLayer(layerKey) {
  state.layer = layerKey;
  const select = el("layerSelect");
  if (select) select.value = layerKey;
  document.querySelectorAll(".layer-pill").forEach((pill) => {
    pill.classList.toggle("is-active", pill.dataset.layer === layerKey);
  });
  showFrame(state.frameIndex);
}

/** สลับการเล่นภาพเคลื่อนไหว Timelapse */
function toggleTimelapse() {
  const btn = el("framePlay");
  if (state.isPlaying) {
    state.isPlaying = false;
    clearInterval(state.playTimer);
    state.playTimer = null;
    if (btn) {
      btn.innerHTML = "▶";
      btn.classList.remove("is-playing");
      btn.setAttribute("title", "เล่นภาพเคลื่อนไหว Timelapse (Spacebar)");
    }
  } else {
    state.isPlaying = true;
    if (btn) {
      btn.innerHTML = "⏸";
      btn.classList.add("is-playing");
      btn.setAttribute("title", "หยุดภาพเคลื่อนไหว (Spacebar)");
    }
    state.playTimer = setInterval(() => {
      if (!state.frames.length) return;
      let nextIndex = state.frameIndex + 1;
      if (nextIndex >= state.frames.length) nextIndex = 0;
      showFrame(nextIndex);
    }, 700);
  }
}

/** แปลงชื่อเฟรม "YYYYMMDD_HHMMSS" เป็น Date */
function frameTime(stamp) {
  return new Date(Date.UTC(
    +stamp.slice(0, 4), +stamp.slice(4, 6) - 1, +stamp.slice(6, 8),
    +stamp.slice(9, 11), +stamp.slice(11, 13),
  ));
}

/** เลือกเฟรมที่ใกล้ปลายช่วงเวลาที่กรองที่สุด
 *  ทั้งหน้าใช้ช่วงเวลาเดียวกัน — ภาพ magnetogram จึงไม่ควรค้างอยู่คนละปีกับกราฟ */
function syncFrameToRange() {
  if (!state.frames.length) return;
  const end = el("goesEnd").value;
  if (!end) return;

  const target = new Date(`${end}T00:00:00Z`).getTime();
  let best = 0;
  let bestGap = Infinity;
  state.frames.forEach((stamp, index) => {
    const gap = Math.abs(frameTime(stamp).getTime() - target);
    if (gap < bestGap) { bestGap = gap; best = index; }
  });
  if (best !== state.frameIndex) showFrame(best);
}

/** เติมตัวเลือกเลเยอร์ใหม่ตามความพร้อมของเฟรมปัจจุบัน
 *  ความพร้อมเป็นราย ๆ เฟรม (ดาวน์โหลดทีละส่วนได้) จึงต้องอัปเดตทุกครั้งที่เปลี่ยนเฟรม
 *  ช่องที่ยังไม่มีข้อมูลยังคงแสดงอยู่แต่กดไม่ได้ — ผู้ใช้จะได้รู้ว่ามีอะไรให้ดูได้บ้าง */
function syncLayerOptions(layers, activeLayer) {
  const select = el("layerSelect");
  select.innerHTML = layers
    .map((layer) => {
      const suffix = layer.available ? "" : " — ไม่มีข้อมูล";
      return `<option value="${layer.key}"${layer.available ? "" : " disabled"}>`
        + `${layer.label}${suffix}</option>`;
    })
    .join("");
  select.value = activeLayer;
  state.layer = activeLayer;

  // อัปเดต layer pills
  document.querySelectorAll(".layer-pill").forEach((pill) => {
    const matched = layers.find((l) => l.key === pill.dataset.layer);
    const available = matched ? matched.available : false;
    pill.disabled = !available;
    pill.classList.toggle("is-active", pill.dataset.layer === activeLayer);
    pill.setAttribute("title", available ? `${matched.label} (${matched.region})` : "ยังไม่มีข้อมูลเฟรมนี้");
  });
}

/** ส่งออกข้อมูล Flare เป็นไฟล์ CSV */
function exportFlaresCsv() {
  if (!state.goes || !state.goes.events || !state.goes.events.length) {
    alert("ไม่มีรายการ Flare ในช่วงเวลาที่เลือก");
    return;
  }
  const headers = ["goes_class", "start_time", "peak_time", "end_time", "peak_flux", "noaa_ar", "harpnum", "lat", "lon"];
  const rows = state.goes.events.map((e) => [
    e.goes_class ?? "",
    e.start_time ?? "",
    e.peak_time ?? "",
    e.end_time ?? "",
    e.peak_flux ?? "",
    e.noaa_ar ?? "",
    e.harpnum ?? "",
    e.lat ?? "",
    e.lon ?? "",
  ]);

  const csvContent = "data:text/csv;charset=utf-8,"
    + [headers.join(","), ...rows.map((r) => r.map((c) => `"${c}"`).join(","))].join("\n");

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `flares_${el("goesStart").value}_to_${el("goesEnd").value}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/** เลื่อนไปเฟรมที่ index — ตัวเดียวที่แตะ state.frameIndex */
async function showFrame(index) {
  if (!state.frames.length) return;
  const clamped = Math.max(0, Math.min(state.frames.length - 1, index));
  state.frameIndex = clamped;

  el("frameSelect").value = String(clamped);
  el("framePrev").disabled = clamped === 0;
  el("frameNext").disabled = clamped === state.frames.length - 1;
  el("frameStamp").textContent = `เฟรม ${clamped + 1} / ${state.frames.length}`;

  const timestamp = state.frames[clamped];
  const panel = el("framePanel");
  panel.innerHTML = `<p class="empty">กำลังโหลด…</p>`;

  const useTruth = el("useTruth").checked;
  // นับคำขอแทนการเทียบ frameIndex: การสลับเลเยอร์หรือติ๊ก mask ไม่ทำให้ index ขยับ
  // แต่ก็ยิงคำขอใหม่เหมือนกัน ตัวนับจึงเป็นตัวเดียวที่บอกได้ว่าผลไหนคือผลล่าสุด
  const requestId = ++state.frameRequest;

  try {
    const data = await fetchJson(
      `/api/segment?timestamp=${timestamp}&use_ground_truth=${useTruth}&layer=${state.layer}`
    );
    if (requestId !== state.frameRequest) return;

    syncLayerOptions(data.layers, data.layer);

    const dice = data.dice_vs_ground_truth !== null && data.dice_vs_ground_truth !== undefined
      ? ` · Dice เทียบ mask จริง ${data.dice_vs_ground_truth.toFixed(3)}`
      : "";
    const active = data.layers.find((l) => l.key === data.layer);
    panel.innerHTML = `
      <img src="data:image/png;base64,${data.image_png}" alt="${active?.label ?? ""} ${timestamp}">
      <div class="frame__meta">
        พบ ${data.n_regions} active region ·
        ครอบคลุม ${(data.predicted_area_fraction * 100).toFixed(2)}% ของภาพ${dice}<br>
        <span class="hint">${active ? `${active.label} · ${active.region} · ` : ""}${useTruth ? "mask จาก SHARP (ground truth)" : "mask ที่ U-Net ทำนาย"}</span>
      </div>`;

    updateFrameMarker();
  } catch (error) {
    if (requestId !== state.frameRequest) return;
    panel.innerHTML = `<p class="empty">${error.message}</p>`;
  }
}

/* ── โครงหน้า: แถบบน, การเผยตัว, การกระโดดข้าม section ────────── */

function setupChrome() {
  const topbar = el("topbar");

  // แถบบนทึบขึ้นเมื่อพ้น hero — อ่านตัวอักษรทับกราฟได้โดยไม่ต้องมีพื้นทึบตลอดเวลา
  const onScroll = () => topbar.classList.toggle("is-stuck", window.scrollY > 40);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ลิงก์ใน nav สว่างตาม section ที่อยู่ในสายตา
  const links = new Map(
    [...document.querySelectorAll(".topnav a")].map((a) => [a.getAttribute("href").slice(1), a])
  );
  const spy = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((link, id) => link.classList.toggle("is-current", id === entry.target.id));
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((_, id) => {
    const section = el(id);
    if (section) spy.observe(section);
  });

  // การ์ดค่อย ๆ ปรากฏเมื่อเลื่อนถึง — ยกเลิกการเฝ้าเมื่อเห็นแล้ว ไม่วิ่งซ้ำตอนเลื่อนกลับ
  const reveal = new IntersectionObserver((entries, observer) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    }
  }, { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll("[data-reveal]").forEach((node) => reveal.observe(node));

  // คีย์ลัดแป้นพิมพ์: ซ้าย/ขวา เลื่อนเฟรม, Spacebar เล่น/หยุด Timelapse
  window.addEventListener("keydown", (e) => {
    if (["input", "select", "textarea"].includes(document.activeElement?.tagName?.toLowerCase())) return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      showFrame(state.frameIndex - 1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      showFrame(state.frameIndex + 1);
    } else if (e.code === "Space") {
      e.preventDefault();
      toggleTimelapse();
    }
  });
}

/* ── เริ่มทำงาน ──────────────────────────────────────────────── */

/** วาดใหม่ทุกแผงที่ผูกกับช่วงเวลา — เรียกทุกครั้งที่แถบกรองเปลี่ยน */
async function applyRange() {
  const button = el("goesReload");
  button.disabled = true;
  try {
    syncFrameToRange();   // ไม่ await — ภาพโหลดคู่ขนานไปกับกราฟ
    await Promise.all([loadGoes(), loadHeroRisk(), loadExtraction()]);
  } finally {
    button.disabled = false;
  }
}

async function init() {
  setupChrome();

  try {
    await loadHealth();
  } catch (error) {
    showBanner(`เชื่อมต่อ backend ไม่สำเร็จ: ${error.message}`);
    return;
  }

  el("onlyFlaring").addEventListener("change", renderHarps);
  el("harpSearch").addEventListener("input", renderHarps);
  el("goesReload").addEventListener("click", applyRange);
  
  const exportBtn = el("exportFlaresBtn");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportFlaresCsv);
  }

  const arSelect = el("extractionArSelect");
  if (arSelect) {
    arSelect.addEventListener("change", () => {
      if (state.extraction) renderExtraction(state.extraction);
    });
  }

  const metricSelect = el("extractionMetricSelect");
  if (metricSelect) {
    metricSelect.addEventListener("change", () => {
      if (state.extraction) renderExtraction(state.extraction);
    });
  }

  // ผูกปุ่ม Presets เหตุการณ์สำคัญ
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");

      el("goesStart").value = btn.dataset.start;
      el("goesEnd").value = btn.dataset.end;

      if (btn.dataset.harp) {
        const harpnum = Number(btn.dataset.harp);
        await selectHarp(harpnum, { syncRange: false });
      }

      await applyRange();
    });
  });

  document.querySelectorAll(".sort").forEach((button) => {
    button.addEventListener("click", () => setSort(button.dataset.sort));
  });

  // สลับโมเดล/split ของแผง confusion matrix — แต่ละกลุ่มเป็น toggle เดี่ยว
  // (ไม่ใช่ checkbox) จึงจัดการ is-active เองแทนใช้ :checked ของ radio จริง
  document.querySelectorAll("[data-cm-model]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.cmModel === state.confusionModel) return;
      state.confusionModel = button.dataset.cmModel;
      button.parentElement.querySelectorAll("[data-cm-model]")
        .forEach((b) => b.classList.toggle("is-active", b === button));
      loadConfusionMatrix();
    });
  });
  document.querySelectorAll("[data-cm-split]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.cmSplit === state.confusionSplit) return;
      state.confusionSplit = button.dataset.cmSplit;
      button.parentElement.querySelectorAll("[data-cm-split]")
        .forEach((b) => b.classList.toggle("is-active", b === button));
      loadConfusionMatrix();
    });
  });

  await loadHarps();
  await loadInfo();       // ต้องรอ loadHarps เพราะ hero นับจำนวน HARP จากรายการที่โหลดมา
  await loadConfusionMatrix();
  await loadFrames();

  // เลือก AR ที่เกิด flare มากที่สุดให้อัตโนมัติ เพื่อให้หน้าเว็บมีอะไรแสดงทันที
  const firstFlaring = visibleHarps().find((h) => h.n_positive > 0) ?? state.harps[0];
  if (firstFlaring) {
    await selectHarp(firstFlaring.harpnum, { syncRange: true });
  } else {
    await applyRange();
  }
}

document.addEventListener("DOMContentLoaded", init);

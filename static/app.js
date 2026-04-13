/* ═══════════════════════════════════════════════════
   CreditAI — Frontend Logic
   ═══════════════════════════════════════════════════ */

const API = "/api/predict";

// ── Preset profiles ───────────────────────────────────────────────────────────
const PRESETS = {
  safe: {
    week_num: 26, month: 6, is_q3: false,
    mobile_phones: 2, home_phones: 1,
    annuity: 6000, num_payments: 48, payments_count: 200, payments_sum: 500000,
    late_pct: 0.5, days90: 0, total_settled: 450000, eir: 8.5,
  },
  medium: {
    week_num: 15, month: 4, is_q3: false,
    mobile_phones: 1, home_phones: 0,
    annuity: 12000, num_payments: 24, payments_count: 60, payments_sum: 150000,
    late_pct: 18, days90: 15, total_settled: 80000, eir: 22,
  },
  high: {
    week_num: 42, month: 10, is_q3: true,
    mobile_phones: 0, home_phones: 0,
    annuity: 22000, num_payments: 12, payments_count: 20, payments_sum: 40000,
    late_pct: 65, days90: 120, total_settled: 10000, eir: 48,
  },
};

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

// ── Range input live value display ───────────────────────────────────────────
function syncRange(inputId, displayId) {
  const inp = $(inputId), disp = $(displayId);
  if (!inp || !disp) return;
  disp.textContent = inp.value;
  on(inp, "input", () => (disp.textContent = inp.value));
}

// ── Apply preset ──────────────────────────────────────────────────────────────
function applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  Object.entries(p).forEach(([k, v]) => {
    const el = $(k);
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!v;
    else el.value = v;
    el.dispatchEvent(new Event("input"));
  });
  showToast(`Loaded: ${key.charAt(0).toUpperCase() + key.slice(1)} applicant profile`);
}

// ── Collect form data ─────────────────────────────────────────────────────────
// ── Collect form data ─────────────────────────────────────────────────────────
function safeInt(id, fallback)   { const v = parseInt($(id).value);   return isNaN(v) ? fallback : v; }
function safeFloat(id, fallback) { const v = parseFloat($(id).value); return isNaN(v) ? fallback : v; }

function getFormData() {
  return {
    week_num:       safeInt("week_num",   26),
    month:          safeInt("month",       6),
    is_q3:          $("is_q3").checked,
    mobile_phones:  safeInt("mobile_phones", 0),
    home_phones:    safeInt("home_phones",   0),
    annuity:        safeFloat("annuity",     0),
    num_payments:   safeInt("num_payments",  0),
    payments_count: safeInt("payments_count", 0),
    payments_sum:   safeFloat("payments_sum", 0),
    late_pct:       safeFloat("late_pct",    0),
    days90:         safeInt("days90",        0),
    total_settled:  safeFloat("total_settled", 0),
    eir:            safeFloat("eir",         0),
  };
}

// ── Gauge SVG drawing ─────────────────────────────────────────────────────────
function drawGauge(pct, color) {
  const arcFill  = $("gauge-arc-fill");
  const arcTrack = $("gauge-arc-track");
  const cx = 130, cy = 130, r = 110;
  const startAngle = -210, totalAngle = 240;
  const endAngle   = startAngle + totalAngle;          // 30°
  // Clamp fill so arc always has a minimum 2° sweep (prevents degenerate SVG path)
  const angle = startAngle + Math.max((pct / 100) * totalAngle, 2);

  function polar(deg, radius) {
    const rad = (deg * Math.PI) / 180;
    return [cx + radius * Math.cos(rad), cy + radius * Math.sin(rad)];
  }

  function arcPath(aDeg, bDeg, thick = 18) {
    const s  = polar(aDeg, r),           e  = polar(bDeg, r);
    const is = polar(aDeg, r - thick),   ie = polar(bDeg, r - thick);
    const large = bDeg - aDeg > 180 ? 1 : 0;
    return `M ${s[0].toFixed(2)} ${s[1].toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e[0].toFixed(2)} ${e[1].toFixed(2)} L ${ie[0].toFixed(2)} ${ie[1].toFixed(2)} A ${r - thick} ${r - thick} 0 ${large} 0 ${is[0].toFixed(2)} ${is[1].toFixed(2)} Z`;
  }

  // Draw track using same polar() function — fill always lies exactly on top of it
  arcTrack.setAttribute("d", arcPath(startAngle, endAngle));

  arcFill.setAttribute("d",    arcPath(startAngle, angle));
  arcFill.setAttribute("fill", color);

  // Zones label positions
  const zoneLabels = [
    { deg: -185, text: "0", col: "#22C55E" },
    { deg: -110, text: "30", col: "#22C55E" },
    { deg:  -30, text: "60", col: "#F59E0B" },
    { deg:   30, text: "100", col: "#EF4444" },
  ];
  const labelsG = $("gauge-labels");
  labelsG.innerHTML = "";
  zoneLabels.forEach(({ deg, text, col }) => {
    const [x, y] = polar(deg, r + 20);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x); t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-size", "10");
    t.setAttribute("fill", col);
    t.textContent = text;
    labelsG.appendChild(t);
  });

  // Needle
  const [nx, ny] = polar(angle, r - 9);
  const [cx2, cy2] = [cx, cy];
  $("gauge-needle").setAttribute("x1", cx2);
  $("gauge-needle").setAttribute("y1", cy2);
  $("gauge-needle").setAttribute("x2", nx);
  $("gauge-needle").setAttribute("y2", ny);
  $("gauge-needle").setAttribute("stroke", color);

  // Center dot
  $("gauge-dot").setAttribute("cx", cx2);
  $("gauge-dot").setAttribute("cy", cy2);
  $("gauge-dot").setAttribute("fill", color);
}

// ── Render prediction result ──────────────────────────────────────────────────
function renderResult(data) {
  const { probability, risk_tier, risk_color, advice, decisions, risk_factors } = data;

  // Show result panel
  $("welcome-panel").style.display  = "none";
  $("result-panel").style.display   = "grid";

  // Gauge
  drawGauge(probability, risk_color);
  $("gauge-pct").textContent   = probability.toFixed(1) + "%";
  $("gauge-pct").style.color   = risk_color;
  $("gauge-tier").textContent  = risk_tier;
  $("gauge-tier").style.color  = risk_color;

  // Risk banner
  const banner = $("risk-banner");
  banner.style.borderColor     = risk_color;
  banner.style.background      = risk_color + "18";
  $("risk-title").textContent  = risk_tier;
  $("risk-title").style.color  = risk_color;
  $("risk-advice").textContent = advice;

  // Progress bar
  $("progress-fill").style.width      = probability + "%";
  $("progress-fill").style.background = `linear-gradient(90deg, ${risk_color}aa, ${risk_color})`;
  $("progress-pct").textContent       = probability.toFixed(1) + "%";
  $("progress-pct").style.color       = risk_color;

  // Decision cards
  const THRESH_META = {
    "Conservative (0.30)": { desc: "Catch max defaults",    icon: "🛡️" },
    "Balanced (0.45)":     { desc: "Standard practice",     icon: "⚖️" },
    "Cost-Optimal (0.60)": { desc: "Best business ROI",     icon: "💰" },
    "High Precision (0.75)":{ desc: "Minimize false alarms",icon: "🎯" },
  };
  const dl = $("decision-list");
  dl.innerHTML = "";
  Object.entries(decisions).forEach(([label, verdict]) => {
    const meta     = THRESH_META[label] || { desc: "", icon: "🔹" };
    const approved = verdict === "APPROVE";
    const dColor   = approved ? "#22C55E" : "#EF4444";
    const dText    = approved ? "✅ APPROVE" : "❌ DECLINE";
    dl.innerHTML += `
      <div class="decision-row">
        <div class="decision-row-left">
          <span class="strategy">${meta.icon} ${label.split("(")[0].trim()}</span>
          <span class="thresh">${label.match(/\((.*?)\)/)?.[0] || ""}</span>
          <div class="desc">${meta.desc}</div>
        </div>
        <span class="decision-badge ${approved ? "approve" : "decline"}">${dText}</span>
      </div>`;
  });

  // Risk factor bars
  const fl  = $("factor-list");
  const max = Math.max(...risk_factors.map(f => f.value), 0.01);
  fl.innerHTML = "";
  risk_factors.forEach(f => {
    fl.innerHTML += `
      <div class="factor-row">
        <div class="factor-header">
          <span class="factor-label">${f.label}</span>
          <span class="factor-val" style="color:${f.color}">${f.value.toFixed(1)}%</span>
        </div>
        <div class="factor-bar-track">
          <div class="factor-bar-fill"
               style="width:${(f.value/max*100).toFixed(1)}%;background:${f.color}"></div>
        </div>
      </div>`;
  });
}

// ── Submit prediction ─────────────────────────────────────────────────────────
async function submitPrediction() {
  const btn = $("predict-btn");
  btn.disabled = true;
  $("spinner").classList.add("show");

  try {
    const payload  = getFormData();
    const response = await fetch(API, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Prediction failed");
    }
    const data = await response.json();
    renderResult(data);
    switchTab("predict");
    showToast("Analysis complete!");
    // Kick off RAG explanation asynchronously (non-blocking)
    fetchExplanation(payload);
  } catch (e) {
    showToast("Error: " + e.message, true);
    console.error(e);
  } finally {
    btn.disabled = false;
    $("spinner").classList.remove("show");
  }
}

// ── RAG Explanation ───────────────────────────────────────────────────────────
async function fetchExplanation(payload) {
  const panel   = $("ai-panel");
  const spinner = $("ai-spinner");
  const body    = $("ai-body");

  // Show panel with loading state
  panel.style.display = "block";
  if (spinner) spinner.classList.add("spinning");
  if (body)    body.style.opacity = "0.4";

  try {
    const res = await fetch("/api/explain", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Explain API error");
    const data = await res.json();
    renderExplanation(data);
  } catch (e) {
    const narr = $("ai-narrative");
    if (narr) narr.textContent = "AI analysis unavailable — " + e.message;
  } finally {
    if (spinner) spinner.classList.remove("spinning");
    if (body)    body.style.opacity = "1";
  }
}

function renderExplanation(data) {
  // Narrative
  const narr = $("ai-narrative");
  if (narr) {
    narr.innerHTML = data.narrative
      .replace(/\[CRITICAL\]/g, '<span class="risk-badge critical">CRITICAL</span>')
      .replace(/\[HIGH\]\s*/g,  '<span class="risk-badge high">HIGH</span>')
      .replace(/\[MODERATE\]/g, '<span class="risk-badge medium">MODERATE</span>')
      .replace(/\[LOW\]\s*/g,   '<span class="risk-badge low-badge">LOW</span>')
      .replace(/\n/g, "<br>");
  }

  // Key drivers
  const drv = $("ai-drivers");
  if (drv && data.key_drivers && data.key_drivers.length) {
    drv.innerHTML = "<div class='ai-drivers-title'>Feature-Level Insights</div>" +
      data.key_drivers.map(d => `
        <div class="ai-driver-card risk-${d.risk.toLowerCase()}">
          <div class="ai-driver-top">
            <span class="ai-driver-name">${d.feature}</span>
            <span class="ai-driver-val">${d.value}</span>
            <span class="ai-driver-risk risk-badge ${d.risk.toLowerCase()}">${d.risk}</span>
          </div>
          <div class="ai-driver-insight">${d.insight}</div>
        </div>`).join("");
  }

  // RAG transparency footer
  const info = $("ai-rag-info");
  if (info) {
    const snippets = (data.rag_snippets || []).slice(0, 1);
    info.innerHTML =
      `<span class="rag-tag">RAG</span> ` +
      `${data.rag_docs_used} definitions retrieved from ChromaDB knowledge base` +
      (snippets.length
        ? `<details class="rag-detail"><summary>View retrieved context</summary>` +
          snippets.map(s => `<p class="rag-snippet">${s.substring(0, 180)}…</p>`).join("") +
          `</details>`
        : "");
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(id) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  const btn   = document.querySelector(`.tab-btn[data-tab="${id}"]`);
  const panel = $(`tab-${id}`);
  if (btn)   btn.classList.add("active");
  if (panel) panel.classList.add("active");
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.style.borderColor = isError ? "#EF4444" : "rgba(99,179,237,0.35)";
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}

// ── Model comparison mini-bars (Insights tab) ─────────────────────────────────
function renderInsights() {
  // Model comparison bars
  const modelData = [
    { name: "AUC-ROC",  lgbm: 0.803, lr: 0.500 },
    { name: "Recall",   lgbm: 0.696, lr: 1.000 },
    { name: "Accuracy", lgbm: 0.747, lr: 0.031 },
  ];
  const mc = $("model-comparison");
  mc.innerHTML = "";
  modelData.forEach(d => {
    mc.innerHTML += `
      <div style="margin-bottom:1rem;">
        <div style="font-size:0.75rem;color:var(--muted);font-weight:600;margin-bottom:5px;">${d.name}</div>
        <div class="bar-row" style="gap:0.5rem;margin-bottom:3px;">
          <span style="width:130px;font-size:0.72rem;color:#93C5FD;flex-shrink:0;">LightGBM ✓</span>
          <div class="bar-track" style="flex:1">
            <div class="bar-fill" style="width:${d.lgbm*100}%;background:#3B82F6;height:8px;border-radius:99px;"></div>
          </div>
          <span style="font-size:0.72rem;font-weight:700;color:#93C5FD;width:38px;">${d.lgbm.toFixed(3)}</span>
        </div>
        <div class="bar-row" style="gap:0.5rem;">
          <span style="width:130px;font-size:0.72rem;color:var(--faint);flex-shrink:0;">Logistic Regression</span>
          <div class="bar-track" style="flex:1">
            <div class="bar-fill" style="width:${d.lr*100}%;background:#475569;height:8px;border-radius:99px;"></div>
          </div>
          <span style="font-size:0.72rem;font-weight:700;color:var(--faint);width:38px;">${d.lr.toFixed(3)}</span>
        </div>
      </div>`;
  });

  // Threshold curve with canvas
  drawThresholdChart();

  // SHAP bars
  const shapData = [
    { name: "education_1103M", val: 14.8 },
    { name: "WEEK_NUM",        val: 14.6 },
    { name: "mobilephncnt",    val: 12.1 },
    { name: "late_pct_1d",     val: 10.8 },
    { name: "homephncnt",      val: 10.3 },
    { name: "pmtnum_254L",     val:  8.6 },
    { name: "lastreject_M",    val:  7.7 },
    { name: "pmtssum_45A",     val:  7.5 },
    { name: "days90_310L",     val:  7.3 },
    { name: "pmtscount_423L",  val:  7.2 },
  ];
  const max = shapData[0].val;
  const sb  = $("shap-bars");
  sb.innerHTML = "";
  shapData.forEach(d => {
    const blues = ["#60A5FA","#3B82F6","#2563EB","#1D4ED8","#1E40AF"];
    const col   = blues[Math.min(Math.floor((d.val / max) * 5), 4)];
    sb.innerHTML += `
      <div class="factor-row">
        <div class="factor-header">
          <span class="factor-label">${d.name}</span>
          <span class="factor-val" style="color:${col}">${d.val.toFixed(1)}%</span>
        </div>
        <div class="factor-bar-track">
          <div class="factor-bar-fill" style="width:${(d.val/max*100).toFixed(1)}%;background:${col}"></div>
        </div>
      </div>`;
  });
}

function drawThresholdChart() {
  const canvas = $("thresh-canvas");
  if (!canvas) return;
  // requestAnimationFrame ensures the canvas is visible and laid out before
  // we read its dimensions (fixes blank chart when Insights tab is opened first time)
  requestAnimationFrame(() => {
  const ctx    = canvas.getContext("2d");
  canvas.width  = canvas.clientWidth  || 440;
  canvas.height = 220;
  const W = canvas.width, H = canvas.height;
  const pad = { t: 20, r: 20, b: 36, l: 36 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

  const thresholds = [0.10,0.20,0.30,0.40,0.45,0.50,0.60,0.70,0.75,0.80,0.90];
  const datasets   = [
    { label: "Precision", data: [0.036,0.042,0.052,0.064,0.074,0.083,0.108,0.140,0.161,0.215,0.325], color: "#3B82F6" },
    { label: "Recall",    data: [0.991,0.960,0.930,0.880,0.760,0.696,0.565,0.420,0.308,0.180,0.054], color: "#F59E0B" },
    { label: "F1",        data: [0.070,0.081,0.099,0.119,0.135,0.148,0.181,0.201,0.212,0.208,0.093], color: "#22C55E" },
  ];

  const toX = (t) => pad.l + ((t - 0.1) / 0.8) * iW;
  const toY = (v) => pad.t + (1 - v) * iH;

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth   = 1;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(W - pad.r, toY(v)); ctx.stroke();
    ctx.fillStyle = "rgba(148,163,184,0.7)"; ctx.font = "10px Inter";
    ctx.fillText((v * 100) + "%", 4, toY(v) + 4);
  });

  // Vertical markers
  [0.60, 0.75].forEach((t, i) => {
    ctx.strokeStyle = i === 0 ? "rgba(59,130,246,0.5)" : "rgba(34,197,94,0.5)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(toX(t), pad.t); ctx.lineTo(toX(t), H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = i === 0 ? "#3B82F6" : "#22C55E";
    ctx.font = "9px Inter"; ctx.textAlign = "center";
    ctx.fillText(i === 0 ? "Cost-Opt" : "Best F1", toX(t), pad.t - 4);
  });

  // Lines
  datasets.forEach(ds => {
    ctx.beginPath();
    ctx.strokeStyle = ds.color; ctx.lineWidth = 2.2; ctx.lineJoin = "round";
    thresholds.forEach((t, i) => {
      i === 0 ? ctx.moveTo(toX(t), toY(ds.data[i])) : ctx.lineTo(toX(t), toY(ds.data[i]));
    });
    ctx.stroke();
    // Dots
    thresholds.forEach((t, i) => {
      ctx.beginPath();
      ctx.arc(toX(t), toY(ds.data[i]), 3, 0, Math.PI * 2);
      ctx.fillStyle = ds.color; ctx.fill();
    });
  });

  // X-axis
  ctx.fillStyle = "rgba(148,163,184,0.7)"; ctx.font = "10px Inter"; ctx.textAlign = "center";
  thresholds.forEach(t => ctx.fillText(t.toFixed(2), toX(t), H - pad.b + 14));
  ctx.fillStyle = "#94A3B8"; ctx.font = "11px Inter";
  ctx.fillText("Threshold", W / 2, H - 4);

  // Legend
  const lx = pad.l + 4;
  datasets.forEach((ds, i) => {
    const lxp = lx + i * 90;
    ctx.fillStyle   = ds.color;
    ctx.fillRect(lxp, pad.t + 2, 16, 4);
    ctx.fillStyle   = "#CBD5E0"; ctx.font = "9px Inter"; ctx.textAlign = "left";
    ctx.fillText(ds.label, lxp + 20, pad.t + 8);
  });
  }); // end requestAnimationFrame
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Range displays
  syncRange("late_pct", "late_pct_val");
  syncRange("week_num", "week_num_val");
  syncRange("eir",      "eir_val_disp");

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    on(btn, "click", () => {
      switchTab(btn.dataset.tab);
      if (btn.dataset.tab === "insights") renderInsights();
    });
  });

  // Preset buttons
  on($("btn-safe"),   "click", () => applyPreset("safe"));
  on($("btn-medium"), "click", () => applyPreset("medium"));
  on($("btn-high"),   "click", () => applyPreset("high"));

  // Predict
  on($("predict-btn"), "click", submitPrediction);

  // Allow Enter key to predict
  document.querySelectorAll("input, select").forEach(el =>
    on(el, "keydown", (e) => { if (e.key === "Enter") submitPrediction(); })
  );

  // Resize canvas when insights tab opens
  window.addEventListener("resize", () => {
    if ($("tab-insights").classList.contains("active")) drawThresholdChart();
  });
});

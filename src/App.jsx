import { useState, useMemo } from "react";

const C = {
  navy:       "#1B2A4A",
  navy2:      "#253D6B",
  gold:       "#C8A84B",
  goldLight:  "#F5E9C0",
  offWhite:   "#F8F6F0",
  lightGray:  "#E8E4DC",
  gray:       "#6B7280",
  green:      "#2E6B3E",
  greenLight: "#D4EDDA",
  red:        "#B82424",
  redLight:   "#FDEAEA",
  amber:      "#C77B0D",
  amberLight: "#FEF3DC",
  white:      "#FFFFFF",
  htext:      "#8BA0C0",
  teal:       "#1A5C6B",
  tealLight:  "#D0EAF0",
  purple:     "#5B3A8A",
  purpleLight:"#E8DEFF",
};

const METHOD_COLORS = {
  sma:  "#1B2A4A",
  wma:  "#2E6B3E",
  ses:  "#C77B0D",
  holt: "#1A5C6B",
  hw:   "#5B3A8A",
};

const METHOD_LABELS = {
  sma:  "Simple Moving Average",
  wma:  "Weighted Moving Average",
  ses:  "Single Exponential Smoothing",
  holt: "Holt's Double (Trend)",
  hw:   "Holt-Winters Triple (Trend + Seasonality)",
};

const METHOD_SHORT = {
  sma:  "SMA",
  wma:  "WMA",
  ses:  "SES",
  holt: "Holt's",
  hw:   "Holt-Winters",
};

const METHOD_BEST_FOR = {
  sma:  "Stable demand, no trend, no seasonality",
  wma:  "Stable to mild trend, emphasizes recent periods",
  ses:  "Irregular demand, no consistent trend",
  holt: "Data with a trend but no seasonality",
  hw:   "Data with both trend and seasonality",
};

// ── Forecasting math ──────────────────────────────────────────────────────────

function calcSMA(data, period) {
  const fitted = new Array(data.length).fill(null);
  for (let i = period; i < data.length; i++) {
    const slice = data.slice(i - period, i);
    fitted[i] = slice.reduce((a, b) => a + b, 0) / period;
  }
  return fitted;
}

function calcWMA(data, period) {
  const fitted = new Array(data.length).fill(null);
  const weights = Array.from({ length: period }, (_, i) => i + 1);
  const wSum = weights.reduce((a, b) => a + b, 0);
  for (let i = period; i < data.length; i++) {
    const slice = data.slice(i - period, i);
    fitted[i] = slice.reduce((sum, v, j) => sum + v * weights[j], 0) / wSum;
  }
  return fitted;
}

function calcSES(data, alpha) {
  const fitted = new Array(data.length).fill(null);
  let s = data[0];
  fitted[0] = s;
  for (let i = 1; i < data.length; i++) {
    fitted[i] = s;
    s = alpha * data[i] + (1 - alpha) * s;
  }
  return { fitted, lastLevel: s };
}

function calcHolt(data, alpha, beta) {
  const fitted = new Array(data.length).fill(null);
  let l = data[0];
  let t = data.length > 1 ? data[1] - data[0] : 0;
  fitted[0] = l;
  for (let i = 1; i < data.length; i++) {
    fitted[i] = l + t;
    const prevL = l;
    l = alpha * data[i] + (1 - alpha) * (l + t);
    t = beta * (l - prevL) + (1 - beta) * t;
  }
  return { fitted, lastLevel: l, lastTrend: t };
}

function calcHW(data, alpha, beta, gamma, season) {
  if (data.length < season * 2) return { fitted: new Array(data.length).fill(null), lastLevel: data[data.length - 1], lastTrend: 0, lastSeasonals: new Array(season).fill(1) };

  // Initialize seasonals
  const nSeasons = Math.floor(data.length / season);
  const seasonAvgs = [];
  for (let i = 0; i < nSeasons; i++) {
    const slice = data.slice(i * season, (i + 1) * season);
    seasonAvgs.push(slice.reduce((a, b) => a + b, 0) / season);
  }
  const rawSeasonals = [];
  for (let i = 0; i < nSeasons; i++) {
    for (let j = 0; j < season; j++) {
      const idx = i * season + j;
      if (idx < data.length) {
        rawSeasonals.push(data[idx] / (seasonAvgs[i] || 1));
      }
    }
  }
  const initSeasonals = [];
  for (let j = 0; j < season; j++) {
    const vals = [];
    for (let i = j; i < rawSeasonals.length; i += season) vals.push(rawSeasonals[i]);
    initSeasonals.push(vals.reduce((a, b) => a + b, 0) / vals.length);
  }

  const seasonals = [...initSeasonals];
  let l = data.slice(0, season).reduce((a, b) => a + b, 0) / season;
  let t = 0;
  const fitted = new Array(data.length).fill(null);

  for (let i = 0; i < data.length; i++) {
    const si = i % season;
    const f = (l + t) * seasonals[si];
    if (i >= season) fitted[i] = f;
    const prevL = l;
    l = alpha * (data[i] / (seasonals[si] || 1)) + (1 - alpha) * (l + t);
    t = beta * (l - prevL) + (1 - beta) * t;
    seasonals[si] = gamma * (data[i] / (l || 1)) + (1 - gamma) * seasonals[si];
  }

  const lastSeasonals = seasonals;
  return { fitted, lastLevel: l, lastTrend: t, lastSeasonals };
}

function errorMetrics(actual, fitted) {
  const pairs = actual.map((a, i) => ({ a, f: fitted[i] })).filter(p => p.f !== null && p.a !== null && !isNaN(p.f));
  if (pairs.length === 0) return { mad: null, mse: null, mape: null };
  const n = pairs.length;
  const mad  = pairs.reduce((s, p) => s + Math.abs(p.a - p.f), 0) / n;
  const mse  = pairs.reduce((s, p) => s + Math.pow(p.a - p.f, 2), 0) / n;
  const mape = pairs.reduce((s, p) => s + Math.abs((p.a - p.f) / (p.a || 1)), 0) / n * 100;
  return { mad, mse, mape };
}

function generateForecast(method, state, horizon, season) {
  const forecast = [];
  if (method === "sma" || method === "wma") {
    // Last fitted value repeated
    const lastVal = state.lastFitted;
    for (let h = 1; h <= horizon; h++) forecast.push(Math.max(0, lastVal));
  } else if (method === "ses") {
    for (let h = 1; h <= horizon; h++) forecast.push(Math.max(0, state.lastLevel));
  } else if (method === "holt") {
    for (let h = 1; h <= horizon; h++) forecast.push(Math.max(0, state.lastLevel + h * state.lastTrend));
  } else if (method === "hw") {
    for (let h = 1; h <= horizon; h++) {
      const si = h % season;
      forecast.push(Math.max(0, (state.lastLevel + h * state.lastTrend) * (state.lastSeasonals[si] || 1)));
    }
  }
  return forecast;
}

// ── Optimization: find best alpha for SES, alpha+beta for Holt ───────────────
function optimizeSES(data) {
  let bestAlpha = 0.3, bestMAPE = Infinity;
  for (let a = 0.1; a <= 0.9; a += 0.05) {
    const { fitted } = calcSES(data, a);
    const { mape } = errorMetrics(data, fitted);
    if (mape !== null && mape < bestMAPE) { bestMAPE = mape; bestAlpha = a; }
  }
  return bestAlpha;
}

function optimizeHolt(data) {
  let bestA = 0.3, bestB = 0.1, bestMAPE = Infinity;
  for (let a = 0.1; a <= 0.9; a += 0.1) {
    for (let b = 0.05; b <= 0.5; b += 0.05) {
      const { fitted } = calcHolt(data, a, b);
      const { mape } = errorMetrics(data, fitted);
      if (mape !== null && mape < bestMAPE) { bestMAPE = mape; bestA = a; bestB = b; }
    }
  }
  return { alpha: bestA, beta: bestB };
}

function optimizeHW(data, season) {
  let bestA = 0.3, bestB = 0.1, bestG = 0.1, bestMAPE = Infinity;
  if (data.length < season * 2) return { alpha: 0.3, beta: 0.1, gamma: 0.1 };
  for (let a of [0.1, 0.3, 0.5, 0.7]) {
    for (let b of [0.05, 0.1, 0.2]) {
      for (let g of [0.05, 0.1, 0.2, 0.3]) {
        const { fitted } = calcHW(data, a, b, g, season);
        const { mape } = errorMetrics(data, fitted);
        if (mape !== null && mape < bestMAPE) { bestMAPE = mape; bestA = a; bestB = b; bestG = g; }
      }
    }
  }
  return { alpha: bestA, beta: bestB, gamma: bestG };
}

// ── Safe stock calc ───────────────────────────────────────────────────────────
function safetyStock(mad, leadTimePeriods, serviceLevelZ) {
  return serviceLevelZ * mad * Math.sqrt(leadTimePeriods);
}

const SERVICE_LEVELS = [
  { label: "90%", z: 1.28 },
  { label: "95%", z: 1.645 },
  { label: "97%", z: 1.88 },
  { label: "99%", z: 2.326 },
];

// ── SVG Chart ────────────────────────────────────────────────────────────────
function ForecastChart({ actual, allFitted, forecast, bestMethod, labels, forecastLabels }) {
  const W = 600, H = 260, PL = 48, PR = 16, PT = 16, PB = 32;
  const cW = W - PL - PR, cH = H - PT - PB;

  const allVals = [
    ...actual.filter(v => v !== null),
    ...Object.values(allFitted).flat().filter(v => v !== null),
    ...(forecast || []).filter(v => v !== null),
  ];
  const minV = Math.min(...allVals) * 0.9;
  const maxV = Math.max(...allVals) * 1.1 || 1;
  const range = maxV - minV || 1;

  const totalPts = actual.length + (forecast?.length || 0);
  const xStep = cW / (totalPts - 1 || 1);

  const toX = (i) => PL + i * xStep;
  const toY = (v) => PT + cH - ((v - minV) / range) * cH;

  const pathFor = (vals, offset = 0) => {
    const pts = vals.map((v, i) => v !== null ? `${toX(i + offset)},${toY(v)}` : null).filter(Boolean);
    if (pts.length < 2) return "";
    return "M" + pts.join("L");
  };

  const yTicks = 5;
  const yTickVals = Array.from({ length: yTicks }, (_, i) => minV + (range * i) / (yTicks - 1));

  return (
    <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 340, display: "block" }}>
        {/* Grid */}
        {yTickVals.map((v, i) => (
          <g key={i}>
            <line x1={PL} y1={toY(v)} x2={W - PR} y2={toY(v)} stroke={C.lightGray} strokeWidth={0.5} />
            <text x={PL - 4} y={toY(v) + 4} textAnchor="end" fontSize={9} fill={C.gray}>
              {Math.round(v).toLocaleString()}
            </text>
          </g>
        ))}

        {/* Forecast region */}
        {forecast && forecast.length > 0 && (
          <rect
            x={toX(actual.length - 1)} y={PT}
            width={toX(actual.length + forecast.length - 1) - toX(actual.length - 1)}
            height={cH}
            fill="#F5E9C0" opacity={0.3}
          />
        )}

        {/* Fitted lines for non-best methods (faint) */}
        {Object.entries(allFitted).map(([method, vals]) => method !== bestMethod && (
          <path key={method} d={pathFor(vals)} fill="none"
            stroke={METHOD_COLORS[method]} strokeWidth={1} opacity={0.2} strokeDasharray="3,3" />
        ))}

        {/* Best method fitted */}
        {allFitted[bestMethod] && (
          <path d={pathFor(allFitted[bestMethod])} fill="none"
            stroke={METHOD_COLORS[bestMethod]} strokeWidth={2} opacity={0.8} />
        )}

        {/* Forecast line */}
        {forecast && forecast.length > 0 && allFitted[bestMethod] && (
          <path
            d={`M${toX(actual.length - 1)},${toY(allFitted[bestMethod].filter(v => v !== null).slice(-1)[0] || actual[actual.length - 1])}` +
               forecast.map((v, i) => `L${toX(actual.length + i)},${toY(v)}`).join("")}
            fill="none" stroke={METHOD_COLORS[bestMethod]} strokeWidth={2.5} strokeDasharray="6,3"
          />
        )}

        {/* Actual line */}
        <path d={pathFor(actual)} fill="none" stroke={C.navy} strokeWidth={2} />

        {/* Actual dots */}
        {actual.map((v, i) => v !== null && (
          <circle key={i} cx={toX(i)} cy={toY(v)} r={2.5} fill={C.navy} />
        ))}

        {/* Forecast dots */}
        {forecast && forecast.map((v, i) => (
          <circle key={i} cx={toX(actual.length + i)} cy={toY(v)} r={3}
            fill={METHOD_COLORS[bestMethod]} stroke={C.white} strokeWidth={1.5} />
        ))}

        {/* X axis labels - show every Nth */}
        {labels && labels.map((lbl, i) => {
          const skip = Math.ceil(labels.length / 8);
          if (i % skip !== 0 && i !== labels.length - 1) return null;
          return (
            <text key={i} x={toX(i)} y={H - 4} textAnchor="middle" fontSize={8} fill={C.gray}>
              {lbl}
            </text>
          );
        })}
        {forecastLabels && forecastLabels.map((lbl, i) => {
          const skip = Math.ceil(forecastLabels.length / 4);
          if (i % skip !== 0 && i !== forecastLabels.length - 1) return null;
          return (
            <text key={i} x={toX(actual.length + i)} y={H - 4} textAnchor="middle" fontSize={8} fill={METHOD_COLORS[bestMethod]}>
              {lbl}
            </text>
          );
        })}

        {/* Divider line */}
        {forecast && forecast.length > 0 && (
          <line x1={toX(actual.length - 1)} y1={PT} x2={toX(actual.length - 1)} y2={H - PB}
            stroke={C.gold} strokeWidth={1} strokeDasharray="4,2" />
        )}

        {/* Legend */}
        <g transform={`translate(${PL + 8}, ${PT + 8})`}>
          <rect x={0} y={0} width={120} height={36} rx={3} fill="white" opacity={0.85} />
          <line x1={6} y1={10} x2={18} y2={10} stroke={C.navy} strokeWidth={2} />
          <circle cx={12} cy={10} r={2} fill={C.navy} />
          <text x={22} y={14} fontSize={8} fill={C.navy}>Actual</text>
          <line x1={6} y1={26} x2={18} y2={26} stroke={METHOD_COLORS[bestMethod]} strokeWidth={2} strokeDasharray="4,2" />
          <circle cx={12} cy={26} r={2.5} fill={METHOD_COLORS[bestMethod]} stroke="white" strokeWidth={1} />
          <text x={22} y={30} fontSize={8} fill={METHOD_COLORS[bestMethod]}>
            {METHOD_SHORT[bestMethod]} (best fit)
          </text>
        </g>
      </svg>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function DemandForecaster() {
  const [activeTab, setActiveTab]       = useState("data");
  const [rawInput, setRawInput]         = useState("");
  const [periodLabel, setPeriodLabel]   = useState("Month");
  const [startLabel, setStartLabel]     = useState("Jan 2023");
  const [horizon, setHorizon]           = useState(6);
  const [smaPeriod, setSmaPeriod]       = useState(3);
  const [season, setSeason]             = useState(12);
  const [leadTime, setLeadTime]         = useState(4);
  const [serviceLevel, setServiceLevel] = useState(1);
  const [showAllMethods, setShowAllMethods] = useState(false);

  // Parse input data
  const data = useMemo(() => {
    return rawInput
      .split(/[\n,\t;]+/)
      .map(v => parseFloat(v.trim()))
      .filter(v => !isNaN(v) && v >= 0);
  }, [rawInput]);

  const hasData = data.length >= 6;
  const hasEnoughForHW = data.length >= season * 2;

  // Generate period labels
  const periodLabels = useMemo(() => {
    if (!hasData) return [];
    // Try to parse start label as month/year
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const match = startLabel.match(/^([A-Za-z]+)\s+(\d{4})$/);
    if (match) {
      const mIdx = months.findIndex(m => m.toLowerCase() === match[1].toLowerCase().slice(0,3));
      const yr = parseInt(match[2]);
      if (mIdx >= 0 && yr) {
        return data.map((_, i) => {
          const totalM = mIdx + i;
          return months[totalM % 12] + " " + (yr + Math.floor(totalM / 12));
        });
      }
    }
    return data.map((_, i) => `${periodLabel} ${i + 1}`);
  }, [data, startLabel, periodLabel, hasData]);

  const forecastLabels = useMemo(() => {
    if (!hasData || !periodLabels.length) return [];
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const last = periodLabels[periodLabels.length - 1];
    const match = last.match(/^([A-Za-z]+)\s+(\d{4})$/);
    if (match) {
      const mIdx = months.findIndex(m => m === match[1].slice(0,3));
      const yr = parseInt(match[2]);
      if (mIdx >= 0) {
        return Array.from({ length: horizon }, (_, i) => {
          const totalM = mIdx + i + 1;
          return months[totalM % 12] + " " + (yr + Math.floor(totalM / 12));
        });
      }
    }
    return Array.from({ length: horizon }, (_, i) => `${periodLabel} ${data.length + i + 1}`);
  }, [data, periodLabels, horizon, periodLabel, hasData]);

  // Run all methods
  const results = useMemo(() => {
    if (!hasData) return null;

    const smaFitted = calcSMA(data, Math.min(smaPeriod, data.length - 1));
    const wmaFitted = calcWMA(data, Math.min(smaPeriod, data.length - 1));

    const bestAlpha = optimizeSES(data);
    const { fitted: sesFitted, lastLevel: sesLevel } = calcSES(data, bestAlpha);

    const { alpha: hA, beta: hB } = optimizeHolt(data);
    const { fitted: holtFitted, lastLevel: holtLevel, lastTrend: holtTrend } = calcHolt(data, hA, hB);

    let hwFitted = new Array(data.length).fill(null);
    let hwState = { lastLevel: data[data.length-1], lastTrend: 0, lastSeasonals: new Array(season).fill(1) };
    let hwParams = { alpha: 0.3, beta: 0.1, gamma: 0.1 };
    if (hasEnoughForHW) {
      hwParams = optimizeHW(data, season);
      const hwResult = calcHW(data, hwParams.alpha, hwParams.beta, hwParams.gamma, season);
      hwFitted = hwResult.fitted;
      hwState = { lastLevel: hwResult.lastLevel, lastTrend: hwResult.lastTrend, lastSeasonals: hwResult.lastSeasonals };
    }

    const smaErrors = errorMetrics(data, smaFitted);
    const wmaErrors = errorMetrics(data, wmaFitted);
    const sesErrors = errorMetrics(data, sesFitted);
    const holtErrors = errorMetrics(data, holtFitted);
    const hwErrors = errorMetrics(data, hwFitted);

    const methods = {
      sma:  { fitted: smaFitted,  errors: smaErrors,  state: { lastFitted: smaFitted.filter(v=>v!==null).slice(-1)[0] || data[data.length-1] } },
      wma:  { fitted: wmaFitted,  errors: wmaErrors,  state: { lastFitted: wmaFitted.filter(v=>v!==null).slice(-1)[0] || data[data.length-1] } },
      ses:  { fitted: sesFitted,  errors: sesErrors,  state: { lastLevel: sesLevel }, params: { alpha: bestAlpha } },
      holt: { fitted: holtFitted, errors: holtErrors, state: { lastLevel: holtLevel, lastTrend: holtTrend }, params: { alpha: hA, beta: hB } },
      hw:   { fitted: hwFitted,   errors: hwErrors,   state: hwState, params: hwParams, available: hasEnoughForHW },
    };

    // Best method = lowest MAPE
    const ranked = Object.entries(methods)
      .filter(([k, v]) => v.errors.mape !== null && (k !== "hw" || hasEnoughForHW))
      .sort(([, a], [, b]) => a.errors.mape - b.errors.mape);

    const bestMethod = ranked[0]?.[0] || "ses";
    const bestState = methods[bestMethod].state;

    const forecast = generateForecast(bestMethod, bestState, horizon, season);

    // Safety stock from best method MAD
    const bestMAD = methods[bestMethod].errors.mad || 0;
    const slZ = SERVICE_LEVELS[serviceLevel].z;
    const ss = safetyStock(bestMAD, leadTime, slZ);
    const rop = (data.reduce((a,b) => a+b,0)/data.length) * leadTime + ss;

    return { methods, bestMethod, forecast, ranked, bestMAD, ss, rop, bestAlpha, hA, hB, hwParams };
  }, [data, smaPeriod, season, horizon, leadTime, serviceLevel, hasData, hasEnoughForHW]);

  const allFitted = results ? Object.fromEntries(Object.entries(results.methods).map(([k,v]) => [k, v.fitted])) : {};

  const fmtN = (n, d=1) => n === null ? "—" : Number(n).toFixed(d);
  const fmtPct = (n) => n === null ? "—" : n.toFixed(1) + "%";

  return (
    <div style={{ fontFamily: "'Georgia', serif", background: C.offWhite, minHeight: "100vh", color: C.navy }}>

      {/* Header */}
      <div style={{ background: C.navy, borderBottom: `3px solid ${C.gold}`, padding: "16px 16px 12px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 18, fontWeight: "bold", color: C.white, letterSpacing: 1 }}>DEMAND FORECASTING</span>
          <span style={{ fontSize: 10, color: C.gold, fontFamily: "sans-serif", letterSpacing: 2, fontWeight: "bold" }}>TOOL</span>
        </div>
        <div style={{ fontSize: 10, color: C.htext, marginTop: 3, fontFamily: "sans-serif" }}>
          Matthew Flanagan, CPSM · Flanagan Sourcing Intelligence Portfolio
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: C.navy2, borderBottom: `2px solid ${C.gold}`, overflowX: "auto" }}>
        <div style={{ display: "flex", minWidth: "max-content" }}>
          {[
            { key: "data",     label: "1. Data" },
            { key: "settings", label: "2. Settings" },
            { key: "results",  label: "3. Forecast" },
            { key: "safety",   label: "4. Safety Stock" },
          ].map(tab => (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
              background: activeTab === tab.key ? C.gold : "transparent",
              color: activeTab === tab.key ? C.navy : C.htext,
              border: "none", padding: "10px 16px", fontSize: 12,
              fontFamily: "sans-serif", fontWeight: "bold", letterSpacing: 1,
              cursor: "pointer", whiteSpace: "nowrap",
            }}>{tab.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: "16px 16px 48px", maxWidth: 800, margin: "0 auto" }}>

        {/* ── DATA TAB ── */}
        {activeTab === "data" && (
          <div>
            <div style={{ background: C.navy2, border: `1px solid ${C.gold}`, borderRadius: 6, padding: "12px 14px", marginBottom: 16, fontSize: 12, color: C.htext, fontFamily: "sans-serif", lineHeight: 1.6 }}>
              Enter your historical demand data below. Paste from Excel or type values separated by commas, line breaks, or tabs. Minimum 6 periods. 12 to 36 periods gives the most reliable results. The tool runs five forecasting methods simultaneously and recommends the best fit based on lowest error.
            </div>

            <Card title="Historical Demand Data" subtitle="Paste or type your demand values, one per line or comma-separated.">
              <textarea
                value={rawInput}
                onChange={e => setRawInput(e.target.value)}
                placeholder={"Paste your data here. For example:\n1240\n1380\n1190\n1420\n1350\n1480\n1290\n1510\n1400\n1560\n1320\n1610"}
                style={{
                  width: "100%", minHeight: 180, border: `1px solid ${C.lightGray}`,
                  borderRadius: 4, padding: "10px", fontSize: 13,
                  fontFamily: "sans-serif", color: C.navy, background: C.white,
                  outline: "none", resize: "vertical", boxSizing: "border-box",
                  lineHeight: 1.6,
                }}
              />
              {data.length > 0 && (
                <div style={{ marginTop: 8, display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <Stat label="Periods entered" value={data.length} />
                  <Stat label="Average demand" value={Math.round(data.reduce((a,b)=>a+b,0)/data.length).toLocaleString()} />
                  <Stat label="Min" value={Math.min(...data).toLocaleString()} />
                  <Stat label="Max" value={Math.max(...data).toLocaleString()} />
                  <Stat label="Status" value={data.length >= 6 ? "✓ Ready" : `Need ${6 - data.length} more`} color={data.length >= 6 ? C.green : C.amber} />
                </div>
              )}
            </Card>

            <Card title="Period Labels" subtitle="Help the tool generate readable axis labels.">
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div>
                  <FieldLabel>Period type</FieldLabel>
                  <select value={periodLabel} onChange={e => setPeriodLabel(e.target.value)} style={inputSt}>
                    <option>Month</option>
                    <option>Week</option>
                    <option>Quarter</option>
                    <option>Day</option>
                  </select>
                </div>
                <div>
                  <FieldLabel>First period (e.g. Jan 2023)</FieldLabel>
                  <input value={startLabel} onChange={e => setStartLabel(e.target.value)} placeholder="Jan 2023" style={inputSt} />
                </div>
              </div>
            </Card>

            <button onClick={() => setActiveTab("settings")} disabled={!hasData} style={{
              ...nextBtnSt,
              background: hasData ? C.navy : C.lightGray,
              color: hasData ? C.white : C.gray,
              border: `2px solid ${hasData ? C.gold : C.lightGray}`,
              cursor: hasData ? "pointer" : "not-allowed",
            }}>PROCEED TO SETTINGS →</button>
          </div>
        )}

        {/* ── SETTINGS TAB ── */}
        {activeTab === "settings" && (
          <div>
            <div style={{ background: C.navy2, border: `1px solid ${C.gold}`, borderRadius: 6, padding: "12px 14px", marginBottom: 16, fontSize: 12, color: C.htext, fontFamily: "sans-serif", lineHeight: 1.6 }}>
              Adjust these settings to match your data and business context. The tool will automatically optimize smoothing parameters (alpha, beta, gamma) to minimize forecast error. You just need to set the structural parameters below.
            </div>

            <Card title="Moving Average Period" subtitle="Used by Simple and Weighted Moving Average methods.">
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[2, 3, 4, 6].map(p => (
                  <button key={p} onClick={() => setSmaPeriod(p)} style={{
                    flex: 1, minWidth: 50, padding: "10px 8px", borderRadius: 4,
                    border: `2px solid ${smaPeriod === p ? C.gold : C.lightGray}`,
                    background: smaPeriod === p ? C.navy : C.white,
                    color: smaPeriod === p ? C.gold : C.gray,
                    fontWeight: "bold", fontFamily: "sans-serif", fontSize: 13, cursor: "pointer",
                  }}>{p}-period</button>
                ))}
              </div>
              <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 8 }}>
                Shorter periods react faster to change. Longer periods smooth more noise. 3 is a good default for monthly data.
              </div>
            </Card>

            <Card title="Seasonality Period" subtitle="Required for Holt-Winters. How many periods in one seasonal cycle?">
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[4, 12, 52].map(p => (
                  <button key={p} onClick={() => setSeason(p)} style={{
                    flex: 1, minWidth: 60, padding: "10px 8px", borderRadius: 4,
                    border: `2px solid ${season === p ? C.gold : C.lightGray}`,
                    background: season === p ? C.navy : C.white,
                    color: season === p ? C.gold : C.gray,
                    fontWeight: "bold", fontFamily: "sans-serif", fontSize: 13, cursor: "pointer",
                  }}>{p === 4 ? "4 (Quarterly)" : p === 12 ? "12 (Monthly)" : "52 (Weekly)"}</button>
                ))}
              </div>
              {!hasEnoughForHW && (
                <div style={{ background: C.amberLight, border: `1px solid ${C.amber}`, borderRadius: 4, padding: "8px 10px", marginTop: 10, fontSize: 11, fontFamily: "sans-serif", color: C.amber, fontWeight: "bold" }}>
                  ⚠ Holt-Winters requires at least {season * 2} periods of data. You have {data.length}. The tool will run the other four methods and exclude Holt-Winters.
                </div>
              )}
            </Card>

            <Card title="Forecast Horizon" subtitle="How many periods forward to forecast?">
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[3, 6, 12].map(h => (
                  <button key={h} onClick={() => setHorizon(h)} style={{
                    flex: 1, padding: "10px 8px", borderRadius: 4,
                    border: `2px solid ${horizon === h ? C.gold : C.lightGray}`,
                    background: horizon === h ? C.navy : C.white,
                    color: horizon === h ? C.gold : C.gray,
                    fontWeight: "bold", fontFamily: "sans-serif", fontSize: 13, cursor: "pointer",
                  }}>{h} periods</button>
                ))}
              </div>
            </Card>

            <button onClick={() => setActiveTab("results")} style={nextBtnSt}>RUN FORECAST →</button>
          </div>
        )}

        {/* ── RESULTS TAB ── */}
        {activeTab === "results" && (
          <div>
            {!hasData ? (
              <EmptyState>Enter demand data on the Data tab to run the forecast.</EmptyState>
            ) : !results ? (
              <EmptyState>Processing...</EmptyState>
            ) : (
              <>
                {/* Winner banner */}
                <div style={{ background: C.navy, border: `2px solid ${C.gold}`, borderRadius: 6, padding: "16px", marginBottom: 16 }}>
                  <div style={{ fontSize: 10, color: C.gold, fontFamily: "sans-serif", fontWeight: "bold", letterSpacing: 2, marginBottom: 8 }}>BEST-FIT FORECASTING METHOD</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                    <div style={{ width: 14, height: 14, borderRadius: "50%", background: METHOD_COLORS[results.bestMethod], flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 16, fontWeight: "bold", color: C.white }}>{METHOD_LABELS[results.bestMethod]}</div>
                      <div style={{ fontSize: 11, color: C.htext, fontFamily: "sans-serif", marginTop: 2 }}>Best for: {METHOD_BEST_FOR[results.bestMethod]}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 10, color: C.htext, fontFamily: "sans-serif" }}>MAPE</div>
                      <div style={{ fontSize: 20, fontWeight: "bold", color: C.gold, fontFamily: "sans-serif" }}>
                        {fmtPct(results.methods[results.bestMethod].errors.mape)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Chart */}
                <Card title="Actual vs. Forecast" subtitle={`${data.length} periods of history · ${horizon}-period forecast (dashed)`}>
                  <ForecastChart
                    actual={data}
                    allFitted={allFitted}
                    forecast={results.forecast}
                    bestMethod={results.bestMethod}
                    labels={periodLabels}
                    forecastLabels={forecastLabels}
                  />
                </Card>

                {/* Forecast values */}
                <Card title="Forecast Values" subtitle={`${horizon}-period forward forecast using ${METHOD_SHORT[results.bestMethod]}`}>
                  <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
                    <div style={{ display: "flex", gap: 8, minWidth: "max-content" }}>
                      {results.forecast.map((v, i) => (
                        <div key={i} style={{
                          minWidth: 80, background: C.navy, border: `1.5px solid ${METHOD_COLORS[results.bestMethod]}`,
                          borderRadius: 4, padding: "10px 8px", textAlign: "center",
                        }}>
                          <div style={{ fontSize: 9, color: C.htext, fontFamily: "sans-serif", marginBottom: 4 }}>{forecastLabels[i] || `Period +${i+1}`}</div>
                          <div style={{ fontSize: 16, fontWeight: "bold", fontFamily: "sans-serif", color: C.gold }}>
                            {Math.round(v).toLocaleString()}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
                    <Stat label="Forecast avg" value={Math.round(results.forecast.reduce((a,b)=>a+b,0)/results.forecast.length).toLocaleString()} />
                    <Stat label="Historical avg" value={Math.round(data.reduce((a,b)=>a+b,0)/data.length).toLocaleString()} />
                    <Stat label="Trend" value={(() => {
                      const fAvg = results.forecast.reduce((a,b)=>a+b,0)/results.forecast.length;
                      const hAvg = data.reduce((a,b)=>a+b,0)/data.length;
                      const pct = ((fAvg - hAvg) / hAvg) * 100;
                      return (pct >= 0 ? "▲ +" : "▼ ") + pct.toFixed(1) + "%";
                    })()} color={results.forecast.reduce((a,b)=>a+b,0)/results.forecast.length >= data.reduce((a,b)=>a+b,0)/data.length ? C.green : C.amber} />
                  </div>
                </Card>

                {/* Method comparison */}
                <Card
                  title="Method Comparison"
                  subtitle="All five methods scored. Lower error = better fit."
                  extra={
                    <button onClick={() => setShowAllMethods(!showAllMethods)} style={{ background: "none", border: `1px solid ${C.lightGray}`, borderRadius: 10, padding: "2px 10px", fontSize: 11, color: C.gray, fontFamily: "sans-serif", cursor: "pointer" }}>
                      {showAllMethods ? "Hide detail" : "Show detail"}
                    </button>
                  }
                >
                  <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "sans-serif", minWidth: 360 }}>
                      <thead>
                        <tr style={{ background: C.navy }}>
                          <th style={{ padding: "8px", textAlign: "left", color: C.white, fontSize: 11 }}>METHOD</th>
                          <th style={{ padding: "8px", textAlign: "center", color: C.gold, fontSize: 11 }}>MAPE</th>
                          <th style={{ padding: "8px", textAlign: "center", color: C.white, fontSize: 11 }}>MAD</th>
                          {showAllMethods && <th style={{ padding: "8px", textAlign: "center", color: C.white, fontSize: 11 }}>MSE</th>}
                          <th style={{ padding: "8px", textAlign: "center", color: C.white, fontSize: 11 }}>RANK</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.ranked.map(([method, mData], idx) => {
                          const isBest = method === results.bestMethod;
                          const isUnavail = method === "hw" && !hasEnoughForHW;
                          return (
                            <tr key={method} style={{ background: isBest ? C.greenLight : idx % 2 === 0 ? C.white : "#F2EFE8" }}>
                              <td style={{ padding: "8px" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: METHOD_COLORS[method], flexShrink: 0 }} />
                                  <div>
                                    <div style={{ fontSize: 12, fontWeight: isBest ? "bold" : "normal", color: C.navy }}>{METHOD_SHORT[method]}</div>
                                    {showAllMethods && <div style={{ fontSize: 10, color: C.gray }}>{METHOD_BEST_FOR[method]}</div>}
                                  </div>
                                </div>
                              </td>
                              <td style={{ padding: "8px", textAlign: "center", fontWeight: "bold", color: isBest ? C.green : C.navy }}>
                                {isUnavail ? "N/A" : fmtPct(mData.errors.mape)}
                              </td>
                              <td style={{ padding: "8px", textAlign: "center", color: C.navy }}>
                                {isUnavail ? "N/A" : fmtN(mData.errors.mad, 0)}
                              </td>
                              {showAllMethods && (
                                <td style={{ padding: "8px", textAlign: "center", color: C.gray }}>
                                  {isUnavail ? "N/A" : fmtN(mData.errors.mse, 0)}
                                </td>
                              )}
                              <td style={{ padding: "8px", textAlign: "center" }}>
                                {isBest ? (
                                  <span style={{ background: C.greenLight, border: `1px solid ${C.green}`, borderRadius: 3, padding: "2px 8px", fontSize: 10, fontWeight: "bold", color: C.green }}>BEST</span>
                                ) : isUnavail ? (
                                  <span style={{ fontSize: 10, color: C.gray }}>Insuf. data</span>
                                ) : (
                                  <span style={{ fontSize: 12, color: C.gray }}>#{idx + 1}</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {showAllMethods && (
                    <div style={{ marginTop: 12, padding: "10px 12px", background: "#F2EFE8", borderRadius: 4, fontSize: 11, fontFamily: "sans-serif", color: C.navy, lineHeight: 1.7 }}>
                      <strong>MAPE</strong> (Mean Absolute Percentage Error) — the primary ranking metric. Expresses average error as a % of actual demand. Lower is better. Under 10% is excellent; under 20% is acceptable for most planning purposes.<br/>
                      <strong>MAD</strong> (Mean Absolute Deviation) — average absolute error in units. Used directly in safety stock calculation.<br/>
                      <strong>MSE</strong> (Mean Squared Error) — penalizes large errors more heavily. Useful when large misses are especially costly.
                    </div>
                  )}
                </Card>

                {results.methods[results.bestMethod].params && (
                  <Card title="Optimized Parameters" subtitle="Auto-tuned to minimize forecast error on your data.">
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {Object.entries(results.methods[results.bestMethod].params).map(([k, v]) => (
                        <div key={k} style={{ flex: 1, minWidth: 80, background: C.navy, borderRadius: 4, padding: "10px", textAlign: "center" }}>
                          <div style={{ fontSize: 10, color: C.htext, fontFamily: "sans-serif", marginBottom: 4 }}>{k.toUpperCase()}</div>
                          <div style={{ fontSize: 18, fontWeight: "bold", fontFamily: "sans-serif", color: C.gold }}>{Number(v).toFixed(2)}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 8 }}>
                      Alpha controls responsiveness to recent changes. Beta controls trend sensitivity. Gamma controls seasonal adjustment speed. Values closer to 1.0 respond faster; values closer to 0 rely more on history.
                    </div>
                  </Card>
                )}

                <button onClick={() => setActiveTab("safety")} style={nextBtnSt}>PROCEED TO SAFETY STOCK →</button>
              </>
            )}
          </div>
        )}

        {/* ── SAFETY STOCK TAB ── */}
        {activeTab === "safety" && (
          <div>
            {!hasData || !results ? (
              <EmptyState>Run the forecast first to calculate safety stock recommendations.</EmptyState>
            ) : (
              <>
                <div style={{ background: C.navy2, border: `1px solid ${C.gold}`, borderRadius: 6, padding: "12px 14px", marginBottom: 16, fontSize: 12, color: C.htext, fontFamily: "sans-serif", lineHeight: 1.6 }}>
                  Safety stock is calculated using the forecast error (MAD) from the best-fit method. The formula is: Safety Stock = Z × MAD × √Lead Time. The reorder point adds average demand during lead time to the safety stock.
                </div>

                <Card title="Lead Time" subtitle="How many periods between placing an order and receiving it?">
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {[1, 2, 3, 4, 6, 8, 12].map(lt => (
                      <button key={lt} onClick={() => setLeadTime(lt)} style={{
                        flex: 1, minWidth: 42, padding: "10px 6px", borderRadius: 4,
                        border: `2px solid ${leadTime === lt ? C.gold : C.lightGray}`,
                        background: leadTime === lt ? C.navy : C.white,
                        color: leadTime === lt ? C.gold : C.gray,
                        fontWeight: "bold", fontFamily: "sans-serif", fontSize: 13, cursor: "pointer",
                      }}>{lt}</button>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 6 }}>Periods (same unit as your demand data)</div>
                </Card>

                <Card title="Service Level" subtitle="How often do you want to avoid a stockout?">
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {SERVICE_LEVELS.map((sl, i) => (
                      <button key={i} onClick={() => setServiceLevel(i)} style={{
                        flex: 1, padding: "10px 6px", borderRadius: 4,
                        border: `2px solid ${serviceLevel === i ? C.gold : C.lightGray}`,
                        background: serviceLevel === i ? C.navy : C.white,
                        color: serviceLevel === i ? C.gold : C.gray,
                        fontWeight: "bold", fontFamily: "sans-serif", fontSize: 13, cursor: "pointer",
                      }}>{sl.label}</button>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 6 }}>
                    95% is standard for most MRO and production components. 99% for critical sole-source items.
                  </div>
                </Card>

                {/* Results */}
                <div style={{ background: C.navy, border: `2px solid ${C.gold}`, borderRadius: 6, padding: "16px", marginBottom: 16 }}>
                  <div style={{ fontSize: 10, color: C.gold, fontFamily: "sans-serif", fontWeight: "bold", letterSpacing: 2, marginBottom: 12 }}>SAFETY STOCK RECOMMENDATION</div>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {[
                      { label: "Forecast MAD", value: Math.round(results.bestMAD).toLocaleString(), sub: "units of avg error" },
                      { label: "Safety Stock", value: Math.round(results.ss).toLocaleString(), sub: `at ${SERVICE_LEVELS[serviceLevel].label} service level`, highlight: true },
                      { label: "Reorder Point", value: Math.round(results.rop).toLocaleString(), sub: "units on hand to trigger order", highlight: true },
                    ].map((item, i) => (
                      <div key={i} style={{ flex: 1, minWidth: 100, background: item.highlight ? "rgba(200,168,75,0.15)" : "rgba(255,255,255,0.05)", borderRadius: 4, padding: "12px 10px", textAlign: "center", border: item.highlight ? `1px solid ${C.gold}` : "none" }}>
                        <div style={{ fontSize: 10, color: C.htext, fontFamily: "sans-serif", marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 22, fontWeight: "bold", fontFamily: "sans-serif", color: item.highlight ? C.gold : C.white }}>{item.value}</div>
                        <div style={{ fontSize: 10, color: C.htext, fontFamily: "sans-serif", marginTop: 4 }}>{item.sub}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Formula explanation */}
                <Card title="Calculation Detail" subtitle="How the safety stock and reorder point are derived.">
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {[
                      { label: "Formula", value: `SS = Z × MAD × √Lead Time` },
                      { label: "Z-score (service level)", value: `${SERVICE_LEVELS[serviceLevel].z} (${SERVICE_LEVELS[serviceLevel].label} service level)` },
                      { label: "MAD from best-fit method", value: `${fmtN(results.bestMAD, 1)} units` },
                      { label: "Lead time", value: `${leadTime} periods` },
                      { label: "Safety stock", value: `${SERVICE_LEVELS[serviceLevel].z} × ${fmtN(results.bestMAD, 1)} × √${leadTime} = ${Math.round(results.ss).toLocaleString()} units` },
                      { label: "Average demand per period", value: `${Math.round(data.reduce((a,b)=>a+b,0)/data.length).toLocaleString()} units` },
                      { label: "Demand during lead time", value: `${Math.round(data.reduce((a,b)=>a+b,0)/data.length * leadTime).toLocaleString()} units` },
                      { label: "Reorder point", value: `${Math.round(data.reduce((a,b)=>a+b,0)/data.length * leadTime).toLocaleString()} + ${Math.round(results.ss).toLocaleString()} = ${Math.round(results.rop).toLocaleString()} units` },
                    ].map((row, i) => (
                      <div key={i} style={{ display: "flex", gap: 10, padding: "8px 10px", background: i % 2 === 0 ? C.white : "#F2EFE8", border: `1px solid ${C.lightGray}`, borderRadius: 4, flexWrap: "wrap" }}>
                        <div style={{ fontSize: 12, fontWeight: "bold", color: C.navy, fontFamily: "sans-serif", minWidth: 180, flexShrink: 0 }}>{row.label}</div>
                        <div style={{ fontSize: 12, color: C.gray, fontFamily: "sans-serif" }}>{row.value}</div>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* Service level comparison */}
                <Card title="Safety Stock by Service Level" subtitle="How safety stock changes across service level targets.">
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {SERVICE_LEVELS.map((sl, i) => {
                      const ss = Math.round(safetyStock(results.bestMAD, leadTime, sl.z));
                      const rop = Math.round(data.reduce((a,b)=>a+b,0)/data.length * leadTime + ss);
                      const isCurrent = i === serviceLevel;
                      return (
                        <div key={i} style={{ flex: 1, minWidth: 80, background: isCurrent ? C.greenLight : C.white, border: `2px solid ${isCurrent ? C.green : C.lightGray}`, borderRadius: 4, padding: "10px 8px", textAlign: "center" }}>
                          <div style={{ fontSize: 11, fontWeight: "bold", color: isCurrent ? C.green : C.navy, fontFamily: "sans-serif", marginBottom: 6 }}>{sl.label}</div>
                          <div style={{ fontSize: 16, fontWeight: "bold", fontFamily: "sans-serif", color: isCurrent ? C.green : C.navy }}>{ss.toLocaleString()}</div>
                          <div style={{ fontSize: 10, color: C.gray, fontFamily: "sans-serif", marginTop: 2 }}>safety stock</div>
                          <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 6 }}>ROP: {rop.toLocaleString()}</div>
                        </div>
                      );
                    })}
                  </div>
                </Card>

                <button onClick={() => { setRawInput(""); setActiveTab("data"); }} style={{ background: "transparent", border: `1px solid ${C.lightGray}`, borderRadius: 4, padding: "10px", fontSize: 12, fontFamily: "sans-serif", color: C.gray, cursor: "pointer", letterSpacing: 1, width: "100%" }}>
                  ↺ START NEW FORECAST
                </button>
              </>
            )}
          </div>
        )}
      </div>

      <div style={{ background: C.navy, borderTop: `2px solid ${C.gold}`, padding: "10px 16px", textAlign: "center" }}>
        <span style={{ fontSize: 10, color: C.htext, fontFamily: "sans-serif", letterSpacing: 1 }}>FLANAGAN SOURCING INTELLIGENCE PORTFOLIO · MATTHEW FLANAGAN, CPSM</span>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function Stat({ label, value, color }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ fontSize: 10, color: C.gray, fontFamily: "sans-serif" }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: "bold", fontFamily: "sans-serif", color: color || C.navy }}>{value}</div>
    </div>
  );
}

function EmptyState({ children }) {
  return (
    <div style={{ textAlign: "center", padding: "40px", color: C.gray, fontFamily: "sans-serif", fontSize: 13, background: C.white, borderRadius: 6, border: `1px dashed ${C.lightGray}` }}>
      {children}
    </div>
  );
}

function FieldLabel({ children }) {
  return <div style={{ fontSize: 12, fontWeight: "bold", color: C.navy, fontFamily: "sans-serif", marginBottom: 4 }}>{children}</div>;
}

function Card({ title, subtitle, extra, children }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.lightGray}`, borderRadius: 6, marginBottom: 16, overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
      <div style={{ background: "#F2EFE8", borderBottom: `2px solid ${C.gold}`, padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: "bold", color: C.navy, fontFamily: "sans-serif" }}>{title}</div>
          {subtitle && <div style={{ fontSize: 11, color: C.gray, fontFamily: "sans-serif", marginTop: 2 }}>{subtitle}</div>}
        </div>
        {extra}
      </div>
      <div style={{ padding: "14px" }}>{children}</div>
    </div>
  );
}

const inputSt = {
  border: "1px solid #E8E4DC", borderRadius: 4, padding: "8px 10px",
  fontSize: 13, fontFamily: "sans-serif", color: "#1B2A4A",
  background: "#FFFFFF", outline: "none", width: "100%", boxSizing: "border-box",
};

const nextBtnSt = {
  background: "#1B2A4A", color: "#FFFFFF", border: "2px solid #C8A84B",
  borderRadius: 4, padding: "12px", fontSize: 13, fontFamily: "sans-serif",
  fontWeight: "bold", letterSpacing: 1, cursor: "pointer", width: "100%",
};
import { useEffect, useMemo, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import {
  basePlotLayout, PLOT_CONFIG, EXTRACTION_ROWS, EXTRACTION_DOMAINS, AR_COLORS, SNAP_DAYS,
} from "../../lib/plotConstants.js";
import { seqColor, frameStampToIso } from "../../lib/format.js";
import { useReveal } from "../../hooks/useReveal.js";
import PlotlyChart from "../PlotlyChart.jsx";

const METRIC_OPTIONS = [
  { value: "all_aia", label: "ทุกชั้นบรรยากาศ + สนามแม่เหล็ก" },
  { value: "b_peak", label: "b_peak (max-in-mask / Gauss)" },
  { value: "171", label: "AIA 171 Å (โคโรนา)" },
  { value: "304", label: "AIA 304 Å (โครโมสเฟียร์)" },
  { value: "1600", label: "AIA 1600 Å (โฟโตสเฟียร์)" },
  { value: "flux", label: "ฟลักซ์แม่เหล็ก (Total Flux)" },
];

const METRIC_META = {
  b_peak: { yTitle: "B_peak (Gauss)", yUnit: " G", label: "b_peak (max-in-mask)", color: "#8ba2ff", fill: "rgba(138, 162, 255, 0.10)" },
  flux: { yTitle: "Total Unsigned Flux (Mx)", yUnit: " Mx", label: "Total Magnetic Flux", color: "#ffb454", fill: "rgba(255, 180, 84, 0.10)" },
  "171": { yTitle: "AIA 171 Å (DN/s)", yUnit: " DN/s", label: "AIA 171 Å · โคโรนา", color: THEME.aia["171"], fill: "rgba(255, 200, 80, 0.08)" },
  "304": { yTitle: "AIA 304 Å (DN/s)", yUnit: " DN/s", label: "AIA 304 Å · โครโมสเฟียร์", color: THEME.aia["304"], fill: "rgba(232, 89, 12, 0.08)" },
  "1600": { yTitle: "AIA 1600 Å (DN/s)", yUnit: " DN/s", label: "AIA 1600 Å · โฟโตสเฟียร์", color: THEME.aia["1600"], fill: "rgba(90, 160, 90, 0.08)" },
};

function trackMaxBPeak(track) {
  return track.b_peak && track.b_peak.length ? Math.max(...track.b_peak) : 0;
}

/** วันที่ YYYY-MM-DD -> วันที่ท้องถิ่น (แยกส่วนเองแทน new Date(str) เพื่อไม่ให้เขตเวลาเลื่อนวัน) */
function localDate(isoDay) {
  const [year, month, day] = isoDay.split("-").map(Number);
  return new Date(year, month - 1, day);
}
function formatLocalDate(dt) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
}

export default function ExtractionCard() {
  const { goes, extraction, frames, frameIndex, useTruth, applyRange } = useApp();
  const revealRef = useReveal();
  const [arId, setArId] = useState("all");
  const [metric, setMetric] = useState("all_aia");

  const tracks = useMemo(
    () => (extraction.data?.tracks ?? []).filter((t) => t.n_points > 0),
    [extraction.data],
  );
  const sortedTracks = useMemo(
    () => [...tracks].sort((a, b) => trackMaxBPeak(b) - trackMaxBPeak(a)),
    [tracks],
  );

  // คงค่าที่เลือกไว้ก่อนหน้าถ้ายังอยู่ในชุดใหม่ ไม่งั้นเลือก AR ที่ b_peak สูงสุดแทน
  useEffect(() => {
    if (!tracks.length) return;
    const stillThere = tracks.some((t) => String(t.track_id) === arId);
    if (!stillThere) setArId(String(sortedTracks[0].track_id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks]);

  const filteredTracks = arId === "all" ? tracks : tracks.filter((t) => String(t.track_id) === arId);
  const displayTracks = filteredTracks.length ? filteredTracks : tracks;

  const frameMarkerShapes = useMemo(() => {
    const stamp = frames[frameIndex];
    if (!stamp) return [];
    return [{
      type: "line", xref: "x", yref: "paper",
      x0: frameStampToIso(stamp), x1: frameStampToIso(stamp), y0: 0, y1: 1,
      line: { color: "rgba(255, 255, 255, 0.85)", width: 1.5, dash: "dot" },
    }];
  }, [frames, frameIndex]);

  const { flareShapes, flareAnnotations } = useMemo(() => {
    const shapes = [];
    const annotations = [];
    for (const evt of goes.data?.events ?? []) {
      if (!evt.peak_time || !evt.goes_class) continue;
      const color = seqColor(evt.goes_class);
      const major = evt.goes_class[0] === "M" || evt.goes_class[0] === "X";
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: evt.peak_time, x1: evt.peak_time, y0: 0, y1: 1,
        line: { color, width: major ? 1.4 : 1, dash: "dash" },
        opacity: major ? 0.85 : 0.4,
      });
      if (major) {
        annotations.push({
          xref: "x", yref: "paper", x: evt.peak_time, y: 0.995,
          text: evt.goes_class, textangle: -90, showarrow: false,
          font: { size: 9, color, family: THEME.fontMono },
          xanchor: "center", yanchor: "top",
        });
      }
    }
    return { flareShapes: shapes, flareAnnotations: annotations };
  }, [goes.data]);

  const built = useMemo(() => {
    if (!displayTracks.length) return null;

    if (metric === "all_aia") {
      const traces = [];
      const annotations = [];

      EXTRACTION_ROWS.forEach((row, index) => {
        const axis = index === 0 ? "" : String(index + 1);
        const isLog = row.scale === "log";

        displayTracks.forEach((track, order) => {
          const raw = row.key === "b_peak" ? (track.b_peak || []) : ((track.series && track.series[row.key]) || []);
          if (!raw.some((v) => v !== null && v > 0)) return;
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

      const titleLabel = displayTracks.length === 1 ? displayTracks[0].label : `${displayTracks.length} AR`;
      annotations.push({
        xref: "paper", yref: "paper", x: 0, y: 1.055, xanchor: "left", yanchor: "bottom",
        text: `<b>${titleLabel}</b> · ความเข้มแสงต่อชั้นบรรยากาศตามเวลา`,
        showarrow: false,
        font: { size: 12, color: THEME.textPrimary, family: THEME.fontSans },
      });

      if (!traces.length) return { traces: [], layout: null };

      const layout = {
        ...basePlotLayout(),
        margin: { l: 56, r: 14, t: 30, b: 34 },
        showlegend: displayTracks.length > 1,
        legend: { orientation: "h", y: -0.12, x: 0, font: { size: 10 } },
        xaxis: {
          ...basePlotLayout().xaxis,
          type: "date",
          anchor: `y${EXTRACTION_ROWS.length}`,
          tickfont: { size: 10 },
        },
        annotations: [...annotations, ...flareAnnotations],
        shapes: [...flareShapes, ...frameMarkerShapes],
      };
      EXTRACTION_DOMAINS.forEach((domain, index) => {
        const axis = index === 0 ? "" : String(index + 1);
        const isLog = EXTRACTION_ROWS[index].scale === "log";
        layout[`yaxis${axis}`] = {
          ...basePlotLayout().yaxis,
          domain,
          type: isLog ? "log" : "linear",
          rangemode: isLog ? undefined : "tozero",
          dtick: isLog ? 1 : undefined,
          exponentformat: isLog ? "power" : undefined,
          tickfont: { size: 10 },
        };
      });

      return { traces, layout };
    }

    // กราฟเดี่ยว: metric เดียวต่อ AR
    let meta = METRIC_META[metric] ?? METRIC_META.b_peak;
    let metricLabel = meta.label;
    const traces = [];

    displayTracks.forEach((track, order) => {
      let values = [];
      if (metric === "b_peak") values = track.b_peak || [];
      else if (metric === "flux") values = track.flux || [];
      else values = (track.series && track.series[metric]) || [];

      const hasValues = values.some((v) => v !== null && v !== undefined && v > 0);
      if (!hasValues) {
        if (metric !== "b_peak" && metric !== "flux" && track.b_peak && track.b_peak.length) {
          values = track.b_peak;
          metricLabel = "b_peak (AIA ไม่มีข้อมูล — แสดง B_peak แทน)";
          meta = METRIC_META.b_peak;
        } else {
          return;
        }
      }

      const trackColor = displayTracks.length > 1 ? AR_COLORS[order % AR_COLORS.length] : meta.color;
      const trackFill = displayTracks.length > 1 ? "none" : meta.fill;

      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: track.label,
        x: track.times,
        y: values,
        fill: trackFill,
        fillcolor: trackFill !== "none" ? trackFill : undefined,
        line: { shape: "spline", smoothing: 0.7, width: 2.5, color: trackColor },
        marker: { size: 5, color: trackColor, line: { width: 1.5, color: "rgba(255,255,255,0.5)" } },
        connectgaps: false,
        hovertemplate: `<b>${track.label}</b><br>%{x|%d %b %Y %H:%M UT}<br><b>%{y:,.1f}${meta.yUnit}</b><extra></extra>`,
      });
    });

    if (!traces.length) return { traces: [], layout: null, noDataMetric: metricLabel };

    const labelColor = displayTracks.length === 1 ? meta.color : THEME.textSecondary;
    const labelText = displayTracks.length === 1 ? `<b>${displayTracks[0].label}</b>` : `<b>${displayTracks.length} AR</b>`;

    const layout = {
      ...basePlotLayout(),
      margin: { l: 65, r: 20, t: 30, b: 38 },
      showlegend: displayTracks.length > 1,
      legend: { orientation: "h", y: -0.18, x: 0, font: { size: 10 } },
      xaxis: { ...basePlotLayout().xaxis, type: "date", tickfont: { size: 10 }, hoverformat: "%d %b %Y %H:%M UT" },
      yaxis: {
        ...basePlotLayout().yaxis,
        title: { text: meta.yTitle, font: { size: 10, color: THEME.textMuted } },
        tickfont: { size: 10 },
        tickformat: metric === "flux" ? ".2s" : ",",
        rangemode: "tozero",
      },
      annotations: [
        {
          xref: "paper", yref: "paper", x: 0.99, y: 1.04, xanchor: "right", yanchor: "bottom",
          text: `<b>${metricLabel}</b>`, showarrow: false,
          font: { size: 11.5, color: THEME.textMuted, family: THEME.fontMono },
        },
        {
          xref: "paper", yref: "paper", x: 0.01, y: 0.96, xanchor: "left", yanchor: "top",
          text: labelText, showarrow: false,
          font: { size: 13, color: labelColor, family: THEME.fontMono },
        },
        ...flareAnnotations,
      ],
      shapes: [...flareShapes, ...frameMarkerShapes],
    };

    return { traces, layout };
  }, [displayTracks, metric, flareShapes, flareAnnotations, frameMarkerShapes]);

  const data = extraction.data;
  const source = useTruth ? "mask SHARP (Ground Truth)" : "mask U-Net (Segmentation)";

  let hint = "";
  let note = "";
  let bodyOverride = null;

  if (extraction.status === "loading") {
    hint = "กำลังคำนวณ…";
  } else if (extraction.status === "error") {
    bodyOverride = <p className="empty">{extraction.error}</p>;
  } else if (data && !tracks.length) {
    const message = data.note
      || (data.channels?.some((c) => c.available) ? "ไม่พบ active region ในช่วงเวลานี้" : "ช่วงเวลานี้ยังไม่มีข้อมูลสำหรับ active region");
    bodyOverride = <EmptyExtraction message={message} nearestFrame={data.nearest_frame} onSnap={applyRange} />;
  } else if (data && built && !built.traces.length) {
    bodyOverride = <p className="empty">{built.noDataMetric ? `ไม่มีข้อมูล ${built.noDataMetric} ใน AR ที่เลือก` : "ไม่มีข้อมูล"}</p>;
  } else if (data) {
    hint = `${displayTracks.length} AR · ${data.n_frames} เฟรม · ${source}`;
    note = data.note || "";
  }

  return (
    <section ref={revealRef} id="extraction" className="card card--extraction" aria-labelledby="h-extraction" data-reveal>
      <div className="card__head">
        <div className="card__head-title">
          <h2 id="h-extraction" title="ความเข้มแสงและสนามแม่เหล็กราย active region จาก mask">AR Intensity · U-Net Mask</h2>
          <span className="hint" id="extractionHint" title={data?.note || ""}>{hint}</span>
        </div>
      </div>
      <div className="controls controls--extraction">
        <label className="field">
          <span className="field__label">เลือก AR</span>
          <select value={arId} onChange={(e) => setArId(e.target.value)}>
            <option value="all">ทุก AR (แสดงรวม)</option>
            {sortedTracks.map((t) => {
              const bPeakMax = trackMaxBPeak(t);
              const bStr = bPeakMax > 0 ? ` · ${bPeakMax.toFixed(0)} G` : "";
              const areaKPx = (t.max_area_px / 1000).toFixed(1);
              return (
                <option key={t.track_id} value={t.track_id}>
                  {t.label}{bStr} ({areaKPx}k px)
                </option>
              );
            })}
          </select>
        </label>
        <label className="field">
          <span className="field__label">เมตริก</span>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRIC_OPTIONS.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        </label>
      </div>
      <p className={`proton__summary${note ? "" : " hidden"}`} id="extractionNote">{note}</p>
      {bodyOverride ? (
        <div id="extractionChart" className="chart chart--extraction">{bodyOverride}</div>
      ) : (
        <PlotlyChart
          id="extractionChart"
          className="chart chart--extraction"
          data={built?.traces ?? []}
          layout={built?.layout ?? null}
          config={PLOT_CONFIG}
          empty={!built?.traces?.length}
          emptyMessage=""
        />
      )}
      <p className="hint extraction__caption">
        เส้นประ = เวลาที่เกิด flare ในช่วงที่กรองไว้ (สีตามระดับ B&lt;C&lt;M&lt;X · ป้ายกำกับเฉพาะ M/X)
      </p>
    </section>
  );
}

function EmptyExtraction({ message, nearestFrame, onSnap }) {
  let button = null;
  if (nearestFrame) {
    const d = localDate(nearestFrame);
    const snapStart = formatLocalDate(new Date(d.getTime() - SNAP_DAYS * 86400000));
    const snapEnd = formatLocalDate(new Date(d.getTime() + SNAP_DAYS * 86400000));
    button = (
      <button
        type="button" className="btn btn--sm btn--snap"
        title={`ขยายช่วงเวลาให้รอบเฟรมที่ใกล้ที่สุด (${nearestFrame})`}
        onClick={() => onSnap({ start: snapStart, end: snapEnd })}
      >
        📅 กระโดดไปช่วง {nearestFrame} ±{SNAP_DAYS} วัน
      </button>
    );
  }
  return (
    <>
      <p className="empty">{message}</p>
      {button}
    </>
  );
}

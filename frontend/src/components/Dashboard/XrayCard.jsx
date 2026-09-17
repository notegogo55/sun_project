import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { api } from "../../lib/api.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG, GOES_LEVELS, DENSE_EVENT_THRESHOLD, BACKGROUND_FLUX, XRAY_Y_FLOOR } from "../../lib/plotConstants.js";
import { formatDateTime, markerSize, seqColor } from "../../lib/format.js";
import { flaresToCsv, downloadCsv } from "../../lib/csv.js";
import { useReveal } from "../../hooks/useReveal.js";
import PlotlyChart from "../PlotlyChart.jsx";

const DEFAULT_LEGEND_NOTE =
  "สีสว่างขึ้น = แรงขึ้น · เส้นสร้างจากเวลาเริ่ม–peak–สิ้นสุดของแต่ละเหตุการณ์ ไม่ใช่สัญญาณดิบรายนาที";

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

export default function XrayCard() {
  const { goes, range, health, selectFlareEvent, selectHarp } = useApp();
  const revealRef = useReveal();
  const chartRef = useRef(null);
  const [legendNote, setLegendNote] = useState(DEFAULT_LEGEND_NOTE);
  const xrayRequest = useRef(0);
  const events = goes.data?.events ?? [];

  const { traces, layout, dense } = useMemo(() => {
    if (!events.length) return { traces: [], layout: null, dense: false };

    const isDense = events.length > DENSE_EVENT_THRESHOLD;
    const reconstructed = reconstructedXrayCurve(events);
    const curve = {
      x: reconstructed.x, y: reconstructed.y,
      type: "scatter", mode: "lines", connectgaps: false,
      line: { color: THEME.series1, width: isDense ? 1.8 : 1.4, shape: "linear" },
      opacity: isDense ? 0.95 : 1,
      hoverinfo: "skip",
    };

    const peakEntries = events
      .map((e, index) => ({ e, index }))
      .filter(({ e }) => !isDense || e.goes_class[0] === "M" || e.goes_class[0] === "X");

    const peaks = {
      x: peakEntries.map(({ e }) => e.peak_time),
      y: peakEntries.map(({ e }) => e.peak_flux),
      text: peakEntries.map(({ e }) => `${e.goes_class}${e.noaa_ar ? ` · AR ${e.noaa_ar}` : ""}`),
      customdata: peakEntries.map(({ index }) => index),
      type: "scatter", mode: "markers",
      marker: {
        size: peakEntries.map(({ e }) => markerSize(e.peak_flux) * (isDense ? 0.65 : 1)),
        color: peakEntries.map(({ e }) => seqColor(e.goes_class)),
        line: { width: isDense ? 1 : 2, color: THEME.surface1 },
      },
      opacity: isDense ? 0.9 : 1,
      hovertemplate: "%{x|%d %b %Y %H:%M}<br>%{text}<extra></extra>",
    };

    return {
      dense: isDense,
      traces: [curve, peaks],
      layout: {
        ...basePlotLayout(),
        transition: { duration: 300, easing: "cubic-in-out" },
        yaxis: {
          ...basePlotLayout().yaxis,
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
      },
    };
  }, [events]);

  // อัปเกรดเป็นฟลักซ์ต่อเนื่องจริงรายนาที (GOES-15 หรือ GOES-16 แล้วแต่ช่วงเวลา) แบบ progressive เบื้องหลัง
  useEffect(() => {
    setLegendNote(DEFAULT_LEGEND_NOTE);
    if (!goes.data || !health?.xray_flux) return undefined;
    let cancelled = false;
    const requestId = ++xrayRequest.current;
    (async () => {
      let payload;
      try {
        payload = await api.xray(goes.data.start, goes.data.end);
      } catch {
        return;
      }
      if (cancelled || requestId !== xrayRequest.current || !payload.n_samples) return;
      chartRef.current?.restyle({
        x: [payload.times],
        y: [payload.flux.map((v) => (v === null ? null : Math.max(v, XRAY_Y_FLOOR)))],
        connectgaps: false,
      }, [0]);
      const decimateNote = payload.decimated ? " (ย่อข้อมูลด้วยค่าสูงสุดต่อช่วง — ช่วงยาวเกินไป)" : "";
      setLegendNote(`สีสว่างขึ้น = แรงขึ้น · เส้นคือฟลักซ์จริงรายนาทีจาก GOES${decimateNote} · จุดคือค่า peak ของแต่ละเหตุการณ์`);
    })();
    return () => { cancelled = true; };
  }, [goes.data, health]);

  function handleClick(click) {
    const index = click.points?.[0]?.customdata;
    if (typeof index !== "number") return;
    const flare = events[index];
    selectFlareEvent(flare);
    if (flare.harpnum) selectHarp(Number(flare.harpnum));
  }

  function handleExport() {
    if (!events.length) {
      window.alert("ไม่มีรายการ Flare ในช่วงเวลาที่เลือก");
      return;
    }
    downloadCsv(`flares_${range.start}_to_${range.end}.csv`, flaresToCsv(events));
  }

  const hint = goes.data
    ? `${goes.data.n_events.toLocaleString()} เหตุการณ์ · ${formatDateTime(goes.data.start)} – ${formatDateTime(goes.data.end)}`
      + (dense ? " · แสดงจุดเฉพาะ M/X (ช่วงยาวเกินไปสำหรับจุดทั้งหมด)" : "")
    : "";

  return (
    <section ref={revealRef} id="xray" className="card card--wide" aria-labelledby="h-xray" data-reveal>
      <div className="card__head">
        <div className="card__head-title">
          <h2 id="h-xray">GOES X-Ray Light Curve</h2>
          <span className="hint" id="xrayHint">{hint}</span>
        </div>
        <button type="button" className="btn btn--sm btn--export" id="exportFlaresBtn" title="ดาวน์โหลดรายการ Flare ในช่วงนี้เป็นไฟล์ CSV" onClick={handleExport}>
          <span>⬇</span> ส่งออก CSV
        </button>
      </div>
      <div className="legend" id="goesLegend">
        <span className="legend__item"><i className="legend__swatch legend__swatch--series" />ฟลักซ์ตามเวลา</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--seq-1" />B</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--seq-2" />C</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--seq-3" />M</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--seq-4" />X</span>
        <span className="legend__note" id="xrayLegendNote">{legendNote}</span>
      </div>
      <PlotlyChart
        ref={chartRef}
        id="xrayChart"
        className="chart"
        data={traces}
        layout={layout}
        config={PLOT_CONFIG}
        onClick={handleClick}
        empty={!events.length}
        emptyMessage={goes.status === "error" ? goes.error : "ไม่พบ flare ในช่วงเวลานี้"}
      />
    </section>
  );
}

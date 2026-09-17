import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { api } from "../../lib/api.js";
import { idleAsync, loadingAsync, readyAsync, errorAsync } from "../../lib/asyncState.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG, GOES_LEVELS, XRAY_Y_FLOOR } from "../../lib/plotConstants.js";
import { CLASS_STYLE } from "../../lib/flareClass.js";
import PlotlyChart from "../PlotlyChart.jsx";

const HOUR = 3600000;
const BEFORE_H = 6;
const AFTER_H = 12;
/** รอให้เมาส์หยุดก่อนค่อยยิงคำขอ — กวาดเมาส์ผ่านจุดหลายสิบจุดไม่ควรยิงหลายสิบคำขอ */
const DEBOUNCE_MS = 180;

const isoUTC = (ms) => new Date(ms).toISOString().slice(0, 19);

/** กราฟ X-ray 1–8 Å รอบเวลาพีคของ flare ที่ชี้/ตรึงอยู่ — คู่กับแผนที่แบบ XrayChart ของ
 *  PositionFlare แต่ดึงจากฟลักซ์รายนาทีใน backend ของโปรเจคนี้ (/api/xray) */
export default function FlareXrayChart({ row }) {
  const { health } = useApp();
  const [state, setState] = useState(idleAsync());
  const requestRef = useRef(0);

  useEffect(() => {
    if (!row || !health?.xray_flux) return undefined;
    const requestId = ++requestRef.current;
    const timer = setTimeout(async () => {
      setState((prev) => loadingAsync(prev));
      try {
        const data = await api.xray(isoUTC(row.t - BEFORE_H * HOUR), isoUTC(row.t + AFTER_H * HOUR));
        if (requestId === requestRef.current) setState(readyAsync(data, { row }));
      } catch (error) {
        if (requestId === requestRef.current) setState(errorAsync(error.message, { row }));
      }
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // ผูกกับ id อย่างเดียว — object ของแถวเดิมถูกสร้างใหม่ไม่ได้ แต่กันไว้ไม่ให้ยิงซ้ำเปล่า ๆ
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [row?.id, health?.xray_flux]);

  const shown = state.row;
  const data = state.data;
  const empty = !data || !data.n_samples;

  const { traces, layout } = useMemo(() => {
    if (empty || !shown) return { traces: [], layout: null };
    const color = CLASS_STYLE[shown.ci].color;
    const { ev } = shown;
    const band = ev.start_time && ev.end_time ? [{
      type: "rect", xref: "x", yref: "paper", x0: ev.start_time, x1: ev.end_time, y0: 0, y1: 1,
      fillcolor: color, opacity: 0.1, line: { width: 0 }, layer: "below",
    }] : [];
    return {
      traces: [{
        x: data.times,
        y: data.flux.map((v) => (v === null ? null : Math.max(v, XRAY_Y_FLOOR))),
        type: "scatter",
        mode: "lines",
        connectgaps: false,
        line: { color: THEME.series1, width: 1.5 },
        hovertemplate: "%{x|%H:%M} UT<br>%{y:.2e} W/m²<extra></extra>",
      }],
      layout: {
        ...basePlotLayout(),
        margin: { l: 34, r: 10, t: 6, b: 26 },
        xaxis: {
          ...basePlotLayout().xaxis,
          type: "date",
          range: [isoUTC(shown.t - BEFORE_H * HOUR), isoUTC(shown.t + AFTER_H * HOUR)],
          tickfont: { size: 10 },
          fixedrange: true,
        },
        yaxis: {
          ...basePlotLayout().yaxis,
          type: "log",
          range: [Math.log10(XRAY_Y_FLOOR), Math.log10(2e-3)],
          tickvals: GOES_LEVELS.map((l) => l.flux),
          ticktext: GOES_LEVELS.map((l) => l.label),
          tickfont: { size: 10 },
          fixedrange: true,
        },
        shapes: [
          ...band,
          ...GOES_LEVELS.map((level) => ({
            type: "line", xref: "paper", x0: 0, x1: 1,
            y0: Math.log10(level.flux), y1: Math.log10(level.flux),
            line: { color: THEME.baseline, width: 1, dash: "dot" },
          })),
          {
            type: "line", xref: "x", yref: "paper", x0: ev.peak_time, x1: ev.peak_time, y0: 0, y1: 1,
            line: { color, width: 1.5 },
          },
        ],
      },
    };
  }, [data, shown, empty]);

  if (!health?.xray_flux) return null;

  let message = "เลือก flare เพื่อดู X-ray รอบเวลาพีค";
  if (row && state.status === "loading" && empty) message = "กำลังโหลด…";
  else if (state.status === "error") message = state.error;
  else if (row && state.status === "ready" && empty) message = "ไม่มีฟลักซ์ X-ray รายนาทีครอบช่วงนี้";

  return (
    <div className="panel">
      <p className="panel__title">X-ray flux</p>
      <div className="hero__minichart-label">
        <span className="panel__sub" style={{ marginBottom: 0 }}>1–8 Å · −{BEFORE_H} → +{AFTER_H} ชม. รอบพีค</span>
        <span className="hero__minichart-hint">
          {shown && !empty ? `${shown.ev.goes_class}${data.decimated ? " · ย่อข้อมูล" : ""}` : ""}
        </span>
      </div>
      <PlotlyChart
        className="chart chart--flare-xray"
        data={traces}
        layout={layout}
        config={PLOT_CONFIG}
        empty={empty}
        emptyMessage={message}
      />
    </div>
  );
}

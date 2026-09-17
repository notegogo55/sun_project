import { useMemo } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG, S_SCALE_LEVELS } from "../../lib/plotConstants.js";
import { formatPfu, shiftIsoHours } from "../../lib/format.js";
import { protonYRange, protonXRange } from "../../lib/proton.js";
import PlotlyChart from "../PlotlyChart.jsx";

/** กราฟโปรตอนย่อข้างจานสุริยะในหน้า HUB — เวอร์ชันย่อของ ProtonCard แสดงเฉพาะช่อง >10 MeV
 *  (ช่องอ้างอิงของ NOAA S-scale) เส้นเดียว พื้นที่แคบเกินจะใส่ทั้งสี่เส้น + legend */
export default function ProtonMiniChart() {
  const { proton, pinnedFlare } = useApp();
  const { data, event, status } = proton;

  const reference = data?.channels.find((c) => c.key === "p10") ?? data?.channels[1];
  const hasReference = reference?.values.some((v) => v !== null);
  // ตัดสินจากการมีข้อมูลอยู่ไหม ไม่ใช่จาก status — ระหว่างโหลดรอบใหม่ (hover ไปจุด flare อื่น)
  // ต้องคงกราฟเดิมไว้ ไม่ใช่กะพริบเป็นค่าว่างทุกครั้ง (ดูเหตุผลเดียวกันใน ProtonCard.jsx)
  const empty = !data || !data.has_data || !hasReference;

  const { layout, hint } = useMemo(() => {
    if (empty || !data) return { layout: null, hint: "" };

    const [low, high] = protonYRange(reference.values);

    const tickHours = [...new Set([-data.before_hours, 0, 24, 48, data.after_hours])]
      .filter((h) => h >= -data.before_hours && h <= data.after_hours)
      .sort((a, b) => a - b);

    const layoutOut = {
      ...basePlotLayout(),
      margin: { l: 30, r: 6, t: 4, b: 18 },
      transition: { duration: 300, easing: "cubic-in-out" },
      xaxis: {
        ...basePlotLayout().xaxis,
        type: "date",
        range: protonXRange(data.peak_time, data.before_hours, data.after_hours),
        tickvals: tickHours.map((h) => shiftIsoHours(data.peak_time, h)),
        ticktext: tickHours.map((h) => `${h > 0 ? "+" : ""}${h} ชม.`),
        tickfont: { size: 8, color: THEME.textMuted },
        hoverformat: "%Y-%m-%d %H:%M UT",
      },
      yaxis: {
        ...basePlotLayout().yaxis,
        type: "log",
        range: [low, high],
        dtick: 2,
        exponentformat: "power",
        tickfont: { size: 8, color: THEME.textMuted },
      },
      shapes: [
        ...S_SCALE_LEVELS
          .filter((level) => Math.log10(level.pfu) >= low && Math.log10(level.pfu) <= high)
          .map((level) => ({
            type: "line", xref: "paper", x0: 0, x1: 1,
            y0: Math.log10(level.pfu), y1: Math.log10(level.pfu),
            line: { color: THEME.baseline, width: 0.8, dash: "dot" },
          })),
        {
          type: "line", xref: "x", yref: "paper",
          x0: data.peak_time, x1: data.peak_time, y0: 0, y1: 1,
          line: { color: THEME.textSecondary, width: 1.2 },
        },
      ],
    };

    const label = event?.goes_class ?? data.goes_class;
    const prefix = label ? `${label} · ` : "";
    const pinTag = pinnedFlare?.peak_time === data.peak_time ? " · ตรึง" : "";
    const hintText = (data.s_scale
      ? `${prefix}${data.s_scale} สูงสุด ${formatPfu(data.max_after)} pfu`
      : `${prefix}ต่ำกว่า S1`) + pinTag;

    return { layout: layoutOut, hint: hintText };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, event, pinnedFlare, empty]);

  const trace = !empty && data ? [{
    x: data.times,
    y: reference.values,
    customdata: reference.values.map((v) => formatPfu(v)),
    type: "scatter",
    mode: "lines",
    connectgaps: false,
    line: { color: THEME.pflux[1], width: 1.4, shape: "linear" },
    hovertemplate: "<b>%{customdata}</b> pfu<extra></extra>",
  }] : [];

  return (
    <div className="panel" id="heroMinichart">
      <p className="panel__title">Proton flux</p>
      <div className="hero__minichart-label">
        <span className="panel__sub" style={{ marginBottom: 0 }}>&gt;10 MeV รอบ flare ที่เลือก</span>
        <span className="hero__minichart-hint" id="heroProtonHint">{hint}</span>
      </div>
      <PlotlyChart
        id="heroProtonChart"
        className="chart chart--hero-proton"
        data={trace}
        layout={layout}
        config={PLOT_CONFIG}
        empty={empty}
        emptyMessage={
          status === "error" ? proton.error
            : status === "loading" ? "กำลังโหลด…"
              : "ไม่มีข้อมูลโปรตอนช่วงนี้"
        }
      />
    </div>
  );
}

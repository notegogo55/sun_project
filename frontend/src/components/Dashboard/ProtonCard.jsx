import { useMemo } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG, S_SCALE_LEVELS } from "../../lib/plotConstants.js";
import { formatDateTime, formatPfu, shiftIsoHours } from "../../lib/format.js";
import { protonYRange, protonXRange } from "../../lib/proton.js";
import { useReveal } from "../../hooks/useReveal.js";
import PlotlyChart from "../PlotlyChart.jsx";

function Sep() {
  return <span className="proton__sep">·</span>;
}

/** บรรทัดสรุปเหนือกราฟ — เดิมประกอบเป็น HTML string แล้วยัดผ่าน dangerouslySetInnerHTML
 *  เพราะตอนนั้นยังเป็น vanilla JS ไม่มี JSX ตอนนี้เขียนเป็น JSX ตรงๆ ได้แล้ว ไม่ต้องพึ่ง
 *  innerHTML (ค่าที่แทรก เช่น data.sources, data.s_scale มาจาก backend ที่เราคุมเองก็จริง
 *  แต่การประกอบ HTML ด้วยมือยังเสี่ยงพลาดรูปแบบ/escape โดยไม่ตั้งใจอยู่ดี) */
function ProtonSummary({ status, data, event, pinnedFlare }) {
  if (status === "loading") return "กำลังโหลด…";
  if (!data) return null;

  const goesClass = event?.goes_class ?? data.goes_class;
  const headline = `${goesClass ? `flare ${goesClass}` : "flare"} ${formatDateTime(data.peak_time)} UT`;
  const pinned = pinnedFlare?.peak_time === data.peak_time;

  if (!data.has_data) {
    return (
      <>
        {headline}<Sep />ไม่มีข้อมูลโปรตอนช่วงนี้
        {pinned && <><Sep />ตรึงไว้ (คลิกซ้ำเพื่อปลด)</>}
      </>
    );
  }

  const parts = [
    headline,
    <>&gt;10 MeV ขณะนั้น <b>{formatPfu(data.flux_at_peak)}</b> pfu</>,
  ];
  if (data.max_after !== null) {
    parts.push(
      <>สูงสุดใน +{data.after_hours} ชม. <b>{formatPfu(data.max_after)}</b> pfu{data.s_scale ? ` (${data.s_scale})` : ""}</>,
    );
  }
  if (data.sources.length) parts.push(`ที่มา ${data.sources.join(" + ")}`);

  return (
    <>
      {parts.map((part, index) => (
        // eslint-disable-next-line react/no-array-index-key
        <span key={index}>{index > 0 && <Sep />}{part}</span>
      ))}
      {pinned && <><Sep />ตรึงไว้ (คลิกซ้ำเพื่อปลด)</>}
    </>
  );
}

export default function ProtonCard() {
  const { health, proton, pinnedFlare } = useApp();
  const revealRef = useReveal();
  const { data, event, status } = proton;

  // `empty` ตัดสินจาก "มีข้อมูลอยู่ไหม" ไม่ใช่จาก status — ระหว่างโหลดรอบใหม่ (hover ผ่านจุด
  // flare อื่น, debounce 90ms ยิงถี่มาก) ต้องคงกราฟเก่าไว้จนกว่าของใหม่จะมาแทน ไม่งั้นกราฟจะ
  // กะพริบเป็นค่าว่างทุกครั้ง (บรรทัดสรุปยังบอก "กำลังโหลด…" เสมอตามต้นฉบับ ดู ProtonSummary)
  const empty = !data || !data.has_data;

  const { traces, layout } = useMemo(() => {
    if (empty || !data) return { traces: [], layout: null };

    const trs = data.channels.map((channel, index) => ({
      x: data.times,
      y: channel.values,
      customdata: channel.values.map((v) => formatPfu(v)),
      name: channel.label,
      type: "scatter",
      mode: "lines",
      connectgaps: false,
      line: { color: THEME.pflux[index], width: 2, shape: "linear" },
      hovertemplate: `<b>%{customdata}</b> pfu<extra>${channel.label}</extra>`,
    }));

    const [low, high] = protonYRange(data.channels.flatMap((channel) => channel.values));

    const tickvals = [];
    const ticktext = [];
    for (let hour = -data.before_hours; hour <= data.after_hours; hour += 12) {
      tickvals.push(shiftIsoHours(data.peak_time, hour));
      ticktext.push(`${hour > 0 ? "+" : ""}${hour} ชม.`);
    }
    const inRange = (pfu) => Math.log10(pfu) >= low && Math.log10(pfu) <= high;

    return {
      traces: trs,
      layout: {
        ...basePlotLayout(),
        margin: { l: 58, r: 34, t: 20, b: 40 },
        hovermode: "x unified",
        transition: { duration: 300, easing: "cubic-in-out" },
        xaxis: {
          ...basePlotLayout().xaxis,
          type: "date",
          range: protonXRange(data.peak_time, data.before_hours, data.after_hours),
          tickvals, ticktext,
          hoverformat: "%Y-%m-%d %H:%M UT",
          fixedrange: true,
        },
        yaxis: {
          ...basePlotLayout().yaxis,
          type: "log", title: "ฟลักซ์ (pfu)",
          range: [low, high], dtick: 1, exponentformat: "power", fixedrange: true,
        },
        shapes: [{
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
          ...S_SCALE_LEVELS.filter((level) => inRange(level.pfu)).map((level) => ({
            xref: "paper", x: 1.008, xanchor: "left",
            yref: "y", y: Math.log10(level.pfu),
            text: level.label, showarrow: false,
            font: { size: 9.5, color: THEME.textMuted, family: THEME.fontMono },
          })),
        ],
      },
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, event, empty]);

  if (!health?.proton_flux) return null;

  return (
    <section ref={revealRef} className="card card--wide" aria-labelledby="h-proton" id="proton" data-reveal>
      <div className="card__head">
        <h2 id="h-proton" title="ฟลักซ์โปรตอนรอบเวลาที่เกิด flare">Proton Flux · Flare Window</h2>
        <span className="hint" id="protonHint">ชี้เมาส์ที่จุด flare บนจานสุริยะหรือบนเส้น X-ray เพื่อดูตัวอย่าง · คลิกเพื่อตรึงไว้</span>
      </div>
      <p className="proton__summary" id="protonSummary">
        <ProtonSummary status={status} data={data} event={event} pinnedFlare={pinnedFlare} />
      </p>
      <div className="legend" id="protonLegend">
        <span className="legend__item"><i className="legend__swatch legend__swatch--pflux-1" />&gt; 1 MeV</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--pflux-2" />&gt; 10 MeV</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--pflux-3" />&gt; 50 MeV</span>
        <span className="legend__item"><i className="legend__swatch legend__swatch--pflux-4" />&gt; 100 MeV</span>
        <span className="legend__note">
          ช่องพลังงานเป็นแบบสะสม (&gt;1 ครอบ &gt;10 ครอบ &gt;50) · ช่องว่างบนเส้น = ไม่มีข้อมูล ไม่ใช่ศูนย์ ·
          S1–S5 คือมาตราพายุรังสีสุริยะของ NOAA วัดที่ &gt;10 MeV
        </span>
      </div>
      <PlotlyChart
        id="protonChart"
        className="chart"
        data={traces}
        layout={layout}
        config={PLOT_CONFIG}
        empty={empty}
        emptyMessage={
          status === "error" ? proton.error
            : status === "loading" ? "กำลังโหลด…"
              : status === "empty" ? "ไม่พบ flare ในช่วงเวลานี้"
                : "คลังโปรตอนไม่ครอบคลุมช่วงเวลานี้ — ข้อมูลเริ่มกลางปี 1998"
        }
      />
    </section>
  );
}

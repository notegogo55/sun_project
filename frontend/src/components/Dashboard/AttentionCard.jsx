import { useMemo } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG } from "../../lib/plotConstants.js";
import { formatValue } from "../../lib/format.js";
import { useReveal } from "../../hooks/useReveal.js";
import PlotlyChart from "../PlotlyChart.jsx";

/** คำอธิบายน้ำหนักตามวิธี pooling ของโมเดล — last/mean ไม่มีน้ำหนักที่เรียนรู้ได้ กราฟจึงแบน/มีแท่งเดียว
 *  โดยธรรมชาติ ต้องบอกผู้ใช้ ไม่งั้นอ่านผิดว่า "โมเดลสนใจแค่ชั่วโมงล่าสุด" */
const POOLING_HINT = {
  attention: "โมเดลสนใจช่วงเวลาไหน · น้ำหนัก attention ย้อนหลัง 24 ชม.",
  last: "โมเดลนี้ใช้สถานะ ณ ชั่วโมงล่าสุดเท่านั้น (pooling: last) — ไม่มีน้ำหนัก attention ที่เรียนรู้ได้",
  mean: "โมเดลนี้เฉลี่ยทุกชั่วโมงเท่ากัน (pooling: mean) — ไม่มีน้ำหนัก attention ที่เรียนรู้ได้",
};

export default function AttentionCard() {
  const { risk, forecastModelInfo } = useApp();
  const revealRef = useReveal();
  const latest = risk.data?.latest ?? null;
  const modelLabel = forecastModelInfo?.label ?? "Model";
  const hint = POOLING_HINT[forecastModelInfo?.pooling] ?? POOLING_HINT.attention;

  const { traces, layout, empty } = useMemo(() => {
    if (!latest?.attention?.length) return { traces: [], layout: null, empty: true };

    const n = latest.attention.length;
    // แกน x เป็นชั่วโมง "ก่อนเวลาออกพยากรณ์" — 0 คือปัจจุบัน, -23 คือเมื่อวาน
    const hours = latest.attention.map((_, i) => -(n - 1 - i));
    const peak = Math.max(...latest.attention);

    return {
      empty: false,
      traces: [{
        x: hours,
        y: latest.attention,
        type: "bar",
        marker: {
          // ชั่วโมงเด่นเป็นส้ม (THEME.sun) ตัดกับแท่งน้ำเงิน — THEME.brand เป็นน้ำเงินไปแล้วในธีมใหม่
          color: latest.attention.map((w) => (w >= peak * 0.92 ? THEME.sun : THEME.series1)),
        },
        hovertemplate: "%{x} ชม. ก่อนหน้า<br>น้ำหนัก %{y:.3f}<extra></extra>",
      }],
      layout: {
        ...basePlotLayout(),
        margin: { l: 46, r: 14, t: 6, b: 34 },
        xaxis: { ...basePlotLayout().xaxis, title: "ชั่วโมงก่อนเวลาพยากรณ์" },
        yaxis: { ...basePlotLayout().yaxis, title: "น้ำหนัก" },
        bargap: 0.25,
      },
    };
  }, [latest]);

  return (
    <section ref={revealRef} id="attention" className="card card--wide" aria-labelledby="h-attn" data-reveal>
      <div className="card__head">
        <h2 id="h-attn">{modelLabel} Attention</h2>
        <span className="hint">{hint}</span>
      </div>
      <PlotlyChart
        id="attentionChart"
        className="chart chart--short"
        data={traces}
        layout={layout}
        config={PLOT_CONFIG}
        empty={empty}
        emptyMessage=""
      />

      <div className="card__head card__head--sub">
        <h2>SHARP Parameters</h2>
        <span className="hint">ค่าล่าสุดแบบดิบ ก่อน normalize</span>
      </div>
      <div id="featureList" className="feature-grid">
        {!latest?.features ? (
          <p className="empty">เลือก active region เพื่อดูค่า</p>
        ) : (
          Object.entries(latest.features).map(([name, value]) => (
            <div className="feature" key={name}>
              <span className="feature__name">{name}</span>
              <span className="feature__value">{formatValue(value)}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

import { useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { formatDateTime } from "../../lib/format.js";
import { useReveal } from "../../hooks/useReveal.js";
import { CLASS_MODES, SPLIT_TEXT, levelMeta } from "../../lib/classForecast.js";
import ClassModeToggle from "../ClassModeToggle.jsx";

/** การ์ดของโมเดลหลัก — ระดับคลาสของ flare ที่แรงที่สุดใน 24 ชม. ถัดไป (<M / M / X) ของ HARP ที่เลือก
 *
 *  คำพยากรณ์ออกทุก 12 ชม. (study dataset) ไม่ใช่รายชั่วโมงแบบแผงความเสี่ยงข้างล่าง แถบเวลาจึงเป็นช่องราย 12 ชม.:
 *  แถวบน = ระดับที่ทำนาย, แถวล่าง = ระดับที่เกิดจริง · คลิกช่องเพื่อดูรายละเอียดของคำพยากรณ์นั้นทางซ้าย
 *  (ค่าเริ่มต้นคือคำพยากรณ์ล่าสุด) */
export default function ClassForecastCard() {
  const { classSummary, classSeries, selectedHarp } = useApp();
  const revealRef = useReveal();

  let body;
  if (!classSummary) {
    body = <p className="empty">กำลังโหลดโมเดลหลัก…</p>;
  } else if (!classSummary.available) {
    body = (
      <p className="empty">
        โมเดลหลักยังไม่พร้อม{classSummary.hint ? <> — รัน <code>{classSummary.hint}</code></> : null}
      </p>
    );
  } else if (classSeries.status === "idle") {
    body = <p className="empty">{selectedHarp == null ? "เลือก active region จากตารางด้านล่าง" : "—"}</p>;
  } else if (classSeries.status === "loading") {
    body = <p className="empty">กำลังคำนวณ…</p>;
  } else if (classSeries.status === "error" || !classSeries.data?.points?.length) {
    body = <p className="empty">{classSeries.error ?? "ไม่มีข้อมูล"}</p>;
  } else {
    const series = classSeries.data;
    // key ผูกกับ HARP + จุดทำงาน — เปลี่ยนเมื่อไรช่องที่เลือกกลับไปที่คำพยากรณ์ล่าสุด
    body = <ClassForecastBody key={`${series.harpnum}:${series.mode}`} series={series} summary={classSummary} />;
  }

  return (
    <section ref={revealRef} id="class" className="card card--wide card--class" aria-labelledby="h-class" data-reveal>
      <div className="card__head">
        <div className="card__head-title">
          <h2 id="h-class">Flare Class Forecast</h2>
          <span className="hint">
            โมเดลหลัก {classSummary?.label ?? "LSTM + V3"} · ระดับของ flare ที่แรงที่สุดใน 24 ชม. ถัดไป
            {classSummary?.cadence_hours ? ` · ออกคำพยากรณ์ทุก ${classSummary.cadence_hours} ชม.` : ""}
          </span>
        </div>
        <ClassModeToggle />
      </div>
      {body}
    </section>
  );
}

function ClassForecastBody({ series, summary }) {
  const points = series.points;
  const [picked, setPicked] = useState(points.length - 1);
  const point = points[Math.min(picked, points.length - 1)];
  const meta = levelMeta(point.level);
  const truth = levelMeta(point.true_level);
  const seeds = summary.n_seeds ?? {};
  const hit = point.level === point.true_level;

  return (
    <div className="class-forecast">
      <div className={`class-forecast__now class-level--${meta.key}`}>
        <div className="class-forecast__badge" aria-live="polite">{meta.short}</div>
        <div className="class-forecast__label">{meta.text}</div>
        <p className="class-forecast__hint">{meta.hint}</p>
        <dl className="class-forecast__meta">
          <dt>ออกเมื่อ</dt>
          <dd>{formatDateTime(point.issue_time)} UTC</dd>
          <dt>โอกาส ≥M1.0</dt>
          <dd>{(point.prob_m * 100).toFixed(1)}% · เตือน {point.n_alarm_m}/{seeds.M ?? "?"} seed</dd>
          <dt>โอกาส ≥X1.0</dt>
          <dd>
            {(point.prob_x * 100).toFixed(1)}%
            {series.mode === "sensitive"
              ? ` · เตือน ${point.n_alarm_x}/${seeds.X ?? "?"} seed`
              : ` · เกณฑ์ ${(summary.strict_threshold * 100).toFixed(1)}%`}
          </dd>
          <dt>HARP</dt>
          <dd>{series.harpnum}{series.noaa_ar ? ` · NOAA ${series.noaa_ar}` : ""}</dd>
        </dl>
        <span className={`outcome ${hit ? "outcome--no" : "outcome--yes"}`}>
          {hit ? "✓" : "✗"} เกิดจริง: {truth.text}
        </span>
        {point.split === "train" && (
          <p className="class-forecast__warn">ช่วงนี้อยู่ใน train — โมเดลเคยเห็นข้อมูลนี้ตอนเทรน</p>
        )}
      </div>

      <div className="class-forecast__timeline">
        <p className="class-forecast__mode-note">{CLASS_MODES[series.mode]?.hint}</p>
        <div className="class-strip" style={{ "--n": points.length }}>
          <span className="class-strip__rowhead">ทำนาย</span>
          <div className="class-strip__row">
            {points.map((p, i) => {
              const m = levelMeta(p.level);
              return (
                <button
                  key={p.issue_time}
                  type="button"
                  className={`class-strip__cell class-level--${m.key}${i === picked ? " is-picked" : ""}`}
                  aria-pressed={i === picked}
                  aria-label={`${formatDateTime(p.issue_time)} ทำนาย ${m.text} เกิดจริง ${levelMeta(p.true_level).text}`}
                  onClick={() => setPicked(i)}
                >
                  {m.short}
                </button>
              );
            })}
          </div>
          <span className="class-strip__rowhead">เกิดจริง</span>
          <div className="class-strip__row" aria-hidden="true">
            {points.map((p) => {
              const m = levelMeta(p.true_level);
              return (
                <span key={p.issue_time} className={`class-strip__truth class-level--${m.key}`}>{m.short}</span>
              );
            })}
          </div>
        </div>
        <div className="class-strip__axis">
          <span>{formatDateTime(points[0].issue_time)}</span>
          <span>{formatDateTime(points[points.length - 1].issue_time)}</span>
        </div>
        <p className="hint class-forecast__foot">
          แต่ละช่องคือคำพยากรณ์ 24 ชม. ที่ออก ณ เวลานั้น (ช่องถัดกันห่าง {summary.cadence_hours ?? 12} ชม. จึงคาบเกี่ยวกัน) ·
          split ของ HARP นี้: {SPLIT_TEXT[point.split] ?? point.split} · &lt;M ไม่ได้แปลว่าจะเกิด C
        </p>
      </div>
    </div>
  );
}

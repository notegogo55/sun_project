import { useEffect, useRef } from "react";
import { useApp } from "../../state/AppContext.js";
import { formatDateTime, RISK_LABEL } from "../../lib/format.js";
import { useCountUp } from "../../hooks/useCountUp.js";
import { useReveal } from "../../hooks/useReveal.js";

/** เปลือกการ์ดร่วมของทุกสถานะ (idle/loading/error/ready) — เดิมแต่ละสถานะก็อปวางเปลือก
 *  เดียวกันซ้ำ 4 ครั้ง ทำให้แก้ header หรือ reveal ทีต้องแก้ 4 ที่ */
function RiskShell({ revealRef, children }) {
  const { forecastModelInfo } = useApp();
  const modelText = forecastModelInfo ? ` · โมเดล ${forecastModelInfo.label}` : "";
  return (
    <section ref={revealRef} id="risk" className="card card--gauge" aria-labelledby="h-risk" data-reveal>
      <div className="card__head">
        <div className="card__head-title">
          <h2 id="h-risk">Flare Risk</h2>
          <span className="hint">โอกาส ≥M1.0 รายชั่วโมง (18 SHARP) ของ active region ที่เลือก{modelText}</span>
        </div>
      </div>
      {children}
    </section>
  );
}

export default function RiskCard() {
  const { risk } = useApp();
  const revealRef = useReveal();
  const fillRef = useRef(null);

  if (risk.status === "idle") {
    return (
      <RiskShell revealRef={revealRef}>
        <div id="riskPanel" className="risk">
          <p className="empty">เลือก active region จากตารางด้านขวา</p>
        </div>
      </RiskShell>
    );
  }

  if (risk.status === "loading") {
    return (
      <RiskShell revealRef={revealRef}>
        <div id="riskPanel" className="risk"><p className="empty">กำลังคำนวณ…</p></div>
      </RiskShell>
    );
  }

  if (risk.status === "error" || !risk.data?.latest) {
    return (
      <RiskShell revealRef={revealRef}>
        <div id="riskPanel" className="risk">
          <p className="empty">{risk.error ?? "ไม่มีข้อมูล"}</p>
        </div>
      </RiskShell>
    );
  }

  const series = risk.data;
  const latest = series.latest;

  // สเกลแท่งเทียบกับ 3 เท่าของ threshold เพื่อให้เห็นความต่างชัด — ค่าดิบมักต่ำ
  // เพราะ positive จริงมีเพียง ~2% โมเดลจึงแทบไม่เคยให้ค่าเกิน 0.5
  const scale = Math.max(latest.threshold * 3, 0.05);
  const fillPct = Math.min(100, (latest.probability / scale) * 100);
  const thresholdPct = Math.min(100, (latest.threshold / scale) * 100);

  return (
    <RiskShell revealRef={revealRef}>
      <RiskPanelBody latest={latest} series={series} fillPct={fillPct} thresholdPct={thresholdPct} fillRef={fillRef} />
    </RiskShell>
  );
}

function RiskPanelBody({ latest, series, fillPct, thresholdPct, fillRef }) {
  const valueRef = useCountUp(latest.probability * 100, (v) => `${v.toFixed(1)}%`);

  // ตั้งความกว้างในเฟรมถัดไปเพื่อให้ transition ของ CSS ได้ทำงาน (0 -> ค่าจริง)
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      if (fillRef.current) fillRef.current.style.width = `${fillPct}%`;
    });
    return () => cancelAnimationFrame(frame);
  }, [fillPct, fillRef]);

  let outcome = null;
  if (latest.actual_label !== null && latest.actual_label !== undefined) {
    outcome = latest.actual_label === 1
      ? <span className="outcome outcome--yes">ผลจริง: เกิด flare ≥ {series.positive_class}</span>
      : <span className="outcome outcome--no">ผลจริง: ไม่เกิด flare</span>;
  }

  return (
    <div id="riskPanel" className={`risk risk--${latest.risk_level}`}>
      <div className="risk__value" ref={valueRef}>0.0%</div>
      <div className="risk__label">{RISK_LABEL[latest.risk_level] ?? latest.risk_level}</div>
      <div className="risk__bar">
        <div className="risk__fill" ref={fillRef} style={{ width: 0 }} />
        <div className="risk__threshold" style={{ left: `${thresholdPct}%` }} title="เกณฑ์ตัดสิน" />
      </div>
      <div className="risk__meta">
        HARP {series.harpnum}{series.noaa_ar ? ` · NOAA ${series.noaa_ar}` : ""}<br />
        ณ {formatDateTime(latest.issue_time)}<br />
        โอกาสเกิด flare ≥ {series.positive_class} ใน {series.horizon_hours} ชม.<br />
        <span className="hint">เกณฑ์ตัดสิน {(latest.threshold * 100).toFixed(1)}% (ปรับจาก validation)</span>
      </div>
      {outcome}
    </div>
  );
}

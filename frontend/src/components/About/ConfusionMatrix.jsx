import { useEffect, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { api } from "../../lib/api.js";
import { CM_CELL_META } from "../../lib/plotConstants.js";
import { formatDateTime } from "../../lib/format.js";
import { scrollToDashboard } from "../../lib/motion.js";
import { idleAsync, loadingAsync, readyAsync, errorAsync } from "../../lib/asyncState.js";

const CELL_LAYOUT = [
  ["tp", "fp"],
  ["fn", "tn"],
];

/** แผงนี้ตรึงอยู่กับ split ของตัวเอง ไม่ขยับตามช่วงเวลาที่เลือกในแถบกรอง (ต่างจากการ์ด
 *  อื่นทั้งหมดในหน้านี้) จึงโหลดใหม่เฉพาะตอนสลับโมเดล/split เท่านั้น — การนับ
 *  TP/FP/TN/FN ทั้งหมดมาจาก backend (sunseg.metrics.threshold_sweep) ฝั่งนี้แค่เปิด
 *  index ตามตำแหน่งสไลเดอร์แล้วคูณ/หารเลขที่ backend นับมาให้แล้ว */
export default function ConfusionMatrix() {
  const { selectHarp, applyRange } = useApp();
  const [model, setModel] = useState("lstm");         // "lstm" | "baseline"
  const [split, setSplit] = useState("test");          // "val" | "test"
  const [confusion, setConfusion] = useState(idleAsync()); // ผล /api/forecast/confusion-matrix (sweep ทั้งกริด)
  const [index, setIndex] = useState(null);            // ตำแหน่งบนกริดที่สไลเดอร์ชี้อยู่ตอนนี้
  const [cell, setCell] = useState(null);               // ช่องที่คลิกค้างไว้ ("tp"|"fp"|"fn"|"tn") — null = ไม่มีลิสต์เปิดอยู่
  const [samples, setSamples] = useState(idleAsync());  // ผล /api/forecast/confusion-matrix/samples ของ cell ที่เปิดอยู่

  useEffect(() => {
    let cancelled = false;
    setCell(null);
    setSamples(idleAsync());
    setConfusion((prev) => loadingAsync(prev));
    (async () => {
      try {
        const data = await api.confusionMatrix(model, split);
        if (cancelled) return;
        setConfusion(readyAsync(data));
        setIndex(data.frozen_index);
      } catch (err) {
        if (!cancelled) setConfusion(errorAsync(err.message));
      }
    })();
    return () => { cancelled = true; };
  }, [model, split]);

  async function handleCellClick(cellKey) {
    if (cell === cellKey) {
      setCell(null);
      setSamples(idleAsync());
      return;
    }
    const threshold = confusion.data.thresholds[index];
    setCell(cellKey);
    setSamples((prev) => loadingAsync(prev));
    try {
      const data = await api.confusionSamples(confusion.data.model, confusion.data.split, threshold, cellKey);
      setSamples(readyAsync(data));
    } catch (err) {
      setSamples(errorAsync(err.message));
    }
  }

  async function jumpToSample(harpnum, isoTime) {
    const center = new Date(isoTime);
    const spanMs = 7 * 24 * 3600 * 1000;
    const start = new Date(center.getTime() - spanMs).toISOString().slice(0, 10);
    const end = new Date(center.getTime() + spanMs).toISOString().slice(0, 10);
    await selectHarp(harpnum, { syncRange: false });
    await applyRange({ start, end });
    scrollToDashboard();
  }

  if (confusion.status === "error") return <p className="empty">{confusion.error}</p>;
  if (confusion.status !== "ready" || index === null) return <p className="empty">กำลังโหลด…</p>;

  const cm = confusion.data;
  const tp = cm.tp[index], fp = cm.fp[index], tn = cm.tn[index], fn = cm.fn[index];
  const total = tp + fp + tn + fn;
  const pct = (n) => (total ? `${(100 * n / total).toFixed(1)}%` : "—");
  const counts = { tp, fp, fn, tn };

  const recall = tp + fn ? tp / (tp + fn) : 0;
  const fpr = fp + tn ? fp / (fp + tn) : 0;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const tssValue = recall - fpr;
  const hss2Num = 2 * (tp * tn - fn * fp);
  const hss2Den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn);
  const hss2 = hss2Den ? hss2Num / hss2Den : 0;

  const threshold = cm.thresholds[index];
  const isFrozen = index === cm.frozen_index;
  const modelLabel = cm.model === "lstm" ? "LSTM" : "logistic baseline";
  const signed = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(4)}`;

  return (
    <>
      <div className="cm__head">
        <h3 className="results__title">Confusion Matrix <span>ปรับ threshold ดูการแลกเปลี่ยนระหว่าง hit กับ false alarm ได้ทันที</span></h3>
        <div className="cm__toggles">
          <div className="layer-pills cm__toggle-group" role="radiogroup" aria-label="เลือกโมเดล">
            <button type="button" className={`layer-pill${model === "lstm" ? " is-active" : ""}`} onClick={() => setModel("lstm")}>LSTM</button>
            <button type="button" className={`layer-pill${model === "baseline" ? " is-active" : ""}`} onClick={() => setModel("baseline")}>Logistic baseline</button>
          </div>
          <div className="layer-pills cm__toggle-group" role="radiogroup" aria-label="เลือกชุดข้อมูล">
            <button type="button" className={`layer-pill${split === "val" ? " is-active" : ""}`} onClick={() => setSplit("val")}>val</button>
            <button type="button" className={`layer-pill${split === "test" ? " is-active" : ""}`} onClick={() => setSplit("test")}>test</button>
          </div>
        </div>
      </div>

      <div id="cmBody">
        {split === "val" && (
          <p className="cm__warn"><b>val</b> คือชุดที่ใช้เลือก threshold นี้เอง — ตัวเลขในโหมดนี้ไม่ใช่ผลที่ควรอ้างเป็นผลงาน ดูที่ <b>test</b> แทน</p>
        )}
        <div className="cm__grid">
          <table className="cm__table" aria-label={`confusion matrix ของ ${modelLabel} บนชุด ${cm.split}`}>
            <thead>
              <tr><th /><th colSpan={2} className="cm__group">เกิดขึ้นจริง</th></tr>
              <tr><th /><th className="num">positive</th><th className="num">negative</th></tr>
            </thead>
            <tbody>
              {CELL_LAYOUT.map((rowKeys, rowIndex) => (
                <tr key={rowKeys[0]}>
                  <th className="cm__rowhead">ทำนาย<br />{rowIndex === 0 ? "positive" : "negative"}</th>
                  {rowKeys.map((key) => (
                    <CmCell key={key} cellKey={key} count={counts[key]} pctText={pct(counts[key])} selected={cell === key} onClick={handleCellClick} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <div className="cm__side">
            <div className="cm__slider">
              <div className="cm__slider-head">
                <span>threshold = <b>{threshold.toFixed(3)}</b></span>
                <button
                  type="button" className="cm__reset" disabled={isFrozen}
                  onClick={() => { setIndex(cm.frozen_index); setCell(null); setSamples(idleAsync()); }}
                >
                  กลับไป {cm.frozen_threshold.toFixed(3)}
                </button>
              </div>
              <input
                type="range" min="0" max={cm.thresholds.length - 1} step="1" value={index}
                aria-label={`threshold ของ ${modelLabel}`}
                onChange={(e) => { setIndex(Number(e.target.value)); setCell(null); setSamples(idleAsync()); }}
              />
              <div className="cm__slider-note">{isFrozen ? "ค่าที่ freeze ไว้จาก validation (ค่าที่ระบบใช้จริง)" : "ต่างจากค่าที่ freeze ไว้ — ดูเพื่อสำรวจเท่านั้น"}</div>
            </div>

            <dl className="cm__metrics">
              <div><dt>TSS</dt><dd>{signed(tssValue)}</dd></div>
              <div><dt>recall</dt><dd>{recall.toFixed(3)}</dd></div>
              <div><dt>precision</dt><dd>{precision.toFixed(3)}</dd></div>
              <div><dt>HSS2</dt><dd>{signed(hss2)}</dd></div>
            </dl>

            <p className="cm__meta">
              AUC {cm.auc.toFixed(4)} <span className="hint">(ไม่ขึ้นกับ threshold)</span><br />
              {cm.n.toLocaleString()} sample · positive {cm.n_positive.toLocaleString()}
            </p>
          </div>
        </div>

        {cell && (
          <CmSamples cellKey={cell} samples={samples} onClose={() => { setCell(null); setSamples(idleAsync()); }} onJump={jumpToSample} />
        )}
      </div>
    </>
  );
}

function CmCell({ cellKey, count, pctText, selected, onClick }) {
  const meta = CM_CELL_META[cellKey];
  return (
    <td
      className={`cm__cell cm__cell--${meta.role}${selected ? " cm__cell--selected" : ""}`}
      tabIndex={0} role="button" aria-pressed={selected}
      aria-label={`${meta.stat} (${meta.term}) ${count.toLocaleString()} sample — คลิกเพื่อดูรายการ`}
      onClick={() => onClick(cellKey)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(cellKey); } }}
    >
      <span className="cm__cell-stat">{meta.stat}</span>
      <span className="cm__cell-term">{meta.term}</span>
      <span className="cm__cell-count">{count.toLocaleString()}</span>
      <span className="cm__cell-pct">{pctText}</span>
    </td>
  );
}

function CmSamples({ cellKey, samples, onClose, onJump }) {
  const meta = CM_CELL_META[cellKey];

  let body;
  if (samples.status === "loading" || samples.status === "idle") {
    body = <p className="empty">กำลังโหลด…</p>;
  } else if (samples.status === "error") {
    body = <p className="empty">{samples.error}</p>;
  } else if (!samples.data.samples.length) {
    body = <p className="empty">ไม่มี sample ในช่องนี้ ณ threshold ปัจจุบัน</p>;
  } else {
    body = (
      <div className="table-wrap">
        <table className="cm__sample-table">
          <thead>
            <tr><th>HARP</th><th>NOAA AR</th><th>เวลาออกพยากรณ์</th><th className="num">ความน่าจะเป็น</th></tr>
          </thead>
          <tbody>
            {samples.data.samples.map((row) => (
              <tr
                key={`${row.harpnum}-${row.issue_time}`} tabIndex={0}
                onClick={() => onJump(row.harpnum, row.issue_time)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onJump(row.harpnum, row.issue_time); } }}
              >
                <td>{row.harpnum}</td>
                <td>{row.noaa_ar ?? "—"}</td>
                <td>{formatDateTime(row.issue_time)}</td>
                <td className="num">{(row.probability * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const count = samples.status === "ready"
    ? <span className="hint">
      · {samples.data.total.toLocaleString()} รายการ
      {samples.data.samples.length < samples.data.total ? ` (แสดง ${samples.data.samples.length.toLocaleString()} รายการแรกที่มั่นใจที่สุด)` : ""}
    </span>
    : null;

  return (
    <div className="cm__samples">
      <div className="cm__samples-head">
        <h4>{meta.stat} <span className="hint">{meta.term}</span> {count}</h4>
        <button type="button" className="cm__samples-close" aria-label="ปิดรายการ" onClick={onClose}>ปิด ✕</button>
      </div>
      {body}
    </div>
  );
}

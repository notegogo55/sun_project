import { useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { CLASS_MODES, levelMeta } from "../../lib/classForecast.js";
import ClassModeToggle from "../ClassModeToggle.jsx";

const fixed = (value, digits = 3) => (value != null ? value.toFixed(digits) : "—");
const pct = (value) => (value != null ? `${Math.round(value * 100)}%` : "—");

/** แถวของตารางเทียบจุดทำงาน — ค่าของแต่ละโหมดมาจาก evaluation[mode][split] ของ /api/class-forecast/summary */
const METRIC_ROWS = [
  ["≥M TSS", (e) => fixed(e.thresholds.M.tss)],
  ["≥M recall / precision", (e) => `${pct(e.thresholds.M.recall)} / ${pct(e.thresholds.M.precision)}`],
  ["≥X TSS", (e) => fixed(e.thresholds.X.tss)],
  ["≥X จับได้", (e) => `${e.thresholds.X.tp} / ${e.thresholds.X.n_positive}`],
  ["≥X precision", (e) => pct(e.thresholds.X.precision)],
  ["HSS 3 คลาส", (e) => fixed(e.hss_multiclass)],
  ["ทายระดับถูกเป๊ะ (event ≥M)", (e) => pct(e.exact_on_events)],
];

/** ผลของโมเดลหลัก (LSTM + V3 แยกระดับ <M/M/X) — ตาราง 3×3 ของจุดทำงานที่เลือก + ตัวชี้วัดของทั้งสองจุดเคียงกัน
 *  ตัวเลขชุดเดียวกับ artifacts/class_forecast/report.md (สร้างจาก service ตัวเดียวกัน) */
export default function ClassForecastResults() {
  const { classSummary, classMode } = useApp();
  const [split, setSplit] = useState("test");

  if (!classSummary?.available) {
    return (
      <>
        <h2 className="results__title">Main model <span>LSTM + V3 · ระดับ &lt;M / M / X</span></h2>
        <p className="empty">
          โมเดลหลักยังไม่พร้อม{classSummary?.hint ? <> — รัน <code>{classSummary.hint}</code></> : null}
        </p>
      </>
    );
  }

  const evaluation = classSummary.evaluation;
  const current = evaluation[classMode][split];
  const levels = current.levels;
  const test = evaluation[classMode].test;
  const nX = test.thresholds.X.n_positive;

  return (
    <>
      <div className="cm__head">
        <h2 className="results__title">
          Main model <span>{classSummary.label} · ระดับของ flare ที่แรงที่สุดใน 24 ชม. · ensemble {classSummary.n_seeds.M} + {classSummary.n_seeds.X} seed</span>
        </h2>
        <div className="cm__toggles">
          <ClassModeToggle />
          <div className="layer-pills cm__toggle-group" role="radiogroup" aria-label="ชุดข้อมูล">
            {["test", "val"].map((s) => (
              <button
                key={s}
                type="button"
                role="radio"
                aria-checked={s === split}
                className={`layer-pill${s === split ? " is-active" : ""}`}
                onClick={() => setSplit(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="class-results">
        <div className="table-wrap table-wrap--auto">
          <table className="class-cm">
            <caption className="hint">
              {CLASS_MODES[classMode].label} · {split} · แถว = ระดับที่เกิดจริง, คอลัมน์ = ระดับที่ทำนาย (% ของแถว)
            </caption>
            <thead>
              <tr>
                <th scope="col">จริง \ ทำนาย</th>
                {levels.map((l) => <th scope="col" key={l} className="num">{levelMeta(l).short}</th>)}
              </tr>
            </thead>
            <tbody>
              {current.confusion.map((row, i) => {
                const total = Math.max(1, row.reduce((a, b) => a + b, 0));
                return (
                  <tr key={levels[i]}>
                    <th scope="row">{levelMeta(levels[i]).text}</th>
                    {row.map((count, j) => (
                      <td
                        key={levels[j]}
                        className={`num class-cm__cell${i === j ? " is-diag" : ""} class-level--${levelMeta(levels[j]).key}`}
                        style={{ "--share": count / total }}
                      >
                        <b>{count.toLocaleString()}</b>
                        <span>{Math.round((100 * count) / total)}%</span>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="table-wrap table-wrap--auto">
          <table className="result-table">
            <thead>
              <tr>
                <th>{split}</th>
                {classSummary.modes.map((m) => (
                  <th key={m} className="num">{CLASS_MODES[m]?.label ?? m}{m === classMode ? " ●" : ""}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(([label, value]) => (
                <tr key={label}>
                  <td>{label}</td>
                  {classSummary.modes.map((m) => (
                    <td key={m} className="num">{value(evaluation[m][split])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="note">
        สองจุดทำงานต่างกันแค่ระดับ X: <em>เตือนไว</em> ใช้ threshold ราย seed ที่ให้ TSS สูงสุด (โหวตข้างมาก) ·{" "}
        <em>ระมัดระวัง</em> ใช้ความน่าจะเป็นเฉลี่ย ≥ {(classSummary.strict_threshold * 100).toFixed(1)}% ที่เลือกบน validation
        ให้ HSS 3 คลาสสูงสุด · X ใน test มีแค่ {nX} sample ตัวเลขของระดับ X จึงแกว่งมาก · &lt;M ไม่ได้แปลว่าจะเกิด C ·
        รายงานเต็มที่ <code>artifacts/class_forecast/report.md</code>
      </p>
    </>
  );
}

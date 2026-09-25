import { useApp } from "../state/AppContext.js";
import { PageHeader, StatTile } from "../components/ui/Headers.jsx";
import ConfusionMatrix from "../components/Model/ConfusionMatrix.jsx";
import ClassForecastResults from "../components/Model/ClassForecastResults.jsx";
import { CLASS_MODES } from "../lib/classForecast.js";
import ForecastModelPicker from "../components/ForecastModelPicker.jsx";

const fixed = (value, digits = 3) => (value != null ? value.toFixed(digits) : "—");

const POOLING_TEXT = { attention: "attention pooling", last: "last-step", mean: "mean pooling" };

/** หนึ่งแถวของตารางผล — โมเดลที่ยังไม่ได้เทรนแสดงคำสั่งที่ต้องรันแทนตัวเลข */
function ModelRow({ model, selected }) {
  const hint = [
    model.n_parameters ? `${model.n_parameters.toLocaleString()} พารามิเตอร์` : null,
    POOLING_TEXT[model.pooling] ?? null,
  ].filter(Boolean).join(" · ");

  if (!model.available) {
    return (
      <tr>
        <td>{model.label} <span className="hint">ยังไม่ได้เทรน</span></td>
        <td className="num" colSpan={4}><code>{model.train_hint}</code></td>
      </tr>
    );
  }
  return (
    <tr>
      <td>
        {model.label}{selected ? " ●" : ""}
        <span className="hint">{hint || "ประวัติ 24 ชม."}</span>
      </td>
      <td className="num">{fixed(model.val_tss)}</td>
      <td className="num strong">{fixed(model.test?.tss)}</td>
      <td className="num">{fixed(model.test?.auc)}</td>
      <td className="num">{fixed(model.test?.recall, 2)}</td>
    </tr>
  );
}

/** ผลการทดลอง — ตัวเลขของโมเดลพยากรณ์ทุกตัวและ logistic baseline อ่านสดจาก /api/forecast/models
 *  (ตัวชี้วัดใน checkpoint + ไฟล์ <ชื่อ>.json ที่สคริปต์เทรนเขียนไว้) ส่วน Dice ของ U-Net อ่านจาก /api/info */
export default function ModelPage() {
  const { info, forecastModels, forecastModelInfo, classSummary, classMode } = useApp();
  // ตัวเลขสรุปบนสุดเป็นของโมเดลหลัก (test, จุดทำงานที่เลือก) — โมเดลรายชั่วโมงอยู่ในตารางด้านล่าง
  const mainTest = classSummary?.available ? classSummary.evaluation[classMode].test : null;
  const modeText = CLASS_MODES[classMode]?.label ?? classMode;
  const dice = info?.segmentation?.metrics?.test?.dice;

  const selected = forecastModelInfo;
  // baseline เทรนคู่กับทุกโมเดลบนข้อมูลชุดเดียวกัน — อ่านจากโมเดลปริยายก่อน แล้วค่อยตัวอื่นที่มี
  const baselineSource = [...forecastModels].sort((a, b) => Number(b.default) - Number(a.default))
    .find((m) => m.baseline_test);
  const baseline = baselineSource?.baseline_test ?? null;

  // จำนวนต่อ split อ่านสดจาก sequence dataset (/api/info) — ชุดเดียวกับที่ทุกโมเดลเทรน/วัดผล
  const splits = info?.data?.splits ?? [];
  const splitTotal = Math.max(1, splits.reduce((sum, row) => sum + row.n, 0));

  const trained = forecastModels.filter((m) => m.available && m.test?.tss != null);
  const winners = baseline?.tss != null ? trained.filter((m) => m.test.tss > baseline.tss) : [];

  return (
    <div className="page">
      <PageHeader
        kicker="// SECTION_04 / RESULTS"
        title="MODEL EVALUATION"
        sub="โมเดลหลัก LSTM + V3 แยกระดับ <M / M / X · โมเดลรายชั่วโมง LSTM · TCN · Transformer · DA-RNN เทียบกับ logistic baseline · แบ่งข้อมูลแบบ HARP-disjoint"
      >
        <ForecastModelPicker />
      </PageHeader>

      <div className="stat-strip">
        <StatTile
          label="LSTM + V3 test TSS ≥M" value={fixed(mainTest?.thresholds.M.tss)}
          unit={modeText} title="โมเดลหลัก — True Skill Statistic ของระดับ ≥M1.0 บน test"
        />
        <StatTile
          label="LSTM + V3 test TSS ≥X" tone="sun" value={fixed(mainTest?.thresholds.X.tss)}
          unit={mainTest ? `จับ X ${mainTest.thresholds.X.tp}/${mainTest.thresholds.X.n_positive}` : ""}
        />
        <StatTile
          label="ทายระดับถูกเป๊ะ" tone="green"
          value={mainTest?.exact_on_events != null ? `${Math.round(mainTest.exact_on_events * 100)}%` : "—"}
          unit="ของ event ≥M บน test"
        />
        <StatTile label="U-Net test Dice" tone="violet" value={dice != null ? dice.toFixed(3) : "—"} />
      </div>

      <div className="results">
        <div className="results__block results__block--wide" id="class-results">
          <ClassForecastResults />
        </div>

        <div className="results__block">
          <h2 className="results__title">Hourly risk models <span>flare ≥M1.0 ใน 24 ชม. · 18 SHARP ราย 1 ชม.</span></h2>
          <div className="table-wrap table-wrap--auto">
            <table className="result-table">
              <thead>
                <tr>
                  <th>โมเดล</th>
                  <th className="num">val TSS</th>
                  <th className="num">test TSS</th>
                  <th className="num">test AUC</th>
                  <th className="num">test recall</th>
                </tr>
              </thead>
              <tbody>
                {forecastModels.map((m) => (
                  <ModelRow key={m.name} model={m} selected={m.name === selected?.name} />
                ))}
                <tr>
                  <td>Logistic regression <span className="hint">ค่า ณ เวลาเดียว</span></td>
                  <td className="num">—</td>
                  <td className="num strong">{fixed(baseline?.tss)}</td>
                  <td className="num">{fixed(baseline?.auc)}</td>
                  <td className="num">{fixed(baseline?.recall, 2)}</td>
                </tr>
                <tr>
                  <td>U-Net <span className="hint">Dice / IoU</span></td>
                  <td className="num" colSpan={4}>{dice != null ? `Dice ${dice.toFixed(3)}` : "ยังไม่ได้เทรน"}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="note">
            ● = โมเดลที่เลือกอยู่ (สลับได้ที่ปุ่มด้านบน มีผลกับ dashboard และ confusion matrix ด้วย) ·
            logistic regression ใช้ค่า ณ เวลาเดียว — โมเดลลำดับเวลาที่ชนะแถวนี้ไม่ชัดเจนแปลว่ามิติเวลาไม่ได้ช่วย
            {baseline?.tss != null && trained.length > 0 && (
              winners.length
                ? <> · บน test ตอนนี้ <em>{winners.map((m) => m.label).join(", ")}</em> ชนะ baseline</>
                : <> · บน test ตอนนี้ <em>ยังไม่มีโมเดลลำดับเวลาตัวไหนชนะ baseline</em></>
            )}
            {" "}· ตัวอย่างที่เป็นอิสระต่อกันจริงคือจำนวน HARP ที่เคยเกิด flare ไม่ใช่จำนวน sample
            ส่วนต่างเล็ก ๆ ระหว่างโมเดลจึงอาจเป็นแค่ noise
          </p>
        </div>

        <div className="results__block">
          <h2 className="results__title">Data split <span>HARP-disjoint</span></h2>
          <ul className="splits">
            {splits.length === 0 && <li className="split"><span className="split__name">—</span> ยังไม่มีข้อมูล sequence</li>}
            {splits.map((row) => (
              <li className="split" key={row.split} title={`${row.first.slice(0, 10)} → ${row.last.slice(0, 10)}`}>
                <span className="split__name">{row.split}</span>
                <span className="split__bar"><i style={{ width: `${((100 * row.n) / splitTotal).toFixed(1)}%` }} /></span>
                <span className="split__num">{row.n.toLocaleString()}</span>
                <span className="split__pos">
                  {row.n_positive.toLocaleString()} · {((100 * row.n_positive) / Math.max(row.n, 1)).toFixed(2)}%
                </span>
              </li>
            ))}
          </ul>
          <p className="note">
            แบ่งตามเวลา <em>และ</em> ให้ HARP ไม่ซ้ำข้าม split · กรอง <code>|LON| &lt; 68°</code> กับ <code>QUALITY == 0</code> ·
            คำนวณ normalization statistics จาก train set เท่านั้น
          </p>
        </div>

        <div className="results__block results__block--wide" id="confusion">
          <ConfusionMatrix />
        </div>
      </div>
    </div>
  );
}

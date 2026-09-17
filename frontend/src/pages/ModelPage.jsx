import { useApp } from "../state/AppContext.js";
import { PageHeader, StatTile } from "../components/ui/Headers.jsx";
import ConfusionMatrix from "../components/Model/ConfusionMatrix.jsx";

/** ผลการทดลอง — ตัวเลขในตารางเป็นผลที่วัดไว้ (README หัวข้อ "ตัวชี้วัด") ยกเว้น test TSS ของ
 *  LSTM และ Dice ของ U-Net ที่อ่านสดจาก /api/info ถ้าโมเดลโหลดอยู่ */
export default function ModelPage() {
  const { info } = useApp();
  const tss = info?.forecast?.metrics?.test?.tss;
  const dice = info?.segmentation?.metrics?.test?.dice;

  return (
    <div className="page">
      <PageHeader
        kicker="// SECTION_04 / RESULTS"
        title="MODEL EVALUATION"
        sub="LSTM vs logistic baseline · flare ≥M1.0 ใน 24 ชม. · แบ่งข้อมูลแบบ HARP-disjoint"
      />

      <div className="stat-strip">
        <StatTile label="LSTM test TSS" value={tss != null ? tss.toFixed(3) : "0.472"} title="True Skill Statistic บน test set" />
        <StatTile label="LSTM test AUC" tone="green" value="0.937" />
        <StatTile label="Test positives" tone="sun" value="37" unit="จาก 9,886 sample" />
        <StatTile label="U-Net test Dice" tone="violet" value={dice != null ? dice.toFixed(3) : "—"} />
      </div>

      <div className="results">
        <div className="results__block">
          <h2 className="results__title">Experiment results <span>flare ≥M1.0 ใน 24 ชม.</span></h2>
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
                <tr>
                  <td>LSTM <span className="hint">ประวัติ 24 ชม.</span></td>
                  <td className="num">0.845</td>
                  <td className="num strong">{tss != null ? tss.toFixed(3) : "0.472"}</td>
                  <td className="num">0.937</td>
                  <td className="num">0.51</td>
                </tr>
                <tr>
                  <td>Logistic regression <span className="hint">ค่า ณ เวลาเดียว</span></td>
                  <td className="num">0.827</td>
                  <td className="num strong">0.908</td>
                  <td className="num">0.988</td>
                  <td className="num">0.97</td>
                </tr>
                <tr>
                  <td>U-Net <span className="hint">Dice / IoU</span></td>
                  <td className="num" colSpan={4}>{dice != null ? `Dice ${dice.toFixed(3)}` : "ยังไม่ได้เทรน"}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="note">
            logistic regression ที่ใช้ค่า ณ เวลาเดียวทำได้ดีกว่า LSTM ที่เห็นประวัติ 24 ชม. — เป็นผลที่วัดได้จริง
            สาเหตุหลักคือ test set มี positive เพียง 37 ตัว และตัวอย่างที่เป็นอิสระต่อกันจริงคือจำนวน HARP ที่เคยเกิด flare (~114 ดวง) ไม่ใช่จำนวน sample
          </p>
        </div>

        <div className="results__block">
          <h2 className="results__title">Data split <span>HARP-disjoint</span></h2>
          <ul className="splits">
            <li className="split">
              <span className="split__name">train</span>
              <span className="split__bar"><i style={{ width: "85.2%" }} /></span>
              <span className="split__num">208,033</span>
              <span className="split__pos">3,854 · 1.85%</span>
            </li>
            <li className="split">
              <span className="split__name">val</span>
              <span className="split__bar"><i style={{ width: "10.8%" }} /></span>
              <span className="split__num">26,255</span>
              <span className="split__pos">408 · 1.55%</span>
            </li>
            <li className="split">
              <span className="split__name">test</span>
              <span className="split__bar"><i style={{ width: "4.0%" }} /></span>
              <span className="split__num">9,886</span>
              <span className="split__pos">37 · 0.37%</span>
            </li>
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

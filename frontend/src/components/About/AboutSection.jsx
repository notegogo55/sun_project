import { useApp } from "../../state/AppContext.js";
import { useReveal } from "../../hooks/useReveal.js";
import ConfusionMatrix from "./ConfusionMatrix.jsx";

export default function AboutSection() {
  const { info } = useApp();
  const tss = info?.forecast?.metrics?.test?.tss;
  const dice = info?.segmentation?.metrics?.test?.dice;
  const headRef = useReveal();
  const pillar1Ref = useReveal();
  const pillar2Ref = useReveal();
  const pillar3Ref = useReveal();
  const resultsRef = useReveal();
  const splitsRef = useReveal();
  const confusionRef = useReveal();

  return (
    <section className="about" id="about">
      <div className="about__inner">
        <header ref={headRef} className="about__head" data-reveal>
          <h2>เกี่ยวกับงานนี้</h2>
          <p className="hint">แบ่งการทำงานเป็น 3 ส่วน — PyTorch + FastAPI</p>
        </header>

        <div className="pillars">
          <article ref={pillar1Ref} className="pillar" data-reveal>
            <p className="pillar__index">01 <span>Segmentation</span></p>
            <h3>U-Net</h3>
            <p>แบ่งส่วน active region จากภาพ magnetogram เต็มดวงของ SDO/HMI โดยใช้ bitmap จาก SHARP เป็น ground truth</p>
          </article>
          <article ref={pillar2Ref} className="pillar" data-reveal>
            <p className="pillar__index">02 <span>Tracking</span></p>
            <h3>Hungarian + differential rotation</h3>
            <p>จับคู่ AR แต่ละดวงข้ามเวลา โดยชดเชยการหมุนของดวงอาทิตย์ที่เร็วไม่เท่ากันในแต่ละละติจูด</p>
          </article>
          <article ref={pillar3Ref} className="pillar" data-reveal>
            <p className="pillar__index">03 <span>Forecasting</span></p>
            <h3>LSTM</h3>
            <p>ทำนายโอกาสเกิด flare ≥M1.0 ภายใน 24 ชม. จาก SHARP magnetic parameters ย้อนหลัง 24 จุด</p>
          </article>
        </div>

        <div className="results">
          <div ref={resultsRef} className="results__block" data-reveal>
            <h3 className="results__title">ผลการทดลอง <span>flare ≥M1.0 ใน 24 ชม.</span></h3>
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

          <div ref={splitsRef} className="results__block" data-reveal>
            <h3 className="results__title">การแบ่งข้อมูล <span>HARP-disjoint</span></h3>
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

          <div ref={confusionRef} className="results__block results__block--wide" data-reveal>
            <ConfusionMatrix />
          </div>
        </div>
      </div>
    </section>
  );
}

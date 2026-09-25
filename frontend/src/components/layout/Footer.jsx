import { Link } from "react-router-dom";
import { useApp } from "../../state/AppContext.js";

export default function Footer() {
  const { info, classSummary, classMode } = useApp();

  // ตัวเลขของโมเดลหลัก (LSTM + V3) บน test ในจุดทำงานที่เลือก
  const mainTest = classSummary?.available ? classSummary.evaluation[classMode].test : null;
  const dice = info?.segmentation?.metrics?.test?.dice;
  const metrics = [
    ["LSTM + V3 test TSS ≥M", mainTest ? mainTest.thresholds.M.tss.toFixed(4) : "—"],
    ["LSTM + V3 test TSS ≥X", mainTest ? mainTest.thresholds.X.tss.toFixed(4) : "—"],
    ["U-Net test Dice", dice != null ? dice.toFixed(4) : "—"],
    ["พยากรณ์ล่วงหน้า", "24 ชม. · ระดับ <M / M / X"],
  ];

  return (
    <footer className="footer">
      <div className="footer__grid">
        <div>
          <h3>About SUNSEG</h3>
          <p>
            ระบบแบ่งส่วน active region จากภาพ magnetogram เต็มดวงของ SDO/HMI ติดตามแต่ละดวงผ่านการหมุน
            ของดวงอาทิตย์ แล้วพยากรณ์ระดับของ solar flare (&lt;M / M / X) ภายใน 24 ชั่วโมงข้างหน้าด้วยโมเดลหลัก LSTM + V3
          </p>
        </div>

        <div>
          <h3>Data Sources</h3>
          <ul>
            <li><a href="http://jsoc.stanford.edu/" target="_blank" rel="noreferrer">JSOC · SDO/HMI SHARP</a></li>
            <li><a href="https://jsoc1.stanford.edu/data/aia/synoptic/" target="_blank" rel="noreferrer">SDO/AIA synoptic</a></li>
            <li><a href="https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/" target="_blank" rel="noreferrer">NOAA GOES XRS</a></li>
            <li><a href="https://www.swpc.noaa.gov/" target="_blank" rel="noreferrer">NOAA SWPC</a></li>
          </ul>
        </div>

        <div>
          <h3>Model Status</h3>
          <div className="footer__metrics" id="modelSummary">
            {metrics.map(([label, value]) => (
              <div key={label} className="footer__metric"><span>{label}</span><b>{value}</b></div>
            ))}
          </div>
          <Link to="/model" className="btn btn--quiet btn--sm">ดูผลการทดลอง →</Link>
        </div>
      </div>

      <div className="footer__bottom">
        <div><strong>SUNSEG</strong> · Solar Active Region Segmentation · Tracking · Flare Forecasting</div>
        <div>PyTorch + FastAPI + React · <a href="/docs" target="_blank" rel="noopener">เอกสาร API ↗</a></div>
      </div>
    </footer>
  );
}

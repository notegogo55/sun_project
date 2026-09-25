import { Link } from "react-router-dom";
import { dashboardHref } from "../lib/nav.js";
import { PageHeader, SectionHead } from "../components/ui/Headers.jsx";

const PILLARS = [
  {
    index: "01", label: "Segmentation", title: "U-Net",
    text: "แบ่งส่วน active region จากภาพ magnetogram เต็มดวงของ SDO/HMI โดยใช้ bitmap จาก SHARP เป็น ground truth",
    foot: "hmi.M_720s → mask ราย pixel", to: dashboardHref("frame"),
  },
  {
    index: "02", label: "Tracking", title: "Hungarian + differential rotation",
    text: "จับคู่ AR แต่ละดวงข้ามเวลา โดยชดเชยการหมุนของดวงอาทิตย์ที่เร็วไม่เท่ากันในแต่ละละติจูด",
    foot: "track ยืนยันเมื่อเห็นติดกัน ≥ 2 เฟรม", to: dashboardHref("extraction"),
  },
  {
    index: "03", label: "Forecasting", title: "LSTM + V3 — ระดับ <M / M / X",
    text: "โมเดลหลัก LSTM + V3 (SHARP + ความเข้มแสง AIA + X-ray ทุก 12 ชม.) ทำนายระดับของ flare ที่แรงที่สุดใน 24 ชม. ถัดไป — ensemble 25 seed ต่อระดับ พร้อมโมเดลความเสี่ยง ≥M1.0 รายชั่วโมงอีกสี่สถาปัตยกรรม (LSTM · TCN · Transformer · DA-RNN)",
    foot: "threshold freeze จาก validation set", to: dashboardHref("class"),
  },
];

const SOURCES = [
  ["JSOC hmi.sharp_cea_720s", "SHARP magnetic parameters — features ของโมเดลพยากรณ์ (พิกัด CEA แก้ผลการฉายแล้ว)"],
  ["JSOC hmi.sharp_720s", "bitmap segment — ground-truth mask ของ U-Net (พิกัด CCD ตรงกับภาพเต็มดวง)"],
  ["JSOC hmi.M_720s", "ภาพ magnetogram เต็มดวง — input ของ U-Net"],
  ["AIA synoptic", "ภาพ AIA 1600/304/171 Å — เลเยอร์ชั้นบรรยากาศใน dashboard", "https://jsoc1.stanford.edu/data/aia/synoptic/"],
  ["NGDC GOES XRS reports", "รายการ flare — labels ของโมเดลพยากรณ์ (ค่าเริ่มต้น ครอบคลุม 1975–2017)", "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"],
  ["HARPNUM ↔ NOAA", "เชื่อม SHARP เข้ากับ flare catalog", "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"],
  ["GOES particle (5 นาที)", "ฟลักซ์โปรตอนรอบเวลาที่เกิด flare — แผง Proton flux"],
  ["NOAA NCEI xrsf-l2-avg1m", "ฟลักซ์ X-ray ต่อเนื่องรายนาที — เส้น GOES X-ray ใน dashboard"],
];

const STACK = ["PyTorch", "FastAPI", "React + Vite", "React Router", "Plotly", "nginx", "Docker Compose"];

export default function AboutPage() {
  return (
    <div className="page">
      <PageHeader
        kicker="// SECTION_05 / PROJECT"
        title="ABOUT SUNSEG"
        sub="Solar Active Region Segmentation · Tracking · Flare Forecasting"
      />

      <SectionHead
        index="01"
        label="METHOD"
        title={<>แบ่งงานเป็น<em>สามส่วน</em></>}
        lead="ส่วนแรกหาว่า active region อยู่ตรงไหน ส่วนที่สองรู้ว่าเป็นดวงเดิมข้ามเวลา ส่วนสุดท้ายทำนายว่าจะปะทุหรือไม่"
      />
      <div className="pillars">
        {PILLARS.map((p) => (
          <article key={p.index} className="pillar">
            <p className="pillar__index">{p.index} <span>{p.label}</span></p>
            <h3>{p.title}</h3>
            <p>{p.text}</p>
            <p className="pillar__foot">
              {p.foot} · <Link to={p.to}>ดูใน dashboard →</Link>
            </p>
          </article>
        ))}
      </div>

      <SectionHead index="02" label="DATA SOURCES" title={<>ข้อมูลจาก<em>ดาวเทียมจริง</em></>} />
      <div className="results__block" style={{ marginBottom: 36 }}>
        <div className="table-wrap table-wrap--auto">
          <table className="source-table">
            <thead><tr><th>แหล่ง</th><th>ใช้ทำอะไร</th></tr></thead>
            <tbody>
              {SOURCES.map(([name, use, href]) => (
                <tr key={name}>
                  <td>{href ? <a href={href} target="_blank" rel="noreferrer">{name}</a> : name}</td>
                  <td>{use}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <SectionHead index="03" label="STACK" title={<>เครื่องมือที่<em>ใช้สร้าง</em></>} />
      <ul className="stack-list">
        {STACK.map((name) => <li key={name}><span className="badge badge--info">{name}</span></li>)}
      </ul>
    </div>
  );
}

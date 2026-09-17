import { useState } from "react";
import { Link } from "react-router-dom";
import { dashboardHref } from "../../lib/nav.js";
import { SectionHead } from "../ui/Headers.jsx";

const STEPS = [
  {
    tag: "SEGMENTATION",
    icon: "◎",
    title: "U-Net · Active Region Mask",
    sub: "SDO/HMI · FULL-DISK MAGNETOGRAM",
    desc: "แบ่งส่วน active region จากภาพ magnetogram เต็มดวง โดยใช้ bitmap จาก SHARP เป็น ground truth แล้วซ้อน mask ลงบนภาพ AIA สามชั้นบรรยากาศ",
    to: dashboardHref("frame"),
  },
  {
    tag: "TRACKING",
    icon: "⟳",
    title: "Hungarian + Differential Rotation",
    sub: "AR IDENTITY ACROSS FRAMES",
    desc: "จับคู่ AR แต่ละดวงข้ามเวลา โดยชดเชยการหมุนของดวงอาทิตย์ที่เร็วไม่เท่ากันในแต่ละละติจูด แล้วดึงความเข้มแสงราย AR ตามเวลา",
    to: dashboardHref("extraction"),
  },
  {
    tag: "FORECASTING",
    icon: "∿",
    title: "LSTM · Flare ≥M1.0 in 24 h",
    sub: "SHARP PARAMETERS · 24-STEP HISTORY",
    desc: "ทำนายโอกาสเกิด flare ≥M1.0 ภายใน 24 ชม. จาก SHARP magnetic parameters ย้อนหลัง 24 จุด พร้อมน้ำหนัก attention ว่าโมเดลสนใจช่วงไหน",
    to: dashboardHref("risk"),
  },
  {
    tag: "EVALUATION",
    icon: "▦",
    title: "TSS · AUC · Confusion Matrix",
    sub: "HARP-DISJOINT TEST SET",
    desc: "เทียบ LSTM กับ logistic regression บนชุดทดสอบที่ HARP ไม่ซ้ำกับชุดฝึก ปรับ threshold ดูการแลกเปลี่ยนระหว่าง hit กับ false alarm ได้ทันที",
    to: "/model",
  },
];

/** การ์ดขยายตัวแบบ DataCard ของต้นแบบ — ชี้เมาส์/โฟกัสแล้วการ์ดนั้นกางออก คลิกเพื่อไปยังส่วนนั้น */
export default function PipelineCards() {
  const [active, setActive] = useState(null);

  return (
    <section className="hub-section" aria-labelledby="pipeline-title">
      <div className="hub-section__inner">
        <SectionHead
          index="02"
          label="PIPELINE"
          title={<span id="pipeline-title">สี่ขั้นจาก<em>ภาพ</em>ถึง<em>คำพยากรณ์</em></span>}
          lead="แต่ละขั้นเปิดดูผลจริงได้ใน dashboard — ชี้ที่การ์ดเพื่ออ่านรายละเอียด"
        />
        <div className="flex-cards" onMouseLeave={() => setActive(null)}>
          {STEPS.map((step, index) => (
            <Link
              key={step.tag}
              to={step.to}
              className={`glass-card${active === index ? " is-active" : ""}`}
              onMouseEnter={() => setActive(index)}
              onFocus={() => setActive(index)}
            >
              <span className="glass-card__glow" />
              <span className="glass-card__body">
                <span className="glass-card__top">
                  <span className="glass-card__icon" aria-hidden="true">{step.icon}</span>
                  <span>
                    <span className="glass-card__tag">{step.tag}</span>
                    <span className="glass-card__title">{step.title}</span>
                  </span>
                </span>
                <span className="glass-card__desc">
                  <span className="glass-card__sub">{step.sub}</span>
                  <span className="glass-card__text">{step.desc}</span>
                  <span className="glass-card__more">EXPLORE <i /> →</span>
                </span>
                <span className="glass-card__index">0{index + 1} / 0{STEPS.length}</span>
                <span className="glass-card__vert" aria-hidden="true"><i /><span>{step.tag}</span></span>
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

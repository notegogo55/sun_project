import { useState } from "react";
import { useApp } from "../state/AppContext.js";

const PRESETS = [
  { icon: "🌟", label: "พ.ค. 2024 (Gannon X8.7)", start: "2024-05-01", end: "2024-05-31", harp: 11149, title: "พายุสุริยะ Gannon 2024 (AR 13664 ปล่อย X8.7)" },
  { icon: "⚡", label: "ต.ค. 2014 (AR 12192)", start: "2014-10-18", end: "2014-10-31", harp: 1807, title: "Monster AR 12192 (X3.1)" },
  { icon: "💥", label: "ก.ย. 2017 (AR 12673 X9.3)", start: "2017-09-01", end: "2017-09-15", harp: 7115, title: "Solar Storm ก.ย. 2017 (AR 12673 X9.3)" },
  { icon: "🔥", label: "ม.ค. 2014 (Solar Max)", start: "2014-01-01", end: "2014-01-31", harp: null, title: "Solar Maximum ช่วงสูงสุดของ Cycle 24" },
  { icon: "📅", label: "ข้อมูลทั้งหมด (2011–2024)", start: "2011-01-01", end: "2024-05-31", harp: null, title: "ช่วงข้อมูลทั้งหมดที่ประมวลผล" },
];

export default function PresetsBar() {
  const { selectHarp, applyRange } = useApp();
  const [active, setActive] = useState(null);

  async function handleClick(preset) {
    setActive(preset.label);
    if (preset.harp) {
      await selectHarp(preset.harp, { syncRange: false });
    }
    await applyRange({ start: preset.start, end: preset.end });
  }

  return (
    <div className="presets-bar" id="presetsBar" role="group" aria-label="ช่วงเหตุการณ์สำคัญ">
      <span className="presets-bar__label">เหตุการณ์สำคัญ:</span>
      {PRESETS.map((preset) => (
        <button
          key={preset.label}
          type="button"
          className={`preset-btn${active === preset.label ? " is-active" : ""}`}
          title={preset.title}
          onClick={() => handleClick(preset)}
        >
          <span className="preset-btn__icon">{preset.icon}</span> {preset.label}
        </button>
      ))}
    </div>
  );
}

import { useEffect, useState } from "react";
import { useApp } from "../state/AppContext.js";

export default function FilterBar() {
  const { range, applyRange, reloading, goes } = useApp();
  const [start, setStart] = useState(range.start);
  const [end, setEnd] = useState(range.end);
  const [minClass, setMinClass] = useState(range.minClass);

  // ช่วงเวลาที่ context เปลี่ยนจากที่อื่น (presets, เลือก HARP, กระโดดจาก confusion
  // matrix) ต้องสะท้อนกลับมาที่ช่องกรอกด้วย
  useEffect(() => {
    setStart(range.start);
    setEnd(range.end);
    setMinClass(range.minClass);
  }, [range.start, range.end, range.minClass]);

  const rangeNote = goes.data
    ? `${goes.data.n_events.toLocaleString()} flare ในช่วงนี้ · `
      + ["B", "C", "M", "X"].map((c) => `${c}=${goes.data.class_counts[c] ?? 0}`).join("  ")
    : "";

  return (
    <div className="filterbar" id="filterbar">
      <label className="field">
        <span className="field__label">จาก</span>
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
      </label>
      <label className="field">
        <span className="field__label">ถึง</span>
        <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
      </label>
      <label className="field">
        <span className="field__label">flare ขั้นต่ำ</span>
        <select value={minClass} onChange={(e) => setMinClass(e.target.value)}>
          <option value="B1.0">≥ B</option>
          <option value="C1.0">≥ C</option>
          <option value="M1.0">≥ M</option>
        </select>
      </label>
      <button
        className="btn"
        disabled={reloading}
        onClick={() => applyRange({ start, end, minClass })}
      >
        แสดงข้อมูล
      </button>
      <span className="filterbar__note">{rangeNote}</span>
    </div>
  );
}

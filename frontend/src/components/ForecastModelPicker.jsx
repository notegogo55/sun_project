import { useApp } from "../state/AppContext.js";

/** ปุ่มเลือกโมเดลพยากรณ์ (LSTM / TCN / Transformer / DA-RNN) — ตัวเดียวกันทั้ง dashboard และหน้า Model
 *
 *  เลือกแล้วมีผลทั้งแอป (แผงความเสี่ยง, attention, ตัวเลขใน hero, footer) เพราะทุกแผงต้องพูดถึง
 *  โมเดลเดียวกัน ตัวที่ยังไม่ได้เทรนแสดงแต่กดไม่ได้ พร้อมคำสั่งที่ต้องรันใน tooltip — ไม่ซ่อนไป
 *  เฉย ๆ เพื่อให้รู้ว่าระบบรองรับทั้งสี่ตัว */
export default function ForecastModelPicker() {
  const { forecastModels, forecastModel, selectForecastModel } = useApp();
  if (forecastModels.length < 2) return null;

  return (
    <div className="model-picker">
      <span className="model-picker__label">Hourly risk model</span>
      <div className="layer-pills model-picker__pills" role="radiogroup" aria-label="เลือกโมเดลความเสี่ยงรายชั่วโมง">
        {forecastModels.map((m) => {
          const active = m.name === forecastModel;
          const tss = m.test?.tss;
          const title = m.available
            ? `${m.label}${tss != null ? ` · test TSS ${tss.toFixed(3)}` : ""}`
            : `ยังไม่ได้เทรน — รัน ${m.train_hint}`;
          return (
            <button
              key={m.name}
              type="button"
              role="radio"
              aria-checked={active}
              className={`layer-pill${active ? " is-active" : ""}`}
              disabled={!m.available}
              title={title}
              onClick={() => selectForecastModel(m.name)}
            >
              {m.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

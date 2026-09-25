import { useApp } from "../state/AppContext.js";
import { CLASS_MODES } from "../lib/classForecast.js";

/** ปุ่มสลับจุดทำงานของโมเดลหลัก (เตือนไว / ระมัดระวัง) — state ระดับแอป การ์ดบน dashboard กับตารางในหน้า Model
 *  จึงพูดถึงจุดเดียวกันเสมอ · ใช้ .layer-pill แบบเดียวกับ ForecastModelPicker */
export default function ClassModeToggle() {
  const { classSummary, classMode, selectClassMode } = useApp();
  const modes = classSummary?.modes ?? Object.keys(CLASS_MODES);

  return (
    <div className="layer-pills class-mode" role="radiogroup" aria-label="จุดทำงานของโมเดลหลัก">
      {modes.map((mode) => (
        <button
          key={mode}
          type="button"
          role="radio"
          aria-checked={mode === classMode}
          className={`layer-pill${mode === classMode ? " is-active" : ""}`}
          title={CLASS_MODES[mode]?.hint}
          onClick={() => selectClassMode(mode)}
        >
          {CLASS_MODES[mode]?.label ?? mode}
        </button>
      ))}
    </div>
  );
}

/** ค่าคงที่ของคำพยากรณ์ระดับคลาสจากโมเดลหลัก (LSTM + V3) — ใช้ร่วมกันระหว่างการ์ดบน dashboard และหน้า Model
 *
 *  ระดับมาจาก backend เป็นสตริง "<M" | "M" | "X" (sunseg.inference.class_forecast.LEVELS) — `key` คือส่วนท้าย
 *  ของชื่อคลาส CSS (`.class-level--x`) เพราะ "<" ใช้ในชื่อคลาสไม่ได้ */
export const CLASS_LEVELS = {
  "<M": { key: "quiet", short: "<M", text: "ต่ำกว่า M", hint: "ไม่มีระดับไหนเตือน — ไม่มี flare หรือมีแค่ B/C" },
  M: { key: "m", short: "M", text: "ระดับ M", hint: "จะเกิด flare ≥ M1.0 แต่ไม่ถึง X" },
  X: { key: "x", short: "X", text: "ระดับ X", hint: "จะเกิด flare ≥ X1.0" },
};

export const levelMeta = (level) => CLASS_LEVELS[level] ?? CLASS_LEVELS["<M"];

/** จุดทำงานทั้งสอง — ต่างกันแค่ระดับ X (ดู artifacts/class_forecast/report.md) */
export const CLASS_MODES = {
  sensitive: {
    label: "เตือนไว",
    hint: "ระดับ X ใช้ threshold ที่ให้ TSS สูงสุด — จับ X ได้เกือบหมด แต่ flare ระดับ M มักถูกเรียกว่า X",
  },
  strict: {
    label: "ระมัดระวัง",
    hint: "ระดับ X ใช้ threshold ที่เลือกบน validation ให้ทายระดับถูกบ่อยที่สุด — แทบไม่เตือน X",
  },
};

export const SPLIT_TEXT = { train: "train", val: "validation", test: "test" };

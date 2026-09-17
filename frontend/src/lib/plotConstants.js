import { THEME } from "./theme.js";

/** เส้นกริด/เส้นฐานของแกน — ค่าตั้งต้นที่ทุกกราฟใช้ร่วมกัน */
const AXIS_BASE = { gridcolor: THEME.gridline, zerolinecolor: THEME.baseline };

/** คืน layout ใหม่ทุกครั้งที่เรียก — Plotly เขียนค่าที่คำนวณเอง (autorange, range, type, …)
 *  กลับลงใน object แกนที่รับไป ถ้าทุกกราฟชี้ก้อนเดียวกัน ค่าที่กราฟก่อนหน้าทิ้งไว้จะรั่ว
 *  ไปทับกราฟถัดไป จึงต้อง spread ก้อนใหม่ทุกครั้งที่ใช้ (เดิมทำด้วย getter ใน object เดียว) */
export function basePlotLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: THEME.textSecondary, family: THEME.fontSans, size: 11 },
    margin: { l: 52, r: 18, t: 12, b: 40 },
    xaxis: { ...AXIS_BASE },
    yaxis: { ...AXIS_BASE },
    showlegend: false,
    hoverlabel: {
      bgcolor: THEME.surface1,
      bordercolor: THEME.border,
      font: { family: THEME.fontSans },
    },
  };
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/** ระดับฟลักซ์พื้นหลังของดวงอาทิตย์ที่สงบ — ใช้เป็นฐานของ light curve */
export const BACKGROUND_FLUX = 1e-8;

/** เส้นแบ่งคลาส GOES — ทำให้อ่านแกน log ได้โดยไม่ต้องแปลงเลขในใจ */
export const GOES_LEVELS = [
  { flux: 1e-7, label: "B" }, { flux: 1e-6, label: "C" },
  { flux: 1e-5, label: "M" }, { flux: 1e-4, label: "X" },
];

/** จำนวนเหตุการณ์ที่มากกว่านี้ถือว่า "หนาแน่น" — ช่วงเวลาที่เลือกยาวหลายปีจะมีจุด
 *  peak หลายพันจุดมาทับกันจนกลบเส้นฟลักซ์ ดูเหมือนกลุ่มจุดกระจัดกระจายแทนที่จะเป็น
 *  เส้นต่อเนื่องแบบที่ spaceweather.gov แสดง โหมดหนาแน่นจึงย่อ/กรองจุดลงและเน้น
 *  เส้นให้เด่นขึ้นแทน (ดูการใช้งานใน Dashboard/XrayCard.jsx) */
export const DENSE_EVENT_THRESHOLD = 200;

export const SEQ_INDEX = { B: 0, C: 1, M: 2, X: 3 };

/** ศัพท์คู่ของ confusion matrix — สถิติ (TP/FP/FN/TN) + วงการ space weather (hit/…) */
export const CM_CELL_META = {
  tp: { stat: "TP", term: "hit", role: "hit" },
  fp: { stat: "FP", term: "false alarm", role: "false-alarm" },
  fn: { stat: "FN", term: "miss", role: "miss" },
  tn: { stat: "TN", term: "correct negative", role: "correct-negative" },
};

/** ขอบล่างจริงของแกน log — ฟลักซ์เงียบสุด (quiet sun) วัดได้ต่ำถึง ~1e-9 W/m² */
export const XRAY_Y_FLOOR = 1e-9;

/** มาตราพายุรังสีสุริยะของ NOAA — วัดที่ช่อง >10 MeV หน่วย pfu */
export const S_SCALE_LEVELS = [
  { pfu: 1e1, label: "S1" }, { pfu: 1e2, label: "S2" }, { pfu: 1e3, label: "S3" },
  { pfu: 1e4, label: "S4" }, { pfu: 1e5, label: "S5" },
];

/** ช่วงแกน y เริ่มต้น (log10 pfu) — ตรึงไว้เพื่อให้เทียบรูปร่างข้าม flare ได้ทันที */
export const PROTON_Y_RANGE = [-2, 5];

export const DISK_CLASSES = ["B", "C", "M", "X"];

/** ลำดับแถวจากบนลงล่าง = จากรากสนามแม่เหล็กที่โฟโตสเฟียร์ ไล่ขึ้นชั้นบรรยากาศ */
export const EXTRACTION_ROWS = [
  { key: "b_peak", label: "HMI |B|", region: "สนามแม่เหล็กที่โฟโตสเฟียร์", scale: "linear" },
  { key: "171", label: "171 Å", region: "โคโรนา", scale: "log" },
  { key: "304", label: "304 Å", region: "โครโมสเฟียร์", scale: "log" },
  { key: "1600", label: "1600 Å", region: "โฟโตสเฟียร์", scale: "log" },
];

export const EXTRACTION_DOMAINS = [
  [0.775, 0.975],
  [0.530, 0.730],
  [0.285, 0.485],
  [0.040, 0.240],
];

/** สีของแต่ละ AR — ใช้ชุดเดียวกันทั้งสามแถว เพื่อให้ตาไล่ AR ดวงเดิมข้ามชั้นได้
 *  ใช้ THEME.sun (ส้มดวงอาทิตย์) ไม่ใช่ THEME.brand — brand ของธีมใหม่เป็นน้ำเงินแทบเท่ากับ
 *  series1 สองเส้นจะแยกกันไม่ออก */
export const AR_COLORS = [
  THEME.series1, THEME.seq[3], THEME.seq[1],
  THEME.sun, THEME.pflux[2], THEME.statusCritical,
];

/** ครึ่งความกว้างของช่วงที่ปุ่ม "กระโดด" ตั้งให้ (วัน)
 *
 *  ต้องกว้างพอให้ได้ **อย่างน้อยสองเฟรมที่ติดกัน** ไม่ใช่แค่เฟรมเดียว — tracker ต้องเห็น
 *  AR ดวงเดิมติดกัน 2 เฟรมจึงยืนยันเป็น track (n_confirm) ถ้ากระโดดไปแล้วได้เฟรมเดียว
 *  กราฟก็ยังว่างอยู่ดี ซึ่งทำให้ปุ่มดูเหมือนกดไม่ติด
 *
 *  วัดรอบเฟรม 2012-06-24 บนคลังหลักที่เก็บภาพทุก 7 วัน:
 *  ±7 วัน → 2 เฟรม 0 track · ±14 → 3 เฟรม 3 track · **±21 → 5 เฟรม 6 track** · ±30 → 7 เฟรม 6 track
 *  ±21 จึงเป็นจุดที่ได้ track เต็มจำนวนโดยไม่คำนวณเกินจำเป็น */
export const SNAP_DAYS = 21;

/** รูปแบบจุดของแต่ละคลาสบนแผนที่ตำแหน่ง flare (หน้า HUB) — ชุดเดียวกับ PositionFlare
 *
 *  4 รายการนี้ไม่ใช่ตัวอักษร GOES ตรง ๆ: "M5" คือ M ที่ mantissa ≥ 5 (ใกล้ X) แยกออกมาให้กรอง/
 *  เห็นได้ ใช้ classIndex() แปลง goes_class เป็น index เสมอ
 *  r = รัศมีจุดเป็นพิกเซล — ขยายจากต้นฉบับเล็กน้อยเพราะแคตตาล็อกนี้มีจุดน้อยกว่า (~3k เทียบ 31k) */
export const CLASS_STYLE = [
  { key: "C", color: "#3b82f6", r: 2.2, alpha: 0.6 },
  { key: "M", color: "#f59e0b", r: 3.1, alpha: 0.78 },
  { key: "M5", color: "#f97316", r: 3.9, alpha: 0.88 },
  { key: "X", color: "#ef4444", r: 4.9, alpha: 0.96 },
];

export function classIndex(goesClass) {
  const letter = goesClass?.[0]?.toUpperCase();
  if (letter === "X") return 3;
  if (letter === "M") return Number.parseFloat(goesClass.slice(1)) >= 5 ? 2 : 1;
  return 0;
}

/** จุดเริ่มของ Solar Cycle (ตามประกาศของ SIDC/NOAA) — ใช้แบ่งยุคในตัวกรอง */
const CYCLE_STARTS = [
  { n: 23, start: Date.UTC(1996, 7, 1) },
  { n: 24, start: Date.UTC(2008, 11, 1) },
  { n: 25, start: Date.UTC(2019, 11, 1) },
];

export function cycleOf(ms) {
  let cycle = 22;
  for (const c of CYCLE_STARTS) if (ms >= c.start) cycle = c.n;
  return cycle;
}

/** เวลาใน API เป็น ISO ที่ไม่มีโซน (หมายถึง UTC) — Date.parse ตรง ๆ จะตีเป็นเวลาท้องถิ่น */
export function isoMs(iso) {
  return Date.parse(/(Z|[+-]\d\d:?\d\d)$/.test(iso) ? iso : `${iso}Z`);
}

/** |lon| ตั้งแต่ค่านี้ขึ้นไปถือว่าใกล้ขอบจาน — ตำแหน่งที่ฉายแล้วเบียดกันและคลาดเคลื่อนสูง */
export const LIMB_LON = 70;

export function durationLabel(startIso, endIso) {
  if (!startIso || !endIso) return "—";
  const minutes = Math.round((isoMs(endIso) - isoMs(startIso)) / 60000);
  if (!Number.isFinite(minutes) || minutes < 0) return "—";
  if (minutes < 60) return `${minutes} นาที`;
  return `${Math.floor(minutes / 60)} ชม. ${minutes % 60} นาที`;
}

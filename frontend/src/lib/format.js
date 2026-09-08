import { THEME } from "./theme.js";
import { SEQ_INDEX } from "./plotConstants.js";

export function formatDateTime(iso) {
  return iso.replace("T", " ").slice(0, 16);
}

/** จัดรูปตัวเลขที่มีช่วงค่ากว้างมาก (SHARP parameters ต่างกันหลาย order of magnitude) */
export function formatValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 1e5 || magnitude < 1e-3) return value.toExponential(2);
  if (magnitude >= 100) return value.toFixed(1);
  return value.toFixed(3);
}

export const RISK_LABEL = {
  low: "ต่ำ", moderate: "ปานกลาง", elevated: "สูงกว่าเกณฑ์", high: "สูงมาก",
};

/** ขนาดจุดตามลอการิทึมของฟลักซ์ — flare แรงจึงเด่นออกมาชัดเจน (ขั้นต่ำ 8px) */
export function markerSize(flux) {
  return 8 + 2.2 * (Math.log10(flux) + 7);
}

export function seqColor(goesClass) {
  return THEME.seq[SEQ_INDEX[goesClass[0]] ?? 0];
}

/** ฟลักซ์กินช่วงค่าหลายหลัก — toPrecision อย่างเดียวจะได้ "1.5e+3" ซึ่งอ่านยากกว่า */
export function formatPfu(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return Math.round(value).toLocaleString("en-US");
  if (value >= 10) return value.toFixed(1);
  return value.toPrecision(2);
}

/** เวลาในข้อมูลเป็น ISO ที่ไม่มีโซน (หมายถึง UTC) — เลื่อนเป็นชั่วโมงแล้วคืนรูปแบบเดิม
 *  ให้ตรงกับค่าที่ Plotly ใช้บนแกน x เป๊ะๆ ไม่ให้เส้นกริดเหลื่อมตามโซนของเครื่องผู้ใช้ */
export function shiftIsoHours(iso, hours) {
  return new Date(Date.parse(`${iso}Z`) + hours * 3600000).toISOString().slice(0, 19);
}

/** พิกัดแบบที่ NOAA ใช้ในรายงาน (S12W40) — อ่านคู่กับ catalogue ต้นทางได้ตรงๆ */
export function formatHeliographic(lat, lon) {
  const pad = (value) => String(Math.round(Math.abs(value))).padStart(2, "0");
  return `${lat >= 0 ? "N" : "S"}${pad(lat)}${lon >= 0 ? "W" : "E"}${pad(lon)}`;
}

/** แปลงชื่อเฟรม "YYYYMMDD_HHMMSS" เป็น Date */
export function frameTime(stamp) {
  return new Date(Date.UTC(
    +stamp.slice(0, 4), +stamp.slice(4, 6) - 1, +stamp.slice(6, 8),
    +stamp.slice(9, 11), +stamp.slice(11, 13),
  ));
}

export function frameLabel(stamp) {
  return `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)} ${stamp.slice(9, 11)}:${stamp.slice(11, 13)}`;
}

/** เวลาของเฟรม -> ISO ที่ Plotly ใช้ในแกน x — ใช้วาดเส้นแนวตั้งบนกราฟความเข้มแสงบอกว่า
 *  การ์ดภาพซ้ายกำลังแสดงเฟรมไหน (ดู ExtractionCard.jsx `frameMarkerShapes`) ผูกสองการ์ด
 *  เข้าด้วยกันด้วยตา */
export function frameStampToIso(stamp) {
  return `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}`
    + `T${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}`;
}

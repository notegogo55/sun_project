import { PROTON_Y_RANGE } from "./plotConstants.js";
import { shiftIsoHours } from "./format.js";

/** หาช่วงแกน y (log10 pfu) ที่ครอบคลุมค่าจริงทั้งหมด แต่ไม่แคบกว่า PROTON_Y_RANGE — ยึด
 *  กรอบมาตรฐานไว้แล้วขยายเฉพาะด้านที่ข้อมูลจริงล้นออกไป ใช้ร่วมกันทั้งกราฟใหญ่ (ProtonCard,
 *  รวมทุกช่อง) และกราฟย่อใน hero (ช่องเดียว) เพื่อให้สเกลนิ่งเทียบรูปร่างข้าม flare ได้ทันที */
export function protonYRange(values) {
  let [low, high] = PROTON_Y_RANGE;
  for (const value of values) {
    if (value === null || value <= 0) continue;
    const exponent = Math.log10(value);
    low = Math.min(low, Math.floor(exponent));
    high = Math.max(high, Math.ceil(exponent));
  }
  return [low, high];
}

/** ช่วงแกน x — ยืดเลยหน้าต่างข้อมูลไปข้างละ 1.5 ชม. กันป้ายแกน/ป้าย S-scale ที่ขอบซ้าย-ขวา
 *  ชนกับ tick สุดท้าย (กริด 5 นาทีปัดลง จุดสุดท้ายจึงมาก่อนขอบหน้าต่างเล็กน้อย) */
export function protonXRange(peakTime, beforeHours, afterHours) {
  return [
    shiftIsoHours(peakTime, -beforeHours - 1.5),
    shiftIsoHours(peakTime, afterHours + 1.5),
  ];
}

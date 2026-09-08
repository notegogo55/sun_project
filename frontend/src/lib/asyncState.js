/** รูปทรง state ของงาน async แบบเดียวที่ใช้ร่วมกันทั้งแอป (risk, goes, extraction, proton,
 *  confusion matrix, …) — ก่อนหน้านี้แต่ละจุดคิดรูปแบบของตัวเองจนต้องเช็คเงื่อนไขประหลาด
 *  ตอนอ่าน (เช่น `samples === "loading"`) ตัวช่วยพวกนี้บังคับให้ทุกที่มีหน้าตาเดียวกัน:
 *  `{ status: "idle"|"loading"|"ready"|"error", data, error, ...extra }`
 *
 *  `loadingAsync` ตั้งใจ**คงข้อมูลเก่าไว้**ระหว่างโหลดรอบใหม่ (ไม่ล้าง data เป็น null) —
 *  จุดที่เคยพังเพราะลืมข้อนี้คือกราฟ GOES/โปรตอนกะพริบเป็นค่าว่างทุกครั้งที่โหลดซ้ำ ทั้งที่
 *  ควรค้างรูปเดิมไว้จนกว่าของใหม่จะมาแทน (ดู state/AppProvider.jsx, Dashboard/ProtonCard.jsx)
 *  ส่วน `errorAsync` ล้าง data ทิ้งตั้งใจ — ของเก่าอาจไม่ตรงกับพารามิเตอร์ใหม่ที่พังไปแล้ว
 */

export function idleAsync(extra = {}) {
  return { status: "idle", data: null, error: null, ...extra };
}

export function loadingAsync(prev, extra = {}) {
  return { ...prev, status: "loading", error: null, ...extra };
}

export function readyAsync(data, extra = {}) {
  return { status: "ready", data, error: null, ...extra };
}

export function errorAsync(message, extra = {}) {
  return { status: "error", data: null, error: message, ...extra };
}

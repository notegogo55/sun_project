/** ฉาย heliographic (lat, lon) ลงจานกลม 2 มิติ — orthographic, B0 = 0
 *
 *  สูตรเดียวกับ PositionFlare (D:\position_flare, src/lib/proj.ts) และแผนที่ Plotly เดิมของ
 *  โปรเจคนี้ ละเลยการเอียงแกนดวงอาทิตย์ (B0 จริง ±7.25°) คืนหน่วยรัศมีดวงอาทิตย์:
 *  x บวก = ตะวันตก, y บวก = เหนือ */
const RAD = Math.PI / 180;

export function project(latDeg, lonDeg) {
  const la = latDeg * RAD;
  const lo = lonDeg * RAD;
  return [Math.cos(la) * Math.sin(lo), Math.sin(la)];
}

/** ดัชนีเชิงพื้นที่แบบ uniform grid หาจุดที่ใกล้เคอร์เซอร์ที่สุด — ไล่ดูทีละจุดทุกครั้ง
 *  ที่เมาส์ขยับทำให้หน่วงเมื่อจุดมีหลายพัน จึงหั่นพื้นที่เป็นช่องแล้วดูเฉพาะช่องรอบเคอร์เซอร์
 *
 *  @param xs, ys  พิกัดหน้าจอของจุดที่ k
 *  @param ids     ค่าที่คืนเมื่อจุดที่ k เป็นคำตอบ (id ของ flare) */
export function buildPointIndex(xs, ys, ids, w, h, cell = 16) {
  const cols = Math.max(1, Math.ceil(w / cell));
  const rows = Math.max(1, Math.ceil(h / cell));
  const buckets = Array.from({ length: cols * rows }, () => []);

  for (let k = 0; k < ids.length; k++) {
    const cx = Math.floor(xs[k] / cell);
    const cy = Math.floor(ys[k] / cell);
    if (cx < 0 || cy < 0 || cx >= cols || cy >= rows) continue;
    buckets[cy * cols + cx].push(k);
  }

  return {
    /** id ของจุดที่ใกล้ (px, py) ที่สุดภายใน maxDist พิกเซล, -1 = ไม่เจอ */
    nearest(px, py, maxDist) {
      const cx = Math.floor(px / cell);
      const cy = Math.floor(py / cell);
      const span = Math.max(1, Math.ceil(maxDist / cell));
      let best = -1;
      let bestD = maxDist * maxDist;
      for (let gy = cy - span; gy <= cy + span; gy++) {
        if (gy < 0 || gy >= rows) continue;
        for (let gx = cx - span; gx <= cx + span; gx++) {
          if (gx < 0 || gx >= cols) continue;
          for (const k of buckets[gy * cols + gx]) {
            const dx = xs[k] - px;
            const dy = ys[k] - py;
            const d = dx * dx + dy * dy;
            if (d < bestD) {
              bestD = d;
              best = ids[k];
            }
          }
        }
      }
      return best;
    },
  };
}

import { useEffect, useMemo, useRef } from "react";
import { buildPointIndex, project } from "../../lib/proj.js";
import { CLASS_STYLE } from "../../lib/flareClass.js";
import { setupCanvas, useSize } from "../../hooks/useSize.js";

/** แผนที่ตำแหน่ง flare บนจานสุริยะ — แนวทางเดียวกับ SolarMap ของ PositionFlare
 *
 *  Canvas 2D สองชั้น: ชั้นล่างคือจาน + กริด + จุดทั้งหมด (วาดใหม่เฉพาะตอนตัวกรองหรือขนาดเปลี่ยน)
 *  ชั้นบนคือวงไฮไลต์เท่านั้น — เมาส์ขยับจึงไม่ต้องวาดจุดหลายพันใหม่ทุกเฟรม ส่วนการชี้ใช้
 *  uniform grid index หาจุดใกล้สุด ไม่ไล่ทีละจุด (Plotly/SVG จุดละ element หน่วงเกินที่ขนาดนี้) */

/** สัดส่วนของด้านสั้นที่ใช้เป็นเส้นผ่านศูนย์กลางจาน — เหลือขอบไว้ให้ป้าย N/S/E/W และจุดที่ลิมบ์ */
const DISK_FILL = 0.86;
const HOVER_RADIUS_PX = 14;
const MONO = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace";

function drawDisk(ctx, cx, cy, R) {
  ctx.save();

  // แสงโคโรนารอบขอบ — ให้จานลอยอยู่ในแสงแทนที่จะเป็นวงเส้นบนพื้นดำเปล่า
  const corona = ctx.createRadialGradient(cx, cy, R * 0.92, cx, cy, R * 1.22);
  corona.addColorStop(0, "rgba(255, 150, 50, 0.14)");
  corona.addColorStop(1, "rgba(255, 120, 30, 0)");
  ctx.fillStyle = corona;
  ctx.beginPath();
  ctx.arc(cx, cy, R * 1.22, 0, Math.PI * 2);
  ctx.fill();

  const disk = ctx.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.05, cx, cy, R);
  disk.addColorStop(0, "rgba(255, 236, 200, 0.15)");
  disk.addColorStop(1, "rgba(255, 170, 80, 0.05)");
  ctx.fillStyle = disk;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fill();

  // กริด heliographic ทุก 30° ฉายด้วยสูตรเดียวกับจุด flare เพื่อให้อ่านตำแหน่งตรงกัน
  ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
  ctx.lineWidth = 1;
  for (let lat = -60; lat <= 60; lat += 30) {
    ctx.beginPath();
    for (let lon = -90; lon <= 90; lon += 2) {
      const [x, y] = project(lat, lon);
      if (lon === -90) ctx.moveTo(cx + x * R, cy - y * R);
      else ctx.lineTo(cx + x * R, cy - y * R);
    }
    ctx.stroke();
  }
  for (let lon = -60; lon <= 60; lon += 30) {
    ctx.beginPath();
    for (let lat = -90; lat <= 90; lat += 2) {
      const [x, y] = project(lat, lon);
      if (lat === -90) ctx.moveTo(cx + x * R, cy - y * R);
      else ctx.lineTo(cx + x * R, cy - y * R);
    }
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
  ctx.font = `13px ${MONO}`;
  ctx.textAlign = "center";
  ctx.fillText("N", cx, cy - R - 9);
  ctx.fillText("S", cx, cy + R + 19);
  ctx.textAlign = "right";
  ctx.fillText("E", cx - R - 8, cy + 4);
  ctx.textAlign = "left";
  ctx.fillText("W", cx + R + 8, cy + 4);
  ctx.restore();
}

export default function FlarePositionMap({ points, hover, selected, onHover, onSelect, message }) {
  const [wrapRef, size] = useSize();
  const baseRef = useRef(null);
  const fxRef = useRef(null);

  const geom = useMemo(
    () => ({ cx: size.w / 2, cy: size.h / 2, R: (Math.min(size.w, size.h) * DISK_FILL) / 2 }),
    [size.w, size.h],
  );

  /** พิกัดหน้าจอของทุกจุด คำนวณครั้งเดียวแล้วใช้ซ้ำทั้งการวาดและการชี้ */
  const screen = useMemo(() => {
    const n = points.length;
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const ids = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const { ev, id } = points[k];
      const [x, y] = project(ev.lat, ev.lon);
      xs[k] = geom.cx + x * geom.R;
      ys[k] = geom.cy - y * geom.R;
      ids[k] = id;
    }
    return { xs, ys, ids };
  }, [points, geom]);

  const index = useMemo(
    () => (size.w > 0 && points.length
      ? buildPointIndex(screen.xs, screen.ys, screen.ids, size.w, size.h)
      : null),
    [screen, points.length, size.w, size.h],
  );

  const posOf = useMemo(() => {
    const map = new Map();
    for (let k = 0; k < points.length; k++) map.set(points[k].id, [screen.xs[k], screen.ys[k], points[k].ci]);
    return map;
  }, [points, screen]);

  // ชั้นล่าง: จาน + จุด (points เรียง C → X มาแล้ว X ที่หายากที่สุดจึงวาดทับบนสุด)
  useEffect(() => {
    const canvas = baseRef.current;
    if (!canvas || size.w === 0) return;
    const ctx = setupCanvas(canvas, size.w, size.h);
    if (!ctx) return;
    drawDisk(ctx, geom.cx, geom.cy, geom.R);
    for (let k = 0; k < points.length; k++) {
      const style = CLASS_STYLE[points[k].ci];
      ctx.globalAlpha = style.alpha;
      ctx.fillStyle = style.color;
      ctx.beginPath();
      ctx.arc(screen.xs[k], screen.ys[k], style.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [points, screen, geom, size]);

  // ชั้นบน: วงไฮไลต์ของจุดที่ตรึง (ขาวทึบ) และที่ชี้อยู่ (ขาวจาง)
  useEffect(() => {
    const canvas = fxRef.current;
    if (!canvas || size.w === 0) return;
    const ctx = setupCanvas(canvas, size.w, size.h);
    if (!ctx) return;
    const ring = (id, color, width, extra) => {
      const p = posOf.get(id);
      if (!p) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.arc(p[0], p[1], CLASS_STYLE[p[2]].r + extra, 0, Math.PI * 2);
      ctx.stroke();
    };
    if (selected >= 0) ring(selected, "#f8fafc", 2, 7);
    if (hover >= 0 && hover !== selected) ring(hover, "rgba(248, 250, 252, 0.7)", 1.5, 5);
  }, [hover, selected, posOf, size]);

  const pick = (event) => {
    if (!index) return -1;
    const rect = event.currentTarget.getBoundingClientRect();
    return index.nearest(event.clientX - rect.left, event.clientY - rect.top, HOVER_RADIUS_PX);
  };

  return (
    <div
      ref={wrapRef}
      className="fp-map"
      role="img"
      aria-label="แผนที่ตำแหน่ง flare บนจานสุริยะ — ชี้ที่จุดเพื่อดูรายละเอียด คลิกเพื่อตรึง"
      onMouseMove={(e) => {
        const id = pick(e);
        if (id !== hover) onHover(id);
      }}
      onMouseLeave={() => onHover(-1)}
      onClick={(e) => {
        const id = pick(e);
        if (id >= 0) onSelect(id);
      }}
    >
      <canvas ref={baseRef} style={{ width: size.w, height: size.h }} />
      <canvas ref={fxRef} style={{ width: size.w, height: size.h }} />
      {message && <p className="fp-map__msg">{message}</p>}
    </div>
  );
}

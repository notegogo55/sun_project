import { useLayoutEffect, useRef, useState } from "react";

/** ขนาดจริงของ element ตาม layout — canvas ต้องรู้เพื่อตั้ง backing store ให้คมบนจอ HiDPI
 *
 *  วัดตรง ๆ ครั้งแรกด้วย getBoundingClientRect ก่อนแล้วค่อยพึ่ง ResizeObserver ต่อ — ถ้ารอแต่
 *  callback ของ observer อย่างเดียว แผนที่จะว่างทุกครั้งที่ browser หน่วงการส่ง callback
 *  (แท็บถูกพักการเรนเดอร์ตอนโหลด) — แนวทางเดียวกับ PositionFlare */
export function useSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const apply = (w, h) => setSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
    const rect = el.getBoundingClientRect();
    apply(rect.width, rect.height);
    const observer = new ResizeObserver(([entry]) => apply(entry.contentRect.width, entry.contentRect.height));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}

/** ตั้ง backing store ตาม devicePixelRatio แล้วคืน context ที่ scale และล้างไว้แล้ว */
export function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

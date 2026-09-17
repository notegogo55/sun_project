import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_STYLE, durationLabel } from "../../lib/flareClass.js";
import { formatDateTime, formatHeliographic } from "../../lib/format.js";

/** ตาราง flare ทุกดวงที่ผ่านตัวกรอง — รวมดวงที่ไม่ทราบตำแหน่งซึ่งคลิกจากแผนที่ไม่ได้
 *  (แนวเดียวกับ FlareList ของ PositionFlare) แถวนับหมื่นจึงวาดแบบ virtual: สร้าง DOM
 *  เฉพาะแถวที่มองเห็น + เผื่อบน/ล่าง แถวสูงคงที่เพื่อคำนวณตำแหน่งจาก scrollTop ได้ตรง ๆ */
const ROW_H = 30;
const VIEW_H = 360;
const OVERSCAN = 8;

const COLUMNS = [
  { key: "t", label: "เวลาพีค (UTC)" },
  { key: "flux", label: "คลาส" },
  { key: "pos", label: "ตำแหน่ง" },
  { key: "src", label: "ที่มา" },
  { key: "ar", label: "NOAA AR" },
  { key: "harp", label: "HARP" },
  { key: "dur", label: "ระยะเวลา" },
];

const SORTERS = {
  t: (r) => r.t,
  flux: (r) => r.ev.peak_flux,
  ar: (r) => r.ev.noaa_ar ?? -Infinity,
  harp: (r) => r.ev.harpnum ?? -Infinity,
};

export default function FlareTable({ rows, hover, selected, onHover, onSelect, onPick }) {
  const [sort, setSort] = useState({ key: "t", dir: "desc" });
  const [scrollTop, setScrollTop] = useState(0);
  const bodyRef = useRef(null);

  const sorted = useMemo(() => {
    const get = SORTERS[sort.key];
    const sign = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const x = get(a);
      const y = get(b);
      return x === y ? a.t - b.t : (x < y ? -sign : sign);
    });
  }, [rows, sort]);

  const positionOf = useMemo(() => new Map(sorted.map((r, k) => [r.id, k])), [sorted]);

  // เลือกจากที่อื่น (แผนที่ / คีย์บอร์ด) แล้วเลื่อนแถวนั้นเข้ามาในกรอบ — เลื่อนเฉพาะในกล่องตาราง
  // ไม่ใช่ scrollIntoView ที่จะลากทั้งหน้าลงมาตอนคลิกจุดบนแผนที่
  useEffect(() => {
    const body = bodyRef.current;
    const k = positionOf.get(selected);
    if (!body || k === undefined) return;
    const top = k * ROW_H;
    if (top < body.scrollTop) body.scrollTop = top;
    else if (top + ROW_H > body.scrollTop + body.clientHeight) body.scrollTop = top + ROW_H - body.clientHeight;
  }, [selected, positionOf]);

  function toggleSort(key) {
    if (!SORTERS[key]) return;
    setSort((current) => (current.key === key
      ? { key, dir: current.dir === "desc" ? "asc" : "desc" }
      : { key, dir: key === "ar" || key === "harp" ? "asc" : "desc" }));
  }

  function onKeyDown(event) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    if (!sorted.length) return;
    const k = positionOf.get(selected);
    const step = event.key === "ArrowDown" ? 1 : -1;
    const next = k === undefined ? 0 : Math.max(0, Math.min(sorted.length - 1, k + step));
    onPick(sorted[next].id);
  }

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const last = Math.min(sorted.length, Math.ceil((scrollTop + VIEW_H) / ROW_H) + OVERSCAN);
  const visible = sorted.slice(first, last);

  return (
    <div className="fp-table">
      <div className="fp-table__scroll">
        <div className="fp-thead" role="row">
          {COLUMNS.map((col) => {
            const sortable = Boolean(SORTERS[col.key]);
            const on = sort.key === col.key;
            return sortable ? (
              <button
                key={col.key}
                type="button"
                role="columnheader"
                className={`fp-th${on ? " is-on" : ""}`}
                aria-sort={on ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                onClick={() => toggleSort(col.key)}
              >
                {col.label}{on && <i aria-hidden="true">{sort.dir === "asc" ? " ↑" : " ↓"}</i>}
              </button>
            ) : (
              <span key={col.key} role="columnheader" className="fp-th">{col.label}</span>
            );
          })}
        </div>

        <div
          ref={bodyRef}
          className="fp-tbody"
          style={{ height: VIEW_H }}
          tabIndex={0}
          role="rowgroup"
          aria-label="รายการ flare — ใช้ลูกศรขึ้น/ลงเพื่อเลื่อนแถวที่เลือก"
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
          onKeyDown={onKeyDown}
          onMouseLeave={() => onHover(-1)}
        >
          <div style={{ height: sorted.length * ROW_H, position: "relative" }}>
            {visible.map((r, offset) => {
              const k = first + offset;
              const cls = [
                "fp-tr",
                k % 2 ? "is-alt" : "",
                r.id === hover ? "is-hover" : "",
                r.id === selected ? "is-sel" : "",
              ].filter(Boolean).join(" ");
              return (
                <div
                  key={r.id}
                  role="row"
                  className={cls}
                  style={{ top: k * ROW_H, height: ROW_H }}
                  onMouseEnter={() => onHover(r.id)}
                  onClick={() => onSelect(r.id)}
                >
                  <span>{formatDateTime(r.ev.peak_time)}</span>
                  <span className="fp-cls">
                    <i className="fp-dot" style={{ background: CLASS_STYLE[r.ci].color }} />
                    {r.ev.goes_class}
                  </span>
                  <span className={r.located ? undefined : "fp-dim"}>
                    {r.located ? formatHeliographic(r.ev.lat, r.ev.lon) : "ไม่ทราบ"}
                  </span>
                  <span className="fp-dim" title={r.matched ? undefined : "หาคู่ใน PositionFlare ไม่เจอ"}>
                    {r.ev.pos_source ?? (r.matched ? "—" : "ไม่พบคู่")}
                  </span>
                  <span>{r.ev.noaa_ar ?? "—"}</span>
                  <span>{r.ev.harpnum ?? "—"}</span>
                  <span className="fp-dim">{durationLabel(r.ev.start_time, r.ev.end_time)}</span>
                </div>
              );
            })}
          </div>
          {!sorted.length && <p className="fp-table__empty">ไม่มี flare ที่ผ่านตัวกรอง</p>}
        </div>
      </div>
    </div>
  );
}

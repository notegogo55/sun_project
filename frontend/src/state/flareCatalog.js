import { api } from "../lib/api.js";
import { classIndex, cycleOf } from "../lib/flareClass.js";

/** แคตตาล็อก flare ของแผนที่ตำแหน่งในหน้า HUB — flare ชุดเดียวกับที่โมเดลพยากรณ์ใช้ทำ label ทุกดวง
 *  (คลาสตามแคตตาล็อกของโมเดล) พร้อมพิกัดจาก PositionFlare ที่ backend จับคู่ไว้ล่วงหน้า
 *  (ดู backend/scripts/data/build_flare_positions.py) แยกจาก `goes` ใน AppProvider โดยเจตนา:
 *  `goes` ผูกกับช่วงเวลาที่เลือกใน dashboard ส่วนแผนที่นี้ต้องเห็นทุกดวงทุกปี
 *
 *  API ส่งแบบ columnar (ทุก list ยาวเท่ากัน) — แปลงเป็นแถวที่นี่ครั้งเดียว เก็บ promise ไว้ระดับ
 *  โมดูล กลับมาหน้า HUB ซ้ำไม่ต้องโหลดใหม่ */
const MINUTE = 60000;
let cached = null;

/** ISO แบบไม่มีโซน (UTC) — รูปแบบเดียวกับที่ endpoint อื่นของ backend ใช้ AppProvider จึงส่งต่อ
 *  flare เหล่านี้ไป /api/proton และเทียบ peak_time กับ flare จากที่อื่นได้ตรง ๆ */
const isoOf = (ms) => new Date(ms).toISOString().slice(0, 19);

function toCatalog(p) {
  const epoch = Date.parse(`${p.epoch}Z`);
  const rows = new Array(p.n);
  let lastLocatedYear = null;

  for (let id = 0; id < p.n; id++) {
    const t = epoch + p.t[id] * MINUTE;
    const lat = p.lat[id];
    const lon = p.lon[id];
    const located = lat != null && lon != null;
    const year = new Date(t).getUTCFullYear();
    if (located) lastLocatedYear = year;
    rows[id] = {
      id,
      t,
      ci: classIndex(p.goes_class[id]),
      cycle: p.cycle[id] ?? cycleOf(t),
      year,
      located,
      matched: p.match_dt_min[id] != null,
      ev: {
        peak_time: isoOf(t),
        start_time: p.start[id] == null ? null : isoOf(t + p.start[id] * MINUTE),
        end_time: p.end[id] == null ? null : isoOf(t + p.end[id] * MINUTE),
        goes_class: p.goes_class[id],
        peak_flux: p.peak_flux[id],
        noaa_ar: p.noaa_ar[id] ?? p.pf_active_region[id],
        harpnum: p.harpnum[id],
        n_harps: p.n_harps[id],
        lat,
        lon,
        pf_class: p.pf_goes_class[id],
        pos_source: p.pos_source[id],
        limb: p.limb[id],
        satellite: p.satellite[id],
        match_dt: p.match_dt_min[id],
      },
    };
  }

  return {
    rows,
    tMin: rows.length ? rows[0].t : NaN,
    tMax: rows.length ? rows[rows.length - 1].t : NaN,
    nLocated: p.n_located,
    nMatched: p.n_matched,
    tolerance: p.tolerance_min,
    builtAt: p.built_at,
    lastLocatedYear,
  };
}

export function loadFlareCatalog() {
  if (!cached) {
    const promise = api.flarePositions().then(toCatalog);
    // ล้มเหลวแล้วต้องลองใหม่ได้ตอนกลับมาหน้านี้ ไม่ใช่ค้าง promise ที่ reject ไว้ตลอด
    promise.catch(() => { if (cached === promise) cached = null; });
    cached = promise;
  }
  return cached;
}

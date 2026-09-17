import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../../state/AppContext.js";
import { loadFlareCatalog } from "../../state/flareCatalog.js";
import { idleAsync, loadingAsync, readyAsync, errorAsync } from "../../lib/asyncState.js";
import { CLASS_STYLE } from "../../lib/flareClass.js";
import { dashboardHref } from "../../lib/nav.js";
import { SectionHead } from "../ui/Headers.jsx";
import FlarePositionMap from "./FlarePositionMap.jsx";
import FlareDetail from "./FlareDetail.jsx";
import FlareTable from "./FlareTable.jsx";
import ProtonMiniChart from "./ProtonMiniChart.jsx";
import FlareXrayChart from "./FlareXrayChart.jsx";

const DAY = 86400000;
const int = (n) => n.toLocaleString("en-US");
const dateUTC = (ms) => new Date(ms).toISOString().slice(0, 10);

/** ห้ามปิดหมด — แผนที่ว่างเปล่าไม่ได้บอกอะไรและหาทางกดกลับยาก */
function toggleIn(set, key) {
  const next = new Set(set);
  if (next.has(key)) next.delete(key); else next.add(key);
  return next.size ? next : set;
}

/** หน้าแรก — แผนที่ตำแหน่ง flare ในแนวงาน PositionFlare (D:\position_flare):
 *  แถบตัวกรอง → แผนที่จานสุริยะ + แผงรายละเอียด → ตาราง flare ทุกดวง
 *
 *  flare ที่แสดงคือชุดเดียวกับแคตตาล็อกที่ LSTM ใช้ทำ label (คลาสตามโมเดล) ส่วนพิกัดมาจาก
 *  PositionFlare ซึ่ง backend จับคู่ด้วยเวลาพีคไว้ล่วงหน้า — ดู state/flareCatalog.js
 *
 *  ชี้ = ดูผ่าน ๆ, คลิก = ตรึง (ของที่ตรึงชนะเสมอจนกว่าจะคลิกซ้ำ) ทั้งแผนที่และตารางใช้ state
 *  hover/selected ชุดเดียวกัน จึงไฮไลต์ตามกันเอง */
export default function FlarePosition() {
  const {
    health, pinnedFlare, selectHarp, applyRange, previewFlareEvent, clearProtonPreview, selectFlareEvent,
  } = useApp();
  const navigate = useNavigate();

  const [catalog, setCatalog] = useState(idleAsync());
  const [cycles, setCycles] = useState(() => new Set());
  const [classes, setClasses] = useState(() => new Set([0, 1, 2, 3]));
  const [year, setYear] = useState("all");
  const [harpOnly, setHarpOnly] = useState(false);
  const [hover, setHover] = useState(-1);
  const [selected, setSelected] = useState(-1);

  useEffect(() => {
    let alive = true;
    setCatalog(loadingAsync(idleAsync()));
    loadFlareCatalog().then(
      (data) => {
        if (!alive) return;
        setCatalog(readyAsync(data));
        setCycles(new Set(data.rows.map((r) => r.cycle)));
      },
      (error) => { if (alive) setCatalog(errorAsync(error.message)); },
    );
    return () => { alive = false; };
  }, []);

  const rows = useMemo(() => catalog.data?.rows ?? [], [catalog.data]);
  const cycleList = useMemo(() => [...new Set(rows.map((r) => r.cycle))].sort((a, b) => a - b), [rows]);
  const yearList = useMemo(() => [...new Set(rows.map((r) => r.year))].sort((a, b) => a - b), [rows]);

  const matched = useMemo(() => rows.filter((r) => classes.has(r.ci)
    && cycles.has(r.cycle)
    && (year === "all" || r.year === Number(year))
    && (!harpOnly || r.ev.harpnum)), [rows, classes, cycles, year, harpOnly]);

  /** จุดบนแผนที่ เรียง C → M → M5 → X ให้ X ที่หายากและสำคัญที่สุดวาดทับบนสุด */
  const points = useMemo(() => {
    const located = matched.filter((r) => r.located);
    return [0, 1, 2, 3].flatMap((c) => located.filter((r) => r.ci === c));
  }, [matched]);

  const perClass = useMemo(() => {
    const all = [0, 0, 0, 0];
    const onMap = [0, 0, 0, 0];
    for (const r of matched) {
      all[r.ci] += 1;
      if (r.located) onMap[r.ci] += 1;
    }
    return { all, onMap };
  }, [matched]);

  function handleHover(id) {
    setHover(id);
    if (id >= 0) previewFlareEvent(rows[id].ev);
    else clearProtonPreview();
  }

  function handleSelect(id) {
    setSelected((current) => (current === id ? -1 : id));
    selectFlareEvent(rows[id].ev);
  }

  /** เลือกแบบไม่สลับ (ลูกศรในตาราง) — ถ้าใช้ handleSelect กดลูกศรกลับมาแถวเดิมจะปลดตรึงแทน */
  function handlePick(id) {
    if (id === selected) return;
    setSelected(id);
    selectFlareEvent(rows[id].ev);
  }

  function openInDashboard(row) {
    const start = dateUTC(row.t - 3 * DAY);
    const end = dateUTC(row.t + 3 * DAY);
    if (row.ev.harpnum) selectHarp(row.ev.harpnum, { syncRange: false });
    applyRange({ start, end });
    navigate(dashboardHref(row.ev.harpnum ? "risk" : "xray"));
  }

  const activeId = selected >= 0 ? selected : hover;
  const activeRow = activeId >= 0 ? rows[activeId] : null;
  const hidden = matched.length - points.length;
  const unpaired = useMemo(() => matched.filter((r) => !r.matched).length, [matched]);
  const data = catalog.data;

  let mapMessage = null;
  if (catalog.status === "loading" || catalog.status === "idle") mapMessage = "กำลังโหลดแคตตาล็อก flare…";
  else if (catalog.status === "error") mapMessage = catalog.error;
  else if (!points.length) mapMessage = "ไม่มี flare ที่ทราบตำแหน่งในตัวกรองนี้";

  return (
    <>
      <section className="fp" aria-labelledby="fp-title">
        <header className="fp-head">
          <h1 className="fp-title" id="fp-title">Flare<span>Position</span></h1>
          <div className="fp-meta">
            {data && (
              <>
                <span title="flare ระดับ C ขึ้นไปในแคตตาล็อกที่ LSTM ใช้ทำ label">
                  {int(data.rows.length)} flares ในแคตตาล็อกโมเดล · {dateUTC(data.tMin)} – {dateUTC(data.tMax)}
                </span>
                <span title={`จับคู่ด้วยเวลาพีค ±${data.tolerance} นาที`}>
                  {int(data.nLocated)} ดวงมีตำแหน่งจาก PositionFlare
                </span>
              </>
            )}
            <span className={`fp-live fp-live--${catalog.status === "ready" ? "ok" : catalog.status === "error" ? "off" : "pending"}`}>
              {catalog.status === "ready" ? "PositionFlare × sunseg" : catalog.status === "error" ? "โหลดไม่สำเร็จ" : "กำลังโหลด…"}
            </span>
          </div>
        </header>

        <div className="fp-controls">
          <div className="fp-group">
            <span className="fp-group__label">Solar Cycle</span>
            <div className="fp-chips">
              {cycleList.map((c) => (
                <button key={c} type="button" className="chip" aria-pressed={cycles.has(c)} onClick={() => setCycles((s) => toggleIn(s, c))}>
                  {c}
                </button>
              ))}
            </div>
          </div>
          <span className="fp-div" aria-hidden="true" />
          <div className="fp-group">
            <span className="fp-group__label" title="คลาสตามแคตตาล็อกของโมเดล — ตัวเดียวกับที่ใช้ตัดสิน label ≥ M1.0">คลาส (โมเดล)</span>
            <div className="fp-chips">
              {CLASS_STYLE.map((s, i) => (
                <button
                  key={s.key} type="button" className="chip" style={{ "--chip-color": s.color }}
                  aria-pressed={classes.has(i)} onClick={() => setClasses((set) => toggleIn(set, i))}
                >
                  <i className="chip__dot" aria-hidden="true" />{s.key}
                </button>
              ))}
            </div>
          </div>
          <span className="fp-div" aria-hidden="true" />
          <label className="fp-group">
            <span className="fp-group__label">ปี</span>
            <select value={year} onChange={(e) => setYear(e.target.value)}>
              <option value="all">ทุกปี</option>
              {yearList.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </label>
          <span className="fp-div" aria-hidden="true" />
          <div className="fp-group">
            <span className="fp-group__label">โมเดล</span>
            <div className="fp-chips">
              <button
                type="button" className="chip" aria-pressed={harpOnly}
                title="เฉพาะ flare ที่จับคู่กับ HARP ได้ — มีผลพยากรณ์ของ LSTM ให้เปิดดูต่อใน dashboard"
                onClick={() => setHarpOnly((v) => !v)}
              >
                เฉพาะที่มีคู่ HARP
              </button>
            </div>
          </div>
        </div>

        <div className="fp-stage">
          <div className="fp-map-wrap">
            <FlarePositionMap
              points={points}
              hover={hover}
              selected={selected}
              onHover={handleHover}
              onSelect={handleSelect}
              message={mapMessage}
            />
            <p className="fp-mapnote">
              แสดง {int(points.length)} จาก {int(matched.length)} ดวงที่ผ่านตัวกรอง
              {hidden > 0 && (
                <> · อีก {int(hidden)} ดวงไม่มีพิกัด (หาคู่ใน PositionFlare ไม่เจอ {int(unpaired)}) นับในสถิติแต่ไม่ขึ้นบนแผนที่</>
              )}
              {data && <> · ตำแหน่งจาก PositionFlare จับคู่ด้วยเวลาพีค ±{data.tolerance} นาที</>}
            </p>
          </div>

          <aside className="fp-side">
            <FlareDetail row={activeRow} pinned={selected >= 0 && activeId === selected} onOpen={openInDashboard} />
            <FlareXrayChart row={activeRow} />

            <div className="panel">
              <p className="panel__title">Flare classes</p>
              <p className="panel__sub">ผ่านตัวกรอง · บนแผนที่ / ทั้งหมด</p>
              <div className="class-bars">
                {CLASS_STYLE.map((s, i) => {
                  const max = Math.max(1, ...perClass.all);
                  return (
                    <div key={s.key} className="class-bar" style={{ "--bar-color": s.color }}>
                      <span className="class-bar__name">{s.key}</span>
                      <span className="class-bar__track">
                        <span className="class-bar__fill" style={{ width: `${(100 * perClass.all[i]) / max}%` }} />
                      </span>
                      <span className="class-bar__num">{int(perClass.onMap[i])} / {int(perClass.all[i])}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {health?.proton_flux && pinnedFlare && <ProtonMiniChart />}
          </aside>
        </div>
      </section>

      <section className="hub-section" aria-labelledby="fp-table-title">
        <div className="hub-section__inner">
          <SectionHead
            index="01"
            label="FLARE TABLE"
            title={<span id="fp-table-title">ทุก flare <em>รวมดวงที่ไม่ทราบตำแหน่ง</em></span>}
            lead="คลิกแถวเพื่อตรึง · ↑↓ เลื่อนแถวที่เลือก · คลิกหัวคอลัมน์เพื่อเรียงตามเวลา ความแรง AR หรือ HARP"
          />
          <FlareTable
            rows={matched}
            hover={hover}
            selected={selected}
            onHover={handleHover}
            onSelect={handleSelect}
            onPick={handlePick}
          />
        </div>
      </section>
    </>
  );
}

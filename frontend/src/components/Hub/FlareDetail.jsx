import { CLASS_STYLE, LIMB_LON, durationLabel } from "../../lib/flareClass.js";
import { formatDateTime, formatHeliographic } from "../../lib/format.js";

function Row({ k, children }) {
  return (
    <div className="fp-row">
      <span className="fp-row__k">{k}</span>
      <span className="fp-row__v">{children}</span>
    </div>
  );
}

/** รายละเอียดของ flare ที่ชี้/ตรึงอยู่ — คลาสหลักคือคลาสในแคตตาล็อกของโมเดล (ตัวที่ทำ label)
 *  ส่วนคลาส/ตำแหน่งจาก PositionFlare แสดงคู่กันพร้อมที่มา ให้เห็นว่าข้อมูลส่วนไหนมาจากไหน
 *  และลิงก์ต่อเข้า dashboard ได้ (HARP ที่จับคู่ไว้มีผลพยากรณ์ของ LSTM ให้ดูต่อ) */
export default function FlareDetail({ row, pinned, onOpen }) {
  if (!row) {
    return (
      <div className="fp-detail fp-detail--empty">
        <p>ชี้เมาส์ที่จุดบนแผนที่หรือแถวในตาราง · คลิกเพื่อตรึงรายละเอียดไว้</p>
      </div>
    );
  }

  const { ev } = row;
  const style = CLASS_STYLE[row.ci];
  const limb = row.located && (ev.limb ?? Math.abs(ev.lon) >= LIMB_LON);
  const classDiffers = ev.pf_class && ev.pf_class !== ev.goes_class;

  return (
    <div className="fp-detail">
      <h2 className="fp-detail__title">
        <i className="fp-dot fp-dot--lg" style={{ background: style.color, color: style.color }} />
        {ev.goes_class}
        {ev.noaa_ar && <small>AR {ev.noaa_ar}</small>}
      </h2>
      <p className="fp-detail__pin">{pinned ? "● ตรึงไว้ · คลิกซ้ำเพื่อปลด" : "○ ชี้อยู่ · คลิกเพื่อตรึง"}</p>

      <Row k="เวลาพีค">{formatDateTime(ev.peak_time)} UTC</Row>
      <Row k="ระยะเวลา">{durationLabel(ev.start_time, ev.end_time)}</Row>
      <Row k="คลาส">
        {ev.goes_class} <em>· แคตตาล็อกโมเดล</em>
        {classDiffers && (
          <em title="PositionFlare ใช้ฟลักซ์สเกล science ที่ NOAA reprocess แล้ว ส่วนรายงาน NGDC เดิม (ถึงปี 2017) ยังคูณ 0.7 อยู่">
            {" "}· {ev.pf_class} ใน PositionFlare
          </em>
        )}
      </Row>
      <Row k="ตำแหน่ง">
        {row.located ? (
          <>
            {formatHeliographic(ev.lat, ev.lon)}
            <em> · {ev.lat.toFixed(0)}°, {ev.lon.toFixed(0)}°</em>
            {ev.pos_source && <em> · {ev.pos_source}</em>}
            {limb && <em className="fp-flag"> · ใกล้ขอบจาน</em>}
          </>
        ) : (
          <em>{row.matched ? "PositionFlare ไม่ทราบตำแหน่ง" : "ไม่ทราบตำแหน่ง"}</em>
        )}
      </Row>
      <Row k="PositionFlare">
        {row.matched
          ? <>จับคู่ได้ <em>· เวลาพีคต่างกัน {Math.abs(ev.match_dt)} นาที{ev.satellite ? ` · ${ev.satellite}` : ""}</em></>
          : <em>หาคู่ในแคตตาล็อก PositionFlare ไม่เจอ</em>}
      </Row>
      <Row k="peak flux">{ev.peak_flux.toExponential(2)} W/m²</Row>
      <Row k="Solar Cycle">{row.cycle}</Row>
      <Row k="HARP">
        {ev.harpnum ?? <em>ไม่มีคู่</em>}
        {ev.n_harps > 1 && <em> · แมปได้ {ev.n_harps} HARP</em>}
      </Row>

      <button type="button" className="btn btn--sm fp-detail__open" onClick={() => onOpen(row)}>
        {ev.harpnum ? "เปิด HARP นี้ใน dashboard →" : "ดูช่วงเวลานี้ใน dashboard →"}
      </button>
      {!ev.harpnum && <p className="fp-detail__note">flare นี้ไม่มีคู่ HARP จึงไม่มีผลพยากรณ์ของ LSTM ให้ดู</p>}
    </div>
  );
}

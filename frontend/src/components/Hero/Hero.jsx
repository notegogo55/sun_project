import { useEffect, useMemo, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import { SEQ_INDEX, DISK_CLASSES } from "../../lib/plotConstants.js";
import { useCountUp } from "../../hooks/useCountUp.js";
import { useReveal } from "../../hooks/useReveal.js";
import DiskChart from "./DiskChart.jsx";
import HeroProtonChart from "./HeroProtonChart.jsx";

function HeroStat({ target, format, meta }) {
  const ref = useCountUp(target, format);
  return (
    <>
      <p className="stat__value" ref={ref}>—</p>
      <p className="stat__meta">{meta}</p>
    </>
  );
}

export default function Hero() {
  const { health, info, harps, heroTopPoint, goes } = useApp();
  const [diskClasses, setDiskClasses] = useState(() => new Set(DISK_CLASSES));
  const eyebrowRef = useReveal();
  const titleRef = useReveal();
  const leadRef = useReveal();
  const statsRef = useReveal();
  const ctaRef = useReveal();
  const visualRef = useReveal();

  const located = useMemo(
    () => (goes.data?.events ?? []).filter((e) => e.lat !== null && e.lon !== null),
    [goes.data],
  );

  const counts = useMemo(() => {
    const c = Object.fromEntries(DISK_CLASSES.map((cls) => [cls, 0]));
    for (const event of located) {
      const key = event.goes_class[0];
      if (key in c) c[key] += 1;
    }
    return c;
  }, [located]);

  const present = DISK_CLASSES.filter((cls) => counts[cls] > 0);

  // class ที่ช่วงนี้ไม่มีเลยต้องหลุดจากตัวกรองด้วย ไม่ใช่แค่ไม่แสดงปุ่ม — ไม่งั้นคนที่
  // เคยเลือกไว้แค่ X แล้วเปลี่ยนไปช่วงที่ไม่มี X จะเจอจานว่างโดยไม่มีปุ่มให้กดแก้
  useEffect(() => {
    setDiskClasses((prev) => {
      const next = new Set([...prev].filter((cls) => counts[cls] > 0));
      if (!next.size) present.forEach((cls) => next.add(cls));
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [located]);

  function toggleDiskClass(cls) {
    setDiskClasses((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls); else next.add(cls);
      if (!next.size) next.add(cls);
      return next;
    });
  }

  const shownCount = located.filter((e) => diskClasses.has(e.goes_class[0])).length;
  const diskHint = !located.length
    ? "ไม่มี flare ที่ยืนยันพิกัดในช่วงนี้"
    : shownCount === located.length
      ? `${located.length.toLocaleString()} จาก ${(goes.data?.n_events ?? 0).toLocaleString()} flare มีพิกัด`
      : `แสดง ${shownCount.toLocaleString()} จาก ${located.length.toLocaleString()} `
        + `ที่มีพิกัด (ทั้งหมด ${(goes.data?.n_events ?? 0).toLocaleString()} flare)`;

  const riskMeta = !heroTopPoint
    ? "ไม่มีข้อมูลในช่วงนี้"
    : `HARP ${heroTopPoint.harpnum}${heroTopPoint.noaa_ar ? ` · NOAA ${heroTopPoint.noaa_ar}` : ""}`;
  const flareMeta = goes.data
    ? ["B", "C", "M", "X"].map((c) => `${c}:${goes.data.class_counts[c] ?? 0}`).join(" · ")
    : "B:0 · C:0 · M:0 · X:0";
  const rangeText = info?.data?.time_range?.start && info?.data?.time_range?.end
    ? `${info.data.time_range.start.slice(0, 4)}–${info.data.time_range.end.slice(2, 4)}`
    : "—";
  const rangeMeta = info
    ? `${info.data.n_sequences.toLocaleString()} sample · ${harps.length.toLocaleString()} HARP`
    : "กำลังโหลด…";

  return (
    <section className="hero" id="top">
      <div className="hero__inner">
        <div className="hero__copy">
          <p ref={eyebrowRef} className="eyebrow" data-reveal>
            <span>SDO/HMI</span><i>·</i><span>GOES XRS</span><i>·</i><span>NOAA</span>
          </p>

          <h1 ref={titleRef} className="hero__title" data-reveal>
            พยากรณ์การปะทุ<br />
            <em>บนดวงอาทิตย์</em>
          </h1>

          <p ref={leadRef} className="hero__lead" data-reveal>
            แบ่งส่วน active region จากภาพ magnetogram เต็มดวง ติดตามผ่านการหมุนของดวงอาทิตย์
            แล้วพยากรณ์โอกาสเกิด flare ระดับ ≥M1.0 ภายใน 24 ชั่วโมงข้างหน้า
          </p>

          <div ref={statsRef} className="hero__stats" data-reveal>
            <article className="stat stat--risk">
              <p className="stat__label">ความเสี่ยงสูงสุดตอนนี้</p>
              <HeroStat target={(heroTopPoint?.probability ?? NaN) * 100} format={(v) => `${v.toFixed(1)}%`} meta={riskMeta} />
            </article>
            <article className="stat">
              <p className="stat__label">flare ในช่วงที่เลือก</p>
              <HeroStat target={goes.data?.n_events ?? NaN} format={(v) => Math.round(v).toLocaleString()} meta={flareMeta} />
            </article>
            <article className="stat">
              <p className="stat__label">ช่วงข้อมูล</p>
              <p className="stat__value">{rangeText}</p>
              <p className="stat__meta">{rangeMeta}</p>
            </article>
          </div>

          <a ref={ctaRef} className="btn btn--cta" href="#dashboard" data-scroll data-reveal>
            เปิดหน้า dashboard <span aria-hidden="true">→</span>
          </a>
        </div>

        <figure ref={visualRef} className="hero__visual" data-reveal>
          <DiskChart located={located} diskClasses={diskClasses} />
          <figcaption className="hero__caption">
            ตำแหน่ง flare บนจานสุริยะ · <span id="diskHint">{diskHint}</span>
          </figcaption>
          <div className="diskfilter" id="diskFilter" role="group" aria-label="กรอง flare บนจานตามระดับความรุนแรง">
            {present.map((cls) => (
              <button
                key={cls}
                type="button"
                className="chip"
                style={{ "--chip-color": THEME.seq[SEQ_INDEX[cls]] }}
                aria-pressed={diskClasses.has(cls)}
                onClick={() => toggleDiskClass(cls)}
              >
                <i className="chip__dot" aria-hidden="true" />{cls}-class <b>{counts[cls].toLocaleString()}</b>
              </button>
            ))}
          </div>
          {health?.proton_flux && <HeroProtonChart />}
        </figure>
      </div>

      <a className="hero__scroll" href="#dashboard" data-scroll aria-label="เลื่อนลงไปที่ dashboard">
        <span />
      </a>
    </section>
  );
}

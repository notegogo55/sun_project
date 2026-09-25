import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useApp } from "../../state/AppContext.js";
import { DASHBOARD_GROUPS, dashboardHref } from "../../lib/nav.js";

const SYSTEM_ITEMS = [
  // label ของจุดพยากรณ์ถูกแทนด้วยชื่อโมเดลที่เลือกอยู่ตอน render (ปริยาย LSTM เหมือนเดิม)
  { key: "forecast_model", label: "LSTM", title: "โมเดลพยากรณ์ flare" },
  { key: "segmentation_model", label: "U-NET", title: "โมเดล U-Net แบ่งส่วน active region" },
  { key: "sequence_store", label: "DATA", title: "ข้อมูล SHARP sequence ย้อนหลัง" },
];

/** นาฬิกา UTC แยกเป็น component ของตัวเอง — setState ทุกวินาทีจะได้ re-render แค่ตัวเลข
 *  ไม่ลากทั้ง navbar (และ dropdown) ไปด้วย */
function UtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  return (
    <div className="utc-clock" aria-label="เวลา UTC">
      <span className="utc-clock__dot" aria-hidden="true" />
      <span className="utc-clock__label">UTC</span>
      <time className="utc-clock__time">{now.toISOString().slice(11, 19)}</time>
    </div>
  );
}

export default function Navbar() {
  const { health, forecastModelInfo, forecastModels } = useApp();
  const { pathname, hash } = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const [dropOpen, setDropOpen] = useState(false);
  const dropRef = useRef(null);

  // เปลี่ยนหน้า/anchor แล้วปิดทั้งเมนูมือถือและ dropdown — ไม่งั้นแผงค้างบังเนื้อหาหน้าใหม่
  useEffect(() => {
    setMenuOpen(false);
    setDropOpen(false);
  }, [pathname, hash]);

  useEffect(() => {
    function onPointer(event) {
      if (dropRef.current && !dropRef.current.contains(event.target)) setDropOpen(false);
    }
    function onKey(event) {
      if (event.key === "Escape") { setDropOpen(false); setMenuOpen(false); }
    }
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  // hover เปิด dropdown เฉพาะอุปกรณ์ที่มีเมาส์จริง — บนจอสัมผัส mouseenter ยิงตอนแตะ
  // แล้วชนกับ onClick ของปุ่มลูกศรจนแผงเปิดแล้วปิดทันที
  const canHover = typeof window !== "undefined" && window.matchMedia("(hover: hover)").matches;

  return (
    <nav className={`navbar${menuOpen ? " is-menu-open" : ""}`} aria-label="เมนูหลัก">
      <div className="navbar__left">
        <Link to="/" className="navbar__brand">
          <span className="navbar__logo" aria-hidden="true" />
          SUNSEG
        </Link>

        <div className="navbar__links" id="navLinks">
          <NavLink to="/" end className="nav-link">Hub</NavLink>

          <div
            ref={dropRef}
            className={`nav-drop${dropOpen ? " is-open" : ""}`}
            onMouseEnter={canHover ? () => setDropOpen(true) : undefined}
            onMouseLeave={canHover ? () => setDropOpen(false) : undefined}
          >
            <NavLink to="/dashboard" className="nav-link">
              Dashboard
            </NavLink>
            <button
              type="button"
              className="nav-link"
              style={{ paddingLeft: 0 }}
              aria-expanded={dropOpen}
              aria-controls="dashboardMenu"
              aria-label="แสดงส่วนต่าง ๆ ของ dashboard"
              onClick={() => setDropOpen((open) => !open)}
            >
              <span className="nav-link__caret" aria-hidden="true">▼</span>
            </button>

            {dropOpen && (
              <div className="nav-drop__panel" id="dashboardMenu">
                {DASHBOARD_GROUPS.map((group) => (
                  <div key={group.key} className="nav-drop__group">
                    <p className="nav-drop__title">// {group.title}</p>
                    {group.items.map((item) => (
                      <Link key={item.id} to={dashboardHref(item.id)} className="nav-drop__item">
                        <span>{item.label}</span>
                        <span className="nav-drop__tag">{item.tag}</span>
                      </Link>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>

          <NavLink to="/model" className="nav-link">Model</NavLink>
          <NavLink to="/about" className="nav-link">About</NavLink>
          <a href="/docs" target="_blank" rel="noopener" className="nav-link">API ↗</a>
        </div>
      </div>

      <div className="navbar__right">
        <div className="sys-status" aria-label="สถานะระบบ">
          {SYSTEM_ITEMS.map((item) => {
            let { label, title } = item;
            let ready = health ? Boolean(health[item.key]) : null;
            if (item.key === "forecast_model" && forecastModelInfo) {
              // จุดนี้หมายถึงโมเดลที่หน้าเว็บกำลังใช้ ไม่ใช่ "มีโมเดลไหนสักตัว"
              const nReady = forecastModels.filter((m) => m.available).length;
              label = forecastModelInfo.label.toUpperCase();
              title = `${item.title} ${forecastModelInfo.label} (พร้อม ${nReady} จาก ${forecastModels.length} โมเดล)`;
              ready = forecastModelInfo.available;
            }
            return (
              <span
                key={item.key}
                className={`sys-dot${ready === null ? "" : ready ? " sys-dot--ok" : " sys-dot--off"}`}
                title={`${title}: ${ready === null ? "กำลังตรวจสอบ…" : ready ? "พร้อมใช้งาน" : "ยังไม่พร้อม"}`}
              >
                {label}
              </span>
            );
          })}
        </div>
        <UtcClock />
        <button
          type="button"
          className="navbar__toggle"
          aria-expanded={menuOpen}
          aria-controls="navLinks"
          aria-label={menuOpen ? "ปิดเมนู" : "เปิดเมนู"}
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? "✕" : "☰"}
        </button>
      </div>
    </nav>
  );
}

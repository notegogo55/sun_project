import { useApp } from "../state/AppContext.js";

const STATUS_ITEMS = [
  { key: "forecast", label: "โมเดลพยากรณ์", title: "โมเดล LSTM พยากรณ์ flare" },
  { key: "segmentation", label: "โมเดล segmentation", title: "โมเดล U-Net แบ่งส่วน active region" },
  { key: "sequences", label: "ข้อมูลย้อนหลัง", title: "ข้อมูล SHARP sequence ย้อนหลัง" },
];

export default function Topbar() {
  const { health } = useApp();
  const flags = {
    forecast: health?.forecast_model,
    segmentation: health?.segmentation_model,
    sequences: health?.sequence_store,
  };

  return (
    <header className="topbar" id="topbar">
      <a className="brand" href="#top">
        <span className="logo" aria-hidden="true">☀</span>
        <span className="brand__text">
          <span className="brand__name">sunseg</span>
          <span className="brand__tagline">Solar Active Region Segmentation · Tracking · Flare Forecasting</span>
        </span>
      </a>

      <nav className="topnav" aria-label="ส่วนต่างๆ ของหน้า">
        <a href="#dashboard" data-scroll>dashboard</a>
        <a href="#about" data-scroll>เกี่ยวกับ</a>
      </nav>

      <div className="status-strip" id="statusStrip">
        {STATUS_ITEMS.map((item) => (
          <span
            key={item.key}
            className={`status status--${health ? (flags[item.key] ? "ok" : "off") : "pending"}`}
            title={item.title}
          >
            {item.label}
          </span>
        ))}
      </div>
    </header>
  );
}

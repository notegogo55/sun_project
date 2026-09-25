import { Fragment } from "react";
import { useApp } from "../state/AppContext.js";
import { DASHBOARD_GROUPS } from "../lib/nav.js";
import { PageHeader, StatTile } from "../components/ui/Headers.jsx";
import ForecastModelPicker from "../components/ForecastModelPicker.jsx";
import PresetsBar from "../components/PresetsBar.jsx";
import FilterBar from "../components/FilterBar.jsx";
import FrameCard from "../components/Dashboard/FrameCard.jsx";
import ExtractionCard from "../components/Dashboard/ExtractionCard.jsx";
import XrayCard from "../components/Dashboard/XrayCard.jsx";
import ProtonCard from "../components/Dashboard/ProtonCard.jsx";
import ClassForecastCard from "../components/Dashboard/ClassForecastCard.jsx";
import RiskCard from "../components/Dashboard/RiskCard.jsx";
import HarpListCard from "../components/Dashboard/HarpListCard.jsx";
import AttentionCard from "../components/Dashboard/AttentionCard.jsx";

/** การ์ดของแต่ละกลุ่ม — ลำดับกลุ่ม/ป้ายมาจาก DASHBOARD_GROUPS ตัวเดียวกับ dropdown ใน navbar */
const GROUP_CARDS = {
  imagery: [FrameCard, ExtractionCard],
  flux: [XrayCard, ProtonCard],
  forecast: [ClassForecastCard, RiskCard, HarpListCard, AttentionCard],
};

export default function DashboardPage() {
  const { range, goes, harps, selectedHarp, frames, risk } = useApp();

  const selected = harps.find((h) => h.harpnum === selectedHarp);
  const major = goes.data ? (goes.data.class_counts.M ?? 0) + (goes.data.class_counts.X ?? 0) : null;

  return (
    <div className="page dashboard" id="dashboard">
      <PageHeader
        kicker="// SECTION_03 / LIVE PANELS"
        title="DASHBOARD"
        sub="ภาพดวงอาทิตย์ · ฟลักซ์ GOES · พยากรณ์ระดับคลาสของ flare (<M / M / X) ด้วยโมเดลหลัก LSTM + V3 — ทุกแผงใช้ช่วงเวลาเดียวกันจากแถบกรองด้านล่าง"
      >
        <ForecastModelPicker />
      </PageHeader>

      <div className="stat-strip">
        <StatTile label="Range" value={`${range.start.slice(2)} → ${range.end.slice(2)}`} title="ช่วงเวลาที่ทุกแผงใช้ร่วมกัน" />
        <StatTile
          label="Flares in range" tone="sun"
          value={goes.data ? goes.data.n_events.toLocaleString() : "—"}
          unit={major !== null ? `${major} ≥ M` : ""}
        />
        <StatTile
          label="Selected HARP" tone="green"
          value={selectedHarp ?? "—"}
          unit={selected?.noaa_ar ? `AR ${selected.noaa_ar}` : risk.data?.noaa_ar ? `AR ${risk.data.noaa_ar}` : ""}
        />
        <StatTile label="Image frames" tone="violet" value={frames.length.toLocaleString()} unit="HMI / AIA" />
      </div>

      <PresetsBar />
      <FilterBar />

      <div className="grid">
        {DASHBOARD_GROUPS.map((group, index) => (
          <Fragment key={group.key}>
            <h2 className="grid__label">
              // 0{index + 1} {group.title} <span>{group.lead}</span>
            </h2>
            {/* key ตามตำแหน่ง — รายการคงที่ไม่สลับลำดับ (Card.name ใช้ไม่ได้ เพราะ build ย่อชื่อฟังก์ชันจนซ้ำกันได้) */}
            {GROUP_CARDS[group.key].map((Card, i) => <Card key={i} />)}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

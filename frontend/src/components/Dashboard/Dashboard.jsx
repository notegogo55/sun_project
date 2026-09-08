import PresetsBar from "../PresetsBar.jsx";
import FilterBar from "../FilterBar.jsx";
import FrameCard from "./FrameCard.jsx";
import ExtractionCard from "./ExtractionCard.jsx";
import XrayCard from "./XrayCard.jsx";
import ProtonCard from "./ProtonCard.jsx";
import RiskCard from "./RiskCard.jsx";
import HarpListCard from "./HarpListCard.jsx";
import AttentionCard from "./AttentionCard.jsx";

export default function Dashboard() {
  return (
    <main className="dashboard" id="dashboard">
      <PresetsBar />
      <FilterBar />
      <div className="grid">
        <FrameCard />
        <ExtractionCard />
        <XrayCard />
        <ProtonCard />
        <RiskCard />
        <HarpListCard />
        <AttentionCard />
      </div>
    </main>
  );
}

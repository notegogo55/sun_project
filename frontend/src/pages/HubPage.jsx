import FlarePosition from "../components/Hub/FlarePosition.jsx";
import PipelineCards from "../components/Hub/PipelineCards.jsx";

/** หน้าแรก — แผนที่ตำแหน่ง flare ทั้งคลัง (แนวงาน PositionFlare) + ตาราง flare → การ์ด pipeline */
export default function HubPage() {
  return (
    <>
      <FlarePosition />
      <PipelineCards />
    </>
  );
}

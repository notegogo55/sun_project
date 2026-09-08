import { useApp } from "../state/AppContext.js";

export default function Footer() {
  const { info } = useApp();

  const tss = info?.forecast?.metrics?.test?.tss;
  const dice = info?.segmentation?.metrics?.test?.dice;

  const parts = [];
  if (info) {
    if (tss !== undefined && tss !== null) parts.push(`LSTM test TSS = ${tss.toFixed(4)}`);
    if (dice !== undefined && dice !== null) parts.push(`U-Net test Dice = ${dice.toFixed(4)}`);
    parts.push(`พยากรณ์ล่วงหน้า ${info.data.horizon_hours} ชม. · เกณฑ์ ≥ ${info.data.positive_class}`);
  }

  return (
    <footer className="footer">
      <span id="modelSummary">{parts.length ? parts.join("  ·  ") : "—"}</span>
      <a href="/docs" target="_blank" rel="noopener">เอกสาร API ↗</a>
    </footer>
  );
}

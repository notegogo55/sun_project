/** อ่านสีจาก CSS custom property — single source of truth (เดิมอยู่ใน app.js บนสุด)
 *  ต้องถูก import หลังจาก styles.css ถูกฉีดเข้า DOM แล้วเสมอ (main.jsx import CSS ก่อน App) */

const css = getComputedStyle(document.documentElement);
const cssVar = (name) => css.getPropertyValue(name).trim();

export const THEME = {
  textPrimary: cssVar("--text-primary"),
  textSecondary: cssVar("--text-secondary"),
  textMuted: cssVar("--text-muted"),
  gridline: cssVar("--gridline"),
  baseline: cssVar("--baseline"),
  // --surface-1 เป็นสีโปร่งแสง (การ์ดกระจก) ซึ่ง Plotly เรนเดอร์ได้ไม่ดี
  // จึงมีคู่แฝดแบบทึบไว้ใช้กับ hoverlabel และวงแหวนรอบ marker
  surface1: cssVar("--surface-1-solid"),
  border: cssVar("--border"),
  brand: cssVar("--primary"),
  sun: cssVar("--sun"),
  series1: cssVar("--series-1"),
  series1Wash: cssVar("--series-1-wash"),
  statusCritical: cssVar("--status-critical"),
  seq: [cssVar("--seq-1"), cssVar("--seq-2"), cssVar("--seq-3"), cssVar("--seq-4")],
  pflux: [cssVar("--pflux-1"), cssVar("--pflux-2"), cssVar("--pflux-3"), cssVar("--pflux-4")],
  aia: {
    "171": cssVar("--aia-171"),
    "304": cssVar("--aia-304"),
    "1600": cssVar("--aia-1600"),
  },
  fontSans: cssVar("--font-sans"),
  fontMono: cssVar("--font-mono"),
};

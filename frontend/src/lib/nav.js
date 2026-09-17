/** รายการ section ของหน้า dashboard — แหล่งเดียวที่ทั้ง dropdown ใน Navbar และหัวกลุ่มในกริด
 *  (DashboardPage) อ่านร่วมกัน `id` ต้องตรงกับ id ของการ์ดแต่ละใบ เพราะ ScrollManager
 *  เลื่อนไปหา element นั้นตาม hash ของ URL (/dashboard#xray) */
export const DASHBOARD_GROUPS = [
  {
    key: "imagery",
    title: "Imagery",
    lead: "ภาพดวงอาทิตย์เต็มดวงและความเข้มแสงราย active region",
    items: [
      { id: "frame", label: "Magnetogram & AIA", tag: "HMI" },
      { id: "extraction", label: "ความเข้มแสงราย AR", tag: "AIA" },
    ],
  },
  {
    key: "flux",
    title: "Flux",
    lead: "ฟลักซ์ X-ray และโปรตอนจากดาวเทียม GOES ในช่วงที่เลือก",
    items: [
      { id: "xray", label: "GOES X-ray light curve", tag: "XRS" },
      { id: "proton", label: "Proton flux รอบ flare", tag: "PRT" },
    ],
  },
  {
    key: "forecast",
    title: "Forecast",
    lead: "ผลพยากรณ์ของ LSTM สำหรับ active region ที่เลือก",
    items: [
      { id: "risk", label: "ความเสี่ยงล่าสุด", tag: "LSTM" },
      { id: "harps", label: "รายการ active region", tag: "HARP" },
      { id: "attention", label: "Attention & SHARP", tag: "ATT" },
    ],
  },
];

export const dashboardHref = (id) => ({ pathname: "/dashboard", hash: `#${id}` });

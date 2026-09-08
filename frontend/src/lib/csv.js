/** สร้างเนื้อหา CSV ของรายการ flare — แยกออกมาเป็น pure function ไม่แตะ DOM
 *  (ตัวสั่งดาวน์โหลดจริงอยู่ที่ผู้เรียก ดู Dashboard/XrayCard.jsx) */
export function flaresToCsv(events) {
  const headers = ["goes_class", "start_time", "peak_time", "end_time", "peak_flux", "noaa_ar", "harpnum", "lat", "lon"];
  const rows = events.map((e) => [
    e.goes_class ?? "", e.start_time ?? "", e.peak_time ?? "", e.end_time ?? "",
    e.peak_flux ?? "", e.noaa_ar ?? "", e.harpnum ?? "", e.lat ?? "", e.lon ?? "",
  ]);
  return [headers.join(","), ...rows.map((r) => r.map((c) => `"${c}"`).join(","))].join("\n");
}

/** สั่งดาวน์โหลดสตริง CSV เป็นไฟล์ — สร้าง `<a>` ชั่วคราวแล้วคลิกเอง (วิธีมาตรฐานที่ไม่ต้องพึ่ง
 *  library เพิ่ม) */
export function downloadCsv(filename, csvContent) {
  const link = document.createElement("a");
  link.setAttribute("href", `data:text/csv;charset=utf-8,${encodeURI(csvContent)}`);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

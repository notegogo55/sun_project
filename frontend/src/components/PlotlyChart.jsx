import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

/** ตัวห่อ Plotly แบบบางที่สุด — ทุกกราฟใน dashboard (ยกเว้น DiskChart ที่ต้องคุม
 *  restyle/animate เองแบบละเอียด) ใช้ตัวนี้ร่วมกัน
 *
 *  Plotly ไม่ใช่ไลบรารีที่ออกแบบมาให้ผูกกับ virtual DOM — วิธีที่ทำงานได้ดีที่สุดคือ
 *  ให้ React คุมแค่ node เปล่าๆ หนึ่งอัน แล้ว Plotly.react() จัดการ diff ภายในของมันเอง
 *  (เหมือนเดิมทุกจุดที่เรียก Plotly.react ใน app.js)
 *
 *  เปิด `restyle()` ออกทาง ref อย่างตั้งใจ — สำหรับกราฟที่ต้องอัปเดต trace เดิมแบบเบา (ไม่ผ่าน
 *  props ปกติ) เช่น XrayCard ที่อัปเกรดเส้นฟลักซ์เป็นข้อมูลรายนาทีหลังโหลดเสร็จแบบ progressive
 *  — ผู้เรียกไม่ควรต้อง `document.getElementById` เจาะ node ที่จริงๆ เป็นของ component นี้ */
const PlotlyChart = forwardRef(function PlotlyChart({
  id, data, layout, config, onHover, onUnhover, onClick,
  className, style, empty, emptyMessage,
}, forwardedRef) {
  const ref = useRef(null);

  useImperativeHandle(forwardedRef, () => ({
    restyle(update, traceIndices) {
      if (ref.current) window.Plotly.restyle(ref.current, update, traceIndices);
    },
  }), []);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (empty) {
      window.Plotly.purge(node);
      return;
    }
    window.Plotly.react(node, data, layout, config);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, layout, config, empty]);

  useEffect(() => {
    const node = ref.current;
    if (!node || empty) return undefined;
    if (onHover) node.on("plotly_hover", onHover);
    if (onUnhover) node.on("plotly_unhover", onUnhover);
    if (onClick) node.on("plotly_click", onClick);
    return () => {
      node.removeAllListeners?.("plotly_hover");
      node.removeAllListeners?.("plotly_unhover");
      node.removeAllListeners?.("plotly_click");
    };
  }, [onHover, onUnhover, onClick, empty]);

  useEffect(() => {
    const node = ref.current;
    return () => {
      if (node) window.Plotly.purge(node);
    };
  }, []);

  return (
    <div id={id} ref={ref} className={className} style={style}>
      {empty ? <p className="empty">{emptyMessage}</p> : null}
    </div>
  );
});

export default PlotlyChart;

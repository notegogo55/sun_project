import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { REDUCED_MOTION } from "../../lib/motion.js";

/** จัดการตำแหน่ง scroll เมื่อเปลี่ยนเส้นทาง — ไม่มี hash: ขึ้นบนสุดของหน้าใหม่ทันที,
 *  มี hash (/dashboard#xray): เลื่อนไปหาการ์ดนั้น
 *
 *  ต้องลองซ้ำเป็นช่วงสั้น ๆ เพราะการ์ดบางใบ mount ช้ากว่าการเปลี่ยนหน้า (ProtonCard คืน
 *  null จนกว่า health จาก backend จะมาถึง) ถ้าหาครั้งเดียวแล้วไม่เจอจะค้างอยู่บนสุดเฉย ๆ */
export default function ScrollManager() {
  const { pathname, hash } = useLocation();

  useEffect(() => {
    if (!hash) {
      window.scrollTo({ top: 0, behavior: "instant" });
      return undefined;
    }
    const id = decodeURIComponent(hash.slice(1));
    let tries = 0;
    let timer;
    const attempt = () => {
      const target = document.getElementById(id);
      if (target) {
        target.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
        return;
      }
      tries += 1;
      if (tries < 25) timer = setTimeout(attempt, 120);
    };
    attempt();
    return () => clearTimeout(timer);
  }, [pathname, hash]);

  return null;
}

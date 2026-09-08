import { useEffect, useRef } from "react";
import { REDUCED_MOTION } from "../lib/motion.js";

/** นับขึ้นสู่ค่าเป้าหมาย — ใช้กับตัวเลขสรุปใน hero/risk gauge เท่านั้น
 *  คืน ref ให้แปะกับ node ปลายทาง (เขียน textContent ตรงๆ เหมือนเดิม แทนการ setState
 *  ทุกเฟรม ซึ่งจะ re-render ทั้งต้นไม้ 60 ครั้งต่อวินาทีโดยไม่จำเป็น) */
export function useCountUp(target, format) {
  const ref = useRef(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    if (!Number.isFinite(target)) {
      node.textContent = "—";
      return undefined;
    }
    if (REDUCED_MOTION) {
      node.textContent = format(target);
      return undefined;
    }

    const duration = 700;
    const start = performance.now();
    let frame;
    function step(now) {
      const progress = Math.min(1, (now - start) / duration);
      node.textContent = format(target * (1 - Math.pow(1 - progress, 3)));
      if (progress < 1) frame = requestAnimationFrame(step);
    }
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, format]);

  return ref;
}

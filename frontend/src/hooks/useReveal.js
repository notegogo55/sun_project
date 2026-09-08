import { useEffect, useState } from "react";

/** ผูกกับ element เดียวที่มี attribute `data-reveal` — เพิ่มคลาส `.is-visible` แล้วเลิกสังเกต
 *  ทันทีที่เลื่อนมาอยู่ในสายตา (ไม่วิ่งซ้ำตอนเลื่อนกลับ)
 *
 *  เดิมทำแบบรวมศูนย์ด้วย `document.querySelectorAll("[data-reveal]")` ครั้งเดียวตอน
 *  App mount (ดู useChrome.js เดิม) ซึ่งพังกับ element ที่ mount ทีหลัง (เช่น ProtonCard ที่
 *  คืน `null` จนกว่า health จะมาถึงแบบ async เสมอ) — element พวกนั้นไม่เคยถูก observe เลย
 *  และค้างที่ opacity:0 ตลอดไป
 *
 *  คืนค่าเป็น **callback ref** (ไม่ใช่ object ref จาก useRef) โดยตั้งใจ — object ref ธรรมดา
 *  คู่กับ `useEffect(fn, [])` จะพังแบบเดียวกันอีก: effect รันครั้งเดียวตอน mount ตัว hook เอง
 *  (ตอนที่ query อาจยังคืน null เพราะ component ยัง return null อยู่) แล้วไม่มีวันรู้ว่า
 *  DOM node จริงโผล่มาทีหลังเมื่อไร เพราะ .current เปลี่ยนไม่ทำให้ re-render callback ref
 *  ถูก React เรียกใหม่ทุกครั้งที่ element mount/unmount จริง จึง observe ได้ถูกจังหวะเสมอ
 *  ไม่ว่า component จะ mount ตอนไหนก็ตาม */
export function useReveal() {
  const [node, setNode] = useState(null);

  useEffect(() => {
    if (!node) return undefined;

    const observer = new IntersectionObserver((entries, obs) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        obs.unobserve(entry.target);
      }
    }, { rootMargin: "0px 0px -8% 0px" });

    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);

  return setNode;
}

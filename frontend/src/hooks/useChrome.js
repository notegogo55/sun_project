import { useEffect } from "react";

/** โครงหน้า: แถบบนทึบขึ้นเมื่อเลื่อนพ้น hero, ลิงก์ nav สว่างตาม section ที่เห็น —
 *  ย้ายมาจาก setupChrome() เดิม (ไม่รวมคีย์ลัดเลื่อนเฟรม ซึ่งอยู่ใน FrameCard เพราะผูกกับ
 *  เฟรมโดยตรง, และไม่รวมการเผยตัวของการ์ด — ย้ายไป useReveal.js ผูกทีละ component แทน
 *  เพราะ querySelectorAll ครั้งเดียวตอน mount พลาด element ที่ mount ทีหลัง เช่น ProtonCard) */
export function useChrome() {
  useEffect(() => {
    const topbar = document.getElementById("topbar");
    const onScroll = () => topbar?.classList.toggle("is-stuck", window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    const links = new Map(
      [...document.querySelectorAll(".topnav a")].map((a) => [a.getAttribute("href").slice(1), a]),
    );
    const spy = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach((link, id) => link.classList.toggle("is-current", id === entry.target.id));
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach((_, id) => {
      const section = document.getElementById(id);
      if (section) spy.observe(section);
    });

    return () => {
      window.removeEventListener("scroll", onScroll);
      spy.disconnect();
    };
  }, []);
}

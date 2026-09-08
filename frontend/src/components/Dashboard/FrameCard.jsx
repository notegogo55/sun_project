import { useEffect, useRef, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { api } from "../../lib/api.js";
import { frameLabel } from "../../lib/format.js";
import { useReveal } from "../../hooks/useReveal.js";

const LAYER_PILLS = [
  { key: "mag", label: "Magnetogram", title: "โฟโตสเฟียร์: แผนที่ขั้วแม่เหล็ก (HMI)" },
  { key: "1600", label: "AIA 1600 Å", title: "โฟโตสเฟียร์บน / ทรานซิชัน: AIA 1600 Å" },
  { key: "304", label: "AIA 304 Å", title: "โครโมสเฟียร์: AIA 304 Å" },
  { key: "171", label: "AIA 171 Å", title: "โคโรนาชั้นล่าง: AIA 171 Å" },
];

export default function FrameCard() {
  const { frames, frameIndex, setFrameIndex, layer, setLayer, useTruth, setUseTruth } = useApp();
  const revealRef = useReveal();
  const [frameData, setFrameData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const playTimer = useRef(null);
  const framesRef = useRef(frames);
  framesRef.current = frames;
  const frameIndexRef = useRef(frameIndex);
  frameIndexRef.current = frameIndex;
  // คีย์ของ fetch ล่าสุดที่สำเร็จ — กันยิง /api/segment ซ้ำตอน backend ถอย layer (เช่นขอ 304
  // แต่เฟรมนี้ไม่มี) แล้วเราเรียก setLayer(data.layer) แก้กลับ ซึ่งทำให้ effect นี้รันใหม่ทันที
  // เพราะ layer อยู่ใน deps ถ้า key ตรงกับที่เพิ่งได้มาก็แปลว่าข้อมูลที่มีอยู่คือคำตอบของ
  // คำขอนี้อยู่แล้ว ไม่ต้องขอซ้ำ
  const lastFetchRef = useRef(null);

  function clampIndex(index) {
    return Math.max(0, Math.min(framesRef.current.length - 1, index));
  }

  /* ── ดึงภาพเฟรมปัจจุบัน ─────────────────────────────────────── */
  useEffect(() => {
    if (frameIndex < 0 || !frames.length) return undefined;

    const last = lastFetchRef.current;
    if (last && last.frameIndex === frameIndex && last.useTruth === useTruth && last.layer === layer) {
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const data = await api.segment(frames[frameIndex], useTruth, layer);
        if (cancelled) return;
        setFrameData(data);
        lastFetchRef.current = { frameIndex, useTruth, layer: data.layer };
        // backend อาจถอยไป layer ที่มีข้อมูลจริง (เช่น ขอ 304 แต่เฟรมนี้ไม่มี) — ต้องซิงก์กลับ
        if (data.layer !== layer) setLayer(data.layer);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, frames, useTruth, layer]);

  /* ── Timelapse ──────────────────────────────────────────────── */
  function toggleTimelapse() {
    if (isPlaying) {
      setIsPlaying(false);
      clearInterval(playTimer.current);
      playTimer.current = null;
    } else {
      setIsPlaying(true);
      playTimer.current = setInterval(() => {
        if (!framesRef.current.length) return;
        const next = frameIndexRef.current + 1;
        setFrameIndex(next >= framesRef.current.length ? 0 : next);
      }, 700);
    }
  }
  useEffect(() => () => clearInterval(playTimer.current), []);

  /* ── คีย์ลัด: ซ้าย/ขวาเลื่อนเฟรม, Spacebar เล่น/หยุด Timelapse ── */
  useEffect(() => {
    function onKeyDown(e) {
      if (["input", "select", "textarea"].includes(document.activeElement?.tagName?.toLowerCase())) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setFrameIndex(clampIndex(frameIndexRef.current - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setFrameIndex(clampIndex(frameIndexRef.current + 1));
      } else if (e.code === "Space") {
        e.preventDefault();
        toggleTimelapse();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying]);

  if (!frames.length) {
    return (
      <section ref={revealRef} className="card card--frame" aria-labelledby="h-frame" data-reveal>
        <div className="card__head">
          <h2 id="h-frame">Magnetogram &amp; Solar Atmosphere</h2>
          <span className="hint" />
        </div>
        <div id="framePanel" className="frame">
          <p className="empty">
            ยังไม่มีเฟรมภาพ — รัน <code>backend/scripts/download_images.py</code> แล้ว <code>backend/scripts/build_masks.py</code>
          </p>
        </div>
      </section>
    );
  }

  const active = frameData?.layers?.find((l) => l.key === frameData.layer);
  const diceText = frameData?.dice_vs_ground_truth != null
    ? ` · Dice เทียบ mask จริง ${frameData.dice_vs_ground_truth.toFixed(3)}`
    : "";

  return (
    <section ref={revealRef} className="card card--frame" aria-labelledby="h-frame" data-reveal>
      <div className="card__head">
        <h2 id="h-frame">Magnetogram &amp; Solar Atmosphere</h2>
        <span className="hint">{frameIndex >= 0 ? `เฟรม ${frameIndex + 1} / ${frames.length}` : ""}</span>
      </div>

      <div className="layer-pills" id="layerPills" role="radiogroup" aria-label="เลือกชั้นบรรยากาศ">
        {LAYER_PILLS.map((pill) => {
          const matched = frameData?.layers?.find((l) => l.key === pill.key);
          const available = matched ? matched.available : true;
          return (
            <button
              key={pill.key}
              type="button"
              className={`layer-pill${layer === pill.key ? " is-active" : ""}`}
              disabled={frameData ? !available : false}
              title={matched ? `${matched.label} (${matched.region})` : pill.title}
              onClick={() => setLayer(pill.key)}
            >
              <span className={`layer-pill__dot layer-pill__dot--${pill.key}`} /> {pill.label}
            </button>
          );
        })}
      </div>

      <div className="controls controls--frame">
        <button
          className="icon-btn" type="button" aria-label="เฟรมก่อนหน้า" title="เฟรมก่อนหน้า (ลูกศรซ้าย)"
          disabled={frameIndex <= 0}
          onClick={() => setFrameIndex(clampIndex(frameIndex - 1))}
        >‹</button>
        <button
          className={`icon-btn icon-btn--play${isPlaying ? " is-playing" : ""}`} type="button"
          aria-label="เล่น Timelapse"
          title={isPlaying ? "หยุดภาพเคลื่อนไหว (Spacebar)" : "เล่น/หยุดภาพเคลื่อนไหว (Spacebar)"}
          onClick={toggleTimelapse}
        >{isPlaying ? "⏸" : "▶"}</button>
        <label className="field field--grow">
          <span className="sr-only">เลือกเฟรม</span>
          <select value={frameIndex >= 0 ? frameIndex : 0} onChange={(e) => setFrameIndex(Number(e.target.value))}>
            {frames.map((stamp, index) => (
              <option key={stamp} value={index}>{frameLabel(stamp)}</option>
            ))}
          </select>
        </label>
        <button
          className="icon-btn" type="button" aria-label="เฟรมถัดไป" title="เฟรมถัดไป (ลูกศรขวา)"
          disabled={frameIndex >= frames.length - 1}
          onClick={() => setFrameIndex(clampIndex(frameIndex + 1))}
        >›</button>
        <label className="switch" title="เปิดเพื่อใช้ mask จริงจาก SHARP (Ground Truth) หรือปิดเพื่อใช้ mask ที่ U-Net ทำนาย">
          <input type="checkbox" checked={useTruth} onChange={(e) => setUseTruth(e.target.checked)} />
          <span>mask SHARP (Ground Truth)</span>
        </label>
      </div>

      <div id="framePanel" className="frame">
        {loading && <p className="empty">กำลังโหลด…</p>}
        {!loading && error && <p className="empty">{error}</p>}
        {!loading && !error && frameData && (
          <>
            <img src={`data:image/png;base64,${frameData.image_png}`} alt={`${active?.label ?? ""} ${frames[frameIndex]}`} />
            <div className="frame__meta">
              พบ {frameData.n_regions} active region ·
              ครอบคลุม {(frameData.predicted_area_fraction * 100).toFixed(2)}% ของภาพ{diceText}<br />
              <span className="hint">
                {active ? `${active.label} · ${active.region} · ` : ""}
                {useTruth ? "mask จาก SHARP (ground truth)" : "mask ที่ U-Net ทำนาย"}
              </span>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

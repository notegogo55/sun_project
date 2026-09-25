import { useEffect, useRef, useState } from "react";
import { AppContext } from "./AppContext.js";
import { api } from "../lib/api.js";
import { frameTime } from "../lib/format.js";
import { idleAsync, loadingAsync, readyAsync, errorAsync } from "../lib/asyncState.js";

const DEFAULT_RANGE = { start: "2014-10-18", end: "2014-10-31", minClass: "C1.0" };

/** ศูนย์กลาง state ของทั้งแอป — ย้ายมาจาก `state` object เดี่ยวใน app.js เดิม
 *
 *  ของที่ผูกกันแน่น (ต้องดูพร้อมกัน หรือแผงหนึ่งพึ่งผลของอีกแผง) อยู่ที่นี่ทั้งหมด:
 *  ช่วงเวลา (range), harp ที่เลือก, ผล GOES, เฟรมภาพที่กำลังดู, flare ที่ตรึง/แสดงอยู่
 *  ส่วนที่แยกเดี่ยวได้ (เช่น confusion matrix, ตัวกรองตาราง HARP, ตัวเลือก AR ของกราฟ
 *  ความเข้มแสง) ถูกเก็บไว้เป็น local state ในคอมโพเนนต์ของมันเองแทน — ไม่จำเป็นต้อง
 *  รวมทุกอย่างไว้ที่เดียว
 *
 *  dashboard ใช้ "ช่วงเวลาเดียว" จาก `range` — GOES X-ray, จานสุริยะใน hero และตัวเลข
 *  สรุป วาดจากช่วงเดียวกันเสมอ (ผ่าน `applyRange`) เพื่อให้เทียบข้ามแผงได้โดยตรง
 *
 *  การ์ดภาพ (FrameCard) กับการ์ดความเข้มแสง (ExtractionCard) ผูกกันผ่าน `useTruth`
 *  เท่านั้น ไม่ได้ยิงคำขอร่วมกันแบบต้นฉบับ (เดิมทั้งคู่วาดจาก /api/segment ครั้งเดียวกัน
 *  ใน showFrame() การ์ดขวาไม่ยิง network เอง) — ในนี้ FrameCard เรียก /api/segment และ
 *  ExtractionCard เรียก /api/intensity-series แยกกัน แต่ทั้งคู่อ่าน `useTruth` จาก
 *  context เดียวกันเสมอ ทำให้แหล่ง mask (SHARP ground truth vs U-Net) ตรงกันเสมอ
 *  แม้จะเป็นคนละคำขอ
 *
 *  โมเดลพยากรณ์ (`forecastModel`) เป็น state ระดับแอปด้วยเหตุผลเดียวกัน — แผงความเสี่ยง,
 *  attention, ตัวเลขใน hero และ footer ต้องพูดถึงโมเดลตัวเดียวกันเสมอ เริ่มที่โมเดลปริยายของ
 *  backend (LSTM) หน้าเว็บจึงเหมือนเดิมทุกประการจนกว่าผู้ใช้จะกดสลับ
 */
export function AppProvider({ children }) {
  const [health, setHealth] = useState(null);
  const [bootError, setBootError] = useState(null);
  const [info, setInfo] = useState(null);

  const [harps, setHarps] = useState([]);
  const [harpsError, setHarpsError] = useState(null);
  const [selectedHarp, setSelectedHarp] = useState(null);
  const [risk, setRisk] = useState(idleAsync());

  const [forecastModels, setForecastModels] = useState([]);   // ผล /api/forecast/models
  const [forecastModel, setForecastModel] = useState(null);    // ชื่อโมเดลที่เลือก (null = ปริยายของ backend)

  // โมเดลหลัก (LSTM + V3 แยกระดับ <M/M/X) — คนละ dataset กับโมเดลรายชั่วโมงข้างบน จึงมี state ของตัวเอง
  const [classSummary, setClassSummary] = useState(null);      // ผล /api/class-forecast/summary
  const [classMode, setClassMode] = useState("sensitive");     // จุดทำงาน: sensitive | strict
  const [classSeries, setClassSeries] = useState(idleAsync()); // คำพยากรณ์ระดับคลาสของ HARP ที่เลือก

  const [frames, setFrames] = useState([]);
  const [frameIndex, setFrameIndex] = useState(-1);
  const [layer, setLayer] = useState("mag");
  const [useTruth, setUseTruth] = useState(false);

  const [extraction, setExtraction] = useState(idleAsync());
  const extractionRequest = useRef(0);
  // กันผลของคำขอพยากรณ์เก่ามาทับของใหม่ — สลับโมเดล/HARP เร็ว ๆ ทำให้มีหลายคำขอค้างพร้อมกัน
  const riskRequest = useRef(0);
  const classRequest = useRef(0);
  const classModeRef = useRef("sensitive");
  const classSummaryRef = useRef(null);

  const [range, setRange] = useState(DEFAULT_RANGE);
  const [reloading, setReloading] = useState(false);

  const [goes, setGoes] = useState(idleAsync());
  const [heroTopPoint, setHeroTopPoint] = useState(null);

  const [pinnedFlare, setPinnedFlare] = useState(null);
  const [chartedFlare, setChartedFlare] = useState(null);
  const [proton, setProton] = useState(idleAsync({ event: null }));
  const protonRequest = useRef(0);
  const protonPreviewTimer = useRef(null);

  // ตัวชี้ไปยัง state ล่าสุดเสมอ — จำเป็นเพราะฟังก์ชันเหล่านี้อาจถูกเรียกจาก closure ของ
  // render รอบเก่า (effect ตอน mount ด้านล่างรันครั้งเดียวแล้วเดินหน้าเป็น async chain ยาว,
  // หรือ event handler ของ Plotly ที่ผูกไว้ครั้งเดียวต่อชุดข้อมูลใน DiskChart) ถ้าอ่าน
  // `health`/`range`/`useTruth`/`pinnedFlare`/`goes` ตรงๆ จาก closure นั้นจะได้ค่าตอนที่
  // closure ถูกสร้างตลอดไป (เช่น health = null, pinnedFlare = null) ไม่ว่า state จริงจะ
  // อัปเดตไปแล้วกี่รอบ — อ่านผ่าน ref แทนเพื่อให้ได้ค่าล่าสุดเสมอไม่ว่าจะถูกเรียกจาก closure รุ่นไหน
  const healthRef = useRef(null);
  healthRef.current = health;
  const rangeRef = useRef(DEFAULT_RANGE);
  const useTruthRef = useRef(false);
  useTruthRef.current = useTruth;
  const framesRef = useRef([]);
  framesRef.current = frames;
  const pinnedFlareRef = useRef(null);
  pinnedFlareRef.current = pinnedFlare;
  const goesRef = useRef(idleAsync());
  goesRef.current = goes;
  const forecastModelRef = useRef(null);
  const selectedHarpRef = useRef(null);
  selectedHarpRef.current = selectedHarp;

  /* ── สถานะระบบ + ตัวเลข config/metrics ───────────────────────── */

  useEffect(() => {
    let cancelled = false;
    (async () => {
      let healthData;
      try {
        healthData = await api.health();
      } catch (error) {
        if (!cancelled) setBootError(`เชื่อมต่อ backend ไม่สำเร็จ: ${error.message}`);
        return;
      }
      if (cancelled) return;
      setHealth(healthData);

      // เลือกโมเดลก่อนยิงคำขอพยากรณ์ใด ๆ — ปริยายของ backend ถ้าพร้อม ไม่งั้นตัวแรกที่พร้อม
      try {
        const models = await api.forecastModels();
        if (cancelled) return;
        setForecastModels(models);
        const initial = models.find((m) => m.default && m.available)
          ?? models.find((m) => m.available)
          ?? models.find((m) => m.default);
        if (initial) {
          forecastModelRef.current = initial.name;
          setForecastModel(initial.name);
        }
      } catch {
        /* backend รุ่นเก่าไม่มี endpoint นี้ — ปล่อยให้ใช้โมเดลปริยายของ backend */
      }

      try {
        const summary = await api.classForecastSummary();
        if (cancelled) return;
        classSummaryRef.current = summary;
        setClassSummary(summary);
      } catch {
        /* backend รุ่นเก่าไม่มี endpoint นี้ — การ์ดโมเดลหลักจะแสดงว่ายังไม่พร้อม */
      }

      let harpsData = [];
      if (healthData.sequence_store) {
        try {
          harpsData = await api.harps();
          if (cancelled) return;
          setHarps(harpsData);
        } catch (error) {
          if (!cancelled) setHarpsError(error.message);
        }
      }

      try {
        const infoData = await api.info();
        if (!cancelled) setInfo(infoData);
      } catch {
        /* ไม่สำคัญพอที่จะรบกวนผู้ใช้ */
      }

      if (healthData.n_frames > 0) {
        try {
          const frameList = await api.frames();
          if (!cancelled && frameList.length) setFrames(frameList);
        } catch {
          /* ส่วนภาพจะแสดงข้อความว่างเอง */
        }
      }

      if (cancelled) return;

      const flaring = [...harpsData].filter((h) => h.n_positive > 0)
        .sort((a, b) => b.n_positive - a.n_positive);
      const first = flaring[0] ?? harpsData[0];
      if (first) {
        await selectHarp(first.harpnum, { syncRange: true });
      } else {
        await applyRange({});
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** ข้อความแจ้งขั้นตอนที่ยังขาด — คำนวณจาก health เสมอ (ไม่ต้องเก็บเป็น state แยก) */
  const missingSetup = [];
  if (health) {
    if (!health.sequence_store) {
      missingSetup.push(["python backend/scripts/data/download_metadata.py", "python backend/scripts/data/build_sequences.py"]);
    } else if (!health.forecast_model) {
      missingSetup.push(["python backend/scripts/forecast/train.py --model lstm"]);
    }
    if (!health.segmentation_model && health.n_frames === 0) {
      missingSetup.push(["python backend/scripts/data/download_images.py", "python backend/scripts/build_masks.py"]);
    }
    // แผนที่ตำแหน่ง flare หน้าแรก — ต้องมี flares.parquet ของโมเดลก่อน แล้วค่อยจับคู่กับ PositionFlare
    if (!health.flare_positions) {
      missingSetup.push(["python backend/scripts/data/build_flare_positions.py"]);
    }
  }

  /* ── HARP ที่เลือก + พยากรณ์ ─────────────────────────────────── */

  async function selectHarp(harpnum, { syncRange = false } = {}) {
    setSelectedHarp(harpnum);
    // ต่างจาก goes/proton ตรงนี้ตั้งใจ**ล้าง**ข้อมูลเก่าก่อนโหลดใหม่ (ไม่ใช้ loadingAsync
    // แบบเก็บของเดิม) — ต้นฉบับ (app.js:626-628) ล้างแผงความเสี่ยงเป็น "กำลังคำนวณ…" ทันที
    // ไม่โชว์ตัวเลขของ HARP ก่อนหน้าค้างไว้ระหว่างรอ เพราะเป็นคนละ AR กันเลย ไม่ใช่แค่รีเฟรช
    setRisk(loadingAsync(idleAsync()));
    loadClassSeriesInternal(harpnum);  // ไม่รอ — คนละ endpoint กับแผงความเสี่ยง ไม่ควรถ่วงการขยับช่วงเวลา
    const requestId = ++riskRequest.current;
    try {
      const series = await api.forecast(harpnum, forecastModelRef.current);
      if (requestId !== riskRequest.current) return;
      setRisk(readyAsync(series));
      if (syncRange && series.times.length) {
        await applyRange({
          start: series.times[0].slice(0, 10),
          end: series.times[series.times.length - 1].slice(0, 10),
        });
      }
    } catch (error) {
      if (requestId !== riskRequest.current) return;
      setRisk(errorAsync(error.message));
    }
  }

  /** สลับโมเดลพยากรณ์ — คำนวณแผงความเสี่ยงของ HARP เดิมและตัวเลขใน hero ใหม่ด้วยโมเดลที่เลือก
   *  (ไม่ขยับช่วงเวลา — ผู้ใช้กำลังเทียบโมเดลบน AR/ช่วงเดิม) */
  async function selectForecastModel(name) {
    if (name === forecastModelRef.current) return;
    forecastModelRef.current = name;
    setForecastModel(name);
    const harp = selectedHarpRef.current;
    await Promise.all([
      harp != null ? selectHarp(harp) : Promise.resolve(),
      loadHeroRiskInternal(rangeRef.current.end),
    ]);
  }

  async function loadClassSeriesInternal(harpnum) {
    const requestId = ++classRequest.current;
    if (!classSummaryRef.current?.available) {
      setClassSeries(idleAsync());
      return;
    }
    setClassSeries(loadingAsync(idleAsync()));
    try {
      const series = await api.classForecast(harpnum, classModeRef.current);
      if (requestId !== classRequest.current) return;
      setClassSeries(readyAsync(series));
    } catch (error) {
      if (requestId !== classRequest.current) return;
      setClassSeries(errorAsync(error.message));
    }
  }

  /** สลับจุดทำงานของโมเดลหลัก (เตือนไว/ระมัดระวัง) — มีผลทั้งการ์ดบน dashboard และตารางในหน้า Model */
  async function selectClassMode(mode) {
    if (mode === classModeRef.current) return;
    classModeRef.current = mode;
    setClassMode(mode);
    const harp = selectedHarpRef.current;
    if (harp != null) await loadClassSeriesInternal(harp);
  }

  /* ── ช่วงเวลาที่กรอง (ใช้ร่วมกันทั้งหน้า) ─────────────────────── */

  function syncFrameToRange(framesList, end) {
    if (!framesList.length || !end) return;
    const target = new Date(`${end}T00:00:00Z`).getTime();
    let best = 0;
    let bestGap = Infinity;
    framesList.forEach((stamp, index) => {
      const gap = Math.abs(frameTime(stamp).getTime() - target);
      if (gap < bestGap) { bestGap = gap; best = index; }
    });
    setFrameIndex((current) => (best !== current ? best : current));
  }

  async function loadGoesInternal(next) {
    // คงข้อมูลเก่าไว้ระหว่างโหลด (loadingAsync) แทนที่จะล้างเป็น null ทันที — กราฟ X-ray,
    // จานสุริยะ และตัวเลขสรุปใน hero ล้วนอ่านจาก goes.data โดยตรง ถ้าล้างตรงนี้ทุกแผงจะ
    // กะพริบเป็นสถานะว่างทุกครั้งที่กด "แสดงข้อมูล" ทั้งที่ของเดิม (app.js:1221) ไม่เคยล้าง
    // อะไรก่อน fetch เลย ปล่อยให้ค่าเก่าค้างจนของใหม่มาถึงเหมือนกัน
    setGoes((prev) => loadingAsync(prev));
    try {
      const data = await api.goes(next.start, next.end, next.minClass);
      setGoes(readyAsync(data));
      setPinnedFlare(null);
      setChartedFlare(null);
      autoSelectProtonInternal(data);
    } catch (error) {
      setGoes(errorAsync(error.message));
      setProton(idleAsync({ event: null }));
      setChartedFlare(null);
    }
  }

  async function loadHeroRiskInternal(end) {
    const currentHealth = healthRef.current;
    if (!currentHealth?.forecast_model || !currentHealth?.sequence_store || !end) return;
    try {
      const ranked = await api.forecastAt(end, forecastModelRef.current);
      setHeroTopPoint(ranked[0] ?? null);
    } catch {
      /* ตัวเลขใน hero เป็นข้อมูลเสริม — ล้มเหลวเงียบๆ */
    }
  }

  async function loadExtractionInternal(next) {
    if (!next.start || !next.end) return;
    const requestId = ++extractionRequest.current;
    setExtraction((prev) => loadingAsync(prev));
    try {
      const data = await api.intensitySeries(next.start, next.end, useTruthRef.current);
      if (requestId !== extractionRequest.current) return;
      setExtraction(readyAsync(data));
    } catch (error) {
      if (requestId !== extractionRequest.current) return;
      setExtraction(errorAsync(error.message));
    }
  }

  async function applyRange(overrides) {
    const next = { ...rangeRef.current, ...overrides };
    rangeRef.current = next;
    setRange(next);
    setReloading(true);
    try {
      syncFrameToRange(framesRef.current, next.end);
      await Promise.all([
        loadGoesInternal(next),
        loadHeroRiskInternal(next.end),
        loadExtractionInternal(next),
      ]);
    } finally {
      setReloading(false);
    }
  }

  // useTruth เปลี่ยน (สลับ mask SHARP <-> U-Net) → ตัวเลขความเข้มแสงต้องคำนวณใหม่
  // ข้ามการเรียกครั้งแรกตอน mount เพราะ init effect ด้านบนจัดการโหลดรอบแรกอยู่แล้ว
  const isFirstUseTruth = useRef(true);
  useEffect(() => {
    if (isFirstUseTruth.current) { isFirstUseTruth.current = false; return; }
    loadExtractionInternal(range);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useTruth]);

  /* ── ฟลักซ์โปรตอนรอบเวลาที่เกิด flare ─────────────────────────── */

  async function loadProtonInternal(event) {
    if (!healthRef.current?.proton_flux || !event?.peak_time) return;
    const requestId = ++protonRequest.current;
    setProton((prev) => loadingAsync(prev));
    try {
      const data = await api.proton(event.peak_time, event.goes_class);
      if (requestId !== protonRequest.current) return;
      setChartedFlare(event);
      setProton(readyAsync(data, { event }));
    } catch (error) {
      if (requestId !== protonRequest.current) return;
      setChartedFlare(null);
      setProton(errorAsync(error.message, { event }));
    }
  }

  function autoSelectProtonInternal(data) {
    if (!healthRef.current?.proton_flux) return;
    const strongest = data.events.reduce(
      (best, event) => (best === null || event.peak_flux > best.peak_flux ? event : best),
      null,
    );
    if (strongest) {
      loadProtonInternal(strongest);
    } else {
      setChartedFlare(null);
      setProton(idleAsync({ event: null, status: "empty" }));
    }
  }

  // อ่าน pinnedFlare/goes ผ่าน ref (pinnedFlareRef/goesRef) แทนตัวแปร state ตรงๆ — สอง
  // ฟังก์ชันนี้ถูกเรียกจาก event handler ของ Plotly ใน DiskChart.jsx ที่ผูกไว้ครั้งเดียวต่อ
  // ชุดข้อมูล (deps ไม่มี pinnedFlare) closure ที่ค้างอยู่จึงเห็น pinnedFlare เป็นค่าตอนที่
  // สร้าง handler เสมอ ถ้าอ่านตรงๆ คลิกจุดเดิมซ้ำจะไม่มีวันปลดตรึง (alreadyPinned เป็น false
  // เสมอ) และ hover จุดอื่นจะยิง preview ทับกราฟที่ตรึงไว้ — อ่านผ่าน ref แก้ได้ที่จุดเดียว
  // ไม่ว่า caller จะเป็น closure รุ่นไหน (เดิม app.js:1462, app.js:1471)
  function previewFlareEvent(flare) {
    if (!healthRef.current?.proton_flux || pinnedFlareRef.current) return;
    clearTimeout(protonPreviewTimer.current);
    protonPreviewTimer.current = setTimeout(() => loadProtonInternal(flare), 90);
  }

  function clearProtonPreview() {
    clearTimeout(protonPreviewTimer.current);
  }

  function selectFlareEvent(flare) {
    clearTimeout(protonPreviewTimer.current);
    const alreadyPinned = pinnedFlareRef.current?.peak_time === flare.peak_time;
    setPinnedFlare(alreadyPinned ? null : flare);
    if (!alreadyPinned) {
      loadProtonInternal(flare);
    } else if (goesRef.current.data) {
      autoSelectProtonInternal(goesRef.current.data);
    }
  }

  // ข้อมูลสรุปของโมเดลที่เลือก (label, ผล test, pooling) — null ถ้า backend ยังไม่ตอบ
  const forecastModelInfo = forecastModels.find((m) => m.name === forecastModel) ?? null;

  const value = {
    health, bootError, info, missingSetup,
    harps, harpsError, selectedHarp, risk, selectHarp,
    forecastModels, forecastModel, forecastModelInfo, selectForecastModel,
    classSummary, classMode, selectClassMode, classSeries,
    frames, frameIndex, setFrameIndex, layer, setLayer, useTruth, setUseTruth,
    extraction,
    range, reloading, applyRange,
    goes,
    heroTopPoint,
    pinnedFlare, chartedFlare, proton,
    selectFlareEvent, previewFlareEvent, clearProtonPreview,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

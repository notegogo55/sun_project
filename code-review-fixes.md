# Prompt: แก้ไขผลจาก code review (React migration + Docker split)

> ไฟล์นี้คือ **prompt สำหรับสั่งงานแก้ไข** — ส่งทั้งไฟล์ให้ agent หรืออ่านเองแล้วไล่แก้ทีละข้อก็ได้
> ที่มา: `/code-review` สองแกน (Standards + Spec) เทียบ working tree กับ commit `3c08d97`

---

## บริบทของโค้ดเบส (อ่านก่อนเริ่ม)

`d:\sun_project` — dashboard ติดตาม active region บนดวงอาทิตย์และพยากรณ์ solar flare
backend เป็น FastAPI (`backend/app/`) frontend เป็น React + Vite (`frontend/`)

การเปลี่ยนแปลงที่ถูกรีวิวคือการแปลง frontend จาก vanilla JS (`frontend/app.js` 2,441 บรรทัด)
มาเป็น React แล้วแยก Docker เป็นสองคอนเทนเนอร์ ทั้งหมดยังไม่ commit

**โครงสร้าง frontend ที่ต้องรู้:**

```
frontend/src/
  state/AppProvider.jsx      ศูนย์กลาง state ทั้งแอป (เทียบเท่า object `state` ใน app.js เดิม)
  state/AppContext.js        createContext + useApp()
  components/Hero/           Hero, DiskChart (จานสุริยะ), HeroProtonChart
  components/Dashboard/      FrameCard, ExtractionCard, XrayCard, ProtonCard,
                             RiskCard, HarpListCard, AttentionCard
  components/About/          AboutSection, ConfusionMatrix
  components/PlotlyChart.jsx ตัวห่อ Plotly ที่กราฟส่วนใหญ่ใช้ร่วมกัน
  lib/                       api.js, theme.js, format.js, plotConstants.js, motion.js
  hooks/                     useChrome.js (topbar/scroll-spy/reveal), useCountUp.js
  styles.css                 CSS เดิมยกมาทั้งไฟล์ (class names สำคัญมาก ห้ามแก้ชื่อ)
```

**โค้ดเดิมไว้เทียบเสมอ** เมื่อไม่แน่ใจว่าพฤติกรรมเดิมเป็นอย่างไร:

```bash
git show 3c08d97:frontend/app.js
git show 3c08d97:frontend/index.html
```

⚠️ แต่ระวัง: working tree ก่อนเริ่มแปลงมีงานค้างที่ยังไม่ commit อยู่ด้วย (โหมด dense ของกราฟ X-ray)
ซึ่ง **ไม่มี** ใน `3c08d97` — ดูหัวข้อ "ห้ามทำ" ด้านล่าง

**บั๊กประจำถิ่นของโค้ดชุดนี้ (เจอมาแล้ว 3 รอบ):** stale closure
`AppProvider` ประกาศฟังก์ชันใหม่ทุก render โดยตั้งใจ (ไม่ใช้ `useCallback`) ฟังก์ชันไหนที่ถูกเรียก
จาก effect ที่รันครั้งเดียว (deps `[]`) หรือจาก event handler ที่ผูกไว้ครั้งเดียว จะค้างอยู่กับค่า state
ของ render รอบนั้นตลอดไป วิธีแก้ที่ใช้อยู่แล้วในไฟล์คือ **อ่านผ่าน ref** — ดู `healthRef`, `rangeRef`,
`useTruthRef`, `framesRef` ใน `AppProvider.jsx` เป็นแบบอย่าง ทำแบบเดียวกันเสมอ อย่าเปลี่ยนไปใช้วิธีอื่น

---

## A. บั๊กพฤติกรรม — แก้ก่อน (เรียงตามความร้ายแรง)

### A1. pin/unpin บนจานสุริยะพัง (stale closure) 🔴

**ไฟล์:** `frontend/src/components/Hero/DiskChart.jsx` (effect ที่ผูก plotly event, deps `[located, shown, projected]`)
+ `frontend/src/state/AppProvider.jsx` (`selectFlareEvent`, `previewFlareEvent`)

**อาการ:** handler ของ `plotly_click`/`plotly_hover` ถูกผูกครั้งเดียวต่อชุดข้อมูล และปิดทับ
`selectFlareEvent`/`previewFlareEvent` จาก render ที่ `pinnedFlare === null` ตลอด ผลคือ

- คลิกจุดเดิมซ้ำ **ไม่ปลดตรึง** — `alreadyPinned` ใน `selectFlareEvent` เป็น `false` เสมอ
  เดิม: `app.js:1471` `const alreadyPinned = state.pinnedFlare?.peak_time === flare.peak_time;`
- hover จุดอื่นยัง**ยิง preview ทับกราฟที่ตรึงไว้** (คลิกควรชนะ hover เสมอ)
  เดิม: `app.js:1462` `if (!state.health?.proton_flux || state.pinnedFlare) return;`
- `XrayCard` ไม่พัง (เพราะ `PlotlyChart` re-subscribe ทุก render) → พฤติกรรมสองที่ไม่ตรงกัน

**สิ่งที่ต้องแก้:** แก้ที่ต้นทางใน `AppProvider.jsx` ไม่ใช่ไล่ patch ทีละ handler —
เพิ่ม `pinnedFlareRef` และ `goesRef` ตามแบบ `healthRef` ที่มีอยู่แล้ว แล้วให้ `selectFlareEvent`
กับ `previewFlareEvent` อ่านจาก ref (`selectFlareEvent` อ่าน `goes.data` ตอน unpin ด้วย จึงต้องมี
`goesRef` คู่กัน) แก้ที่เดียวจบทุก caller ไม่ว่าจะถูกเรียกจาก closure รุ่นไหน

**ยืนยันหลังแก้:** คลิกจุด flare บนจาน → กราฟโปรตอนตรึง (สรุปขึ้นคำว่า "ตรึงไว้") →
กวาดเมาส์ผ่านจุดอื่น กราฟต้องไม่เปลี่ยน → คลิกจุดเดิมซ้ำ ต้องปลดตรึงและกลับไปแสดง flare แรงสุดในช่วง

### A2. สถานะ loading ล้างกราฟที่วาดไว้แล้ว 🔴

**ไฟล์:** `AppProvider.jsx` (`loadGoesInternal`), `Dashboard/ProtonCard.jsx`, `Hero/HeroProtonChart.jsx`

**อาการ:**

- `setGoes({ status: "loading", data: null, ... })` → ทุกครั้งที่กด "แสดงข้อมูล" กราฟ X-ray กะพริบเป็น
  ข้อความ "ไม่พบ flare ในช่วงเวลานี้" จานสุริยะโดน `purge` และตัวเลขใน hero เด้งกลับเป็น "—"
  เดิม `loadGoes` (`app.js:1221`) ไม่ล้างอะไรเลยก่อน fetch — ของเก่าค้างจนของใหม่มาถึง
- `ProtonCard.jsx` `const empty = status !== "ready"` → ทุก hover preview (debounce 90 ms) กราฟถูก purge
  แล้ววาดใหม่ พร้อมข้อความ **ผิดความจริง** ว่า "คลังโปรตอนไม่ครอบคลุมช่วงเวลานี้ — ข้อมูลเริ่มกลางปี 1998"
  เดิมแค่ `el("protonSummary").textContent = "กำลังโหลด…"` (`app.js:1436`) กราฟยังอยู่ และ
  `transition: {duration: 300}` ได้ทำงานจริงตามที่ตั้งใจไว้
- `HeroProtonChart.jsx` มีรูปแบบ `status !== "ready"` เดียวกัน → แก้คู่กัน

**สิ่งที่ต้องแก้:** ระหว่างโหลด ให้คงข้อมูลเดิมไว้ (`setGoes(prev => ({ ...prev, status: "loading" }))`)
และให้เงื่อนไข `empty` ของกราฟตัดสินจาก **การมีข้อมูลจริง** (`!data || !data.has_data`) ไม่ใช่จาก status
ส่วนข้อความ "กำลังโหลด…" ให้ไปอยู่ที่บรรทัดสรุป/hint เหมือนเดิม และ `emptyMessage` ต้องแยกกรณี
"ไม่มีคลังข้อมูลช่วงนี้" ออกจาก "กำลังโหลด" ให้ชัด

**ยืนยันหลังแก้:** กดปุ่ม "แสดงข้อมูล" ซ้ำๆ กราฟต้องไม่กะพริบเป็นสถานะว่าง · กวาดเมาส์ผ่านจุด flare
หลายจุดติดกัน กราฟโปรตอนต้องไหลเปลี่ยนรูป ไม่ใช่หายแล้วโผล่

### A3. การ์ด Proton ค้างที่ opacity 0 (มองไม่เห็นทั้งการ์ด) 🟠

**ไฟล์:** `frontend/src/hooks/useChrome.js` + `Dashboard/ProtonCard.jsx`

**อาการ:** `ProtonCard` คืน `null` จนกว่า `health` จะมาถึง (async เสมอ) แต่ `useChrome` เรียก
`document.querySelectorAll("[data-reveal]")` **ครั้งเดียวตอน mount** การ์ดที่ mount ทีหลังจึงไม่เคยถูก
observe ไม่เคยได้คลาส `.is-visible` และค้างที่ `opacity: 0` ตาม `styles.css`
เดิมการ์ดอยู่ใน DOM เสมอ แค่สลับคลาส `hidden` (`app.js:218`)

**สิ่งที่ต้องแก้:** เปลี่ยน reveal ให้เป็นแบบ per-component — ทำ hook `useReveal()` ที่คืน ref
ให้แต่ละ element ที่จะ reveal ใช้ แล้วให้ hook นั้น observe/unobserve ตัวเองตามวงจรชีวิตของ component
(เลิกใช้ `querySelectorAll` รวมศูนย์) วิธีนี้ถูกต้องไม่ว่า component จะ mount ตอนไหน
ทางเลือกที่ง่ายกว่าแต่ตรงกับของเดิม: render `<section>` ไว้เสมอแล้วสลับคลาสซ่อนแทนการคืน `null`

**ยืนยันหลังแก้:** ต้องทดสอบตอน `health.proton_flux === true` เท่านั้น (ใน Docker เป็น `false` เพราะ
คลังโปรตอนอยู่นอก repo) → รัน backend ในเครื่องโดยตั้ง `SUNSEG_PROTON_DIR` แล้วเลื่อนลงไปดูการ์ด
"Proton flux รอบเวลาที่เกิด flare" ต้องมองเห็น

### A4. error ตอนโหลดรายการ HARP เงียบหาย 🟠

**ไฟล์:** `AppProvider.jsx` (init effect, `catch {}` รอบ `api.harps()`) + `Dashboard/HarpListCard.jsx`

**อาการ:** ถ้า `/api/harps` ล้มเหลว ตารางค้างที่ "กำลังโหลด…" ตลอดไป ไม่บอกสาเหตุ
เดิม: `app.js:539` `tbody.innerHTML = \`<tr><td colspan="4" class="empty">${error.message}</td></tr>\``

**สิ่งที่ต้องแก้:** เก็บ error ไว้ใน state (เช่น `harpsError`) ส่งผ่าน context แล้วให้ `HarpListCard`
แสดงข้อความ error นั้นในตารางแทน "กำลังโหลด…"

### A5. ยิง `/api/segment` ซ้ำเมื่อ backend ถอย layer 🟡

**ไฟล์:** `Dashboard/FrameCard.jsx` (effect ดึงภาพเฟรม)

**อาการ:** `if (data.layer !== layer) setLayer(data.layer);` แต่ `layer` อยู่ใน deps ของ effect เดียวกัน
→ พอ backend ถอยไป layer ที่มีข้อมูลจริง (เช่นขอ 304 แล้วเฟรมนั้นไม่มี) จะยิงคำขอซ้ำอีกรอบทันที
เดิม `syncLayerOptions` (`app.js:2185`) แค่เขียน `state.layer = activeLayer` เฉยๆ ไม่ยิงซ้ำ

หมายเหตุ: เกิดเฉพาะตอน backend ถอย layer เท่านั้น ไม่ใช่ทุกครั้งที่เปลี่ยนเฟรม

**สิ่งที่ต้องแก้:** แยก "layer ที่ขอ" ออกจาก "layer ที่ได้จริง" — เช่นเก็บ ref ของคีย์ที่เพิ่ง fetch สำเร็จ
(`frameIndex|useTruth|layer`) แล้วข้าม fetch ถ้า effect ถูกกระตุ้นด้วยค่าที่ response รอบก่อนครอบคลุมอยู่แล้ว

### A6. ข้อความ UI เพี้ยนจากเดิม 🟡

- `Dashboard/ExtractionCard.jsx` — `"mask ที่ U-Net ทำนาย"` เดิมคือ `"mask U-Net (Segmentation)"` (`app.js:1191`)
- `Dashboard/HarpListCard.jsx` — `harpCount` แสดงสตริงว่างตอนยังไม่มีข้อมูล เดิมแสดงบรรทัดนับเสมอ

---

## B. คุณภาพโค้ดและคอนเวนชันของรีโป

> รีโปนี้ไม่มีไฟล์มาตรฐานการเขียนโค้ด แต่มีคอนเวนชันที่ชัดมากจากโค้ดเดิม:
> **คอมเมนต์ภาษาไทยที่อธิบาย "ทำไม"** — เหตุผลของการตัดสินใจที่ไม่ชัดในตัวเอง มักอ้างบั๊กที่เคยเจอ
> หรือตัวเลขที่วัดมาจริง ไม่ใช่การบรรยายซ้ำว่าโค้ดทำอะไร ให้รักษาคอนเวนชันนี้ในทุกไฟล์ที่แตะ

### B1. เลิกประกอบ HTML string แล้วยัดผ่าน `dangerouslySetInnerHTML`

`Dashboard/ProtonCard.jsx` — `protonSummaryHtml()` ประกอบ `<b>`/`<span>` เองแล้วแทรกค่าจาก backend
(`data.sources.join(" + ")`, `data.s_scale`) เข้าไปตรงๆ เดิมทำแบบนั้นเพราะไม่มี JSX — ตอนนี้เขียนเป็น
JSX ได้ทั้งก้อนแล้ว ให้แปลงเป็น JSX

(`Hero/DiskChart.jsx` ใช้ `tip.innerHTML` เหมือนกัน แต่มีคอมเมนต์อธิบายเหตุผลที่ต้อง imperative ไว้ชัด
— อันนั้นเก็บไว้ได้)

### B2. คืนคอมเมนต์ "ทำไม" ที่หายไป

- `app.js:1-19` เดิมมี docblock อธิบายสถาปัตยกรรมทั้งหน้า ("หน้าเว็บแบ่งเป็นสามฉาก…",
  "ทุกส่วนออกแบบให้ degrade ได้…", การผูกกันของการ์ดภาพกับการ์ดความเข้มแสง) — ให้เขียนใหม่ไว้ที่
  `App.jsx` หรือ `AppProvider.jsx`
- `About/ConfusionMatrix.jsx` (~238 บรรทัด, คอมเมนต์ 0 บรรทัด) — คืนคำอธิบายรายฟิลด์ที่ `app.js:44-49`
  เคยมี และเหตุผลว่าทำไมแผงนี้ไม่ผูกกับช่วงเวลาของหน้า (`app.js:274-280`)

### B3. คอมเมนต์ที่ชี้ไปยังไฟล์ที่ถูกลบแล้ว (dangling)

`lib/plotConstants.js` ("ดูคำอธิบายยาวใน app.js เดิม"), `lib/api.js` ("ดู fetchJson เดิม"),
`Dashboard/RiskCard.jsx` ("ดูเหตุผลใน app.js เดิม"), `lib/format.js`, `Hero/Hero.jsx`
— `frontend/app.js` ถูกลบไปแล้วในชุดแก้เดียวกัน ให้ **ย้ายเหตุผลจริงมาเขียนไว้ตรงนั้น** แทนการชี้ทาง

### B4. Plotly โหลดจาก CDN โดยไม่มีคำอธิบาย

`frontend/index.html` โหลด `plotly-2.35.2.min.js` จาก CDN แล้วโค้ดเรียกผ่าน `window.Plotly`
แต่ไม่มีใน `package.json` และไม่มีคอมเมนต์บอกว่าทำไมถึงไม่ bundle
→ เขียนคอมเมนต์อธิบายเหตุผล (ขนาด bundle) ไว้ใน `index.html`

### B5. โค้ดซ้ำ

- คำนวณช่วงแกน y ของ proton ซ้ำคำต่อคำใน `ProtonCard.jsx` กับ `HeroProtonChart.jsx`
  (`low = Math.min(low, Math.floor(exponent))` …) รวมถึง `range: [shiftIsoHours(...), ...]`
  → แยกเป็น helper ร่วม (เช่น `lib/proton.js`)
- `REDUCED_MOTION` ประกาศซ้ำใน `lib/motion.js` และ `hooks/useCountUp.js` → เหลือแหล่งเดียว
- เปลือกการ์ดใน `RiskCard.jsx` (`<section className="card card--gauge">…<h2 id="h-risk">`) ซ้ำ 4 ครั้ง
  ตามจำนวน status → เหลือ wrapper เดียว สลับเฉพาะเนื้อใน

### B6. รูปทรง state ที่กระจัดกระจาย

`{ status, data, error }` เกิดซ้ำ 4 ชุดใน `AppProvider.jsx` → ทำ helper เล็กๆ ร่วมกัน
(`idle()` / `loading(prev)` / `ready(data)` / `failed(msg)`) แล้วใช้ให้ทั่ว
ส่วน `ConfusionMatrix.jsx` แหวกไปใช้ union แปลกๆ (`setSamples("loading")`, `setSamples({ error })`)
จนต้องเช็ก `samples && !samples.error && samples !== "loading"` → ปรับให้ใช้รูปทรงเดียวกับที่อื่น

### B7. เจาะทะลุ encapsulation ของ `PlotlyChart`

`Dashboard/XrayCard.jsx` ใช้ `document.getElementById("xrayChart")` + `Plotly.restyle` เพื่ออัปเกรด
เส้นฟลักซ์เป็นข้อมูลรายนาที ทั้งที่ ref ของ node เป็นของ `PlotlyChart`
→ ให้ `PlotlyChart` เปิด API ออกมาอย่างตั้งใจ (`forwardRef` + `useImperativeHandle` คืน node
หรือเมธอด `restyle()`) แล้ว `XrayCard` เรียกผ่านนั้น

### B8. ชื่อที่ไม่ตรงกับสิ่งที่เป็น

- `AppProvider.jsx` — `selectHarp: selectHarpInternal` (และพวก `*Internal` อื่นๆ) ชื่อบอกว่า internal
  แต่เป็น public API ที่ export ออกไปทั้งดุ้น → ตัดคำว่า Internal ทิ้ง
- `Dashboard/FrameCard.jsx` — ตัวแปร `dice` เก็บ **สตริง** `" · Dice เทียบ mask จริง …"` ไม่ใช่ตัวเลข
  → เปลี่ยนชื่อเป็น `diceText` หรือย้ายไปคำนวณใน JSX

### B9. หน้าที่ปนกันใน AppProvider

`exportFlaresCsv()` สร้าง `<a>` แล้วสั่งดาวน์โหลด — เป็นงาน DOM ที่ไปอยู่ในตัวเก็บ state
→ แยกส่วนสร้างสตริง CSV เป็น pure function (เช่น `lib/csv.js`) แล้วให้ `XrayCard` เป็นคนสั่งดาวน์โหลด

---

## C. ห้ามทำ (สำคัญ — กันแก้ถอยหลัง)

1. **ห้ามเอาโหมด "dense" ของกราฟ X-ray ออก** (`DENSE_EVENT_THRESHOLD` ใน `lib/plotConstants.js`
   และการกรองเหลือ M/X ใน `XrayCard.jsx`) — รีวิวแกน Spec ตั้งธงว่าเป็น scope creep เพราะไปเทียบกับ
   commit `3c08d97` แต่จริงๆ เป็นงานที่ **มีอยู่แล้วใน working tree ก่อนการแปลง** และถูก port มาตามนั้น
   ทุกบรรทัด การเอาออกคือการถอยงานที่เจ้าของตั้งใจทำ
2. **ห้ามแตะ `backend/app/routers/goes.py`** (`limit` default = 20000) — งานค้างเดิม ไม่เกี่ยวกับรีวิวนี้
3. **ห้ามแก้ `frontend/src/styles.css`** — เป็นไฟล์เดิมที่ยกมาทั้งก้อน ชื่อคลาสทุกตัวผูกกับ JSX อยู่
4. **ห้ามรื้อการแยกคอนเทนเนอร์** — `backend/Dockerfile` + `frontend/Dockerfile` (nginx) +
   `docker-compose.yml` สองเซอร์วิส, พอร์ต 3000 (frontend) / 8000 (backend), nginx proxy `/api`,
   และ `base: "/"` ใน `vite.config.js` ทั้งหมดเป็นผลลัพธ์ที่ตั้งใจของงานล่าสุด
5. **ห้ามเอาการเสิร์ฟไฟล์ static กลับเข้า FastAPI** — `backend/app/main.py` เป็น API ล้วนแล้ว
6. **ห้ามเปลี่ยนไลบรารีกราฟ / ห้าม bundle Plotly เข้ามา** เว้นแต่จะถูกสั่งชัดเจน
7. **ห้ามแก้ข้อความไทยบน UI** นอกจากข้อ A6 ที่ระบุให้คืนค่าเดิม
8. **ห้าม `git commit` หรือ `git push`** เว้นแต่จะถูกสั่ง

---

## D. วิธียืนยันว่าแก้แล้วใช้ได้จริง

**1. build ต้องผ่าน**

```bash
cd frontend && npm run build
```

**2. รันเพื่อทดสอบด้วยตา** — เลือกอย่างใดอย่างหนึ่ง

```bash
# ก) โหมดพัฒนา (ต้องใช้โหมดนี้ถ้าจะทดสอบฟีเจอร์ proton — Docker ไม่มีคลังโปรตอน)
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --port 8000   # แยกเทอร์มินัล
cd frontend && npm run dev                                                        # http://localhost:5173

# ข) โหมดใกล้เคียง production (สองคอนเทนเนอร์)
docker compose up --build      # frontend http://localhost:3000 · API http://localhost:8000/docs
```

**3. เช็กลิสต์ในเบราว์เซอร์** (console ต้องไม่มี error สักตัว)

- [ ] โหลดหน้าแรก: ตัวเลข hero ทั้งสามช่องขึ้นค่าจริง ไม่ค้างที่ "—" และไม่มีคำว่า NaN
- [ ] **A1** คลิกจุด flare บนจาน → ตรึง · กวาดเมาส์ผ่านจุดอื่น กราฟโปรตอนไม่เปลี่ยน · คลิกจุดเดิมซ้ำ → ปลดตรึง
- [ ] **A2** กด "แสดงข้อมูล" ซ้ำๆ กราฟ X-ray/จานสุริยะ/ตัวเลข hero ไม่กะพริบเป็นสถานะว่าง
- [ ] **A3** (ต้องมี `SUNSEG_PROTON_DIR`) การ์ด "Proton flux รอบเวลาที่เกิด flare" มองเห็นได้จริง
- [ ] **A5** สลับ layer pill ไปช่องที่เฟรมนั้นไม่มีข้อมูล → เห็นคำขอ `/api/segment` รอบเดียวใน Network tab
- [ ] ตาราง Active Region: ค้นหา/เรียงคอลัมน์/คลิกแถว แล้ว risk gauge + attention chart อัปเดตตาม
- [ ] Confusion matrix: ลาก threshold, สลับ LSTM/baseline, สลับ val/test, คลิกช่องดูรายการ sample,
      คลิกแถว sample แล้วกระโดดกลับไป dashboard ได้
- [ ] ปุ่ม preset ช่วงเหตุการณ์สำคัญ, ปุ่มส่งออก CSV, timelapse (Spacebar) และลูกศรซ้าย/ขวาเลื่อนเฟรม

**4. เทียบกับของเดิมเมื่อไม่แน่ใจ:** `git show 3c08d97:frontend/app.js` คือแหล่งความจริงของพฤติกรรม
(ยกเว้นโหมด dense ตามข้อ C1)

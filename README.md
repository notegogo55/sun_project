# sunseg — Solar Active Region Segmentation, Tracking & Flare Forecasting

ระบบ end-to-end สำหรับติดตามสภาพอวกาศ (space weather) ประกอบด้วย 3 ส่วนที่ทำงานต่อกัน:

| ส่วน | โมเดล | ทำอะไร |
|---|---|---|
| **Segmentation** | U-Net (PyTorch) | แบ่งส่วน active region จากภาพ magnetogram เต็มดวงของ SDO/HMI |
| **Tracking** | Hungarian + differential rotation | ติดตาม AR แต่ละดวงข้ามเวลา โดยชดเชยการหมุนของดวงอาทิตย์ |
| **Forecasting (โมเดลหลัก)** | **LSTM + V3** (PyTorch, ensemble 25 seed × 2 ระดับ) | ทำนาย**ระดับคลาส**ของ flare ที่แรงที่สุดใน 24 ชม. ถัดไป — **<M / M / X** — จาก SHARP + ความเข้มแสง AIA + X-ray ทุก 12 ชม. |
| **Forecasting (รายชั่วโมง)** | LSTM · TCN · Transformer · DA-RNN (PyTorch) | ความเสี่ยง flare ≥M1.0 ภายใน 24 ชม. จาก 18 SHARP ทุก 1 ชม. — สี่สถาปัตยกรรมบนข้อมูลชุดเดียวกัน สลับดูได้ในหน้าเว็บ |

**เป้าหมายของงานวิจัย: เปรียบเทียบโมเดลพยากรณ์เพื่อหาตัวที่ให้ผลดีที่สุด** — โมเดลหนึ่งตัวคือคู่
(สถาปัตยกรรม, ชุด feature) ซึ่งทั้งหมด 16 ตัว (4 สถาปัตยกรรม × 4 ชุด feature) จัดอันดับด้วย TSS เฉลี่ยบน test
ชุด feature (SHARP ล้วน / + ความเข้มแสง AIA / + X-ray / ทั้งหมด) เป็นตัวเลือกหนึ่งของโมเดลเหมือน hyperparameter
ไม่ใช่คำถามวิจัยแยก (เปลี่ยนจากคำถามเดิม "ข้อมูลหลายความยาวคลื่นช่วยไหม" เมื่อ 2026-09-24 — ผลของคำถามเดิม
ยังอยู่ในภาคผนวกของรายงานผล)

**โมเดลหลักของโปรเจค (ตั้งแต่ 2026-09-24) คือ LSTM + V3** — อันดับ 1 ของการเปรียบเทียบ ใช้ทำนายระดับคลาส
<M / M / X (ดูหัวข้อ "โมเดลหลัก" ในผลการทดลอง)

ผลลัพธ์ทั้งหมดแสดงผ่าน **webapp (FastAPI + React)**

---

## ความต้องการของระบบ

- Python **3.12** (PyTorch ยังไม่รองรับ 3.13+)
- NVIDIA GPU (ทดสอบบน GTX 1650 4 GB — config ถูกปรับมาให้พอดีกับ VRAM เท่านี้)
- พื้นที่ดิสก์ว่างประมาณ 10 GB
- **อีเมลที่ลงทะเบียนกับ JSOC** สำหรับดาวน์โหลดข้อมูล → [ลงทะเบียนที่นี่](http://jsoc.stanford.edu/ajax/register_email.html)

## การติดตั้ง

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1

# PyTorch ต้องติดตั้งแยกจาก index ของ CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ส่วนที่เหลือ + ตัวโปรเจคเอง (pyproject.toml อยู่ใน backend/)
uv pip install -e "./backend[dev]"

# ตั้งค่าอีเมล JSOC
Copy-Item .env.example .env    # แล้วแก้ SUNSEG_JSOC_EMAIL ในไฟล์
```

ตรวจสอบว่า CUDA ใช้ได้:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## ขั้นตอนการใช้งาน

```powershell
# 1) ดึงข้อมูล metadata (เร็ว — ไม่กี่สิบ MB): SHARP keywords + flare catalog + ตาราง HARP-NOAA
python backend/scripts/data/download_metadata.py

# 2) สร้าง sequences + labels สำหรับโมเดลพยากรณ์ (ทุกตัวใช้ชุดเดียวกัน)
python backend/scripts/data/build_sequences.py

# 3) เทรนโมเดลพยากรณ์ พร้อมเทียบกับ logistic-regression baseline ทุกครั้ง
#    ชื่อโมเดลและ hyperparameter อยู่ที่ backend/configs/forecast.yaml (lstm, tcn, transformer, darnn)
#    แต่ละตัวเขียนไฟล์ของตัวเอง: artifacts/models/<ชื่อ>.pt, artifacts/metrics/<ชื่อ>.json และ
#    artifacts/metrics/<ชื่อ>_predictions.parquet (แผง Confusion Matrix ในหน้าเว็บต้องใช้ — ดูหัวข้อ "ตัวชี้วัด")
python backend/scripts/forecast/train.py --model lstm          # ไม่กี่นาที
python backend/scripts/forecast/train.py --model all           # ทั้งสี่ตัวเรียงกันทีละตัว (DA-RNN ช้าสุด)

# 4) ดาวน์โหลดภาพ + สร้าง mask สำหรับ U-Net
#    ต้องมีอีเมล JSOC ใน .env  ·  ทดสอบด้วยไม่กี่เฟรมก่อนเสมอ
python backend/scripts/data/download_images.py --start 2014-10-20 --end 2014-10-28 --limit 5
python backend/scripts/segmentation/plot_masks.py          # << ตรวจด้วยตาก่อน! (ดูหมายเหตุด้านล่าง)
python backend/scripts/data/download_images.py     # แล้วค่อยดึงเต็มช่วง (เร็วขึ้นมาก — ดูหมายเหตุด้านล่าง)

# 5) เทรน U-Net
python backend/scripts/segmentation/train.py --overfit-one-batch   # ตรวจสุขภาพโมเดลก่อน
python backend/scripts/segmentation/train.py

# 6) ดึงภาพ AIA สามชั้นบรรยากาศ (ไม่บังคับ แต่ทำให้หน้าเว็บสลับชั้นได้)
python backend/scripts/data/download_aia.py --wcs-only          # ดึง WCS ของเฟรมก่อน (เร็ว)
python backend/scripts/data/download_aia.py --limit 5           # ทดสอบไม่กี่เฟรมก่อนเสมอ
python backend/scripts/checks/plot_aia_alignment.py               # << ตรวจด้วยตาก่อน! (ดูหมายเหตุด้านล่าง)
python backend/scripts/data/download_aia.py                     # แล้วค่อยดึงเต็มช่วง (~45 นาที)

# 6.5) ตำแหน่ง flare สำหรับแผนที่หน้าแรก — ยืมพิกัดจาก PositionFlare มาแปะให้ flare ของแคตตาล็อกโมเดล
#      (ต้องมี flares.parquet จากขั้นตอน 1 และ CSV ของ PositionFlare — ดูหัวข้อ "ตำแหน่ง flare" ด้านล่าง)
python backend/scripts/data/build_flare_positions.py

# 6.7) feature เพิ่มเติมของงานเปรียบเทียบ: ฟลักซ์ X-ray ทั้งช่วง + ความเข้มแสง AIA ราย HARP
#      (ต้องมีภาพจากขั้นตอน 4 + 6 และ U-Net จากขั้นตอน 5 — ดู "ฟลักซ์ X-ray" และ "ภาพ AIA" ด้านล่าง)
python backend/scripts/data/download_xray.py --satellite g15               # 2011-2017
python backend/scripts/data/download_xray.py --satellite g16 --start 2017-01-01
python backend/scripts/study/extract_intensity.py --limit 20               # ทดสอบก่อน
python backend/scripts/study/extract_intensity.py                          # รันต่อจากที่ค้างได้

# 6.8) ผลหลัก: เปรียบเทียบ 16 โมเดล (4 สถาปัตยกรรม × V0-V3 · 25 seed ต่อตัว)
python backend/scripts/study/build_dataset.py        # -> data/processed/study_sequences/ (cadence 12 ชม.)
python backend/scripts/study/train.py                # -> artifacts/model_comparison/report.md (รันต่อได้)
python backend/scripts/study/plot_figures.py         # รูปประกอบรายงาน -> artifacts/model_comparison/figures/
python backend/scripts/study/persistence_baseline.py # baseline แบบ persistence -> persistence_baseline.md

# 6.9) โมเดลหลัก LSTM + V3 แยกระดับ <M/M/X
#      ระดับ M: เซลล์ lstm/V3 จากขั้นตอน 6.8 · ระดับ X: เซลล์เดียวกันที่เทรนด้วย label ≥X1.0
python backend/scripts/study/build_dataset.py --positive-class X1.0 --out-dir data/processed/study_sequences_x
python backend/scripts/study/train.py --data-dir data/processed/study_sequences_x --out-dir artifacts/model_comparison_x `
    --variants V3 --architectures-file <yaml ที่มีแค่ block lstm ของ configs/study/architectures.yaml>
python backend/scripts/study/class_forecast.py   # รายงาน -> artifacts/class_forecast/report.md

# 7) เปิด webapp — backend (FastAPI, backend/app) กับ frontend (React + Vite, frontend/)
#    เป็นคนละ process/container เสมอ ต้องรันคู่กัน
uvicorn app.main:app --app-dir backend --reload &   # backend: http://localhost:8000 (API docs ที่ /docs)
cd frontend && npm install && npm run dev            # frontend: http://localhost:5173 (proxy /api ให้เอง)
```

> **Docker (แนะนำถ้าจะรันแบบ deploy จริง)**: `docker compose up --build` สร้างและรันสองคอนเทนเนอร์
> แยกกันชัดเจน — `backend` (FastAPI พอร์ต 8000) กับ `frontend` (React build เสิร์ฟด้วย nginx พอร์ต
> 3000, proxy `/api`/`/docs` ไปที่ `backend` ให้เอง) เปิด http://localhost:3000 ดูรายละเอียดที่
> `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`

> **ไม่จำเป็นต้องทำขั้นตอน 1–6 ให้ครบก่อนถึงจะเปิดเว็บได้** — แอปออกแบบให้เปิดได้เสมอแม้ยังไม่มี
> โมเดลหรือข้อมูลบางส่วน ส่วนที่ยังไม่พร้อมจะขึ้นสถานะ "ยังไม่พร้อม" พร้อมบอกว่าต้องรันสคริปต์ไหนต่อ
> แทนที่จะพังทั้งหน้า จะรัน `uvicorn` (ขั้นตอนที่ 7) ทันทีหลังติดตั้งเสร็จก็ได้

> **ทำไม `download_images.py` เร็วขึ้นมาก**: แต่เดิม export ภาพเต็มดวง + SHARP bitmap ด้วย
> `protocol="fits"` ต้องเข้าคิว export ของ JSOC ทุกคำขอ (~20-60 วิ/คำขอ วัดจากงานจริง: ดาวน์โหลด
> 5,112 เฟรมช่วง 2011-2017 ที่ cadence 12 ชม. กิน ~9 วัน) ตอนนี้ใช้ `method="url_quick"`,
> `protocol="as-is"` แทน (`JsocClient.export_fast`) ซึ่ง**ไม่เข้าคิวเลย** (`request.id` เป็น `None`
> เสมอ — วัดจริง ~2-20 วิ/คำขอ) แล้วต่อ WCS เข้า header เองจาก keyword query แยกต่างหาก
> (`frame_wcs.fetch_frame_wcs`/`fetch_sharp_wcs`) ด้วยเทคนิคเดียวกับที่ `--wcs-only` ของขั้นตอน 6
> ใช้อยู่แล้ว — ค่าพิกเซลและ keyword ที่ได้เหมือนกันทุกประการเพราะมาจาก DRMS record เดียวกัน
> (JSOC แค่เขียน keyword ลง header ให้ตอน `protocol="fits"` ไม่ได้คำนวณอะไรใหม่) ตรวจแล้วว่า
> จำนวน patch ที่ใช้และ % mask coverage ตรงกับวิธีเดิมเป๊ะในเฟรมทดสอบ

> **ห้ามข้ามขั้นตอน `plot_masks.py`** — mask ถูกสร้างโดยแปลงพิกัดจาก SHARP patch ไปยังภาพเต็มดวงผ่าน WCS
> ถ้าการแปลงผิด mask จะเลื่อนไปจากตำแหน่งจริงทั้งภาพ โมเดลจะยังเทรนได้และ loss จะลดลงสวยงาม แต่เรียนรู้
> สิ่งที่ผิด สิ่งที่ต้องเห็นในภาพคือ **เส้นขอบสีส้มล้อมรอบบริเวณสนามแม่เหล็กเข้ม** (จุดขาว/ดำจัด)

> **ห้ามข้ามขั้นตอน `plot_aia_alignment.py`** ด้วยเหตุผลเดียวกัน — WCS ที่ผิดจะให้ภาพ AIA เต็มดวง
> ที่สวยงามพร้อมเส้นขอบในตำแหน่งที่ดูน่าเชื่อถือ แต่ตัวเลขความเข้มแสงราย AR ที่ได้กลับถูกสุ่มมาจาก
> quiet Sun ทั้งหมด สิ่งที่ต้องเห็นคือ **จุดสว่าง (plage) ของ 1600 Å ทับพิกเซลสนามแม่เหล็กเข้มพอดี**
> สคริปต์ยังตรวจเชิงตัวเลขให้ด้วย โดยเทียบ contrast ของภาพปกติกับภาพที่หมุน 180° แล้ว fail ถ้าภาพหมุนชนะ

## เครื่องมือตรวจสอบ

```powershell
python backend/scripts/checks/check_env.py           # ไลบรารีครบ, CUDA ใช้ได้, config ถูกต้อง
python backend/scripts/checks/check_connectivity.py  # เข้าถึง JSOC / NGDC / HEK ได้หรือไม่
python backend/scripts/checks/inspect_flares.py      # คุณภาพของ label — ดูอัตราการเก็บ flare M+
python backend/scripts/checks/check_api.py           # ยิง API จริงผ่าน HTTP (ต้องเปิดเซิร์ฟเวอร์ก่อน)
python backend/scripts/checks/plot_aia_alignment.py  # ภาพ AIA วางทับกริดของเฟรมตรงตำแหน่งหรือไม่
```

> **ทำไมโมเดลพยากรณ์ก่อน U-Net?** โมเดลพยากรณ์ใช้แค่ข้อมูลตาราง (ไม่กี่ร้อย MB) เทรนเสร็จในไม่กี่นาที
> จึงยืนยันได้เร็วว่า labeling pipeline ถูกต้อง ก่อนจะลงทุนหลายชั่วโมงไปกับการดาวน์โหลดภาพ

## เปิด webapp ด้วย Docker (ไม่ต้องติดตั้ง Python/Node เอง)

ใช้ตอนแค่อยาก **ดู** ผลบนเครื่องอื่น — สอง container นี้ให้บริการ webapp อย่างเดียว ไม่ใช้เทรนโมเดล
(เทรนต้องทำบนเครื่องที่มี GPU ตามขั้นตอนด้านบนก่อน แล้วโมเดล/ข้อมูลถูก mount แบบอ่านอย่างเดียว
เข้า container ทำให้เปลี่ยนโมเดลได้โดยไม่ต้อง build image ใหม่):

```powershell
docker compose up --build
# เปิด http://localhost:3000  (frontend — nginx proxy /api ไปที่ backend ให้เอง)
# เอกสาร API อยู่ที่ http://localhost:8000/docs (backend โดยตรง)
```

`docker compose ps` ควรเห็นสอง container: `sunseg-backend` (FastAPI, พอร์ต 8000) กับ
`sunseg-frontend` (nginx เสิร์ฟ React build, พอร์ต 3000) — แยกกันชัดเจน build/scale/restart
ได้อิสระจากกัน browser คุยกับ frontend container อย่างเดียว ส่วน frontend เป็นคน proxy
`/api` ต่อไปที่ backend ผ่าน Docker network ภายใน (ดู `frontend/nginx.conf`)

---

## แหล่งข้อมูล

| แหล่ง | ใช้ทำอะไร |
|---|---|
| JSOC `hmi.sharp_cea_720s` | SHARP magnetic parameters — features ของโมเดลพยากรณ์ (พิกัด CEA แก้ผลการฉายแล้ว) |
| JSOC `hmi.sharp_720s` | `bitmap` segment — ground-truth mask ของ U-Net (พิกัด CCD ตรงกับภาพเต็มดวง) |
| JSOC `hmi.M_720s` | ภาพ magnetogram เต็มดวง — input ของ U-Net |
| [คลัง AIA synoptic](https://jsoc1.stanford.edu/data/aia/synoptic/) | ภาพ AIA 6 ช่อง — เลเยอร์ชั้นบรรยากาศในหน้าเว็บ + feature ความเข้มแสงของ V1/V3 (ดูด้านล่าง) |
| [NGDC GOES XRS reports](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/) | รายการ flare — labels ของโมเดลพยากรณ์ ช่วง 2011–2017 (คลังครอบคลุม 1975–2017) |
| HEK | รายการ flare ช่วงหลัง 2017 — `--flare-source auto` (ค่าเริ่มต้น) สลับมาใช้ให้เองเมื่อเกินคลังของ NGDC |
| [ตาราง HARPNUM↔NOAA](http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt) | เชื่อม SHARP เข้ากับ flare catalog |
| คลัง GOES particle ราย 5 นาที (ภายนอก) | ฟลักซ์โปรตอนรอบเวลาที่เกิด flare — แผง "Proton flux" ในหน้าเว็บ (ดูด้านล่าง) |
| PositionFlare `flares_all_cycles.csv` (ภายนอก) | ตำแหน่ง flare บนแผนที่หน้าแรก — จับคู่เข้ากับแคตตาล็อกโมเดลด้วยเวลาพีค (ดูด้านล่าง) |
| NOAA NCEI `xrsf-l2-avg1m_science` (GOES-15 และ GOES-16/17/18/19) | ฟลักซ์ X-ray ต่อเนื่องรายนาที — เส้น "GOES X-Ray" ในแดชบอร์ด + feature X-ray ของ V2/V3 (ดูด้านล่าง) |

### ตำแหน่ง flare บนแผนที่หน้าแรก (PositionFlare)

แผนที่ในหน้าแรกแสดง **flare ชุดเดียวกับที่โมเดลพยากรณ์ใช้ทำ label** (`data/interim/flares.parquet`
ระดับ C ขึ้นไป — คลาสและจำนวนดวงตรงกับแคตตาล็อกของโมเดลเป๊ะ) แต่แคตตาล็อกนั้นมีพิกัดแค่ปี ≤ 2017
(รายงาน NGDC) พิกัดจึงยืมมาจากแคตตาล็อกของโปรเจค PositionFlare (`flares_all_cycles.csv` —
SWPC > XRS > XRS-HPC > AR, ครอบ 1996-2026) โดย `build_flare_positions.py` จับคู่ทีละดวงด้วย
**เวลาพีค** (ใกล้สุดภายใน ±10 นาที, ฟลักซ์ต่างกันไม่เกิน 4 เท่า, หนึ่งดวงของ PositionFlare
จับคู่ได้ครั้งเดียว) แล้วเขียน `data/processed/flare_positions.parquet` ให้ API `/api/flare-positions`

| | |
|---|---|
| flare ของโมเดล (C+) | 11,394 ดวง · 2011 → 2025 |
| จับคู่ PositionFlare ได้ | 10,930 (95.9%) — เวลาพีคตรงกันเป๊ะ 10,551 คู่ |
| มีพิกัดบนแผนที่ | 10,637 (93.4%) — เดิม 3,327 ดวงจากรายงาน NGDC อย่างเดียว |

- **ไม่เทียบคลาสตอนจับคู่** เพราะสเกลต่างกัน: รายงาน NGDC เดิมยังคูณ 0.7 ของ GOES-13/15 ส่วน
  PositionFlare ใช้ค่า science (C1.0 ของโมเดล = C1.4 ใน PositionFlare) หน้าเว็บแสดงคลาสของโมเดล
  เป็นหลักและคลาสของ PositionFlare กำกับไว้
- flare ของ PositionFlare ที่ **ไม่อยู่** ในแคตตาล็อกโมเดล (~16,000 ดวงในช่วงเดียวกัน ส่วนใหญ่ปี
  2018-2023 ที่ไม่มีคู่ HARP) ไม่ขึ้นบนแผนที่โดยเจตนา
- CSV ของ PositionFlare อยู่นอก repo ตั้ง path ได้ที่ `position_flare.csv` ใน `configs/data.yaml`
  หรือ env `SUNSEG_POSITION_FLARE_CSV` — ใช้ตอนรันสคริปต์เท่านั้น ตัวแอปอ่านไฟล์ใน `data/processed`
  รันสคริปต์ซ้ำเมื่อแคตตาล็อกฝั่งใดฝั่งหนึ่งอัปเดต แล้วรีสตาร์ต backend

### ภาพ AIA สามชั้นบรรยากาศ (ไม่บังคับ)

magnetogram บอกได้แค่สนามแม่เหล็กที่ผิว แต่ active region แผ่ตัวขึ้นไปทั่วชั้นบรรยากาศ หน้าเว็บจึง
สลับ **เลเยอร์** ของภาพพื้นหลังได้ โดย mask จาก U-Net ทับอยู่ที่เดิมทุกเลเยอร์ ทำให้เห็นว่าโครงสร้าง
สนามแม่เหล็กเดียวกันให้ความร้อนกับพลาสมาชั้นบนแค่ไหน:

| ช่อง | ชั้นบรรยากาศ | อุณหภูมิลักษณะเฉพาะ | feature ของ V1/V3 |
|---|---|---|:---:|
| AIA 4500 Å | โฟโตสเฟียร์ (continuum) | ~5,000 K | ✓ |
| AIA 1600 Å | โฟโตสเฟียร์ / transition region | ~10,000 K | |
| AIA 304 Å | โครโมสเฟียร์ (He II) | ~50,000 K | ✓ |
| AIA 171 Å | โคโรนาสงบ (Fe IX) | ~600,000 K | ✓ |
| AIA 131 Å | โคโรนาร้อน / flare (Fe XXI) | ~10 MK | |
| AIA 94 Å | โคโรนาร้อน (Fe XVIII) | ~6 MK | |

feature ความเข้มแสงของงานเปรียบเทียบคือค่า p95 ราย HARP ของสามช่องที่ติ๊กไว้ สกัดด้วย
`study/extract_intensity.py` (U-Net ทำนาย mask → normalise ด้วย quiet Sun → จับคู่ HARP) ช่อง 4500 แทนที่ 1600
เพราะเป็นโฟโตสเฟียร์จริงที่ไม่ drift ตามรอบสุริยะ ส่วน 94/131 ใช้เฉพาะงานเสริม (`variants_v3_aia.yaml`)

```powershell
python backend/scripts/data/download_aia.py --wcs-only     # ดึง WCS ของเฟรม (เร็ว ไม่ต้องใช้อีเมล JSOC)
python backend/scripts/data/download_aia.py --limit 5      # ทดสอบก่อน
python backend/scripts/checks/plot_aia_alignment.py          # << ตรวจการจัดตำแหน่ง ห้ามข้าม
python backend/scripts/data/download_aia.py                # เต็มช่วง — ~1.2 GB, ราว 45 นาที (6 worker)
```

**ทำไมไม่ใช้คิว export ของ JSOC**: คลัง synoptic ให้ภาพผ่าน HTTP ธรรมดา และภาพในคลังเป็น
**level 1.5 อยู่แล้ว** (registered, de-rolled, `CDELT=2.4″/px`, 1024² จากการ bin 4×4) จึงไม่ต้อง
เพิ่ม dependency `aiapy` เข้ามาเลย เทียบกันแล้วประหยัดกว่ามาก — ~1.2 GB และไม่ถึงชั่วโมง เทียบกับ
~40 GB และ 10-20 ชั่วโมงถ้าใช้ `aia.lev1_euv_12s` ผ่านคิว export — ทางนั้นยังทำได้ถ้าคลัง synoptic
ใช้ไม่ได้ แต่ยังไม่ได้เขียนไว้ในสคริปต์ (ดูรายละเอียด series/segment ที่ต้องใช้ใน docstring ของ
`download_aia.py`)

**WCS ของเฟรมถูกดึงแยก**: `download_images.py` เก็บเฟรมเป็น `.npy` เปล่าแล้วลบ FITS ทิ้ง จึงไม่มี
WCS ติดมาด้วย `--wcs-only` จะ query *keyword* จาก JSOC (ไม่ต้องใช้อีเมล ไม่เข้าคิว) แล้ว cache ไว้ที่
`data/interim/frame_wcs.parquet` ค่าที่จำเป็นจริง ๆ คือ `CROTA2 ≈ 180` (HMI เก็บภาพในทิศ CCD ดิบ
ซึ่งกลับหัว), `CRPIX ≈ (2037.0, 2049.6)` และ `CDELT ≈ 0.5044″/px` — ค่าประมาณแบบเดิมคลาดจากนี้
มากพอที่จะทำให้ภาพ AIA วางผิดตำแหน่งทั้งภาพ

**กราฟความเข้มแสงตามเวลา**: การ์ดขวาของ dashboard วาดสามแถว (หนึ่งแถวต่อหนึ่งชั้นบรรยากาศ
เรียงจากโคโรนาลงมาโฟโตสเฟียร์ เหมือนภาพตัดขวางจริง) แต่ละเส้นคือ active region หนึ่งดวงที่ถูก
ติดตามข้ามเฟรมด้วย tracker ตัวเดียวกับ `/api/track` ค่าความเข้มแสงถูกแนบไปกับ detection **ก่อน**
ป้อนเข้า tracker จึงติดตามไปกับ AR ดวงนั้นได้ (endpoint: `/api/intensity-series`)

ความหนาแน่นของเส้นขึ้นกับ cadence ของเฟรมในช่วงที่เลือก:

| ช่วง | cadence | ผลที่ได้ |
|---|---|---|
| ชุดหลัก 2011-2017 | 7 วัน | เห็นแนวโน้มระยะยาว แต่ AR ที่อายุสั้นกว่า 7 วันจะหายไปทั้งดวง และการจับคู่ข้ามช่องว่างอาศัยการทำนายจากกฎการหมุนเป็นหลัก |
| พ.ค. 2024 | 12 ชม. | เส้นต่อเนื่อง ติดตาม AR ได้หลายสิบจุด — ช่วงที่เหมาะกับการดูวิวัฒนาการมากที่สุด |

API จะแนบ `note` อธิบายข้อจำกัดของช่วงที่เลือกมาด้วยเสมอ และหน้าเว็บแสดงข้อความนั้นให้เห็น

> **ค่าที่ได้เป็น DN/s แบบสัมพัทธ์** — หารด้วย `EXPTIME` แล้วเพื่อให้เทียบข้ามช่องได้ แต่ **ไม่ได้แก้
> instrument degradation** (ความไวของ AIA ลดลงตามปี) กราฟความเข้มแสงจึงใช้เทียบ AR **ภายในเฟรม
> เดียวกัน**เท่านั้น ห้ามนำไปเทียบข้ามปี

### ฟลักซ์ X-ray ต่อเนื่อง (ไม่บังคับ)

เส้น **GOES X-Ray** ในแดชบอร์ดค่าเริ่มต้นวาดจากรายการ flare (แค่ 3 จุดต่อเหตุการณ์: เริ่ม/peak/จบ)
ซึ่งได้เส้นเป็นหนามแหลม ๆ ไม่ใช่ฟลักซ์จริงที่ขึ้นลงต่อเนื่องแบบหน้า
[SWPC](https://www.swpc.noaa.gov/products/goes-x-ray-flux) — รันสคริปต์นี้เพื่อดึงฟลักซ์จริงรายนาที
จาก NOAA NCEI ฟลักซ์ชุดเดียวกันนี้ยังเป็น **feature X-ray ของ V2/V3** ในงานเปรียบเทียบด้วย
(median ของ log₁₀ ฟลักซ์ทั้งดวงในแต่ละช่วง 12 ชม.)

GOES-15 หยุดส่งข้อมูลปี 2020 ช่วงหลังจากนั้นต้องใช้ GOES-R series (g16 → g19 → g18 → g17 ตามลำดับที่ใช้แทนกัน)
ซึ่งอยู่คนละ URL root และสเกลต่างจาก g15 ราว 8% `sunseg.data.xray_flux` ปรับทุกดวงให้เข้าสเกลของ g15 ให้เอง
(ตัวคูณวัดจากช่วงที่ดาวเทียมทับซ้อนกัน — ดู `data/measure_xray_cross_calibration.py`)

```powershell
# ทดสอบด้วยเดือนเดียวก่อน (แนะนำ — g15 ช่วง 2011-2017 มีเกือบ 2,600 ไฟล์ ~150 MB)
python backend/scripts/data/download_xray.py --satellite g15 --start 2014-10-01 --end 2014-10-31

# ดึงเต็มช่วง
python backend/scripts/data/download_xray.py --satellite g15
python backend/scripts/data/download_xray.py --satellite g16 --start 2017-01-01
python backend/scripts/data/download_xray.py --satellite g19 --start 2024-01-01   # ช่วงที่ g16 ไม่มีไฟล์แล้ว
```

ไฟล์ถูกเก็บที่ `data/raw/xrs/` รันซ้ำได้ปลอดภัย (ข้ามไฟล์ที่มีอยู่แล้ว) ถ้ายังไม่ได้รันสคริปต์นี้
หรือช่วงวันที่ที่เลือกยังไม่ได้ดาวน์โหลด กราฟจะ fallback กลับไปใช้เส้นจากรายการ flare
โดยอัตโนมัติ — ไม่ได้บล็อกไม่ให้เปิดเว็บ

### ฟลักซ์โปรตอน (ไม่บังคับ)

แผง **Proton flux** ในหน้าเว็บอ่านจากคลัง GOES particle ราย 5 นาทีที่อยู่ **นอก repo** —
เป็นข้อมูลดิบเกือบ 800 MB ที่โปรเจคอื่นดูแลอยู่ sunseg แค่อ่านอย่างเดียว

ชี้ path ด้วย `SUNSEG_PROTON_DIR` ใน `.env` (หรือแก้ `proton.root` ใน `backend/configs/data.yaml`)
ให้ตรงกับโฟลเดอร์ที่มีโครงสร้าง `<ปี>/<ดาวเทียม>_<ปี>_5m_clean.txt` ถ้าไม่ตั้งและ path ปริยาย
ไม่มีอยู่จริง แอปจะเปิดได้ตามปกติแล้วซ่อนแผงนี้ไป — ส่วนอื่นไม่ได้พึ่งข้อมูลชุดนี้

ทำไมต้องมี: X-ray บอกว่า flare ปะทุแรงแค่ไหน แต่ไม่ได้บอกว่ามีอนุภาคเดินทางมาถึงโลกหรือเปล่า
flare ระดับ X บางดวงตามมาด้วยพายุอนุภาคระดับ S3 ส่วนอีกหลายดวงที่แรงพอ ๆ กันเงียบสนิท
(เช่น AR 12192 เมื่อ ต.ค. 2014 ที่ปล่อย X หลายดวงแต่แทบไม่มี SEP เลย)

### ข้อจำกัดที่ทราบของ label

รายงาน GOES XRS ของ NOAA **ระบุเลข NOAA AR ให้ flare เพียงราว 49%** (flare ที่ไม่มีการยืนยันตำแหน่ง
ทางแสงจะไม่มีทั้งพิกัดและเลข AR) flare เหล่านั้นถูกตัดทิ้งตอนจับคู่กับ HARP

ผลกระทบจริงน้อยกว่าตัวเลขนี้มาก เพราะ label คือ "**มี** flare M+ ในอีก 24 ชม. หรือไม่" — flare ที่หายไป
จะพลิก label ก็ต่อเมื่อไม่มี flare อื่นคลุมหน้าต่างเดียวกัน และ AR ที่ปะทุมักปะทุหลายครั้งติดกัน
ตรวจยืนยันได้จาก **positive rate สุดท้าย 2.32% (ข้อมูล 2011–2025) ซึ่งอยู่ในช่วงที่งานวิจัยรายงาน (1–5%)**

ใช้ `python backend/scripts/checks/inspect_flares.py` เพื่อดูตัวเลขนี้กับข้อมูลของคุณเอง

**flare ช่วงที่มาจาก HEK (2015 เป็นต้นไป) ตกหล่นมากกว่านี้** — `flares.parquet` สร้างก่อนแก้บั๊กอ่านเลข
NOAA AR ของ HEK (ทิ้ง flare ที่ AR = 0 และเลข AR 4 หลัก) ปี 2022 จับคู่ HARP ได้แค่ 22 จาก 193 ครั้งของ M+
`data/rebuild_flares_v2.py` สร้างฉบับแก้แล้วเป็น `data/interim/flares_v2.parquet` จาก cache โดยไม่แตะเครือข่าย
แต่ **ไม่เขียนทับไฟล์เดิม** เพราะผลทุกชุดในหัวข้อ "ผลการทดลอง" เทรนด้วย label จากไฟล์เดิม — งานใหม่เลือกใช้ได้ผ่าน
`study/build_dataset.py --flares-path data/interim/flares_v2.parquet`

## หลักการสำคัญ: การป้องกัน data leakage

โปรเจคนี้บังคับใช้กติกา 4 ข้อทุกจุดของ pipeline — ถ้าละเลย metric จะดูดีเกินจริงอย่างรุนแรง:

1. **แบ่ง train/val/test ตามเวลา *และ* ให้ HARP ไม่ซ้ำข้าม split** — sample จาก HARP เดียวกันมี autocorrelation สูงมาก การสุ่มแบ่งทำให้โมเดลเห็นภาพ AR ดวงเดียวกันทั้งใน train และ test
2. **กรอง `|LON| < 68°`** — SHARP parameters ไม่น่าเชื่อถือใกล้ขอบจาน เพราะ projection effect
3. **กรอง `QUALITY == 0`** — ตัด frame ที่มีปัญหาในการสังเกตการณ์
4. **คำนวณ normalization statistics จาก train set เท่านั้น**

## ตัวชี้วัด

Segmentation ใช้ IoU / Dice ส่วน forecasting ใช้ **TSS (True Skill Statistic)** เป็นหลัก:

```
TSS = TP/(TP+FN) − FP/(FP+TN)
```

TSS เป็นมาตรฐานของวงการ flare forecasting เพราะไม่ถูกหลอกโดย class imbalance
(positive rate จริงอยู่ที่ ~1–3% เท่านั้น — โมเดลที่ทายว่า "ไม่เกิด" ตลอดจะได้ accuracy 97% แต่ TSS = 0)

### Confusion Matrix แบบโต้ตอบได้ในหน้าเว็บ

ตาราง TSS/AUC/recall แถวเดียวในหัวข้อ "ผลการทดลอง" ด้านล่างบอกไม่ได้ว่าโมเดล **ผิดพลาดแบบไหน** —
หน้าเว็บจึงมีแผง confusion matrix ขนาด 2×2 ต่อจากตารางนั้น ลากสไลเดอร์ปรับ threshold แล้วเห็นจำนวน
นับและตัวชี้วัดขยับทันที พร้อมคลิกแต่ละช่องเพื่อดูรายชื่อ sample ที่อยู่ในช่องนั้น

ตารางเรียง **แถวคือสิ่งที่โมเดลทำนาย คอลัมน์คือสิ่งที่เกิดขึ้นจริง** (positive มาก่อน) ตามธรรมเนียม
contingency table ของวงการ space weather verification — ตรงข้ามกับที่ไลบรารี ML ทั่วไป (เช่น
scikit-learn) ซึ่งเรียงคอลัมน์เป็นค่าทำนาย แต่ละช่องมีทั้งชื่อทางสถิติและคำเรียกของวงการกำกับคู่กัน:

| | เกิดจริง: positive | เกิดจริง: negative |
|---|---|---|
| **ทำนาย positive** | TP · *hit* | FP · *false alarm* |
| **ทำนาย negative** | FN · *miss* | TN · *correct negative* |

สลับดูได้ระหว่างโมเดลที่เลือกอยู่ (ปุ่มเลือกโมเดลบนหัวหน้า) กับ logistic baseline — สไลเดอร์รีเซ็ตไปที่ threshold ของโมเดลที่เพิ่งสลับมา
เสมอ เพราะสองโมเดลมีสเกลความน่าจะเป็นคนละแบบ ห้ามใช้สไลเดอร์ตัวเดียวคุมสองโมเดลพร้อมกัน — และ
สลับดูได้ระหว่างชุด val กับ test

> **split `val` ในแผงนี้คือชุดที่ใช้เลือก threshold ตอนเทรน ไม่ใช่ตัวเลขที่ควรอ้างเป็นผลงาน** — ตัวเลข
> ที่รายงานผลจริงต้องดูที่ `test` เสมอ (ดูหัวข้อ "หลักการสำคัญ: การป้องกัน data leakage" ด้านบน)

คลิกช่องใดช่องหนึ่ง (เช่น *miss* เพื่อดู flare ที่โมเดลพลาดทั้งหมด) จะเห็นรายชื่อ HARP, NOAA AR,
เวลาออกพยากรณ์ และค่าความน่าจะเป็นของ sample ในช่องนั้น เรียงจากมั่นใจที่สุดไปน้อยที่สุด คลิก sample
หนึ่งแถวจะเลื่อนขึ้นไปที่ dashboard ด้านบนพร้อมเลือก AR ดวงนั้นและขยับช่วงวันที่ให้ครอบเวลานั้นให้
อัตโนมัติ (threshold ที่ลากเล่นในแผงนี้เป็นของแผงเองเท่านั้น ไม่กระทบ threshold ที่ dashboard ใช้พยากรณ์จริง)

> **ต้องรันขั้นตอนที่ 3 (`forecast/train.py --model <ชื่อ>`) ซ้ำหนึ่งครั้ง** ถึงจะเห็นแผงนี้ของโมเดลนั้น — สคริปต์เทรนบันทึกค่าทำนาย
> ราย sample ของ val/test ไว้ที่ `artifacts/metrics/<ชื่อ>_predictions.parquet` เพิ่มจากไฟล์ตัวชี้วัดเดิม
> (LSTM ที่เทรนก่อนระบบรองรับหลายโมเดลเขียนไว้ที่ `artifacts/metrics/predictions.parquet` — แอปยังอ่านไฟล์นั้นให้)
> ถ้ายังไม่มีไฟล์นี้ (เช่น checkout checkpoint เก่าที่เทรนไว้ก่อนหน้านี้มา) แผงจะขึ้นข้อความบอกวิธีแก้
> แทนตาราง — เป็นพฤติกรรมที่ตั้งใจ ไม่ใช่ bug ส่วนที่เหลือของหน้าเว็บยังเปิดใช้งานได้ตามปกติ

การนับ TP/FP/TN/FN ทั้งหมดทำฝั่ง Python (`sunseg.metrics.threshold_sweep`) ด้วยโค้ดชุดเดียวกับที่
คำนวณตัวเลขในไฟล์ `<ชื่อโมเดล>.json` — สไลเดอร์บนหน้าเว็บแค่เปิดตารางที่คำนวณไว้ล่วงหน้าตามตำแหน่ง ไม่มี
การนับซ้ำฝั่ง JavaScript ตัวเลขในแผงนี้จึงตรงกับไฟล์ตัวชี้วัดเสมอที่ threshold เดียวกัน

## ผลการทดลอง

ข้อมูล production 2011–2025 · 493,174 sample · 4,872 HARP · positive 2.32% · แบ่ง HARP-disjoint

| | n | positive | |
|---|---|---|---|
| train | 384,838 | 5,382 (1.40%) | 2011-01 → 2023-12 |
| val | 54,178 | 4,312 (7.96%) | 2024-01 → 2024-12 |
| test | 54,158 | **1,766 (3.26%)** | 2025-02 → 2025-12 |

### พยากรณ์ flare ≥M1.0 ใน 24 ชม. (โมเดลในระบบ — seed เดียวต่อโมเดล)

ทั้งสี่ตัวใช้ hyperparameter จากการค้นหาด้วยงบเท่ากัน (`configs/forecast.yaml`, ตั้งแต่ 2026-09-23)

| โมเดล | พารามิเตอร์ | val TSS | **test TSS** | test HSS2 | test BSS | test AUC | test recall | test precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 7,265 | 0.796 | **0.756** | 0.262 | −0.019 | 0.939 | 0.891 | 0.182 |
| TCN | 16,121 | 0.794 | **0.754** | 0.262 | −0.184 | 0.941 | 0.889 | 0.182 |
| Transformer | 5,481 | 0.805 | **0.745** | 0.245 | −0.135 | 0.940 | 0.891 | 0.170 |
| DA-RNN | 3,849 | 0.805 | **0.691** | 0.254 | +0.155 | 0.936 | 0.818 | 0.179 |
| Logistic regression (ค่า ณ เวลาเดียว) | 19 | 0.807 | **0.686** | 0.288 | −1.124 | 0.943 | 0.790 | 0.205 |

ที่มา: `artifacts/metrics/<ชื่อ>.json` · โมเดลลำดับเวลาทั้งสี่ตัวได้ TSS สูงกว่า logistic regression แต่ AUC
ไม่สูงกว่า — ความสามารถในการจัดอันดับความเสี่ยงใกล้เคียงกัน ผลต่างของ TSS มาจาก threshold และรูปร่างของการ
กระจายความน่าจะเป็นเป็นหลัก ตารางนี้มี seed เดียวต่อโมเดล (ความผันผวนข้าม seed ของ TSS ราว 0.013–0.032)
จึง**ใช้จัดอันดับไม่ได้** การจัดอันดับโมเดลอยู่ที่ผลหลักด้านล่าง

### ผลหลัก: อันดับของโมเดลทั้ง 16 ตัว (4 สถาปัตยกรรม × 4 ชุด feature · 25 seed ต่อตัว)

`artifacts/model_comparison/report.md` (รูป `figures/ranking.png`) — dataset ของงานเปรียบเทียบ (cadence 12 ชม. ·
8 timestep) · test ปี 2025 3,098 sample (positive 155)

| อันดับ | โมเดล | test TSS (mean ± SD) | อันดับบน validation |
|---:|---|---:|---:|
| **1** | **LSTM + 18 SHARP + intensity + X-ray (V3)** | **0.747 ± 0.016** | 13 |
| 2 | LSTM + 18 SHARP + X-ray (V2) | 0.746 ± 0.032 | 9 |
| 3 | TCN + 18 SHARP + intensity + X-ray (V3) | 0.746 ± 0.016 | 4 |
| 4 | LSTM + 18 SHARP + intensity (V1) | 0.744 ± 0.013 | 14 |
| 5 | LSTM + 18 SHARP (V0) | 0.742 ± 0.029 | 8 |
| … | (ครบ 16 อันดับใน `report.md`) | … | … |
| 16 | DA-RNN + 18 SHARP + X-ray (V2) | 0.697 ± 0.028 | 6 |

- **โมเดลที่ดีที่สุดคือ LSTM + V3** (TSS 0.747) แต่ **13 จาก 15 ตัวที่เหลือแยกจากอันดับ 1 ไม่ได้** (ห่างไม่เกิน SD
  ของผลต่างแบบจับคู่ seed) — แยกได้จริงมีแค่ DA-RNN + V3 และ DA-RNN + V2 ทั้ง 16 ตัวอยู่ในช่วง 0.697–0.747
- **อันดับบน validation กับ test ไม่สอดคล้องกัน** (Spearman ρ = −0.20) — ถ้าเลือกด้วย validation จะได้
  Transformer + V0 ซึ่งอยู่อันดับ 13 บน test ความต่างระหว่างอันดับต้น ๆ จึงเป็น noise เป็นส่วนใหญ่
- ดีที่สุดของแต่ละสถาปัตยกรรม: LSTM 0.747 (V3) · TCN 0.746 (V3) · Transformer 0.739 (V1) · DA-RNN 0.728 (V1) ·
  เฉลี่ยข้ามชุด feature LSTM 0.745 · TCN 0.737 · Transformer 0.729 · DA-RNN 0.718
- DA-RNN ได้ BSS สูงสุดชัดเจนทุกชุด feature (0.165–0.198) — ถ้าต้องการความน่าจะเป็นที่ calibrate ดีกว่า
  ไม่ใช่ TSS DA-RNN คือตัวเลือกที่ดีกว่า
- **เทียบกับ baseline แบบ persistence** (`study/persistence_baseline.py` -> `artifacts/model_comparison/persistence_baseline.md`):
  persistence คลาสสิก (≥M1.0 ใน 24 ชม. ที่ผ่านมา) ได้ test TSS 0.514 แต่ HSS2 0.527 / precision 0.565 สูงกว่าแบบจำลอง ·
  ที่อัตราเตือนผิดเท่ากัน แบบจำลองจับได้ไม่มากกว่า (recall 0.513 ± 0.083 vs 0.535) · **กฎ "มี flare ≥C5.0 จากบริเวณ
  เดียวกันใน 48 ชม. ที่ผ่านมา" ที่เลือกบน validation ได้ test TSS 0.797 · HSS2 0.460 · precision 0.348 ชนะ LSTM + V3
  ทุก seed** (ตระกูลกฎกำหนดหลังเห็นผลบน test บางส่วน — ดู docstring ของสคริปต์) · ประวัติ flare ของบริเวณนั้นเป็นข้อมูลที่
  แบบจำลองทั้ง 16 แบบไม่ได้ใช้
- **แบบจำลองระดับ X**: ใช้เป็นระดับ X ของโมเดลหลัก — ดูหัวข้อถัดไป
- ภาคผนวก (คำถามเดิม): การเติม intensity/X-ray ให้ผลต่าง −0.028 ถึง +0.016 เทียบกับ 18 SHARP ของสถาปัตยกรรม
  เดียวกัน ซึ่งเล็กกว่า SD ของตัวเองทุกค่า

**ข้อควรระวังเพิ่มเติม**: `BSS` ของโมเดลส่วนใหญ่ติดลบหรือใกล้ศูนย์ แปลว่า *ค่าความน่าจะเป็นที่โมเดลให้มายังไม่
calibrate* — เป็นผลโดยตรงจาก focal loss และ `class_weight="balanced"` ที่จงใจบิดสเกลความน่าจะเป็นเพื่อดัน recall
ควรตีความผลลัพธ์เป็น **การจัดอันดับความเสี่ยง** ไม่ใช่ความน่าจะเป็นที่แท้จริง (หากต้องการค่าที่ตีความได้ ให้เพิ่ม
ขั้นตอน isotonic/Platt calibration บน validation set)

### โมเดลหลัก: LSTM + V3 แยกระดับคลาส <M / M / X

`artifacts/class_forecast/report.md` (สร้างด้วย `study/class_forecast.py` · ตัวเลขชุดเดียวกับหน้าเว็บ) — สองแบบจำลอง
ทวิภาคของเซลล์ LSTM + V3 (label ≥M1.0 และ ≥X1.0, 25 seed ต่อระดับ) รวมเป็นระดับเดียว: ระดับที่ทำนาย = ระดับสูงสุดที่
ensemble (seed เกินครึ่ง) เตือน · **<M = ไม่มีระดับไหนเตือน ไม่ได้แปลว่าจะเกิด C** (ไม่มีแบบจำลอง ≥C1.0)

มีสองจุดทำงานให้สลับบนหน้าเว็บ ต่างกันแค่ระดับ X: **เตือนไว** (threshold ราย seed ที่ให้ TSS สูงสุด) กับ
**ระมัดระวัง** (ความน่าจะเป็นเฉลี่ย ≥ 0.316 เลือกบน validation ให้ HSS 3 คลาสสูงสุด)

| test (3,098 sample) | เตือนไว | ระมัดระวัง |
|---|---:|---:|
| ≥M TSS | 0.744 | 0.748 |
| ≥X TSS | 0.703 | 0.066 |
| ≥X จับได้ / precision | 12 จาก 15 / 4% | 1 จาก 15 / 33% |
| HSS 3 คลาส | 0.187 | 0.303 |
| ทายระดับถูกเป๊ะ (event ≥M จริง) | 26% | 83% |

- **ระดับ ≥M ใช้ได้จริง แต่ระดับ X แยก X ออกจาก M ได้ไม่ดี** — โหมดเตือนไวเรียก flare ระดับ M จริง 100 จาก 140
  ว่าเป็น X · โหมดระมัดระวังทายระดับถูกบ่อยแต่แทบไม่เคยเตือน X
- X ใน test มีแค่ 15 sample จาก 5 HARP — ตัวเลขของระดับ X แกว่งได้มาก
- persistence ที่เลือกบน validation ให้ TSS สูงกว่าทั้งสองระดับ (≥M 0.797 · ≥X 0.714)
- การเสนอจุดทำงานที่สองเกิดขึ้นหลังเห็นผลบน test ของโหมดเตือนไว (threshold ของมันเลือกบน validation ล้วน)

### Segmentation

| โมเดล | train / val / test (เฟรม) | test Dice / IoU |
|---|---|---|
| U-Net `models/unet.pt` | 6,870 / 691 / 681 | **0.916 / 0.855** |

## โครงสร้างโปรเจค

แยก `backend/` (Python — โมเดล, data pipeline, FastAPI) กับ `frontend/` (React + Vite)
ออกจากกันชัดเจนที่ระดับ root ส่วน `data/` และ `artifacts/` อยู่นอกทั้งสองฝั่งเพราะเป็น
runtime data ไม่ใช่โค้ด (mount เป็น volume ใน Docker)

```
backend/
  configs/
    data.yaml         แหล่งข้อมูล, path, การแบ่ง split
    unet.yaml         U-Net
    forecast.yaml     โมเดลพยากรณ์ทั้งสี่ตัว (ค่าร่วม + ค่าเฉพาะรายโมเดล, default_model ของหน้าเว็บ)
    tracking.yaml     detection + tracker
    study/            งานเปรียบเทียบ: architectures.yaml (แกนสถาปัตยกรรม), variants*.yaml (แกนชุด feature)
  src/sunseg/
    config.py         โหลด+validate config ด้วย pydantic
    artifacts.py      ชื่อไฟล์ผลลัพธ์ของแต่ละโมเดล (checkpoint/metrics/predictions) — ฝั่งเทรนกับแอปใช้ร่วมกัน
    data/             ดึงข้อมูลจาก JSOC/HEK, สร้าง dataset, อ่านฟลักซ์โปรตอน GOES
    datasets/         PyTorch Dataset (sequence ของ production, study ของงานเปรียบเทียบ, segmentation)
    models/
      unet.py         U-Net
      forecast/       LSTM, TCN, Transformer, DA-RNN + pooling ร่วม + registry (ประกอบจากชื่อ kind)
    training/         loop เทรนโมเดลพยากรณ์ที่ทุกสคริปต์ใช้ร่วมกัน, losses, utils (seed/scheduler/checkpoint)
    inference/        ClassForecastService (โมเดลหลัก <M/M/X), ForecastModels (รายชั่วโมง), ค่าทำนายของแผง confusion matrix, segmentation
    study/            สรุปผลงานเปรียบเทียบ (อันดับโมเดล + Δ แบบจับคู่ seed), ประเมินเวลาต้นทาง
    tracking/         detection, differential rotation, tracker
    report_style.py   สไตล์รูปประกอบรายงาน (จานสี, ฟอนต์ไทย)
  app/                FastAPI backend — API เท่านั้น ไม่รู้จักไฟล์ frontend เลย
  scripts/
    data/             ดาวน์โหลด + สร้าง dataset (download_*, build_sequences, build_flare_positions,
                      rebuild_flares_v2, measure_xray_cross_calibration)
    segmentation/     train.py (U-Net), plot_masks, render_*_video
    forecast/         train.py --model <ชื่อ|all>, plot_example, plot_storm_forecast (คำพยากรณ์ราย 24 ชม. เทียบฟลักซ์จริง)
    study/            build_dataset, train, tune (Optuna + CV), class_forecast (รายงานโมเดลหลัก), plot_figures,
                      persistence_baseline, extract_intensity, evidence_analysis, measure_acquisition_time, ด่านตรวจต่าง ๆ
    checks/           check_env, check_connectivity, check_api, inspect_flares, plot_aia_alignment
  tests/              pytest (632 test)
  pyproject.toml
  Dockerfile        build จาก root ของ repo (ต้องการ README.md นอก backend/)
frontend/           React + Vite — คนละ container/process จาก backend เสมอ
  src/
    state/            AppProvider — ศูนย์กลาง state ของแอป ครอบทุกหน้า (คงอยู่ข้ามการเปลี่ยนหน้า)
    pages/            หน้าตาม route: / (Hub), /dashboard, /model, /about — react-router
    components/       layout/ (navbar, footer, scroll), Hub/ (แผนที่ตำแหน่ง flare แบบ Canvas + ตาราง), Dashboard/, Model/, ui/
    lib/              เรียก API, ค่าคงที่ของกราฟ, ตัวช่วยจัดรูปข้อความ, nav.js (section ของ dashboard)
    styles/           ธีม deep-space: tokens.css (สี/ฟอนต์) + base/chrome/hub/dashboard/model
  Dockerfile        build เป็น nginx image (multi-stage: npm build -> nginx serve)
  nginx.conf        proxy /api, /docs, /openapi.json ไปที่ backend container
data/               ดาวน์โหลด/ประมวลผลจาก JSOC (อยู่นอก backend/ โดยเจตนา)
  processed/frames/   คู่ (magnetogram, mask) ขนาด 512px — input ของ U-Net
  processed/aia/      ภาพ AIA รายช่อง วางบนกริดเดียวกับ frames/ แล้ว (DN/s)
  interim/frame_wcs.parquet   WCS จริงของแต่ละเฟรม (ดู "ภาพ AIA" ด้านบน)
artifacts/          โมเดลที่เทรนแล้ว, log, ผลการวัด (อยู่นอก backend/ โดยเจตนา)
  model_comparison/   ผลหลัก: report.md (อันดับ 16 โมเดล), figures/, persistence_baseline.md
  model_comparison_x/ เซลล์ LSTM + V3 ที่เทรนด้วย label ≥X1.0 (ระดับ X ของโมเดลหลัก)
  class_forecast/     รายงานโมเดลหลัก <M/M/X
docs/adr/           บันทึกการตัดสินใจเชิงออกแบบ (เช่น 0001: จูน hyperparameter แยกรายสถาปัตยกรรม)
```

## การทดสอบ

```powershell
cd backend
pytest
```

## License

MIT

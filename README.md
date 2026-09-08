# sunseg — Solar Active Region Segmentation, Tracking & Flare Forecasting

ระบบ end-to-end สำหรับติดตามสภาพอวกาศ (space weather) ประกอบด้วย 3 ส่วนที่ทำงานต่อกัน:

| ส่วน | โมเดล | ทำอะไร |
|---|---|---|
| **Segmentation** | U-Net (PyTorch) | แบ่งส่วน active region จากภาพ magnetogram เต็มดวงของ SDO/HMI |
| **Tracking** | Hungarian + differential rotation | ติดตาม AR แต่ละดวงข้ามเวลา โดยชดเชยการหมุนของดวงอาทิตย์ |
| **Forecasting** | LSTM (PyTorch) | ทำนายโอกาสเกิด flare ≥M1.0 ภายใน 24 ชม. จาก SHARP magnetic parameters |

ผลลัพธ์ทั้งหมดแสดงผ่าน **webapp (FastAPI + Plotly.js)**

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
python backend/scripts/download_metadata.py

# 2) สร้าง sequences + labels สำหรับ LSTM
python backend/scripts/build_sequences.py

# 3) เทรน LSTM (ไม่กี่นาที) พร้อมเทียบกับ logistic-regression baseline
#    เซฟค่าทำนายราย sample ของ val/test ไว้ด้วย (artifacts/metrics/predictions.parquet)
#    ซึ่งแผง Confusion Matrix ในหน้าเว็บ (ขั้นตอน 7) ต้องใช้ — ดูหัวข้อ "ตัวชี้วัด" ด้านล่าง
python backend/scripts/train_lstm.py

# 4) ดาวน์โหลดภาพ + สร้าง mask สำหรับ U-Net
#    ต้องมีอีเมล JSOC ใน .env  ·  ทดสอบด้วยไม่กี่เฟรมก่อนเสมอ
python backend/scripts/download_images.py --start 2014-10-20 --end 2014-10-28 --limit 5
python backend/scripts/plot_masks.py          # << ตรวจด้วยตาก่อน! (ดูหมายเหตุด้านล่าง)
python backend/scripts/download_images.py     # แล้วค่อยดึงเต็มช่วง (หลายชั่วโมง)

# 5) เทรน U-Net
python backend/scripts/train_unet.py --overfit-one-batch   # ตรวจสุขภาพโมเดลก่อน
python backend/scripts/train_unet.py

# 6) ดึงภาพ AIA สามชั้นบรรยากาศ (ไม่บังคับ แต่ทำให้หน้าเว็บสลับชั้นได้)
python backend/scripts/download_aia.py --wcs-only          # ดึง WCS ของเฟรมก่อน (เร็ว)
python backend/scripts/download_aia.py --limit 5           # ทดสอบไม่กี่เฟรมก่อนเสมอ
python backend/scripts/plot_aia_alignment.py               # << ตรวจด้วยตาก่อน! (ดูหมายเหตุด้านล่าง)
python backend/scripts/download_aia.py                     # แล้วค่อยดึงเต็มช่วง (~45 นาที)
python backend/scripts/download_aia.py --case-study         # เฟรม case study 2024 (อยู่นอก time_range)

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

> **ห้ามข้ามขั้นตอน `plot_masks.py`** — mask ถูกสร้างโดยแปลงพิกัดจาก SHARP patch ไปยังภาพเต็มดวงผ่าน WCS
> ถ้าการแปลงผิด mask จะเลื่อนไปจากตำแหน่งจริงทั้งภาพ โมเดลจะยังเทรนได้และ loss จะลดลงสวยงาม แต่เรียนรู้
> สิ่งที่ผิด สิ่งที่ต้องเห็นในภาพคือ **เส้นขอบสีส้มล้อมรอบบริเวณสนามแม่เหล็กเข้ม** (จุดขาว/ดำจัด)

> **ห้ามข้ามขั้นตอน `plot_aia_alignment.py`** ด้วยเหตุผลเดียวกัน — WCS ที่ผิดจะให้ภาพ AIA เต็มดวง
> ที่สวยงามพร้อมเส้นขอบในตำแหน่งที่ดูน่าเชื่อถือ แต่ตัวเลขความเข้มแสงราย AR ที่ได้กลับถูกสุ่มมาจาก
> quiet Sun ทั้งหมด สิ่งที่ต้องเห็นคือ **จุดสว่าง (plage) ของ 1600 Å ทับพิกเซลสนามแม่เหล็กเข้มพอดี**
> สคริปต์ยังตรวจเชิงตัวเลขให้ด้วย โดยเทียบ contrast ของภาพปกติกับภาพที่หมุน 180° แล้ว fail ถ้าภาพหมุนชนะ

## เครื่องมือตรวจสอบ

```powershell
python backend/scripts/check_env.py           # ไลบรารีครบ, CUDA ใช้ได้, config ถูกต้อง
python backend/scripts/check_connectivity.py  # เข้าถึง JSOC / NGDC / HEK ได้หรือไม่
python backend/scripts/inspect_flares.py      # คุณภาพของ label — ดูอัตราการเก็บ flare M+
python backend/scripts/check_api.py           # ยิง API จริงผ่าน HTTP (ต้องเปิดเซิร์ฟเวอร์ก่อน)
python backend/scripts/plot_aia_alignment.py  # ภาพ AIA วางทับกริดของเฟรมตรงตำแหน่งหรือไม่
```

> **ทำไม LSTM ก่อน U-Net?** LSTM ใช้แค่ข้อมูลตาราง (ไม่กี่ร้อย MB) เทรนเสร็จในไม่กี่นาที
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
| JSOC `hmi.sharp_cea_720s` | SHARP magnetic parameters — features ของ LSTM (พิกัด CEA แก้ผลการฉายแล้ว) |
| JSOC `hmi.sharp_720s` | `bitmap` segment — ground-truth mask ของ U-Net (พิกัด CCD ตรงกับภาพเต็มดวง) |
| JSOC `hmi.M_720s` | ภาพ magnetogram เต็มดวง — input ของ U-Net |
| [คลัง AIA synoptic](https://jsoc1.stanford.edu/data/aia/synoptic/) | ภาพ AIA 1600/304/171 Å — เลเยอร์ชั้นบรรยากาศในหน้าเว็บ (ดูด้านล่าง) |
| [NGDC GOES XRS reports](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/) | รายการ flare — labels ของ LSTM (ค่าเริ่มต้น, ครอบคลุม 1975–2017) |
| HEK | รายการ flare ทางเลือก (`--flare-source hek`) — ยืดหยุ่นกว่าแต่ช้ากว่ามาก |
| [ตาราง HARPNUM↔NOAA](http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt) | เชื่อม SHARP เข้ากับ flare catalog |
| คลัง GOES particle ราย 5 นาที (ภายนอก) | ฟลักซ์โปรตอนรอบเวลาที่เกิด flare — แผง "Proton flux" ในหน้าเว็บ (ดูด้านล่าง) |
| NOAA NCEI `xrsf-l2-avg1m_science` (GOES-15) | ฟลักซ์ X-ray ต่อเนื่องรายนาที — เส้น "GOES X-Ray" ในแดชบอร์ด (ดูด้านล่าง) |

### ภาพ AIA สามชั้นบรรยากาศ (ไม่บังคับ)

magnetogram บอกได้แค่สนามแม่เหล็กที่ผิว แต่ active region แผ่ตัวขึ้นไปทั่วชั้นบรรยากาศ หน้าเว็บจึง
สลับ **เลเยอร์** ของภาพพื้นหลังได้ โดย mask จาก U-Net ทับอยู่ที่เดิมทุกเลเยอร์ ทำให้เห็นว่าโครงสร้าง
สนามแม่เหล็กเดียวกันให้ความร้อนกับพลาสมาชั้นบนแค่ไหน:

| ช่อง | ชั้นบรรยากาศ | อุณหภูมิลักษณะเฉพาะ |
|---|---|---|
| AIA 1600 Å | โฟโตสเฟียร์ / transition region | ~10,000 K |
| AIA 304 Å | โครโมสเฟียร์ (He II) | ~50,000 K |
| AIA 171 Å | โคโรนาสงบ (Fe IX) | ~600,000 K |

```powershell
python backend/scripts/download_aia.py --wcs-only     # ดึง WCS ของเฟรม (เร็ว ไม่ต้องใช้อีเมล JSOC)
python backend/scripts/download_aia.py --limit 5      # ทดสอบก่อน
python backend/scripts/plot_aia_alignment.py          # << ตรวจการจัดตำแหน่ง ห้ามข้าม
python backend/scripts/download_aia.py                # เต็มช่วง — ~1.2 GB, ราว 45 นาที (6 worker)
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
| case study พ.ค. 2024 | 12 ชม. | เส้นต่อเนื่อง ติดตาม AR ได้หลายสิบจุด — ช่วงที่เหมาะกับการดูวิวัฒนาการมากที่สุด |

API จะแนบ `note` อธิบายข้อจำกัดของช่วงที่เลือกมาด้วยเสมอ และหน้าเว็บแสดงข้อความนั้นให้เห็น

> **ค่าที่ได้เป็น DN/s แบบสัมพัทธ์** — หารด้วย `EXPTIME` แล้วเพื่อให้เทียบข้ามช่องได้ แต่ **ไม่ได้แก้
> instrument degradation** (ความไวของ AIA ลดลงตามปี) กราฟความเข้มแสงจึงใช้เทียบ AR **ภายในเฟรม
> เดียวกัน**เท่านั้น ห้ามนำไปเทียบข้ามปี

### ฟลักซ์ X-ray ต่อเนื่อง (ไม่บังคับ)

เส้น **GOES X-Ray** ในแดชบอร์ดค่าเริ่มต้นวาดจากรายการ flare (แค่ 3 จุดต่อเหตุการณ์: เริ่ม/peak/จบ)
ซึ่งได้เส้นเป็นหนามแหลม ๆ ไม่ใช่ฟลักซ์จริงที่ขึ้นลงต่อเนื่องแบบหน้า
[SWPC](https://www.swpc.noaa.gov/products/goes-x-ray-flux) — รันสคริปต์นี้เพื่อดึงฟลักซ์จริงรายนาที
จาก NOAA NCEI (เฉพาะ GOES-15 ช่วง 2011–2017 ซึ่งครอบคลุม `time_range` ของโปรเจคพอดี):

```powershell
# ทดสอบด้วยเดือนเดียวก่อน (แนะนำ — ไฟล์เต็มช่วงมีเกือบ 2,600 ไฟล์ ~150 MB)
python backend/scripts/download_xray.py --start 2014-10-01 --end 2014-10-31

# ดึงเต็มช่วงตาม time_range ใน configs/data.yaml (ใช้เวลานาน)
python backend/scripts/download_xray.py
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
ตรวจยืนยันได้จาก **positive rate สุดท้าย 1.76% ซึ่งอยู่ในช่วงที่งานวิจัยรายงาน (1–5%)**

ใช้ `python scripts/inspect_flares.py` เพื่อดูตัวเลขนี้กับข้อมูลของคุณเอง

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

สลับดูได้ระหว่าง LSTM กับ logistic baseline — สไลเดอร์รีเซ็ตไปที่ threshold ของโมเดลที่เพิ่งสลับมา
เสมอ เพราะสองโมเดลมีสเกลความน่าจะเป็นคนละแบบ ห้ามใช้สไลเดอร์ตัวเดียวคุมสองโมเดลพร้อมกัน — และ
สลับดูได้ระหว่างชุด val กับ test

> **split `val` ในแผงนี้คือชุดที่ใช้เลือก threshold ตอนเทรน ไม่ใช่ตัวเลขที่ควรอ้างเป็นผลงาน** — ตัวเลข
> ที่รายงานผลจริงต้องดูที่ `test` เสมอ (ดูหัวข้อ "หลักการสำคัญ: การป้องกัน data leakage" ด้านบน)

คลิกช่องใดช่องหนึ่ง (เช่น *miss* เพื่อดู flare ที่โมเดลพลาดทั้งหมด) จะเห็นรายชื่อ HARP, NOAA AR,
เวลาออกพยากรณ์ และค่าความน่าจะเป็นของ sample ในช่องนั้น เรียงจากมั่นใจที่สุดไปน้อยที่สุด คลิก sample
หนึ่งแถวจะเลื่อนขึ้นไปที่ dashboard ด้านบนพร้อมเลือก AR ดวงนั้นและขยับช่วงวันที่ให้ครอบเวลานั้นให้
อัตโนมัติ (threshold ที่ลากเล่นในแผงนี้เป็นของแผงเองเท่านั้น ไม่กระทบ threshold ที่ dashboard ใช้พยากรณ์จริง)

> **ต้องรันขั้นตอนที่ 3 (`train_lstm.py`) ซ้ำหนึ่งครั้ง** ถึงจะเห็นแผงนี้ — สคริปต์เทรนบันทึกค่าทำนาย
> ราย sample ของ val/test ไว้ที่ `artifacts/metrics/predictions.parquet` เพิ่มจากไฟล์ตัวชี้วัดเดิม
> ถ้ายังไม่มีไฟล์นี้ (เช่น checkout checkpoint เก่าที่เทรนไว้ก่อนหน้านี้มา) แผงจะขึ้นข้อความบอกวิธีแก้
> แทนตาราง — เป็นพฤติกรรมที่ตั้งใจ ไม่ใช่ bug ส่วนที่เหลือของหน้าเว็บยังเปิดใช้งานได้ตามปกติ

การนับ TP/FP/TN/FN ทั้งหมดทำฝั่ง Python (`sunseg.metrics.threshold_sweep`) ด้วยโค้ดชุดเดียวกับที่
คำนวณตัวเลขในไฟล์ `lstm.json` — สไลเดอร์บนหน้าเว็บแค่เปิดตารางที่คำนวณไว้ล่วงหน้าตามตำแหน่ง ไม่มี
การนับซ้ำฝั่ง JavaScript ตัวเลขในแผงนี้จึงตรงกับไฟล์ตัวชี้วัดเสมอที่ threshold เดียวกัน

## ผลการทดลอง

ข้อมูล 2011–2025 · 496,120 sample · 4,903 HARP · positive 2.34% · แบ่ง HARP-disjoint

| | n | positive | |
|---|---|---|---|
| train | 208,033 | 3,854 (1.85%) | 2011-01 → 2015-06 |
| val | 26,255 | 408 (1.55%) | 2015-07 → 2016-06 |
| test | 261,832 | **7,368 (2.81%)** | 2017-03 → 2025-12 |

### พยากรณ์ flare ≥M1.0 ใน 24 ชม.

| โมเดล | val TSS | **test TSS** | test AUC | test recall | test precision |
|---|---|---|---|---|---|
| LSTM (ประวัติ 24 ชม.) | 0.845 | **0.691** | 0.934 | 0.79 | 0.19 |
| Logistic regression (ค่า ณ เวลาเดียว) | 0.827 | **0.777** | 0.953 | 0.91 | 0.17 |

### ข้อค้นพบ: LSTM ไม่ชนะ baseline

**logistic regression ที่ใช้ SHARP parameters ณ เวลาเดียว ยังคงทำได้ดีกว่า LSTM ที่เห็นประวัติ 24 ชั่วโมง**
นี่คือผลที่วัดได้จริง ไม่ใช่ข้อบกพร่องของการติดตั้ง และสอดคล้องกับงานวิจัยบางส่วนในสาขานี้ ตัวเลขนี้วัดบน
test set ที่มี positive ถึง 7,368 ตัว (มี.ค. 2017 – ธ.ค. 2025) จึงเชื่อถือได้กว่าตัวเลขจากชุดข้อมูลก่อนหน้า
ที่มี positive เพียงหลักสิบมาก สาเหตุที่วิเคราะห์ได้:

1. **effective sample size เล็กกว่าที่เห็นมาก** — sample ที่ห่างกัน 1 ชม. จาก AR ดวงเดียวกันแทบเหมือนกัน
   จำนวนตัวอย่างที่เป็นอิสระต่อกันจริงคือ **จำนวน HARP ที่เคยเกิด flare ซึ่งมีเพียง 357 ดวง** ไม่ใช่
   11,630 sample ที่เป็น positive โมเดลที่มีพารามิเตอร์นับแสนจึงจำข้อมูลแทนที่จะเรียนรู้รูปแบบ
   (ลดจาก 224k → 7k พารามิเตอร์แล้ว test TSS ดีขึ้นอย่างมีนัยสำคัญ แต่ก็ยังแพ้ baseline)
2. **train กับ test อยู่คนละช่วงของวัฏจักรสุริยะ** — train อยู่ในช่วง solar maximum ของ cycle 24
   (2011-2015) ส่วน test ครอบคลุมตั้งแต่ขาลงของ cycle 24 ไปจนถึง cycle 25 ทั้งลูก ซึ่งมีพฤติกรรมต่างกัน

> การปรับ hyperparameter ต่อโดยดูผลบน test set จะเป็นการ overfit test set เสียเอง จึงหยุดไว้ที่การปรับ
> ตามเหตุผลเชิงโครงสร้าง 2 ครั้ง แล้วรายงานผลตามจริง

**ข้อควรระวังเพิ่มเติม**: `BSS` ติดลบทั้งสองโมเดล (LSTM −0.28, baseline −1.97) แปลว่า *ค่าความน่าจะเป็นที่
โมเดลให้มายังไม่ calibrate* — เป็นผลโดยตรงจาก focal loss และ `class_weight="balanced"` ที่จงใจบิดสเกล
ความน่าจะเป็นเพื่อดัน recall ควรตีความผลลัพธ์เป็น **การจัดอันดับความเสี่ยง** ไม่ใช่ความน่าจะเป็นที่แท้จริง
(หากต้องการค่าที่ตีความได้ ให้เพิ่มขั้นตอน isotonic/Platt calibration บน validation set)

### Segmentation

| โมเดล | ตัวชี้วัด | ผลลัพธ์ |
|---|---|---|
| U-Net | Dice / IoU | **0.916 / 0.855** (test) |

## โครงสร้างโปรเจค

แยก `backend/` (Python — โมเดล, data pipeline, FastAPI) กับ `frontend/` (React + Vite)
ออกจากกันชัดเจนที่ระดับ root ส่วน `data/` และ `artifacts/` อยู่นอกทั้งสองฝั่งเพราะเป็น
runtime data ไม่ใช่โค้ด (mount เป็น volume ใน Docker)

```
backend/
  configs/          ไฟล์ตั้งค่า YAML (data, unet, lstm, tracking)
  src/sunseg/
    config.py       โหลด+validate config ด้วย pydantic
    data/           ดึงข้อมูลจาก JSOC/HEK, สร้าง dataset, อ่านฟลักซ์โปรตอน GOES
    datasets/       PyTorch Dataset
    models/         U-Net, LSTM
    tracking/       detection, differential rotation, tracker
    inference/      pipeline รวม 3 ส่วน
  app/              FastAPI backend — API เท่านั้น ไม่รู้จักไฟล์ frontend เลย
  scripts/          CLI entrypoints
  tests/            pytest
  pyproject.toml
  Dockerfile        build จาก root ของ repo (ต้องการ README.md นอก backend/)
frontend/           React + Vite — คนละ container/process จาก backend เสมอ
  src/
    state/            AppProvider — ศูนย์กลาง state ของแอป (เทียบเท่า `state` object เดิม)
    components/       แบ่งตามแผงของ dashboard (Hero, Dashboard/, About/)
    lib/              เรียก API, ค่าคงที่ของกราฟ, ตัวช่วยจัดรูปข้อความ
  Dockerfile        build เป็น nginx image (multi-stage: npm build -> nginx serve)
  nginx.conf        proxy /api, /docs, /openapi.json ไปที่ backend container
data/               ดาวน์โหลด/ประมวลผลจาก JSOC (อยู่นอก backend/ โดยเจตนา)
  processed/frames/   คู่ (magnetogram, mask) ขนาด 512px — input ของ U-Net
  processed/aia/      ภาพ AIA รายช่อง วางบนกริดเดียวกับ frames/ แล้ว (DN/s)
  interim/frame_wcs.parquet   WCS จริงของแต่ละเฟรม (ดู "ภาพ AIA" ด้านบน)
artifacts/          โมเดลที่เทรนแล้ว, log, ผลการวัด (อยู่นอก backend/ โดยเจตนา)
```

## การทดสอบ

```powershell
cd backend
pytest
```

## License

MIT

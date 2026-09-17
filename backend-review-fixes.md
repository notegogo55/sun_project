# Prompt: แก้ไขผลจาก code review (backend — sunseg package + FastAPI)

> ไฟล์นี้คือ **prompt สำหรับสั่งงานแก้ไข** — ส่งทั้งไฟล์ให้ agent หรืออ่านเองแล้วไล่แก้ทีละข้อก็ได้
> ขอบเขต: `backend/src/sunseg/`, `backend/app/`, `backend/scripts/`, `backend/configs/`
> แกนที่รีวิว: correctness · ML pipeline quality · performance · structure
> คนละชุดกับ `code-review-fixes.md` (นั่นคือ frontend/React migration) — ไม่มีข้อไหนซ้ำกัน

---

## บริบทของโค้ดเบส (อ่านก่อนเริ่ม)

`d:\sun_project` — U-Net แบ่งส่วน active region + LSTM พยากรณ์ flare ≥ M1.0 ใน 24 ชม.
ศัพท์และนิยาม (HARP / detection / NOAA AR / sample / issue_time / variant / control /
activity baseline) อยู่ใน `CONTEXT.md` — **ใช้คำตามนั้นเสมอ**

**คอนเวนชันที่ต้องรักษา:** คอมเมนต์ภาษาไทยที่อธิบาย "ทำไม" ไม่ใช่ "ทำอะไร" มักอ้างบั๊กที่เคยเจอ
หรือตัวเลขที่วัดมาจริง ทุกไฟล์ที่แตะต้องคงสไตล์นี้ไว้

**ข้อสังเกตก่อนเริ่ม:** โครงสร้างโดยรวมแข็งแรงกว่างาน ML ทั่วไปมาก — split เป็น HARP-disjoint +
gap จริง, normalisation คำนวณจาก train เท่านั้น, threshold freeze บน val ก่อนแตะ test,
`val/test` loader ไม่ shuffle จึงเรียงตรงกับ `meta`, `threshold_sweep` เป็นแหล่งนับ TP/FP/TN/FN
แห่งเดียว, test suite ครอบคลุมและตั้งชื่อดี ปัญหาด้านล่างจึงเป็นรอยรั่วเฉพาะจุด ไม่ใช่ปัญหาเชิงสถาปัตยกรรม

---

## A. บั๊กที่กระทบตัวเลขผลลัพธ์ — แก้ก่อน

### A1. `find_best_threshold` คืน TSS = 0 เงียบ ๆ เมื่อความน่าจะเป็นทุกตัวต่ำกว่า 0.01 🔴

**ไฟล์:** `src/sunseg/metrics.py:154` · ผู้เรียก `scripts/train_lstm.py:223,247` ·
`scripts/train_study.py:158,169`

**อาการ:** `candidates = np.linspace(0.01, 0.99, n_steps)` มีพื้นเป็น 0.01 ตายตัว
ถ้าโมเดลให้ค่าความน่าจะเป็นต่ำกว่า 0.01 ทุกตัว ทุก candidate จะได้ TP = 0 → `tss` = 0 ทุกจุด →
ฟังก์ชันคืน `(0.01, 0.0)` โดยไม่มี error

ยืนยันด้วยข้อมูลสังเคราะห์ (positive 2%, จัดอันดับได้ดีจริง):

```
all probs < 0.01 -> find_best_threshold คืน thr=0.010 TSS=0.0000   (AUC ของชุดเดียวกัน = 0.877)
```

นี่คือ **บั๊กเดียวกับที่คอมเมนต์ `train_lstm.py:219-222` เขียนไว้ว่าแก้แล้ว** — ตอนนั้นแก้กรณี
threshold ตายตัวที่ 0.5 แต่ย้ายพื้นมาไว้ที่ 0.01 เฉย ๆ ไม่ได้เอาพื้นออก

ร้ายแรงเพราะค่านี้ถูกใช้สามที่พร้อมกัน: (1) `EarlyStopping` เลือก checkpoint,
(2) เงื่อนไขหยุดเทรน, (3) threshold ที่ freeze ลง `lstm.pt` และ `runs/*.json` ทุกไฟล์
โมเดลที่ดีจริงจึงถูกทิ้งได้ หรือทั้งตารางเปรียบเทียบได้ TSS ≈ 0 ทุกแบบ โดยไม่มีอะไรเตือน

เพิ่มเติม แม้ในกรณีปกติกริดก็หยาบกว่าที่เห็น — วัดจากการกระจายความน่าจะเป็นที่สมจริง:

```
prob range [0.0000, 0.1827]; grid points inside that range: 36 of 200
```

**สิ่งที่ต้องแก้:** เลิกใช้กริดตายตัว ให้ candidate มาจากข้อมูลเอง (`np.unique(y_prob)` หรือ
quantile ของมัน) แล้วให้คะแนนทุกจุดในรอบเดียวด้วย `threshold_sweep` ที่มีอยู่แล้วในโมดูลเดียวกัน
(O(N log N) แทน 200 × O(N) — เร็วขึ้นด้วยในตัว) ถ้ายังอยากมีกริดคงที่ ต้องย้ายไปประกาศ
**ที่เดียว** ใน `metrics.py` แล้ว export ออกไป

**พ่วงกันมา:** `src/sunseg/inference/predictions_store.py:29` ประกาศ
`THRESHOLD_GRID = np.linspace(0.01, 0.99, 200)` ซ้ำอีกชุด มีแค่คอมเมนต์ผูกไว้ว่า "กริดเดียวกัน"
ถ้า `metrics.py` เปลี่ยน กริดสองฝั่งจะหลุดจากกันโดยไม่มี error เพราะ `frozen_index` ใช้ `argmin`
หา index ที่ใกล้ที่สุดเสมอ — แผง confusion matrix จะแสดงจำนวนนับที่ threshold คนละค่ากับที่ freeze ไว้
→ ให้ `predictions_store.py` import จาก `metrics.py` แทนการประกาศเอง

**ยืนยันหลังแก้:** ทดสอบว่าชุดที่ทุก prob < 0.01 แต่จัดอันดับถูกสมบูรณ์ ต้องได้ TSS ใกล้ 1
ไม่ใช่ 0 · เพิ่ม unit test ข้อนี้ใน `tests/test_metrics.py`

### A2. ไม่มีการตรวจว่า val มี positive — เทรนไปหาคำตอบ "ไม่ทายอะไรเลย" เงียบ ๆ 🔴

**ไฟล์:** `scripts/train_study.py:483-490` · `scripts/train_lstm.py:177-185`

**อาการ:** ตรวจแค่ว่า val/test ไม่ว่าง และ **train** มี positive แต่ไม่เคยตรวจว่า **val** มี positive
ถ้า val มี positive 0 ตัว:

- `tss()` คืน `0.0 - fpr` (recall = 0 เพราะ `tp+fn == 0`)
- `find_best_threshold` จึงไป maximize `-FPR` → เลือก threshold สูงสุด
- `EarlyStopping` ก็ maximize `-FPR` → checkpoint ที่ถูกเลือกคือตัวที่ไม่ทายบวกเลย
- `_scope_report` แสดง val เป็น `n/a` (ถูกต้อง) แต่ **ตัวเลข test ของ run นั้นถูกรายงานเหมือนใช้ได้**

ไม่ใช่กรณีสมมติ — คอมเมนต์ใน `configs/data.yaml` หัวข้อ `split:` บันทึกไว้เองว่าเส้นแบ่งเดิม
เคยทำให้ "test ได้ 334 sample แต่ positive 0 ตัว" มาแล้ว เส้นแบ่งใหม่ก็ยังเลื่อนได้อีก

**สิ่งที่ต้องแก้:** เพิ่ม `val_pos == 0` (และ `test_pos == 0`) เข้าไปในเงื่อนไขเดียวกัน พร้อมข้อความ
บอกสาเหตุแบบเดียวกับที่มีอยู่ แก้ทั้งสองไฟล์ให้เงื่อนไขตรงกัน

### A3. `run_signature` ไม่นับรายชื่อคอลัมน์ของแบบ — ผลเก่าถูกนำมารวมเงียบ ๆ 🔴

**ไฟล์:** `scripts/train_study.py:124-136` · จุดที่ใช้ `:501-506` และ `:365`

**อาการ:** signature ประกอบด้วย config ของ LSTM, `epochs`, และ `arrays.features` ซึ่งคือ
**รายชื่อคอลัมน์ผู้สมัครทั้งหมดใน dataset** ไม่ใช่ `spec["columns"]` ของแบบนั้น

ผลคือแก้ `configs/study_variants.yaml` แล้ว signature ไม่เปลี่ยน → `main()` เห็นว่า
"มีผลแล้ว — ข้าม" และ `aggregate()` รับไฟล์ `runs/*.json` เก่าเข้ามารวม ตารางเปรียบเทียบจึง
ปนผลจากชุด feature สองชุดภายใต้ชื่อแบบเดียวกัน

เหตุการณ์จริงที่เพิ่งเกิด: การตัด `1600_median`/`1600_p95` ออกจาก `intensity_columns`
(2026-09-13) เป็นการแก้ไฟล์ variants ล้วน ๆ — ถ้าไม่ใช้ `--force` ผล V1/V2/V3 เก่าที่ยังมี 1600
จะถูกนำมารวมต่อ

**สิ่งที่ต้องแก้:** ข้อมูลมีอยู่แล้ว — `record` เก็บ `"columns": spec["columns"]` ไว้ทุกไฟล์
แค่ยังไม่ถูกใช้ตรวจ ทางที่ตรงที่สุดคือให้ `run_signature` รับ `columns` แล้วใส่ลง signature
(กลายเป็น signature รายแบบ) และ/หรือเพิ่มเงื่อนไข `record["columns"] != spec["columns"]`
ทั้งที่จุดข้าม (`:501`) และที่ `aggregate()` (`:365`)

**ยืนยันหลังแก้:** เทรน V1 หนึ่ง seed → แก้ `intensity_columns` ในไฟล์ variants → รันใหม่
โดยไม่ใส่ `--force` ต้องเห็นว่ามันเทรนใหม่ ไม่ใช่ข้าม

---

## B. คุณภาพของ pipeline และการตรวจสอบ

### B1. augmentation ของ U-Net ซ้ำกัน 4 ชุดเพราะ worker แชร์ RNG เดียวกัน 🟠

**ไฟล์:** `src/sunseg/datasets/segmentation.py:47` · `make_segmentation_loaders:165-189` ·
`configs/unet.yaml` (`num_workers: 4`)

**อาการ:** `self._rng = np.random.default_rng(seed)` ถูกสร้างใน `__init__` คือใน process แม่
PyTorch ตั้ง seed ใหม่ให้ **torch RNG** ของแต่ละ worker แต่ไม่แตะ `np.random.Generator` ที่เป็น
attribute ของ Dataset ซึ่งถูก pickle/fork ไปพร้อมกันทั้งก้อน ทุก worker จึงเริ่มจาก state เดียวกัน
(ยิ่ง `persistent_workers=True` ยิ่งค้างแบบนั้นทั้งการเทรน)

ยืนยันแล้ว — เลข 4 ตัวแรกที่แต่ละ worker สุ่มได้:

```
worker0 [0.774, 0.4389, 0.8586, 0.6974]
worker1 [0.774, 0.4389, 0.8586, 0.6974]
worker2 [0.774, 0.4389, 0.8586, 0.6974]
worker3 [0.774, 0.4389, 0.8586, 0.6974]
```

ความหลากหลายของ augmentation จึงเหลือราว 1/4 ของที่ตั้งใจ ซึ่งสำคัญมากกับชุดข้อมูลที่มีเฟรมไม่เยอะ
ผลข้างเคียงอีกอย่าง: `positive_fraction()` ดึงจาก `self._rng` ตัวเดียวกัน เรียกหรือไม่เรียกทำให้
ลำดับ augmentation ทั้งการเทรนเลื่อนไปคนละชุด

**สิ่งที่ต้องแก้:** ใส่ `worker_init_fn` ที่สร้าง `dataset._rng` ใหม่จาก
`torch.utils.data.get_worker_info().seed` หรือสร้าง Generator แบบ lazy ต่อ worker ใน `__getitem__`
และแยก RNG ของ `positive_fraction()` ออกเป็นคนละตัว

**ยืนยันหลังแก้:** โหลด batch เดียวกันด้วย `num_workers=4` สองรอบ ค่า augmentation ของ
sample ที่อยู่คนละ worker ต้องไม่ซ้ำกันเป็นชุด ๆ

### B2. ตัวตรวจ leakage ทั้งสองตัวไม่มีทางจับอะไรได้เลย 🟠

**ไฟล์:** `src/sunseg/data/splits.py:100-107` (`_assert_disjoint`) และ `:122-135`
(`verify_no_harp_overlap`)

**อาการ:** ทั้งสองตัวถูกต้องในฐานะฟังก์ชัน และมี unit test ที่ผ่านจริง แต่**ที่จุดเรียกใช้จริง
มันเป็น tautology**:

- `_assert_disjoint` ตรวจ `assigned.duplicated("HARPNUM")` บนตารางที่มาจาก
  `harp_lifespans()` ซึ่งทำ `groupby("HARPNUM")` → HARPNUM ไม่ซ้ำโดยโครงสร้าง
- `verify_no_harp_overlap` ตรวจสมบัติเดียวกันบนตารางที่ `split` มาจาก
  `assigned.set_index("HARPNUM")["split"]` ซึ่งเป็น map 1:1 → HARP หนึ่งดวงมีได้ split เดียวเสมอ

ยืนยันแล้ว: สร้าง mapping ที่ตั้งใจให้รั่วยังไงก็ผ่าน เพราะรูปทรงข้อมูลไม่เปิดช่องให้รั่วแบบนั้น

ที่สำคัญกว่าคือ **ไม่มีตัวไหนตรวจ leakage แบบที่ docstring ของโมดูลห่วงอยู่จริง** — คือ
"หน้าต่าง `(t, t+horizon]` ของ sample ใน train ต้องไม่ไปทับ sample ของ val/test"
ตอนนี้เรื่องนี้ถูกการันตีด้วย `gap_days: 14` ใน YAML เพียงอย่างเดียว ไม่เคยถูกตรวจกับข้อมูลที่ผลิตออกมาจริง
(ค่าปัจจุบัน 14 วัน > horizon 24 ชม. จึงปลอดภัย — แต่เป็นความปลอดภัยที่ไม่มีอะไรค้ำ ถ้ามีคนลด `gap_days`)

**สิ่งที่ต้องแก้:** เพิ่มการตรวจบน **ผลลัพธ์** ใน `verify_no_harp_overlap` (หรือฟังก์ชันใหม่ข้าง ๆ):
`max(issue_time ของ train) + horizon < min(issue_time ของ val)` และเงื่อนไขเดียวกันสำหรับ val→test
เรียกจาก `build_sequences.py:113` และ `build_study_dataset.py:191` ที่เรียกตัวเดิมอยู่แล้ว
ส่วน `_assert_disjoint` จะเก็บไว้เป็น invariant ก็ได้ แต่ควรเขียนคอมเมนต์บอกตามตรงว่ามันกันอะไร

### B3. `build_sequences.py` ถอยไปใช้ข้อมูลทั้งชุดคำนวณ normalisation และเขียนไฟล์ก่อนคำนวณ 🟠

**ไฟล์:** `scripts/build_sequences.py:128-131` · `scripts/build_study_dataset.py:211-215`

**สองอาการที่คนละที่แต่เรื่องเดียวกัน:**

1. `build_sequences.py:128-131` — ถ้า train ว่าง จะ `train_mask = None` แล้วคำนวณ mean/std
   จาก **ทุก split รวมกัน** โดยแค่ log warning ไฟล์ `norm_stats.npz` ที่ออกมาไม่มีอะไรบอกว่า
   ถูกคำนวณแบบรั่ว และ `load_sequence_splits` ก็ดูไม่ออก
   เทียบกับ `build_study_dataset.py:193-205` ที่เจอสถานการณ์เดียวกันแล้ว **error แล้ว return 1**
   — เส้นทาง production กลับเป็นเส้นทางที่หย่อนกว่า

2. `build_study_dataset.py` เขียน `X.npy` + `meta.parquet` ที่บรรทัด 211-213 **ก่อน**
   `compute_normalisation` ที่บรรทัด 215 ซึ่งโยน exception ได้ (มี `raise` เรื่อง inf/nan อยู่จริง)
   ถ้าพัง จะเหลือ `X.npy` ใหม่คู่กับ `norm_stats.npz` **ของรอบก่อน** และ `load_study_arrays`
   ตรวจแค่ *จำนวน* feature ถ้าจำนวนเท่าเดิมก็ผ่านไปเงียบ ๆ พร้อม stats ผิดชุด

**สิ่งที่ต้องแก้:** (1) ทำให้ `build_sequences.py` error เหมือน `build_study_dataset.py` หรือ
อย่างน้อยบันทึก `train_only=False` ลง `norm_stats.npz` แล้วให้ `load_sequence_splits` warn ดัง ๆ
(2) ย้าย `compute_normalisation` ขึ้นไปก่อนการเขียนไฟล์ทั้งหมด เขียนสี่ไฟล์พร้อมกันหลังคำนวณเสร็จ

### B4. `/api/forecast` คืน 500 แถวแรกสุด แล้วเรียกแถวสุดท้ายว่า "latest" 🟠

**ไฟล์:** `src/sunseg/inference/forecast.py:288` (`find_rows`) · `app/routers/forecast.py:98`

**อาการ:** `find_rows` จบด้วย `.sort_values("issue_time").head(limit)` = เอา `limit` แถว
**เก่าที่สุด** ในช่วง แล้ว `forecast_series` ทำ `latest_row = rows.iloc[-1]` ไปสร้างแผง `latest`
HARP ที่มี sample เกิน `limit` (ปริยาย 500) ในช่วงที่ขอ จะแสดง "ค่าล่าสุด" เป็นจุดที่เก่าที่สุดลำดับ
ที่ 500 โดยไม่มีอะไรบอก — ทั้ง `probability`, `risk_level`, `attention`, `features` ผิดหมด

**สิ่งที่ต้องแก้:** แยก "ตัดจำนวน" ออกจาก "เรียงเวลา" — ถ้าผู้เรียกต้องการช่วงล่าสุดให้ใช้
`.tail(limit)` หรือส่ง argument บอกทิศ แล้วค่อยเรียงกลับ ตรวจผู้เรียกทุกตัวของ `find_rows`
ว่าต้องการปลายไหน (`forecast_at_time` ใช้ `limit=20000` บนหน้าต่าง ±3 ชม. จึงไม่โดน)

### B5. คีย์ config สามตัวถูก parse แต่ไม่เคยถูกอ่าน 🟠

ทั้งสามผ่าน pydantic `_Strict` เรียบร้อยจึงไม่มี error แต่แก้แล้วไม่มีผลอะไรเลย — ขัดกับ
docstring ของ `config.py` เองที่บอกว่ามีไว้ให้ "typo ใน config ระเบิดตั้งแต่ตอนโหลด"

| คีย์ | ประกาศที่ | ทำไมไม่มีผล |
|---|---|---|
| `tracking.yaml → rotation.snodgrass_A/B/C` | `config.py:388-393` | `Track.predict` (`tracker.py:91`) เรียก `rotate_longitude(lon, lat, delta)` โดยไม่ส่งสัมประสิทธิ์ → ค่าคงที่ระดับโมดูลใน `rotation.py:24-26` ชนะเสมอ |
| `lstm.yaml → eval.select_threshold_on` | `config.py:366` | grep ทั้ง repo แล้วไม่มีใครอ่าน ตั้งเป็น `train` ก็ยังเลือกบน val |
| `data.yaml → sequence.stride_hours: 1` | `config.py:247` | ไม่มีใครอ่าน — stride เป็น 1 จุดกริดเสมอโดยโครงสร้างของ `sliding_window_view` |

**สิ่งที่ต้องแก้:** เลือกอย่างใดอย่างหนึ่งต่อคีย์ — ต่อสายให้ใช้จริง หรือลบทั้งใน pydantic และใน YAML
(`extra="forbid"` ทำให้ต้องลบคู่กัน) อย่าทิ้งไว้ให้คนอ่านเข้าใจผิดว่าปรับได้
พ่วง: `rotation.py:33` `CARRINGTON_RATE` ไม่มีใครใช้เช่นกัน

### B6. `primary_metric` อนุญาตค่าที่โค้ดไม่รองรับ และห้ามค่าที่รองรับ 🟠

**ไฟล์:** `src/sunseg/config.py:367` เทียบกับ `src/sunseg/metrics.py:148`

```python
primary_metric: Literal["tss", "hss2", "auc"] = "tss"          # config.py
scorers = {"tss": tss, "hss2": hss2, "f1": ...}                 # metrics.py
```

ตั้ง `primary_metric: auc` ผ่าน validation แล้วไประเบิดเป็น `ValueError` ตอน epoch แรก
— **กลางการเทรน** ซึ่งคือสิ่งที่ docstring ของ `config.py` บอกว่าตั้งใจกันไว้ ส่วน `f1`
ที่โค้ดรองรับกลับถูกปฏิเสธตั้งแต่โหลด

**สิ่งที่ต้องแก้:** เปลี่ยน Literal เป็น `["tss", "hss2", "f1"]` (AUC ไม่ขึ้นกับ threshold
จึงใช้เลือก threshold ไม่ได้อยู่แล้ว) หรือดีกว่านั้น — ให้ Literal อ้างจากคีย์ของ `scorers` โดยตรง
เพื่อไม่ให้หลุดจากกันได้อีก

---

## C. ประสิทธิภาพ (ถูกต้องอยู่แล้ว แต่เปลืองโดยไม่จำเป็น)

### C1. `apply_normalisation` บังคับเป็น float64 ทั้งที่ไม่ต้อง 🟡

`signed_log1p` (`build_sequences.py:345`) cast เป็น float64 เสมอ ซึ่ง **จำเป็นเฉพาะใน
`compute_normalisation`** ที่ต้องยกกำลังสองหาค่า variance (ตามที่ docstring ของมันอธิบายไว้เอง)
แต่ `apply_normalisation` ไม่ได้คำนวณ variance เลย — และ `log1p(1e22) ≈ 50.7` อยู่ในพิสัยของ
float32 สบาย ๆ

วัดแล้ว: input float32 35 MB → intermediate float64 69 MB (**2.0×**) ทุกครั้งที่เรียก

จุดที่เจ็บ: `datasets/sequence.py:109` ทำกับ dataset ทั้งก้อน · `datasets/study.py:83` ทำใหม่
ทุกแบบ (5 รอบ) · `build_sequences.py:134` ทำกับ train ทั้งชุด **เพื่อพิมพ์ log บรรทัดเดียว**

**แก้:** เพิ่มพารามิเตอร์ `dtype` ให้ `signed_log1p` (float32 สำหรับ apply, float64 สำหรับ compute)
และเปลี่ยนบรรทัด sanity log ให้สุ่ม subsample แทนการแปลงทั้งชุด

### C2. `mask_iou` เทียบ mask เต็มภาพทุกคู่ 🟡

`detect.py:136` เก็บ `mask = labels == label_id` เป็น boolean เต็มเฟรม (512×512) ต่อ detection
แล้ว `tracker.py:186` เรียก `mask_iou` ทุกคู่ (track, detection) ที่ผ่านเกณฑ์ระยะ → `logical_and`
+ `logical_or` บนทั้งภาพทุกคู่ ทั้งที่ AR กินพื้นที่ไม่กี่ร้อยพิกเซล

`Detection.bbox` มีอยู่แล้ว — เช็ก bbox ตัดกันก่อน (ไม่ตัดกัน = คืน 0 ทันที) แล้วค่อยคำนวณ IoU
เฉพาะใน bbox ที่ซ้อนทับ ได้ผลเท่าเดิมเป๊ะ กระทบ `render_segmentation_video.py` และ
`render_harp_patch_video.py` ที่วนหลายพันเฟรมมากที่สุด

### C3. `build_unified_window` ให้ทุก HARP วิ่งบนกริดของทั้งหน้าต่าง 🟡

**ไฟล์:** `src/sunseg/data/study_dataset.py:167-201`

วัดจากค่าจริงของ `build_study_dataset.py` (main range 2011-01-01..2018-01-01 @ 12 ชม.):

```
กริดของหน้าต่าง        5,115 จุด
HARP หนึ่งดวงอยู่จริง      28 จุด  (0.55%)
windows ที่สร้างต่อ HARP  5,108 อัน (6.0 MB) — เหลือรอด keep ~21 อัน
x 4,000 HARP           ~24 วินาที และ ~24 GB ของการจองหน่วยความจำชั่วคราว
```

`ffill/bfill` ลากค่าสุดท้ายของ HARP ไปตลอด 7 ปีที่เหลือ แล้ว `sliding_window_view` สร้างหน้าต่าง
จากค่าที่ลากมาทั้งหมด ก่อนจะถูก `keep` (ซึ่งบังคับ `sharp_real_w[:, -1]`) ตัดทิ้งเกือบหมด
**ผลลัพธ์ถูกต้อง** — เป็นงานเปล่าล้วน ๆ

**แก้:** หั่นกริดเหลือ `[t_first, t_last]` ของ HARP นั้น โดย snap เข้ากับ anchor เดิม
(`grid[(grid >= t_first - tol) & (grid <= t_last + tol)]`) — การ snap กับ anchor เดียวกัน
คือหลักการที่ทั้งโมดูลตั้งอยู่บน จึงยังคง exact-timestamp match กับ intensity/X-ray ไว้ได้ครบ
ระวัง: `xray_values` ต้องหั่นด้วย index ชุดเดียวกัน

### C4. `find_rows` คัดลอกตาราง meta ทั้งก้อนทุก request 🟡

`inference/forecast.py:278` `self.meta.reset_index(names="row")` สร้างสำเนาเต็มก่อนกรอง
ทุกครั้งที่มีคนเรียก `/api/forecast` หรือ `/api/forecast/at`
→ ใส่คอลัมน์ `row` ครั้งเดียวใน `_load()` (`self.meta["row"] = np.arange(len(self.meta))`)

พ่วง: `app/routers/forecast.py:159` `rows.sort_values("_gap").drop_duplicates("HARPNUM")`
ใช้ sort ที่ไม่ stable — เมื่อสองแถวห่างเวลาเป้าหมายเท่ากัน แถวที่รอดขึ้นกับ pandas เวอร์ชัน
ใส่ `kind="stable"`

---

## D. รายละเอียดย่อย

| # | ไฟล์ | ปัญหา | แก้ |
|---|---|---|---|
| D1 | `tracking/tracker.py:242-259` | docstring บรรทัด 13-14 บอกว่า track ที่เป็น `tentative` แล้วหายจะถูกลบ **ทันที** แต่โค้ดปล่อยให้อยู่ครบ `max_age_frames` เท่ากับ track ที่ confirmed · `test_transient_false_positive_is_discarded` ผ่านได้เพราะส่ง `max_age_frames=1` ไม่ใช่ค่า 3 ใน `tracking.yaml` · ยืนยันแล้ว: detection ปลอมเฟรมเดียวสองครั้งห่างกัน 36 ชม. ในตำแหน่งที่ตรงกับการหมุน → ได้ track ที่ `confirmed` 1 เส้น | `if track.state == "tentative" and track.age_since_update > 0: continue` แล้วแก้เทสต์ให้ใช้ค่าจาก config |
| D2 | `scripts/train_study.py:296-297, 551` | ยังฝัง "พ.ค. 2024" / "พายุ Gannon" ไว้ตายตัว ทั้งที่คอมเมนต์ของ `SCOPE_TITLES` (บรรทัด 91-94) บันทึกไว้เองว่าเอาป้ายแบบนี้ออกแล้วเพราะหน้าต่างที่สองกลายเป็นช่วง 2022-2025 ทั่วไป — แก้ตกไปสองบรรทัด | ใช้ช่วงเวลาจริงจากตาราง `balance` แทนข้อความตายตัว |
| D3 | `scripts/train_unet.py:96-122` | gradient accumulation ทิ้งเศษท้าย epoch — ถ้า `len(loader) % grad_accum_steps != 0` batch สุดท้ายสูงสุด `grad_accum_steps-1` ชุดสะสม gradient ไว้แล้วไม่เคยถูก `step` แล้วโดนล้างตอนขึ้น epoch ใหม่ | `step()` เพิ่มอีกครั้งหลังจบลูปถ้ายังมี gradient ค้าง |
| D4 | `src/sunseg/config.py:251-254` | `SplitConfig` ไม่มี validator ว่า `train_end < val_end` ต่างจาก `TimeRange` ที่มี `_end_after_start` · สลับค่ากันจะได้ val/test เกือบว่างแบบเงียบ ๆ แทนที่จะ error ตอนโหลด | เพิ่ม `field_validator("val_end")` แบบเดียวกับ `TimeRange` และตรวจ `gap_days * 24 >= flare.horizon_hours` ด้วย |
| D5 | `src/sunseg/metrics.py:116-130` | ลูป `while` ระดับ Python ไล่หา tie ใน `roc_auc` — O(N) iteration ของ Python ล้วน และถูกเรียกทุก request ผ่าน `PredictionsStore.sweep` | ใช้วิธี vectorise (`np.unique(..., return_inverse=True, return_counts=True)`) หรือ cache ผลของ `sweep` ต่อ (model, split) |
| D6 | `scripts/build_study_dataset.py:155` | ขอบของหน้าต่าง intensity ใช้ `.between(start, end)` (รวมปลายทั้งสองข้าง) แต่ SHARP ใช้ `>= start` และ `< end` — ไม่ตรงกัน ตอนนี้ไม่มีผลเพราะสองหน้าต่างไม่ทับกัน แต่จะกัดทันทีที่มีหน้าต่างติดกัน | ใช้ `>= start` และ `< end` ให้เหมือนกันทั้งสองแหล่ง |
| D7 | `src/sunseg/train_utils.py:20-36` | `set_seed(deterministic=True)` ตั้งแค่ flag ของ cuDNN ไม่ได้เรียก `torch.use_deterministic_algorithms(True)` และไม่ตั้ง `CUBLAS_WORKSPACE_CONFIG` — cuDNN LSTM บน GPU จึงยังไม่ deterministic จริง รันซ้ำ (แบบ, seed) เดิมได้ตัวเลขไม่เท่าเดิม ซึ่งขัดกับสมมติฐานของ `paired_summary` ที่ว่า seed เดียวกันหักความผันผวนร่วมออกได้ | เรียก `torch.use_deterministic_algorithms(True, warn_only=True)` และตั้ง env var เมื่อ `deterministic=True` หรือเขียนข้อจำกัดนี้ลงรายงานให้ชัด |

---

## E. ห้ามทำ (กันแก้ถอยหลัง)

1. **ห้ามเปลี่ยนให้ `val`/`test` loader shuffle** — `scope_reports`, `long_predictions`,
   `build_predictions_frame` ทั้งหมดจับคู่ค่าทำนายกับ `meta` ด้วยลำดับแถวล้วน ๆ
   (`sequence.py:142-143` ตั้ง `shuffle=False` ไว้โดยตั้งใจ มีคอมเมนต์กำกับใน `train_lstm.py:135-138`)
2. **ห้ามย้ายการ normalise ไปทำตอน build** — `sequence.py:78-79` อธิบายไว้แล้วว่าจงใจเก็บ `X.npy`
   เป็นค่าดิบเพื่อตรวจย้อนกลับได้ และ `ForecastService.predict_batch` พึ่งพาสัญญานี้
3. **ห้ามแก้เกณฑ์ `keep` ใน `build_unified_window`** (`study_dataset.py:196-201`) —
   "เข้มงวดที่สุดเป็นเกณฑ์ร่วม" คือสิ่งที่ทำให้ทุกแบบเทียบกันได้ ข้อ C3 แก้แค่ขนาดกริด ไม่แตะเกณฑ์
4. **ห้ามเอา `xray_median`/`xray_max`/`xray_min` ออกจาก dataset** — ไม่มีแบบไหนใช้ก็จริง
   (`study_variants.yaml` เลือกแค่ `xray_log10_median`) แต่หลัก "เก็บครบ เลือกใช้ทีหลัง"
   คือสิ่งที่ทำให้เพิ่มแบบใหม่ได้โดยไม่ต้องสร้าง dataset ใหม่
5. **ห้าม `git commit` / `git push`** เว้นแต่ถูกสั่ง

---

## F. วิธียืนยันว่าแก้แล้วใช้ได้จริง

```bash
.venv/Scripts/python.exe -m pytest backend/tests -q          # ต้องผ่านทั้งหมดเหมือนเดิม
.venv/Scripts/python.exe -m ruff check backend
```

เทสต์ที่ควรเพิ่มพร้อมกับการแก้ (แต่ละข้อจับ regression ของตัวเอง):

- [ ] **A1** ชุดที่ทุก prob < 0.01 แต่จัดอันดับสมบูรณ์ → TSS ใกล้ 1 ไม่ใช่ 0
- [ ] **A1** `predictions_store.THRESHOLD_GRID` กับกริดของ `find_best_threshold` มาจากแหล่งเดียวกัน
- [ ] **A2** val ที่ positive = 0 → `main()` คืน 1 พร้อมข้อความ ไม่ใช่เทรนต่อ
- [ ] **A3** เปลี่ยน `columns` ของแบบหนึ่ง → signature เปลี่ยน → ไม่ข้าม
- [ ] **B1** `num_workers=4` แล้ว augmentation ของ sample คนละ worker ไม่ซ้ำกัน
- [ ] **B2** สร้าง meta ที่ train กับ val ทับกันในมิติเวลา → ตัวตรวจใหม่ต้อง raise
- [ ] **B4** HARP ที่มี sample เกิน `limit` → `latest.issue_time` ต้องเป็นจุดล่าสุดจริง
- [ ] **D1** `ARTracker()` ด้วยค่าปริยายจาก `tracking.yaml` → detection ปลอมเฟรมเดียวไม่กลายเป็น track

ส่วนที่ต้องรันของจริงหลังแก้ A1-A3 (เพราะกระทบตัวเลขที่รายงาน):

```bash
python backend/scripts/train_study.py --force          # เทรนใหม่ทุกแบบทุก seed
```

แล้วเทียบ `artifacts/lstm_feature_ablation/report.md` กับของเดิม — ถ้า TSS ขยับมาก
แปลว่า A1 หรือ A2 เคยกัดผลอยู่จริง และตัวเลขในรายงานเดิมใช้ไม่ได้

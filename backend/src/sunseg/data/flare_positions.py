"""ตำแหน่ง flare จาก PositionFlare จับคู่เข้ากับแคตตาล็อกที่โมเดลใช้ทำ label

แคตตาล็อกของโมเดล (``data/interim/flares.parquet``) มีพิกัดแค่บางส่วน — รายงาน NGDC มีตำแหน่ง
ถึงปี 2017 เท่านั้น ส่วนปี 2018+ ที่มาจาก HEK ไม่มีเลย ขณะที่ PositionFlare (``D:/position_flare``)
รวบรวมตำแหน่งจากหลายแหล่ง (SWPC > XRS > XRS-HPC > AR) ครอบ 1996-2026 โมดูลนี้เอาตำแหน่งจาก
ที่นั่นมาแปะให้ flare ของแคตตาล็อกโมเดล **โดยไม่เพิ่มหรือตัด flare ดวงไหน**: ชุด flare และคลาส
ที่แผนที่แสดงคือชุดเดียวกับที่ LSTM ใช้ทำ label เสมอ ดวงที่หาคู่ไม่เจอยังอยู่ครบ แค่ไม่มีพิกัด

จับคู่ด้วย **เวลาพีค** (ใกล้สุดภายใน tolerance) ไม่เทียบคลาสตรง ๆ เพราะสองแคตตาล็อกใช้สเกล
ฟลักซ์ต่างกัน: รายงาน NGDC เดิม (≤ 2017) ยังคูณตัวคูณ 0.7 ของ GOES-13/15 อยู่ ขณะที่
PositionFlare ใช้ค่า science ที่ NOAA reprocess แล้ว (สูงกว่า ~1.43 เท่า) — C1.0 ของโมเดล
จึงเป็น C1.4 ใน PositionFlare อัตราส่วนฟลักซ์ใช้แค่กันจับผิดดวงเท่านั้น ไม่ใช่เกณฑ์หลัก

วัดจริงบนข้อมูลชุดปัจจุบัน (2026-09): 10,553 จาก 11,394 ดวงเวลาพีคตรงกันเป๊ะ, 10,876 ภายใน
±5 นาที, 10,943 ภายใน ±10 นาที — tolerance เริ่มต้น 10 นาทีจึงเก็บเกือบทั้งหมดโดยไม่ต้องกว้าง
จนเสี่ยงจับ flare คนละดวงที่เกิดติด ๆ กัน
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: tolerance เริ่มต้นของเวลาพีค (นาที) — ดูตัวเลขที่วัดได้ใน docstring บนสุด
MATCH_TOLERANCE_MIN = 10.0

#: อัตราส่วนฟลักซ์ (PositionFlare / โมเดล) ที่ยอมรับ — สเกลต่างกันจริงแค่ ~1.43 เท่า
#: เผื่อกว้างไว้ 4 เท่าเพื่อกันเฉพาะกรณีจับได้ดวงที่แรงต่างกันคนละระดับชัดเจน
MAX_FLUX_RATIO = 4.0

#: PositionFlare เก็บเฉพาะ C/M/X — flare ระดับ A/B ของโมเดลไม่มีคู่ให้จับ จึงไม่อยู่ในแผนที่
MAPPED_CLASSES = ("C", "M", "X")

#: จับคู่ซ้ำกี่รอบ — รอบถัดไปให้ดวงที่เสียคู่ใกล้สุดให้ดวงอื่นได้ลองคู่ถัดไปที่ยังว่าง
MATCH_PASSES = 3

#: คอลัมน์ที่อ่านจาก ``flares_all_cycles.csv`` ของ PositionFlare
PF_USECOLS = [
    "flare_id", "cycle", "time_peak", "goes_class", "xrsb_irrad",
    "lat", "lon", "pos_source", "pos_at_limb", "active_region", "satellite",
]

#: คอลัมน์ที่ได้จาก PositionFlare หลังจับคู่ (ดวงที่ไม่มีคู่ = ค่าว่างทั้งหมด)
PF_OUTPUT_COLUMNS = [
    "pf_flare_id", "pf_goes_class", "pf_peak_flux", "lat", "lon",
    "pos_source", "pos_at_limb", "pf_active_region", "satellite", "cycle",
]

OUTPUT_COLUMNS = [
    "peak_time", "start_time", "end_time", "goes_class", "peak_flux", "noaa_ar",
    "harpnum", "n_harps", "matched", "match_dt_min", *PF_OUTPUT_COLUMNS,
]


def read_position_flare_csv(path: Path) -> pd.DataFrame:
    """อ่านแคตตาล็อกของ PositionFlare เรียงตามเวลาพีค (เวลาเป็น UTC แบบไม่มีโซน เหมือนฝั่งโมเดล)"""
    frame = pd.read_csv(path, usecols=PF_USECOLS, low_memory=False)
    frame["peak"] = pd.to_datetime(frame["time_peak"], errors="coerce")
    frame = frame[frame["peak"].notna()]
    return frame.sort_values("peak").reset_index(drop=True)


def model_catalog_events(pairs: pd.DataFrame) -> pd.DataFrame:
    """ยุบไฟล์คู่ (flare, HARP) ของโมเดลกลับเป็น flare ไม่ซ้ำ เฉพาะคลาส C/M/X

    flare ดวงเดียวแมปได้หลาย HARP (NOAA AR เดียวคลุมหลายแพตช์) — เลือก HARP เลขน้อยสุด
    เป็นตัวแทนให้ผลเหมือนกันทุกครั้งที่รัน และเก็บจำนวน HARP ไว้ใน ``n_harps``
    """
    frame = pairs.copy()
    frame["peak_time"] = pd.to_datetime(frame["peak_time"])
    letters = frame["goes_class"].astype("string").str[0].str.upper()
    frame = frame[letters.isin(MAPPED_CLASSES)]

    key = ["peak_time", "goes_class"]
    harps = frame.groupby(key)["HARPNUM"].agg(harpnum="min", n_harps="nunique")
    events = (
        frame.drop_duplicates(key)
        .set_index(key)[["start_time", "end_time", "peak_flux", "noaa_ar"]]
        .join(harps)
        .reset_index()
    )
    events["n_harps"] = events["n_harps"].astype(int)
    return events.sort_values("peak_time").reset_index(drop=True)


def match_positions(
    events: pd.DataFrame,
    position_flare: pd.DataFrame,
    tolerance_min: float = MATCH_TOLERANCE_MIN,
    max_flux_ratio: float = MAX_FLUX_RATIO,
) -> pd.DataFrame:
    """แปะตำแหน่งจาก PositionFlare ให้ flare ของโมเดล — คืนแถวเท่ากับ ``events`` เป๊ะ

    กติกา: เวลาพีคห่างกันไม่เกิน ``tolerance_min``, ฟลักซ์ต่างกันไม่เกิน ``max_flux_ratio``
    เท่า, และ flare ของ PositionFlare หนึ่งดวงจับคู่ได้กับ flare ของโมเดลแค่ดวงเดียว (ดวงที่
    เวลาใกล้กว่าชนะ ดวงที่แพ้ได้ลองคู่ถัดไปในรอบต่อไป)
    """
    left = events.sort_values("peak_time").reset_index(drop=True)
    right = position_flare.rename(columns={
        "peak": "pf_peak",
        "flare_id": "pf_flare_id",
        "goes_class": "pf_goes_class",
        "xrsb_irrad": "pf_peak_flux",
        "active_region": "pf_active_region",
    })[["pf_peak", *PF_OUTPUT_COLUMNS]].sort_values("pf_peak").reset_index(drop=True)

    tolerance = pd.Timedelta(minutes=tolerance_min)
    assigned = pd.Series(-1, index=left.index)          # index แถวของ right ที่จับได้
    dt_min = pd.Series(float("nan"), index=left.index)

    for _ in range(MATCH_PASSES):
        todo = left.index[assigned < 0]
        free = right.drop(index=assigned[assigned >= 0].to_numpy())
        if not len(todo) or free.empty:
            break

        candidate = pd.merge_asof(
            left.loc[todo, ["peak_time", "peak_flux"]].reset_index(names="event"),
            free[["pf_peak", "pf_peak_flux"]].reset_index(names="pf_row"),
            left_on="peak_time", right_on="pf_peak",
            direction="nearest", tolerance=tolerance,
        )
        ratio = candidate["pf_peak_flux"] / candidate["peak_flux"]
        # ฟลักซ์ว่างฝั่งใดฝั่งหนึ่ง = ตรวจอัตราส่วนไม่ได้ — ยอมรับตามเวลาอย่างเดียว
        flux_ok = ratio.isna() | ratio.between(1 / max_flux_ratio, max_flux_ratio)
        candidate = candidate[candidate["pf_row"].notna() & flux_ok].copy()
        if candidate.empty:
            break

        candidate["abs_dt"] = (candidate["pf_peak"] - candidate["peak_time"]).abs()
        winners = candidate.sort_values(["abs_dt", "event"]).drop_duplicates("pf_row")
        rows = winners["pf_row"].astype(int).to_numpy()
        assigned.loc[winners["event"].to_numpy()] = rows
        dt_min.loc[winners["event"].to_numpy()] = (
            (right.loc[rows, "pf_peak"].to_numpy() - winners["peak_time"].to_numpy())
            / pd.Timedelta(minutes=1)
        )

    matched = assigned >= 0
    out = left.copy()
    picked = right.reindex(assigned.where(matched).to_numpy()).reset_index(drop=True)
    for column in PF_OUTPUT_COLUMNS:
        out[column] = picked[column].to_numpy()
    out["matched"] = matched.to_numpy()
    out["match_dt_min"] = dt_min.round(2)
    out["pos_at_limb"] = out["pos_at_limb"].astype("boolean")
    for column in ("pf_flare_id", "pf_active_region", "cycle", "harpnum", "noaa_ar"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round().astype("Int64")
    return out[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------- #
# ฝั่งแอป — อ่านผลที่ scripts/data/build_flare_positions.py เขียนไว้
# --------------------------------------------------------------------------- #

#: epoch ของคอลัมน์เวลาใน payload (นาทีนับจากจุดนี้) — ตัวเลขเล็กกว่า ISO string หลายเท่า
PAYLOAD_EPOCH = pd.Timestamp("2000-01-01")


def meta_path_for(parquet_path: Path) -> Path:
    """ไฟล์ข้อมูลประกอบที่เขียนคู่กับ parquet (ที่มา, tolerance, เวลาสร้าง)"""
    return parquet_path.with_suffix(".json")


def _optional(values, cast):
    return [None if pd.isna(v) else cast(v) for v in values]


class FlarePositionStore:
    """ผลการจับคู่ที่สร้างไว้แล้ว — ไม่มีไฟล์ก็เปิดแอปได้ แค่หน้าแรกบอกให้รันสคริปต์"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.frame: pd.DataFrame | None = None
        self.meta: dict = {}
        self._payload: dict | None = None

        if not self.path.exists():
            logger.warning(
                "ไม่พบตำแหน่ง flare ที่ %s — รัน backend/scripts/data/build_flare_positions.py", self.path
            )
            return
        self.frame = pd.read_parquet(self.path)
        meta_file = meta_path_for(self.path)
        if meta_file.exists():
            self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
        logger.info(
            "โหลดตำแหน่ง flare: %d ดวง (จับคู่ PositionFlare ได้ %d, มีพิกัด %d)",
            len(self.frame), int(self.frame["matched"].sum()), int(self.frame["lat"].notna().sum()),
        )

    @property
    def available(self) -> bool:
        return self.frame is not None

    def info(self) -> dict:
        return {"available": self.available, "path": str(self.path), **self.meta}

    def payload(self) -> dict:
        """ข้อมูลทั้งชุดแบบ columnar — ทุก list ยาว ``n`` เท่ากัน สร้างครั้งเดียวแล้วเก็บไว้"""
        if self.frame is None:
            raise RuntimeError("ยังไม่มีตำแหน่ง flare — รัน backend/scripts/data/build_flare_positions.py")
        if self._payload is not None:
            return self._payload

        f = self.frame
        peak = pd.to_datetime(f["peak_time"])

        def offset(column: str) -> list[int | None]:
            minutes = (pd.to_datetime(f[column]) - peak) / pd.Timedelta(minutes=1)
            return _optional(minutes.round(), int)

        self._payload = {
            "epoch": PAYLOAD_EPOCH.isoformat(),
            "n": len(f),
            "n_matched": int(f["matched"].sum()),
            "n_located": int(f["lat"].notna().sum()),
            "tolerance_min": float(self.meta.get("tolerance_min", MATCH_TOLERANCE_MIN)),
            "source": self.meta.get("source_csv"),
            "built_at": self.meta.get("built_at"),
            "t": ((peak - PAYLOAD_EPOCH) / pd.Timedelta(minutes=1)).round().astype(int).tolist(),
            "start": offset("start_time"),
            "end": offset("end_time"),
            "goes_class": f["goes_class"].astype(str).tolist(),
            "peak_flux": [float(v) for v in f["peak_flux"]],
            "noaa_ar": _optional(f["noaa_ar"], int),
            "harpnum": _optional(f["harpnum"], int),
            "n_harps": [int(v) for v in f["n_harps"]],
            "match_dt_min": _optional(f["match_dt_min"], lambda v: round(float(v), 1)),
            "pf_goes_class": _optional(f["pf_goes_class"], str),
            "lat": _optional(f["lat"], lambda v: round(float(v), 1)),
            "lon": _optional(f["lon"], lambda v: round(float(v), 1)),
            "pos_source": _optional(f["pos_source"], str),
            "limb": _optional(f["pos_at_limb"], bool),
            "pf_active_region": _optional(f["pf_active_region"], int),
            "satellite": _optional(f["satellite"], str),
            "cycle": _optional(f["cycle"], int),
        }
        return self._payload

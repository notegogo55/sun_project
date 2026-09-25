"""ฟังก์ชันบริสุทธิ์ของการสกัดความเข้มแสงตลอดช่วง 2011-2017 + พ.ค. 2024 (ticket 05 ของ
``.scratch/lstm-feature-ablation/``) — แยกจากสคริปต์ ``backend/scripts/study/extract_intensity.py``
เพื่อทดสอบได้โดยไม่ต้องมีเฟรมหรือโมเดลจริง

ด่านตรวจเดือนเดียว (``intensity_gate_check.py``) พิสูจน์ท่อไว้แล้วบน พ.ค. 2024 การขยายไปทั้ง
ช่วงเพิ่มโจทย์สามข้อที่เดือนเดียวไม่มี:

1. **ต้องหยุดแล้วรันต่อได้** — หลายพันเฟรม และเฟรมยังทยอยมาจากการดาวน์โหลด จึงเก็บ ledger
   หนึ่งแถวต่อเฟรมไว้ตัดสินว่าเฟรมไหนเสร็จแล้ว (:func:`completed_stems`)
2. **การหา HARP ที่ควรเห็นในแต่ละเฟรมต้องเร็ว** — ตาราง SHARP มี ~1.2 ล้านแถว วิธีของด่าน
   เดิมที่กรองทั้งตารางทุกเฟรมจะกลายเป็นหลายพันล้านการเปรียบเทียบ
   (:func:`expected_harps_by_frame`)
3. **หลักฐานการลบ drift ต้องมาจากตารางจริงทั้งช่วง** ไม่ใช่ตัวอย่างรายสัปดาห์ 60 เฟรม — ค่าดิบ
   ย้อนคืนได้ตรง ๆ จาก ``normalised × quiet_sun`` เพราะมัธยฐานคูณค่าคงที่เข้าไปได้
   (:func:`drift_overlap`)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

#: สถานะของเฟรมใน ledger — มีแค่ ``ok`` ที่นับว่าเสร็จ ที่เหลือถูกลองใหม่ทุกครั้งที่รัน เพราะ
#: เป็นของที่ยังมาไม่ถึง (ภาพ/แผนที่ระบุตัวตนยังดาวน์โหลดไม่เสร็จ, AIA ยังไม่ครบทุกช่อง)
STATUS_OK = "ok"
STATUS_NO_IMAGE = "skipped_no_image"
STATUS_NO_IDENTITY = "skipped_no_identity_map"
STATUS_MISSING_AIA = "skipped_missing_aia"

#: คอลัมน์ของ ledger — ``missed_unet``/``missed_no_identity`` เป็น list ของ HARPNUM
LEDGER_COLUMNS: tuple[str, ...] = (
    "stem",
    "window",
    "status",
    "n_expected",
    "n_matched",
    "n_extra",
    "n_rows",
    "missed_unet",
    "missed_no_identity",
    "missing_channels",
)


def frame_time(stem: str) -> pd.Timestamp:
    """แปลงชื่อเฟรม ``YYYYMMDD_HHMMSS`` เป็นเวลา"""
    return pd.Timestamp(datetime.strptime(stem, "%Y%m%d_%H%M%S"))


#: ระยะห่างสูงสุดที่ถือว่าแถว SHARP "อยู่ในเฟรม" — เท่ากับของด่านเดือนเดียว
MATCH_TOLERANCE = pd.Timedelta(minutes=30)


def expected_harps_by_frame(
    sharp: pd.DataFrame,
    stems: list[str],
    tolerance: pd.Timedelta = MATCH_TOLERANCE,
) -> dict[str, set[int]]:
    """HARP ที่ควรถูกมองเห็นในแต่ละเฟรม — แถว SHARP ที่ ``|t_rec - เวลาเฟรม| <= tolerance``

    ให้ผลเท่ากับ ``expected_harps_by_frame`` ของด่านเดือนเดียวทุกประการ (รวมขอบทั้งสองข้าง)
    แต่หาช่วงด้วย ``searchsorted`` บนตารางที่เรียงเวลาแล้ว แทนการกรองทั้งตารางทุกเฟรม
    """
    if sharp.empty:
        return {stem: set() for stem in stems}

    ordered = sharp.sort_values("t_rec", kind="stable")
    times = ordered["t_rec"].to_numpy(dtype="datetime64[ns]")
    harps = ordered["HARPNUM"].to_numpy()
    tol = tolerance.to_timedelta64().astype("timedelta64[ns]")

    result: dict[str, set[int]] = {}
    for stem in stems:
        moment = frame_time(stem).to_datetime64().astype("datetime64[ns]")
        lo = np.searchsorted(times, moment - tol, side="left")
        hi = np.searchsorted(times, moment + tol, side="right")
        result[stem] = {int(h) for h in np.unique(harps[lo:hi])}
    return result


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def latest_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """แถวล่าสุดของแต่ละเฟรม — เฟรมเดียวถูกบันทึกได้หลายครั้ง (เช่น รอบแรกยังไม่มี AIA ครบ
    แล้วรอบหลังมีครบ) สถานะที่ใช้คือของรอบหลังสุดเสมอ"""
    if ledger.empty:
        return pd.DataFrame(columns=list(LEDGER_COLUMNS))
    return ledger.drop_duplicates("stem", keep="last").sort_values("stem").reset_index(drop=True)


def completed_stems(ledger: pd.DataFrame) -> set[str]:
    """เฟรมที่สกัดเสร็จแล้วจริง (สถานะล่าสุดเป็น ``ok``) — รอบถัดไปข้ามเฟรมเหล่านี้"""
    latest = latest_ledger(ledger)
    return set(latest.loc[latest["status"] == STATUS_OK, "stem"])


def _ok_frames(ledger: pd.DataFrame) -> pd.DataFrame:
    latest = latest_ledger(ledger)
    return latest[latest["status"] == STATUS_OK]


def overall_match_rate(ledger: pd.DataFrame) -> tuple[int, int]:
    """``(n_matched, n_expected)`` รวมทุกเฟรมที่ ok — ตัวเลขเดียวกับ "อัตราการจับคู่" ของด่านเดิม"""
    ok = _ok_frames(ledger)
    return int(ok["n_matched"].sum()), int(ok["n_expected"].sum())


def match_rate_by_year(ledger: pd.DataFrame) -> pd.DataFrame:
    """อัตราการจับคู่รายปี (เฉพาะเฟรมที่ ok) พร้อมจำนวนที่พลาดแยกตามสาเหตุ

    เป็นตัวเลขที่ต้องใช้วินิจฉัยเมื่ออัตรารวมต่ำกว่าด่านเดือนเดียว — เช่น ถ้าตกเฉพาะบางปีของ
    solar cycle จะชี้ไปที่ AR เล็ก/จางที่ U-Net มองไม่เห็น มากกว่าท่อที่พัง
    """
    columns = [
        "year", "n_frames", "n_expected", "n_matched",
        "n_missed_unet", "n_missed_no_identity", "n_extra", "match_rate",
    ]
    ok = _ok_frames(ledger)
    if ok.empty:
        return pd.DataFrame(columns=columns)

    ok = ok.assign(
        year=ok["stem"].str[:4].astype(int),
        n_missed_unet=ok["missed_unet"].map(len),
        n_missed_no_identity=ok["missed_no_identity"].map(len),
    )
    table = (
        ok.groupby("year")
        .agg(
            n_frames=("stem", "count"),
            n_expected=("n_expected", "sum"),
            n_matched=("n_matched", "sum"),
            n_missed_unet=("n_missed_unet", "sum"),
            n_missed_no_identity=("n_missed_no_identity", "sum"),
            n_extra=("n_extra", "sum"),
        )
        .reset_index()
    )
    table["match_rate"] = table["n_matched"] / table["n_expected"].where(table["n_expected"] > 0)
    return table[columns]


def missed_pairs(ledger: pd.DataFrame) -> pd.DataFrame:
    """คู่ (เฟรม, HARP) ที่ควรเห็นแต่จับคู่ไม่ได้ พร้อมสาเหตุ หนึ่งแถวต่อคู่

    ``reason`` เป็น ``unet_missed`` (HARP อยู่ในแผนที่ระบุตัวตน แต่ U-Net ไม่ทำนาย AR ตรงนั้น)
    หรือ ``not_in_identity_map`` (ไม่มีพิกเซลของ HARP นี้ในแผนที่เลย)
    """
    rows: list[dict] = []
    for row in _ok_frames(ledger).itertuples(index=False):
        for harp in row.missed_unet:
            rows.append({"stem": row.stem, "HARPNUM": int(harp), "reason": "unet_missed"})
        for harp in row.missed_no_identity:
            rows.append({"stem": row.stem, "HARPNUM": int(harp), "reason": "not_in_identity_map"})
    return pd.DataFrame(rows, columns=["stem", "HARPNUM", "reason"])


# --------------------------------------------------------------------------- #
# การลบ drift ข้ามปี — วัดจากตารางความเข้มแสงจริงทั้งช่วง
# --------------------------------------------------------------------------- #


def iqr_overlap(a: pd.Series, b: pd.Series) -> float:
    """สัดส่วนที่ช่วง [p25, p75] ของสองกลุ่มคาบเกี่ยวกัน (0 = ไม่เกี่ยวเลย, 1 = ซ้อนทับสมบูรณ์)

    ตัววัดเดียวกับ ``drift_check`` ของด่านเดือนเดียว เพื่อให้ตัวเลขเทียบกันได้ตรง ๆ
    """
    lo = max(a.quantile(0.25), b.quantile(0.25))
    hi = min(a.quantile(0.75), b.quantile(0.75))
    span = max(hi - lo, 0.0)
    widest = max(a.quantile(0.75) - a.quantile(0.25), b.quantile(0.75) - b.quantile(0.25), 1e-9)
    return float(span / widest)


def _raw_and_normalised(table: pd.DataFrame, channel: str) -> pd.DataFrame:
    """มัธยฐานของ AR ทั้งแบบ normalise แล้วและแบบดิบ (DN/s) ต่อแถว

    ค่าดิบย้อนคืนได้แม่นยำ: ``normalised = raw / quiet_sun`` ทั้งภาพ และมัธยฐานของ
    ``raw / c`` เท่ากับ ``median(raw) / c`` เมื่อ ``c > 0`` จึงไม่ต้องเก็บค่าดิบแยกไว้
    """
    median_col, quiet_col = f"{channel}_median", f"{channel}_quiet_sun"
    df = table[["issue_time", median_col, quiet_col]].dropna()
    return pd.DataFrame(
        {
            "year": df["issue_time"].str[:4].astype(int).to_numpy(),
            "normalised": df[median_col].to_numpy(dtype=float),
            "raw": (df[median_col] * df[quiet_col]).to_numpy(dtype=float),
            "quiet_sun": df[quiet_col].to_numpy(dtype=float),
        }
    )


def drift_overlap(table: pd.DataFrame, channel: str) -> dict:
    """เทียบการกระจายของความเข้มแสง AR ระหว่างปีต้นช่วงกับปลายช่วง ก่อนและหลัง normalise

    ถ้า normalisation ลบ degradation ได้จริง overlap หลัง normalise ต้องมากกว่าก่อน —
    เกณฑ์เดียวกับด่านเดือนเดียว (ปีต้น/ปลาย = หนึ่งในสามแรก/หลังของปีที่มีข้อมูล)
    """
    if table.empty or f"{channel}_median" not in table.columns:
        return {"ok": False, "reason": f"ไม่มีคอลัมน์ {channel}_median"}

    samples = _raw_and_normalised(table, channel)
    years = sorted(samples["year"].unique())
    if len(years) < 2:
        return {"ok": False, "reason": "ต้องมีข้อมูลอย่างน้อย 2 ปีจึงจะเทียบข้ามปีได้"}

    k = max(1, len(years) // 3)
    early_years, late_years = [int(y) for y in years[:k]], [int(y) for y in years[-k:]]
    early = samples[samples["year"].isin(early_years)]
    late = samples[samples["year"].isin(late_years)]

    raw_overlap = iqr_overlap(early["raw"], late["raw"])
    norm_overlap = iqr_overlap(early["normalised"], late["normalised"])
    return {
        "ok": True,
        "channel": channel,
        "early_years": early_years,
        "late_years": late_years,
        "n_early": int(len(early)),
        "n_late": int(len(late)),
        "raw_early_median": float(early["raw"].median()),
        "raw_late_median": float(late["raw"].median()),
        "norm_early_median": float(early["normalised"].median()),
        "norm_late_median": float(late["normalised"].median()),
        "raw_overlap": raw_overlap,
        "norm_overlap": norm_overlap,
        "improved": norm_overlap > raw_overlap,
    }


def drift_by_year(table: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """มัธยฐานรายปีของระดับ quiet Sun, ค่าดิบ และค่า normalise ของ AR ต่อช่อง

    ระดับ quiet Sun ที่ลดลงตามปีคือหน้าตาของ instrument degradation (เด่นที่ 304/171) —
    ค่า normalise ควรนิ่งกว่าค่าดิบถ้า normalisation ทำงาน (แต่ไม่จำเป็นต้องนิ่งสนิท
    เพราะความสว่างของ AR เปลี่ยนตาม solar cycle จริงด้วย)
    """
    if table.empty:
        return pd.DataFrame(columns=["year", "n_rows"])

    per_channel = []
    for channel in channels:
        if f"{channel}_median" not in table.columns:
            continue
        samples = _raw_and_normalised(table, channel)
        per_channel.append(
            samples.groupby("year").agg(
                **{
                    f"{channel}_quiet_sun": ("quiet_sun", "median"),
                    f"{channel}_raw": ("raw", "median"),
                    f"{channel}_normalised": ("normalised", "median"),
                }
            )
        )
    counts = table["issue_time"].str[:4].astype(int).value_counts().rename("n_rows")
    return pd.concat([counts, *per_channel], axis=1).sort_index().rename_axis("year").reset_index()

"""สกัดความเข้มแสงราย HARP ตลอดช่วง 2011-2017 + หน้าต่าง พ.ค. 2024 พร้อมรายงานด่านตรวจบน
ช่วงเต็ม (ticket 05 ของ ``.scratch/lstm-feature-ablation/``)

    # ทดสอบก่อนเสมอ
    python backend/scripts/extract_intensity.py --limit 20

    # รันเต็ม — หยุดกลางคันแล้วรันใหม่ได้ เฟรมที่เสร็จแล้วถูกข้าม
    python backend/scripts/extract_intensity.py

    # รวมผลที่สกัดไว้แล้วเป็นไฟล์/รายงานใหม่ โดยไม่ประมวลผลเฟรมเพิ่ม
    python backend/scripts/extract_intensity.py --report-only

ท่อเดียวกับด่านตรวจเดือนเดียว (``intensity_gate_check.py``) ทุกขั้น: U-Net ทำนาย mask →
``extract_frame_intensities`` (detect → quiet-Sun normalise → จับคู่ HARP ผ่านแผนที่ระบุตัวตน)
ใบนี้คือการขยายสเกล ไม่ใช่การออกแบบใหม่

**เฟรมที่ยังไม่พร้อมไม่ถือว่าเสร็จ** — เฟรมที่ยังไม่มีภาพ, แผนที่ระบุตัวตน หรือ AIA ครบทุกช่อง
ถูกบันทึกสถานะไว้แล้วลองใหม่ทุกครั้งที่รัน ระหว่างที่ ``download_images.py``/``download_aia.py``
ยังทยอยเติมเฟรม จึงรันสคริปต์นี้ซ้ำได้เรื่อย ๆ

ผลลัพธ์ใต้ ``artifacts/intensity_study/full_range/``:

- ``intensity_full_range.parquet`` — schema เดียวกับ ``gate_check/intensity_may2024.parquet``
  (``build_study_dataset.py`` หยิบทุกไฟล์ ``intensity*.parquet`` ใต้ ``intensity_study/`` เอง
  ไม่ต้องแก้สคริปต์นั้น — แถวของ พ.ค. 2024 ที่ซ้ำกับของด่านเดิมถูก dedupe ที่นั่น)
- ``match_report.parquet`` — ledger หนึ่งแถวต่อเฟรม (สถานะล่าสุด)
- ``report.md`` — ด่านตรวจบนช่วงเต็ม
- ``parts/`` — ผลรายรอบที่ทำให้รันต่อได้ ตั้งชื่อ ``rows_*``/``ledger_*`` **ไม่ขึ้นต้นด้วย
  ``intensity``** โดยตั้งใจ ไม่ให้ ``build_study_dataset.py`` หยิบซ้ำ
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.aia import AiaFrameStore  # noqa: E402
from sunseg.data.build_sequences import (  # noqa: E402
    build_flare_lookup,
    clean_sharp_frame,
    label_times,
)
from sunseg.data.frame_wcs import FrameWcsStore  # noqa: E402
from sunseg.data.intensity import extract_frame_intensities  # noqa: E402
from sunseg.data.intensity_range import (  # noqa: E402
    LEDGER_COLUMNS,
    STATUS_MISSING_AIA,
    STATUS_NO_IDENTITY,
    STATUS_NO_IMAGE,
    STATUS_OK,
    completed_stems,
    drift_by_year,
    drift_overlap,
    expected_harps_by_frame,
    latest_ledger,
    match_rate_by_year,
    missed_pairs,
    overall_match_rate,
)
from sunseg.inference.segment import SegmentationService  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("extract_intensity")

OUT_SUBDIR = Path("intensity_study") / "full_range"
GATE_CHECK_TABLE = Path("other_results") / "checks" / "intensity_study_gate_check" / "intensity_may2024.parquet"

#: อัตราการจับคู่ที่ด่านเดือนเดียวเคยได้ (``gate_check/report.md``: 584/621)
GATE_MATCH_RATE = 584 / 621
#: ต่ำกว่าด่านเดือนเดียวเกินเท่านี้ = "ต่ำกว่าอย่างมีนัย" ที่ ticket 05 บังคับให้วินิจฉัย
MATCH_RATE_ALERT = 0.05

STATUS_MEANINGS = {
    STATUS_OK: "สกัดเสร็จ (นับในทุกตัวเลขด้านล่าง)",
    STATUS_NO_IDENTITY: "มีภาพแต่ยังไม่มีแผนที่ระบุตัวตน — รอ download_images.py ดาวน์โหลดใหม่",
    STATUS_MISSING_AIA: "AIA ยังไม่ครบทุกช่อง — รอ download_aia.py (หรือคลัง synoptic ไม่มีไฟล์)",
    STATUS_NO_IMAGE: "ไม่มีภาพ",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2011-01-01", help="จุดเริ่มของช่วงหลัก YYYY-MM-DD")
    p.add_argument("--end", default="2017-12-31", help="จุดสิ้นสุดของช่วงหลัก YYYY-MM-DD (รวมวันนั้น)")
    p.add_argument("--limit", type=int, help="ประมวลผลเฟรมที่ยังไม่เสร็จแค่ N เฟรมแรก (ใช้ตอนทดสอบ)")
    p.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="อุปกรณ์ที่ใช้รัน U-Net (ปริยาย: GPU ถ้ามี)",
    )
    p.add_argument(
        "--flush-every", type=int, default=100,
        help="บันทึกผลลงดิสก์ทุก N เฟรม — ถ้าหยุดกลางคันจะเสียงานไม่เกินเท่านี้",
    )
    p.add_argument("--report-only", action="store_true", help="ไม่ประมวลผลเฟรมเพิ่ม แค่รวมผลที่มีแล้วเขียนรายงานใหม่")
    return p.parse_args()


def _day(text) -> str:
    return pd.Timestamp(text).strftime("%Y%m%d")


def select_stems(images_dir: Path, windows: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """เฟรมบนดิสก์ที่อยู่ในหน้าต่างใดหน้าต่างหนึ่ง คืน ``[(stem, ชื่อหน้าต่าง), ...]``"""
    selected: list[tuple[str, str]] = []
    for path in sorted(images_dir.glob("*.npy")):
        day = path.stem[:8]
        for name, first, last in windows:
            if first <= day <= last:
                selected.append((path.stem, name))
                break
    return selected


def window_of(stems: pd.Series, windows: list[tuple[str, str, str]]) -> pd.Series:
    days = stems.str[:8]
    labels = pd.Series("", index=stems.index)
    for name, first, last in windows:
        labels[(days >= first) & (days <= last)] = name
    return labels


# --------------------------------------------------------------------------- #
# การเก็บผลแบบรันต่อได้
# --------------------------------------------------------------------------- #


class PartWriter:
    """เก็บผลไว้ในหน่วยความจำ แล้วเขียนเป็นไฟล์ part ทุก ``flush_every`` เฟรม

    เขียน rows ก่อน ledger เสมอ — ถ้าเครื่องดับระหว่างสองไฟล์ เฟรมเหล่านั้นจะยังไม่ถูกนับว่า
    เสร็จ (ไม่มีใน ledger) แล้วถูกประมวลผลใหม่ แถวที่ซ้ำถูก dedupe ตอนรวม
    """

    def __init__(self, parts_dir: Path, flush_every: int) -> None:
        self.parts_dir = parts_dir
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, flush_every)
        self.run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.counter = 0
        self.rows: list[dict] = []
        self.entries: list[dict] = []

    def add(self, rows: list[dict], entry: dict) -> None:
        self.rows.extend(rows)
        self.entries.append(entry)
        if len(self.entries) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.entries:
            return
        self.counter += 1
        tag = f"{self.run_id}_{self.counter:04d}"
        if self.rows:
            rows = pd.DataFrame(self.rows)
            for column in ("lon", "lat"):
                rows[column] = rows[column].astype("float64")
            rows.to_parquet(self.parts_dir / f"rows_{tag}.parquet", index=False)
        pd.DataFrame(self.entries, columns=list(LEDGER_COLUMNS)).to_parquet(
            self.parts_dir / f"ledger_{tag}.parquet", index=False
        )
        self.rows, self.entries = [], []


def read_parts(parts_dir: Path, prefix: str) -> pd.DataFrame:
    files = sorted(parts_dir.glob(f"{prefix}*.parquet")) if parts_dir.is_dir() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def consolidate(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """รวมทุก part เป็นไฟล์เดียว — เก็บเฉพาะแถวของเฟรมที่สถานะล่าสุดเป็น ok"""
    parts_dir = out_dir / "parts"
    ledger = read_parts(parts_dir, "ledger_")
    latest = latest_ledger(ledger)
    done = completed_stems(ledger)

    rows = read_parts(parts_dir, "rows_")
    if not rows.empty:
        rows = (
            rows[rows["issue_time"].isin(done)]
            .drop_duplicates(["HARPNUM", "issue_time"], keep="last")
            .sort_values(["issue_time", "HARPNUM"])
            .reset_index(drop=True)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows.empty:
        rows.to_parquet(out_dir / "intensity_full_range.parquet", index=False)
    latest.to_parquet(out_dir / "match_report.parquet", index=False)
    return rows, latest


# --------------------------------------------------------------------------- #
# ประมวลผลหนึ่งเฟรม — ขั้นตอนเดียวกับ run_matching_pass ของด่านเดือนเดียว
# --------------------------------------------------------------------------- #


def process_frame(
    stem: str,
    window: str,
    frames_dir: Path,
    expected_harps: set[int],
    segmentation: SegmentationService,
    aia_store: AiaFrameStore,
    wcs_store: FrameWcsStore,
    channel_keys: list[str],
    min_area_px: int,
) -> tuple[list[dict], dict]:
    entry = {
        "stem": stem,
        "window": window,
        "status": STATUS_OK,
        "n_expected": len(expected_harps),
        "n_matched": 0,
        "n_extra": 0,
        "n_rows": 0,
        "missed_unet": [],
        "missed_no_identity": [],
        "missing_channels": "",
    }

    image_path = frames_dir / "images" / f"{stem}.npy"
    identity_path = frames_dir / "harp_ids" / f"{stem}.npy"
    if not image_path.exists():
        return [], {**entry, "status": STATUS_NO_IMAGE}
    if not identity_path.exists():
        return [], {**entry, "status": STATUS_NO_IDENTITY}

    # เช็ค AIA ก่อนรัน U-Net — แถวที่ขาดช่องใดช่องหนึ่งถูก build_study_dataset ตัดทิ้งอยู่แล้ว
    # (ต้องมี intensity ครบทุกคอลัมน์) จึงไม่มีประโยชน์ที่จะเขียน รอให้ครบแล้วค่อยสกัดทีเดียว
    channel_arrays = {channel: aia_store.load(stem, channel) for channel in channel_keys}
    missing = [channel for channel, array in channel_arrays.items() if array is None]
    if missing:
        return [], {**entry, "status": STATUS_MISSING_AIA, "missing_channels": ",".join(missing)}

    magnetogram = np.load(image_path).astype(np.float32)
    identity_map = np.load(identity_path)
    probability = segmentation.segment(magnetogram)
    predicted_mask = (probability >= segmentation.threshold).astype(np.uint8)

    rows = extract_frame_intensities(
        timestamp=stem,
        magnetogram=magnetogram,
        predicted_mask=predicted_mask,
        identity_map=identity_map,
        channel_arrays=channel_arrays,
        min_area_px=min_area_px,
        solar_map=wcs_store.solar_map(stem, magnetogram),
    )

    matched = {row["HARPNUM"] for row in rows}
    missed = expected_harps - matched
    missed_unet = sorted(h for h in missed if bool((identity_map == h).any()))
    return rows, {
        **entry,
        "n_matched": len(matched & expected_harps),
        "n_extra": len(matched - expected_harps),
        "n_rows": len(rows),
        "missed_unet": missed_unet,
        "missed_no_identity": sorted(missed - set(missed_unet)),
    }


# --------------------------------------------------------------------------- #
# รายงาน
# --------------------------------------------------------------------------- #


def area_by_outcome(table: pd.DataFrame, ledger: pd.DataFrame, sharp: pd.DataFrame) -> pd.DataFrame:
    """มัธยฐาน AREA_ACR ของ HARP ที่จับคู่ได้ เทียบกับที่พลาดแยกตามสาเหตุ — ข้อมูลตั้งต้นของการ
    วินิจฉัย (ถ้าที่พลาดเล็กกว่ามาก ชี้ไปที่ min_ar_area_px/ความไวของ U-Net ไม่ใช่ท่อพัง)"""
    if "AREA_ACR" not in sharp.columns:
        return pd.DataFrame()
    if table.empty:
        matched = pd.DataFrame(columns=["stem", "HARPNUM", "reason"])
    else:
        matched = table[["issue_time", "HARPNUM"]].rename(columns={"issue_time": "stem"}).assign(reason="matched")
    pairs = pd.concat([matched, missed_pairs(ledger)], ignore_index=True)
    if pairs.empty:
        return pd.DataFrame()

    pairs["HARPNUM"] = pairs["HARPNUM"].astype("int64")
    pairs["t"] = pd.to_datetime(pairs["stem"], format="%Y%m%d_%H%M%S").astype("datetime64[ns]")
    ref = sharp[["HARPNUM", "t_rec", "AREA_ACR"]].dropna().copy()
    ref["HARPNUM"] = ref["HARPNUM"].astype("int64")
    ref["t_rec"] = ref["t_rec"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        pairs.sort_values("t"), ref.sort_values("t_rec"),
        left_on="t", right_on="t_rec", by="HARPNUM",
        tolerance=pd.Timedelta(minutes=30), direction="nearest",
    )
    return merged.groupby("reason")["AREA_ACR"].agg(n="count", median="median").reset_index()


def compare_with_gate_check(table: pd.DataFrame, gate_path: Path) -> dict:
    """ท่อเดียวกันบนเฟรมเดียวกันต้องให้ค่าเดิม — เทียบแถวของ พ.ค. 2024 กับที่ด่านเดิมสกัดไว้"""
    if not gate_path.exists():
        return {"ok": False, "reason": f"ไม่พบ {gate_path}"}
    if table.empty:
        return {"ok": False, "reason": "ยังไม่มีแถวที่สกัดได้"}

    gate = pd.read_parquet(gate_path)
    key = ["HARPNUM", "issue_time"]
    ours = table[table["issue_time"].isin(set(gate["issue_time"]))]
    if ours.empty:
        return {"ok": False, "reason": "ยังไม่ได้สกัดเฟรมใดของหน้าต่างที่ด่านเดิมครอบคลุม"}

    # ด่านเดิมถูกสร้างก่อนแก้บั๊ก "หนึ่งแถวต่อ blob" — คู่ที่มีหลายแถวในตารางนั้นคือ AR ที่ U-Net
    # แตกเป็นหลายชิ้น ค่าจึงต่างจากรอบนี้ (ที่วัดจาก union) โดยตั้งใจ เทียบค่าเฉพาะคู่ที่มี blob
    # เดียวซึ่งต้องตรงกันเป๊ะ ส่วนคู่หลาย blob ตรวจแค่ว่าจำนวนแถวเดิมเท่ากับ n_blobs ของรอบนี้
    blob_counts = gate.groupby(key).size().rename("gate_rows").reset_index()
    single_keys = blob_counts.loc[blob_counts["gate_rows"] == 1, key]
    both = ours.merge(gate.merge(single_keys, on=key), on=key, suffixes=("", "_gate"))
    multi = blob_counts[blob_counts["gate_rows"] > 1].merge(ours, on=key)
    ours_keys = set(ours[key].itertuples(index=False, name=None))
    gate_keys = set(blob_counts[key].itertuples(index=False, name=None))

    diffs = {}
    for column in ("1600_median", "304_median", "171_median", "171_p95"):
        if len(both) and f"{column}_gate" in both.columns:
            gate_values = both[f"{column}_gate"].abs().clip(lower=1e-9)
            diffs[column] = float(((both[column] - both[f"{column}_gate"]).abs() / gate_values).max())
    return {
        "ok": True,
        "n_both": int(len(both)),
        "n_only_new": len(ours_keys - gate_keys),
        "n_only_gate": len(gate_keys - ours_keys),
        "n_multi": int((blob_counts["gate_rows"] > 1).sum()),
        "n_multi_blob_match": int((multi["n_blobs"] == multi["gate_rows"]).sum()) if "n_blobs" in multi else 0,
        "max_rel_diff": diffs,
    }


def signal_preview(table: pd.DataFrame, windows, flares, horizon_hours: int, positive_class: str) -> list[dict]:
    if table.empty or "171_median" not in table.columns:
        return []
    lookup = build_flare_lookup(flares, positive_class)
    times = pd.to_datetime(table["issue_time"], format="%Y%m%d_%H%M%S").to_numpy()
    labels = np.zeros(len(table), dtype=np.uint8)
    for harpnum, idx in table.groupby("HARPNUM").indices.items():
        labels[idx] = label_times(times[idx], lookup.get(int(harpnum)), horizon_hours)

    window = window_of(table["issue_time"], windows).to_numpy()
    out = []
    for name, _, _ in windows:
        sel = window == name
        pos = table.loc[sel & (labels == 1), "171_median"].dropna()
        neg = table.loc[sel & (labels == 0), "171_median"].dropna()
        out.append(
            {
                "window": name,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "positive_median": float(pos.median()) if len(pos) else float("nan"),
                "negative_median": float(neg.median()) if len(neg) else float("nan"),
            }
        )
    return out


def _rate(matched: int, expected: int) -> str:
    return f"{matched}/{expected} = {100 * matched / expected:.1f}%" if expected else "n/a"


def render_report(
    stems: list[tuple[str, str]],
    windows: list[tuple[str, str, str]],
    ledger: pd.DataFrame,
    table: pd.DataFrame,
    sharp: pd.DataFrame,
    flares: pd.DataFrame,
    cfg,
    gate_path: Path,
) -> tuple[str, dict]:
    channel_keys = [channel.key for channel in cfg.aia.channels]
    stem_window = dict(stems)
    latest = ledger[ledger["stem"].isin(stem_window)]
    ok_frames = latest[latest["status"] == STATUS_OK]

    lines = [
        "# ด่านตรวจการสกัดความเข้มแสงช่วงเต็ม — 2011-2017 + พ.ค. 2024 (ticket 05)",
        "",
        f"สร้างเมื่อ {datetime.now():%Y-%m-%d %H:%M} · ท่อเดียวกับด่านเดือนเดียวทุกขั้น "
        "(U-Net → detect → quiet-Sun normalise → จับคู่ HARP)",
        "",
        "## 0. ความครอบคลุมของเฟรม",
        "",
        f"เฟรมบนดิสก์ในช่วงที่เลือก: **{len(stems)}** · ประมวลผลแล้วอย่างน้อยหนึ่งครั้ง: {len(latest)}",
        "",
        "| สถานะ | เฟรม | ความหมาย |",
        "|---|---:|---|",
    ]
    counts = latest["status"].value_counts()
    for status, meaning in STATUS_MEANINGS.items():
        lines.append(f"| `{status}` | {int(counts.get(status, 0))} | {meaning} |")
    lines.append(f"| (ยังไม่ได้ประมวลผล) | {len(stems) - len(latest)} | เช่น รันด้วย `--limit` |")
    lines.append("")

    disk_years = pd.Series([stem[:4] for stem, _ in stems]).value_counts().sort_index()
    ok_years = ok_frames["stem"].str[:4].value_counts()
    lines += ["| ปี | เฟรมบนดิสก์ | สกัดเสร็จ |", "|---|---:|---:|"]
    for year, n_disk in disk_years.items():
        lines.append(f"| {year} | {n_disk} | {int(ok_years.get(year, 0))} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    matched, expected = overall_match_rate(latest)
    rate = matched / expected if expected else float("nan")
    lines += [
        "## 1. อัตราการจับคู่ AR เข้ากับ HARP (เกณฑ์ตัดสิน)",
        "",
        f"**รวม: {_rate(matched, expected)}** · ด่านเดือนเดียว (พ.ค. 2024) เคยได้ {100 * GATE_MATCH_RATE:.1f}%",
        "",
    ]
    for name, _, _ in windows:
        m, e = overall_match_rate(latest[latest["window"] == name])
        lines.append(f"- หน้าต่าง `{name}`: {_rate(m, e)}")
    lines.append("")
    if expected and rate < GATE_MATCH_RATE - MATCH_RATE_ALERT:
        lines += [
            f"**⚠ ต่ำกว่าด่านเดือนเดียวเกิน {100 * MATCH_RATE_ALERT:.0f} จุด — ต้องวินิจฉัยสาเหตุก่อนไปต่อ** "
            "(ticket 05 ห้ามรายงานตัวเลขแล้วผ่านไป) เริ่มจากตารางรายปีและขนาด AR ด้านล่าง",
            "",
        ]
    elif expected:
        lines += ["อยู่ในระดับเดียวกับด่านเดือนเดียว", ""]

    by_year = match_rate_by_year(latest)
    if not by_year.empty:
        lines += [
            "| ปี | เฟรม | ควรเห็น | จับคู่ได้ | อัตรา | พลาด: U-Net ไม่เห็น | พลาด: ไม่อยู่ในแผนที่ | AR เกิน |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in by_year.itertuples(index=False):
            lines.append(
                f"| {row.year} | {row.n_frames} | {row.n_expected} | {row.n_matched} | "
                f"{100 * row.match_rate:.1f}% | {row.n_missed_unet} | {row.n_missed_no_identity} | {row.n_extra} |"
            )
        lines.append("")

    area = area_by_outcome(table, latest, sharp)
    if not area.empty:
        lines += [
            "ขนาดของ HARP (AREA_ACR ของ SHARP ณ เวลาเฟรม) แยกตามผลการจับคู่:",
            "",
            "| ผล | จำนวนคู่ | มัธยฐาน AREA_ACR |",
            "|---|---:|---:|",
        ]
        for row in area.itertuples(index=False):
            lines.append(f"| {row.reason} | {row.n} | {row.median:.1f} |")
        lines.append("")

    # ------------------------------------------------------------------ #
    lines += [
        "## 2. การลบ drift ข้ามปี (เกณฑ์ตัดสิน)",
        "",
        "overlap ของ IQR ระหว่างปีต้นช่วงกับปลายช่วง — ตัววัดเดียวกับด่านเดือนเดียว "
        "(ค่าดิบย้อนคืนจาก normalised × quiet_sun)",
        "",
        "| ช่อง | ปีต้น / ปลาย | overlap ก่อน normalise | หลัง normalise | ผล |",
        "|---|---|---:|---:|---|",
    ]
    drift_results = {}
    for channel in channel_keys:
        result = drift_overlap(table, channel)
        drift_results[channel] = result
        if result.get("ok"):
            lines.append(
                f"| {channel} | {result['early_years']} / {result['late_years']} | "
                f"{result['raw_overlap']:.2f} | {result['norm_overlap']:.2f} | "
                f"{'ผ่าน' if result['improved'] else '**ไม่ผ่าน**'} |"
            )
        else:
            lines.append(f"| {channel} | — | — | — | ตรวจไม่ได้: {result.get('reason')} |")
    lines.append("")

    yearly = drift_by_year(table, channel_keys)
    if not yearly.empty and len(yearly.columns) > 2:
        header = (
            ["ปี", "แถว"]
            + [f"{c} quiet Sun (DN/s)" for c in channel_keys]
            + [f"{c} AR (เท่าของ quiet Sun)" for c in channel_keys]
        )
        lines += [
            "มัธยฐานรายปี — quiet Sun ที่ลดลงตามปีคือ degradation ที่ normalisation หารออก:",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for row in yearly.to_dict("records"):
            cells = [str(row["year"]), str(int(row["n_rows"]))]
            cells += [f"{row.get(f'{c}_quiet_sun', float('nan')):.2f}" for c in channel_keys]
            cells += [f"{row.get(f'{c}_normalised', float('nan')):.2f}" for c in channel_keys]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ------------------------------------------------------------------ #
    n_harps = int(table["HARPNUM"].nunique()) if not table.empty else 0
    mean_per_frame = float(ok_frames["n_matched"].mean()) if not ok_frames.empty else float("nan")
    lines += [
        "## 3. จำนวน (เกณฑ์ตัดสิน)",
        "",
        f"แถวความเข้มแสง (HARP, เวลา) ที่ได้: **{len(table)}** · จำนวน HARP: {n_harps} · "
        f"AR ที่จับคู่ได้เฉลี่ยต่อเฟรม: {mean_per_frame:.2f}",
        "",
    ]
    if not table.empty:
        per_window = window_of(table["issue_time"], windows).value_counts()
        for name, _, _ in windows:
            lines.append(f"- หน้าต่าง `{name}`: {int(per_window.get(name, 0))} แถว")
        lines.append("")

    # ------------------------------------------------------------------ #
    consistency = compare_with_gate_check(table, gate_path)
    lines += [
        "## 4. ความสอดคล้องกับด่านเดือนเดียว",
        "",
        "ท่อเดียวกันบนเฟรมเดียวกันต้องให้ค่าเดิม — เทียบแถวของเฟรมที่ด่านเดิมเคยสกัดไว้",
        "",
    ]
    if consistency["ok"]:
        lines += [
            f"คู่ (HARP, เวลา) ที่มีแต่รอบนี้: {consistency['n_only_new']} · มีแต่ด่านเดิม: {consistency['n_only_gate']}",
            f"- คู่ที่ด่านเดิมมี blob เดียว เทียบค่าได้ตรง ๆ: {consistency['n_both']} คู่",
        ]
        for column, value in consistency["max_rel_diff"].items():
            lines.append(f"  - `{column}` ต่างกันสูงสุด {100 * value:.4f}% (สัมพัทธ์)")
        lines.append(
            f"- คู่ที่ด่านเดิมเก็บแยกราย blob (สร้างก่อนแก้บั๊ก — ไม่เทียบค่า): {consistency['n_multi']} คู่ · "
            f"จำนวนแถวเดิมเท่ากับ `n_blobs` ของรอบนี้ {consistency['n_multi_blob_match']}/{consistency['n_multi']}"
        )
        # 1e-6 เผื่อแค่การปัดเศษ float32 — U-Net บน CPU กับ GPU ให้ mask เหมือนกันทุกพิกเซล
        # (ตรวจแล้ว) ความต่างที่เกินนี้จึงแปลว่าท่อหรือข้อมูลตั้งต้นไม่เหมือนกันจริง
        worst = max(consistency["max_rel_diff"].values(), default=0.0)
        if (
            consistency["n_only_new"] or consistency["n_only_gate"] or worst > 1e-6
            or consistency["n_multi_blob_match"] != consistency["n_multi"]
        ):
            lines += ["", "**⚠ ไม่ตรงกัน — ท่อเดียวกันบนข้อมูลเดียวกันต้องให้ค่าเดิม ต้องหาสาเหตุก่อนใช้ตาราง**"]
        else:
            lines += ["", "ตรงกัน"]
    else:
        lines.append(f"ยังเทียบไม่ได้: {consistency['reason']}")
    lines.append("")

    # ------------------------------------------------------------------ #
    lines += [
        "## 5. สัญญาณเบื้องต้น (ข้อมูลประกอบ — **ห้ามใช้ตัดสินว่าจะไปต่อหรือไม่**)",
        "",
        "นั่นคือสิ่งที่การทดลอง 5 แบบกำลังจะวัด การเอามาเป็นเกณฑ์คือการตัดสินล่วงหน้า",
        "",
    ]
    for item in signal_preview(table, windows, flares, cfg.flare.horizon_hours, cfg.flare.positive_goes_class):
        lines.append(
            f"- `{item['window']}`: positive {item['n_positive']} แถว (171Å median {item['positive_median']:.3f}) · "
            f"negative {item['n_negative']} แถว (171Å median {item['negative_median']:.3f}) — หน่วย: เท่าของ quiet Sun"
        )
    lines.append("")

    summary = {"match_rate": rate, "n_rows": len(table), "n_harps": n_harps, "drift": drift_results}
    return "\n".join(lines), summary


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "extract_intensity.log")

    out_dir = cfg.paths.artifacts / OUT_SUBDIR
    frames_dir = cfg.paths.processed / "frames"
    windows = [("main", _day(args.start), _day(args.end))]

    logger.info("=" * 70)
    logger.info("สกัดความเข้มแสงช่วงเต็ม: %s", " + ".join(f"{n} {a}..{b}" for n, a, b in windows))
    logger.info("=" * 70)

    stems = select_stems(frames_dir / "images", windows)
    logger.info("เฟรมบนดิสก์ในช่วงที่เลือก: %d", len(stems))

    sharp = clean_sharp_frame(pd.read_parquet(cfg.paths.interim / "sharp_keywords.parquet"), cfg)

    if not args.report_only:
        done = completed_stems(read_parts(out_dir / "parts", "ledger_"))
        pending = [(stem, window) for stem, window in stems if stem not in done]
        if args.limit:
            pending = pending[: args.limit]
        logger.info("สกัดเสร็จแล้ว %d เฟรม · รอบนี้จะประมวลผล %d เฟรม", len(done), len(pending))

        if pending:
            device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            segmentation = SegmentationService(
                cfg.paths.artifacts / "models" / "unet.pt", frames_dir=frames_dir, device=device
            )
            if not segmentation.available:
                logger.error("ยังไม่มีโมเดล segmentation — รัน backend/scripts/train_unet.py ก่อน")
                return 1
            logger.info("U-Net รันบน %s", device)

            aia_store = AiaFrameStore(cfg.aia.root)
            wcs_store = FrameWcsStore(cfg.paths.interim / "frame_wcs.parquet")
            channel_keys = [channel.key for channel in cfg.aia.channels]
            expected = expected_harps_by_frame(sharp, [stem for stem, _ in pending])

            writer = PartWriter(out_dir / "parts", args.flush_every)
            status_counts: dict[str, int] = {}
            started = time.time()
            try:
                for index, (stem, window) in enumerate(pending, start=1):
                    try:
                        rows, entry = process_frame(
                            stem, window, frames_dir, expected.get(stem, set()), segmentation,
                            aia_store, wcs_store, channel_keys, cfg.fulldisk.min_ar_area_px,
                        )
                    except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
                        logger.error("[%d/%d] %s ล้มเหลว (จะลองใหม่รอบหน้า): %s", index, len(pending), stem, exc)
                        continue
                    writer.add(rows, entry)
                    status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

                    if index % 25 == 0 or index == len(pending):
                        per_min = index / max(time.time() - started, 1e-9) * 60
                        logger.info(
                            "  [%d/%d] %s · %.0f เฟรม/นาที · เหลือ ~%.0f นาที · %s",
                            index, len(pending), stem, per_min, (len(pending) - index) / per_min,
                            ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
                        )
            finally:
                writer.flush()

    logger.info("รวมผลทุกรอบ + เขียนรายงาน")
    table, ledger = consolidate(out_dir)
    flares = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    report, summary = render_report(
        stems, windows, ledger, table, sharp, flares, cfg, cfg.paths.artifacts / GATE_CHECK_TABLE
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    logger.info("=" * 70)
    logger.info(
        "อัตราการจับคู่รวม: %.1f%% (ด่านเดือนเดียว %.1f%%)", 100 * summary["match_rate"], 100 * GATE_MATCH_RATE
    )
    for channel, result in summary["drift"].items():
        if result.get("ok"):
            logger.info(
                "  drift %s: overlap %.2f -> %.2f (%s)", channel, result["raw_overlap"],
                result["norm_overlap"], "ผ่าน" if result["improved"] else "ไม่ผ่าน",
            )
    logger.info("แถว %d จาก %d HARP · รายงาน: %s", summary["n_rows"], summary["n_harps"], out_dir / "report.md")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

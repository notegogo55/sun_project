"""วัดตัวคูณสเกลระหว่างดาวเทียม GOES สองรุ่น จากช่วงเวลาที่ทั้งสองดวงทำงานทับกันจริง

    python backend/scripts/data/measure_xray_cross_calibration.py

โปรเจกต์นี้ต้องใช้ GOES-15 (ชุดเทรนหลัก 2011-2017) และ GOES-16 (หน้าต่าง case study
พ.ค. 2024) ร่วมกัน แต่ทั้งสองดวงคาลิเบรตกันคนละวิธี (ดู docstring ของ
``sunseg.data.xray_flux``) ห้ามเดาตัวคูณสเกลจากเอกสารหรือความจำ — สคริปต์นี้ดาวน์โหลด
ช่วงเวลาที่ GOES-15 กับ GOES-16 ยังทำงานทับกันอยู่จริง (2017 ถึง 2020-03-04, ค่าเริ่มต้น
คือ 15 ส.ค. - 15 ก.ย. 2017 ซึ่งครอบคลุม flare X9.3 วันที่ 6 ก.ย. 2017 ไว้ให้ตรวจสอบ
ข้ามวิธี) แล้ววัดอัตราส่วนฟลักซ์จริงที่ timestamp เดียวกัน

ผลลัพธ์ที่ได้ใช้เป็นค่า ``SATELLITE_SCALE["g16"]`` ใน ``sunseg.data.xray_flux`` — ถ้ารัน
สคริปต์นี้ซ้ำแล้วได้ค่าต่างจากที่ hardcode ไว้ตอนนี้อย่างมีนัย ต้องอัปเดตค่าคงที่ตรงนั้นด้วย
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.xray_flux import _EPOCH  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("measure_xray_cross_calibration")

REPORT_DIR_NAME = "xray_cross_calibration"

# ระดับฟลักซ์ที่ถือว่าเป็น M-class ขึ้นไป (หน่วย W/m^2) — ใช้แยกตรวจว่าอัตราส่วนที่
# ระดับพื้นหลัง (มีสัญญาณรบกวนเยอะ หารเลขเล็กด้วยเลขเล็ก) กับระดับ flare จริงตรงกันไหม
M_CLASS_FLUX = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", default="g15", help="ดาวเทียมอ้างอิง (สเกล = 1.0)")
    p.add_argument("--target", default="g16", help="ดาวเทียมที่จะวัดตัวคูณสเกลเทียบกับ reference")
    p.add_argument("--start", type=date.fromisoformat, default=date(2017, 8, 15))
    p.add_argument("--end", type=date.fromisoformat, default=date(2017, 9, 15))
    return p.parse_args()


def _read_day_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """เหมือน ``xray_flux._read_day`` แต่**ไม่คูณตัวคูณสเกล** — สคริปต์นี้กำลังจะวัด
    ตัวคูณนั้น ใช้ค่าที่คูณไปแล้วมาวัดซ้ำจะเป็นการเวียนเทียน (circular)"""
    import h5py

    with h5py.File(path, "r") as handle:
        seconds = handle["time"][:].astype("float64")
        flux = handle["xrsb_flux"][:].astype("float64")
        flag = handle["xrsb_flag"][:]

    times = np.array(_EPOCH, dtype="datetime64[s]") + seconds.astype("timedelta64[s]")
    bad = (flag != 0) | ~np.isfinite(flux) | (flux <= 0) | (flux <= -9998.0)
    return times, np.where(bad, np.nan, flux)


def _read_satellite_window(root: Path, satellite: str, start: date, end: date) -> tuple[np.ndarray, np.ndarray]:
    from datetime import timedelta

    all_times, all_flux = [], []
    day = start
    while day <= end:
        year_dir = root / str(day.year)
        matches = sorted(year_dir.glob(f"sci_xrsf-l2-avg1m_{satellite}_d{day:%Y%m%d}_v*.nc"))
        if matches:
            t, f = _read_day_raw(matches[-1])
            all_times.append(t)
            all_flux.append(f)
        day += timedelta(days=1)
    if not all_times:
        return np.array([], dtype="datetime64[s]"), np.array([])
    return np.concatenate(all_times), np.concatenate(all_flux)


def _ratio_stats(v_ref: np.ndarray, v_target: np.ndarray, label: str) -> dict:
    ratio = v_target / v_ref
    stats = {
        "label": label,
        "n": int(len(ratio)),
        "median": float(np.median(ratio)) if len(ratio) else float("nan"),
        "mean": float(np.mean(ratio)) if len(ratio) else float("nan"),
        "p10": float(np.percentile(ratio, 10)) if len(ratio) else float("nan"),
        "p90": float(np.percentile(ratio, 90)) if len(ratio) else float("nan"),
    }
    logger.info(
        "  %-14s n=%-7d median=%.4f mean=%.4f p10/p90=%.4f/%.4f",
        label, stats["n"], stats["median"], stats["mean"], stats["p10"], stats["p90"],
    )
    return stats


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_data_config()

    root = cfg.paths.raw / "xrs"
    out_dir = cfg.paths.artifacts / REPORT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info(
        "วัดตัวคูณสเกล %s -> %s จากช่วงทับซ้อน %s..%s",
        args.target, args.reference, args.start, args.end,
    )
    logger.info("=" * 62)

    # ดาวน์โหลดช่วงทับซ้อนของทั้งสองดวงถ้ายังไม่มี (ใช้ตัวดาวน์โหลดตัวเดียวกับที่ใช้จริง)
    from download_xray import (
        download as download_xray,  # noqa: E402  (sys.path มี scripts/ อยู่แล้วผ่าน __main__)
    )

    for sat in (args.reference, args.target):
        download_xray(args.start, args.end, root, satellite=sat)

    t_ref, f_ref = _read_satellite_window(root, args.reference, args.start, args.end)
    t_tgt, f_tgt = _read_satellite_window(root, args.target, args.start, args.end)
    logger.info("%s: %d จุด (%d ดี)", args.reference, len(t_ref), int(np.isfinite(f_ref).sum()))
    logger.info("%s: %d จุด (%d ดี)", args.target, len(t_tgt), int(np.isfinite(f_tgt).sum()))

    if len(t_ref) == 0 or len(t_tgt) == 0:
        logger.error("ดาวน์โหลดไม่สำเร็จหรือช่วงที่เลือกไม่มีข้อมูลของดวงใดดวงหนึ่ง — วัดต่อไม่ได้")
        return 1

    d_ref = dict(zip(t_ref.astype("datetime64[m]"), f_ref))
    d_tgt = dict(zip(t_tgt.astype("datetime64[m]"), f_tgt))
    common = sorted(set(d_ref) & set(d_tgt))
    logger.info("timestamp ร่วมกัน: %d นาที", len(common))

    v_ref = np.array([d_ref[t] for t in common])
    v_tgt = np.array([d_tgt[t] for t in common])
    both_good = np.isfinite(v_ref) & np.isfinite(v_tgt)
    logger.info("จุดที่มีข้อมูลดีทั้งสองดวง: %d", int(both_good.sum()))

    logger.info("อัตราส่วน %s/%s:", args.target, args.reference)
    overall = _ratio_stats(v_ref[both_good], v_tgt[both_good], "ทั้งช่วง")

    high = both_good & (v_ref > M_CLASS_FLUX)
    m_class = (
        _ratio_stats(v_ref[high], v_tgt[high], f"M-class+ (>{M_CLASS_FLUX:.0e})")
        if high.sum() > 0
        else {"label": "M-class+", "n": 0, "median": float("nan"), "mean": float("nan"), "p10": float("nan"), "p90": float("nan")}
    )

    # จุดสูงสุดของ reference ในช่วงนี้ (แนวโน้มจะเป็น flare ที่แรงที่สุดในหน้าต่าง)
    common_arr = np.array(common)
    peak_idx = int(np.nanargmax(np.where(both_good, v_ref, np.nan)))
    peak_time = str(common_arr[peak_idx])
    peak_ratio = float(v_tgt[peak_idx] / v_ref[peak_idx])
    logger.info(
        "  จุดพีคของ %s ในช่วงนี้ (%s): %s=%.3e, %s=%.3e, อัตราส่วน=%.4f",
        args.reference, peak_time, args.reference, v_ref[peak_idx], args.target, v_tgt[peak_idx], peak_ratio,
    )

    chosen = overall["median"]

    report = f"""# วัดตัวคูณสเกลข้ามดาวเทียม {args.target} -> {args.reference}

**ช่วงที่ใช้วัด:** {args.start} ถึง {args.end} (ช่วงทับซ้อนจริงของทั้งสองดวง)
**timestamp ร่วมกัน:** {len(common)} นาที · **มีข้อมูลดีทั้งสองดวง:** {int(both_good.sum())}

## อัตราส่วน {args.target}/{args.reference}

| กลุ่ม | n | median | mean | p10 | p90 |
|---|---|---|---|---|---|
| ทั้งช่วง | {overall['n']} | {overall['median']:.4f} | {overall['mean']:.4f} | {overall['p10']:.4f} | {overall['p90']:.4f} |
| {m_class['label']} | {m_class['n']} | {m_class['median']:.4f} | {m_class['mean']:.4f} | {m_class['p10']:.4f} | {m_class['p90']:.4f} |

จุดพีคของ {args.reference} ในช่วงนี้ ({peak_time}): {args.reference}={v_ref[peak_idx]:.3e}, {args.target}={v_tgt[peak_idx]:.3e}, อัตราส่วน={peak_ratio:.4f}

## ค่าที่ใช้จริง

`SATELLITE_SCALE["{args.target}"] = 1.0 / {chosen:.4f}` (median ทั้งช่วง) ใน
`sunseg.data.xray_flux` — ถ้าค่านี้ต่างจากที่ hardcode ไว้ในโค้ดตอนนี้อย่างมีนัย
(เกิน ~1%) ต้องอัปเดตค่าคงที่ในไฟล์นั้นด้วย ไม่ใช่แค่รายงานนี้
"""
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("เขียนรายงานที่ %s", report_path)
    logger.info("ค่าที่ควรใช้: SATELLITE_SCALE['%s'] = 1.0 / %.4f", args.target, chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())

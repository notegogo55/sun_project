"""ประเมินเวลาดาวน์โหลด/สกัด feature ต้นทางของงานเปรียบเทียบ feature (V0-V5)

    python backend/scripts/study/measure_acquisition_time.py

ขั้นต้นทางทำไว้ก่อนมีการจับเวลา จึงประเมินจากหลักฐานที่เหลืออยู่ (log + เวลาแก้ไขไฟล์ผลลัพธ์ — วิธีและข้อจำกัดดู
``sunseg.study.acquisition_time``) ยกเว้นสองขั้นที่รันซ้ำได้จึงวัดสด: แปลงไฟล์ X-ray เป็น feature และประกอบ dataset

ผลลัพธ์คือ ``acquisition_time.json`` ข้าง ``report.md`` ของ ``study/train.py`` (ปริยาย
``artifacts/model_comparison/``) — ``study/train.py`` อ่านไฟล์นี้แล้วใส่หัวข้อ "ตั้งแต่ต้นทางจนได้ผล" ให้เอง
แต่ละขั้นมี ``hours`` (ส่วนที่จำเป็นต่อการทดลองนี้) และ ``hours_upper`` (ทุกอย่างที่ทำไปจริง รวมรอบซ้ำ/งานข้าง ๆ)

**เวลาที่ได้คือเวลาที่ใช้ตอนทำจริง** ไม่ใช่เวลาที่จะใช้ถ้าทำซ้ำ — ขึ้นกับ JSOC/NOAA เครือข่าย และจำนวนโปรเซสที่รัน
พร้อมกันตอนนั้น

รันทีละขั้นต่อเนื่อง ไม่ขนาน (ขั้นประกอบ dataset เป็น subprocess ที่รอจนจบก่อนไปต่อ) ใช้ ``--skip-assembly``
เพื่อข้ามขั้นประกอบ dataset (~4 นาที) ตอนรันซ้ำแค่เพื่ออัปเดตตัวเลขจาก log
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parents[1] / "src"))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sunseg.config import STUDY_VARIANTS_FILE, load_data_config  # noqa: E402
from sunseg.data.study_dataset import STUDY_CADENCE_HOURS, load_variants  # noqa: E402
from sunseg.data.xray_flux import XrayFluxStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.study.acquisition_time import Session, busy_seconds, log_sessions  # noqa: E402
from sunseg.study.summary import feature_group  # noqa: E402

logger = logging.getLogger("measure_acquisition_time")

DEFAULT_VARIANTS_FILE = STUDY_VARIANTS_FILE
STUDY_DIR_NAME = "model_comparison"  # โฟลเดอร์ผลของ study/train.py (ไม่ import มา เพราะลาก torch เข้ามาด้วย)
CASE_STUDY_TAG = "[CASE STUDY]"
# ช่วงของ dataset การทดลอง — ปริยายเดียวกับ study/build_dataset.py
MAIN_START, MAIN_END = "2011-01-01", "2026-01-01"
# epoch ของ U-Net ในรอบที่ใหญ่ยาวเกิน 2 ชม. — เกณฑ์ตัดช่วงหยุดต้องสูงกว่านั้น ไม่งั้น epoch ถูกนับเป็นเครื่องพัก
UNET_IDLE_MINUTES = 400.0


def _read_sessions(path: Path, title: str, **kwargs) -> list[Session]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return log_sessions(handle, title, **kwargs)


def _hours(seconds: float) -> float:
    return seconds / 3600


def _component(key: str, label: str, group: str, hours: float, hours_upper: float, method: str, source: str,
               volume: str = "", detail: str = "") -> dict:
    return {
        "key": key, "label": label, "group": group, "hours": hours, "hours_upper": hours_upper,
        "method": method, "source": source, "volume": volume, "detail": detail,
    }


def _files(root: Path, suffix: str) -> list[tuple[str, float, int]]:
    """(ชื่อไม่รวมนามสกุล, mtime, ขนาดไบต์) ของทุกไฟล์ใต้ ``root``"""
    found: list[tuple[str, float, int]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        if not directory.exists():
            continue
        for entry in directory.iterdir():
            if entry.is_dir():
                pending.append(entry)
            elif entry.name.endswith(suffix):
                stat = entry.stat()
                found.append((entry.name[: -len(suffix)], stat.st_mtime, stat.st_size))
    return found


def measure_sharp(logs: Path, interim: Path) -> dict:
    sessions = _read_sessions(logs / "download_metadata.log", r"ดาวน์โหลด metadata:")
    kept = [s for s in sessions if CASE_STUDY_TAG not in s.title]
    rows = pq.ParquetFile(interim / "sharp_keywords.parquet").metadata.num_rows
    return _component(
        "sharp_metadata", "SHARP keywords + ตาราง HARP-NOAA + flare catalog (JSOC/NOAA)", "shared",
        _hours(sum(s.active_seconds for s in kept)), _hours(sum(s.active_seconds for s in sessions)),
        "log (รวมเวลาที่ทำงานอยู่ของทุก session)", "artifacts/logs/download_metadata.log",
        volume=f"{rows:,} แถว SHARP",
        detail=f"{len(kept)} session (ตัด case study {len(sessions) - len(kept)} session ออกจากค่าหลัก)",
    )


def measure_hmi(logs: Path, frames: Path, gap: float) -> tuple[dict, set[str]]:
    files = _files(frames / "images", ".npy")
    stems = {name for name, _, _ in files}
    sessions = _read_sessions(logs / "download_images.log", r"ดาวน์โหลดภาพและสร้าง mask:")
    component = _component(
        "hmi_images", "ภาพ HMI magnetogram + bitmap SHARP → mask (JSOC ขอ export ทีละเฟรม)", "intensity",
        _hours(busy_seconds([m for _, m, _ in files], gap)), _hours(sum(s.active_seconds for s in sessions)),
        f"เวลาไฟล์ (ช่วงห่างไม่เกิน {gap:.0f} นาที) · ขอบบน = log ทุก session",
        "data/processed/frames/images · artifacts/logs/download_images.log",
        volume=f"{len(files):,} เฟรม (กริด 12 ชม. — ตรงกับเฟรมที่การทดลองใช้ทั้งหมด)",
        detail=f"ขอบบนรวมเฟรมรายสัปดาห์ของชุดเทรน U-Net, case study และรอบที่ซ้ำ ({len(sessions)} session)",
    )
    return component, stems


def measure_aia(aia: Path, channels: list[str], study_stems: set[str], gap: float) -> dict:
    mtimes, in_study, total, size = [], 0, 0, 0
    for channel in channels:
        for name, mtime, nbytes in _files(aia / channel, ".npy"):
            mtimes.append(mtime)
            total += 1
            size += nbytes
            in_study += name in study_stems
    share = in_study / total if total else 0.0
    busy = _hours(busy_seconds(mtimes, gap))
    return _component(
        "aia_images", f"ภาพ AIA {', '.join(channels)} (ดาวน์โหลด + แปลงเป็น DN/s)", "intensity",
        busy * share, busy,
        f"เวลาไฟล์ (ช่วงห่างไม่เกิน {gap:.0f} นาที) × สัดส่วนไฟล์ที่เป็นเฟรมของการทดลอง",
        "data/processed/aia",
        volume=f"{total:,} ไฟล์ · {size / 1e9:.0f} GB",
        detail=f"เฟรมของการทดลอง {share:.0%} ของไฟล์ (ที่เหลือเป็นเฟรมของงานทดลองอื่นที่โหลดปนกัน) — "
        "ประมาณตามสัดส่วนไฟล์ เพราะสองชุดถูกโหลดสลับกันตามเวลา แยกช่วงเวลาตรง ๆ ไม่ได้",
    )


def measure_unet(logs: Path) -> dict:
    sessions = _read_sessions(
        logs / "train_unet.log", r"เทรน U-Net แบ่งส่วน active region",
        idle_minutes=UNET_IDLE_MINUTES, done=r"ใช้เวลาเทรน",
    )
    # โมเดลปัจจุบันมาจาก session สุดท้าย ถ้ามันไม่ได้เทรนจนจบเอง (เทรนต่อจากโมเดลเดิมแล้วหยุด) ก็ต้องนับรอบเต็ม
    # ล่าสุดก่อนหน้าที่เป็นจุดตั้งต้นด้วย
    chain = [sessions[-1]]
    if not sessions[-1].completed:
        earlier = [s for s in sessions[:-1] if s.completed]
        if earlier:
            chain.append(earlier[-1])
    minutes = ", ".join(f"{s.active_seconds / 60:.0f}" for s in reversed(chain))
    return _component(
        "unet_training", "เทรน U-Net แบ่ง active region (ตัวแบ่ง AR ที่ intensity ต้องใช้)", "intensity",
        _hours(sum(s.active_seconds for s in chain)), _hours(sum(s.active_seconds for s in sessions)),
        "log (รอบเทรนที่ให้โมเดลปัจจุบัน) · ขอบบน = ทุกรอบเทรนที่เคยทำ", "artifacts/logs/train_unet.log",
        volume="GTX 1650",
        detail=f"โมเดลปัจจุบัน = รอบเทรนต่อเนื่อง {minutes} นาที · ทั้งหมด {len(sessions)} รอบเทรน",
    )


def measure_extraction(logs: Path, n_frames: int) -> dict:
    sessions = _read_sessions(logs / "extract_intensity.log", r"สกัดความเข้มแสง")
    longest = max(sessions, key=lambda s: s.active_seconds)
    return _component(
        "intensity_extraction", "สกัด intensity (U-Net หา AR → p95 ต่อช่อง → normalise quiet Sun → จับคู่ HARP)",
        "intensity", _hours(longest.active_seconds), _hours(sum(s.active_seconds for s in sessions)),
        "log (รอบเต็มรอบเดียวที่นานที่สุด) · ขอบบน = ทุก session", "artifacts/logs/extract_intensity.log",
        volume=f"{n_frames:,} เฟรม",
        detail=f"{len(sessions)} session (รอบแรก ๆ ข้ามเฟรมที่ยังไม่มี AIA แล้วกลับมาเติม) · "
        f"ความเร็วรอบเต็ม ~{n_frames / (longest.active_seconds / 60):.0f} เฟรม/นาที",
    )


def measure_xray_download(xrs: Path, gap: float) -> dict:
    files = _files(xrs, ".nc")
    by_satellite: dict[str, list[float]] = {}
    for name, mtime, _ in files:
        match = re.search(r"_g(\d+)_d", name)
        by_satellite.setdefault(f"g{match.group(1)}" if match else "?", []).append(mtime)
    hours = _hours(busy_seconds([m for _, m, _ in files], gap))
    per_satellite = " · ".join(
        f"{sat} {_hours(busy_seconds(times, gap)) * 60:.0f} นาที" for sat, times in sorted(by_satellite.items())
    )
    return _component(
        "xray_download", "ดาวน์โหลด GOES XRS รายนาที g15-g19 (NOAA NCEI/NGDC)", "xray",
        hours, hours,
        f"เวลาไฟล์ (ช่วงห่างไม่เกิน {gap:.0f} นาที) — ไม่มี log ของรอบ g15 เลย จึงใช้วิธีนี้วิธีเดียว",
        "data/raw/xrs",
        volume=f"{len(files):,} ไฟล์ · {sum(s for _, _, s in files) / 1e6:.0f} MB",
        detail=f"แยกดาวเทียม: {per_satellite}",
    )


def measure_xray_binning(xrs: Path) -> tuple[dict, float]:
    """คืน (ขั้น, วินาทีของรอบแรก) — วินาทีถูกหักออกจากขั้นประกอบ dataset ที่รวมการแปลงนี้ไว้ในตัว"""
    store = XrayFluxStore(xrs)
    start, end = pd.Timestamp(MAIN_START), pd.Timestamp(MAIN_END)
    timings = []
    for _ in range(2):
        began = time.perf_counter()
        bins = store.bin_series(start, end, cadence_hours=STUDY_CADENCE_HOURS)
        timings.append(time.perf_counter() - began)
    return _component(
        "xray_binning", "แปลงไฟล์ X-ray เป็น feature ต่อ timestep (XrayFluxStore.bin_series)", "xray",
        _hours(timings[0]), _hours(timings[0]),
        "วัดสด (รอบแรก) — ขึ้นกับแคชไฟล์ของระบบปฏิบัติการ", "sunseg.data.xray_flux",
        volume=f"{len(bins):,} bin (cadence {STUDY_CADENCE_HOURS} ชม.)",
        detail=f"รอบแรก {timings[0]:.0f} วินาที · รอบซ้ำ {timings[1]:.0f} วินาที (แคชอุ่นแล้ว)",
    ), timings[0]


def measure_assembly(xray_binning_seconds: float) -> dict:
    """ประกอบ dataset ทั้งชุดที่ทุกแบบใช้ร่วมกัน — รัน ``study/build_dataset.py`` จริงลงโฟลเดอร์ชั่วคราว"""
    with tempfile.TemporaryDirectory(prefix="study_build_") as tmp:
        began = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "build_dataset.py"), "--out-dir", tmp],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        seconds = time.perf_counter() - began
    if result.returncode != 0:
        raise RuntimeError(f"study/build_dataset.py ล้ม (exit {result.returncode}):\n{result.stderr[-1500:]}")
    return _component(
        "dataset_assembly", "ประกอบ dataset ก้อนเดียว (SHARP + intensity + X-ray บนกริดเวลาเดียวกัน)", "shared",
        _hours(max(seconds - xray_binning_seconds, 0.0)), _hours(seconds),
        "วัดสด (รัน study/build_dataset.py ลงโฟลเดอร์ชั่วคราว)", "backend/scripts/study/build_dataset.py",
        detail=f"ทั้งกระบวนการ {seconds:.0f} วินาที รวมแปลง X-ray {xray_binning_seconds:.0f} วินาที "
        "(ส่วนนั้นนับแยกในกลุ่ม X-ray)",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="ไฟล์ผลลัพธ์ (ปริยาย: artifacts/model_comparison/acquisition_time.json)")
    p.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS_FILE)
    p.add_argument("--gap-minutes", type=float, default=30.0, help="ช่วงห่างระหว่างไฟล์ที่ยาวกว่านี้ถือว่างานหยุด")
    p.add_argument("--skip-assembly", action="store_true", help="ข้ามขั้นประกอบ dataset (~4 นาที)")
    args = p.parse_args()

    cfg = load_data_config()
    setup_logging()
    out = args.out or cfg.paths.artifacts / STUDY_DIR_NAME / "acquisition_time.json"
    logs = cfg.paths.artifacts / "logs"

    variants = load_variants(args.variants_file)
    intensity_columns = {c for spec in variants.values() for c in spec["columns"] if feature_group(c) == "intensity"}
    channels = sorted({c.split("_")[0] for c in intensity_columns})

    logger.info("วัดเวลาต้นทาง: ช่อง intensity %s · เกณฑ์ช่วงหยุด %.0f นาที", channels, args.gap_minutes)
    components = [measure_sharp(logs, cfg.paths.interim)]
    hmi, study_stems = measure_hmi(logs, cfg.paths.processed / "frames", args.gap_minutes)
    components += [
        hmi,
        measure_aia(cfg.paths.processed / "aia", channels, study_stems, args.gap_minutes),
        measure_unet(logs),
        measure_extraction(logs, len(study_stems)),
        measure_xray_download(cfg.paths.raw / "xrs", args.gap_minutes),
    ]
    binning, binning_seconds = measure_xray_binning(cfg.paths.raw / "xrs")
    components.append(binning)
    if args.skip_assembly:
        logger.warning("ข้ามขั้นประกอบ dataset ตามที่สั่ง — ต้นทุนร่วมทุกแบบจะขาดส่วนนี้")
    else:
        components.append(measure_assembly(binning_seconds))

    for c in components:
        logger.info("  %-22s %-9s %8.2f ชม. (ขอบบน %8.2f)  %s", c["key"], c["group"], c["hours"], c["hours_upper"], c["volume"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "gap_minutes": args.gap_minutes,
                "components": components,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("เขียน %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ตรวจคุณภาพของรายการ flare ที่ดึงมา ก่อนนำไปสร้าง label

คำถามสำคัญที่สคริปต์นี้ตอบ: **flare ระดับ M ขึ้นไป (ซึ่งเป็น positive class ของเรา)
ระบุ NOAA AR ได้กี่เปอร์เซ็นต์?**

flare ที่ไม่มีเลข AR จะถูกตัดทิ้งตอนจับคู่กับ HARP ซึ่งหมายความว่า sample ที่ควรเป็น
positive จะถูก label เป็น negative — เป็น label noise ที่กดเพดานประสิทธิภาพของโมเดล
ถ้าตัวเลขนี้ต่ำเกินไป ต้องเปลี่ยนแหล่งข้อมูลหรือหาวิธีเชื่อมตำแหน่ง flare กับ HARP แทน

    python backend/scripts/checks/inspect_flares.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.goes_class import goes_class_to_flux  # noqa: E402
from sunseg.logging_utils import force_utf8_stdio  # noqa: E402

force_utf8_stdio()


def main() -> int:
    cfg = load_data_config()
    path = cfg.paths.interim / "flares.parquet"
    if not path.exists():
        print(f"ไม่พบ {path} — รัน backend/scripts/data/download_metadata.py ก่อน")
        return 1

    # ไฟล์นี้ถูกคลี่เป็นคู่ (flare, HARP) แล้ว จึงต้องยุบกลับเป็น event ที่ไม่ซ้ำ
    mapped = pd.read_parquet(path)
    print("=" * 74)
    print("ตรวจคุณภาพรายการ flare")
    print("=" * 74)
    print(f"\nไฟล์: {path}")
    print(f"คู่ (flare, HARP) ทั้งหมด : {len(mapped):,}")
    print(f"HARP ที่มี flare อย่างน้อย 1 ครั้ง : {mapped['HARPNUM'].nunique():,}")

    _report_by_class(mapped, "flare ที่จับคู่ HARP สำเร็จ (ใช้สร้าง label ได้)")

    # --- เทียบกับรายการดิบ เพื่อดูว่าสูญเสียไปเท่าไรตอนจับคู่ ---
    raw_dir = cfg.paths.raw / "cache" / "ngdc"
    raw_files = sorted(raw_dir.glob("goes-xrs-report_*.txt")) if raw_dir.exists() else []
    if not raw_files:
        print("\n(ไม่พบไฟล์ NGDC ดิบ — ข้ามการเปรียบเทียบอัตราการสูญเสีย)")
        return 0

    from sunseg.data.noaa_flares import parse_report_file

    raw = pd.concat([parse_report_file(p) for p in raw_files], ignore_index=True)
    start = pd.Timestamp(cfg.time_range.start)
    end = pd.Timestamp(cfg.time_range.end)
    if not mapped.empty:
        start = mapped["peak_time"].min().normalize()
        end = mapped["peak_time"].max().normalize() + pd.Timedelta(days=1)
    raw = raw[(raw["peak_time"] >= start) & (raw["peak_time"] < end)]

    print("\n" + "-" * 74)
    _report_by_class(raw, "flare ทั้งหมดในรายการดิบ (ก่อนจับคู่)")

    print("\n" + "=" * 74)
    print("อัตราการเก็บได้ (retention) แยกตามคลาส — ตัวเลขคลาส M/X คือตัวที่สำคัญ")
    print("=" * 74)
    print(f"{'คลาส':<8}{'ดิบ':>10}{'มี AR':>10}{'เก็บได้':>12}")

    unique_mapped = mapped.drop_duplicates(subset=["peak_time", "goes_class"])
    for letter in ["B", "C", "M", "X"]:
        n_raw = int((raw["goes_class"].str[0] == letter).sum())
        n_kept = int((unique_mapped["goes_class"].str[0] == letter).sum())
        pct = f"{100 * n_kept / n_raw:.1f}%" if n_raw else "-"
        print(f"{letter:<8}{n_raw:>10,}{n_kept:>10,}{pct:>12}")

    # --- ข้อสรุปสำหรับ positive class ที่เราสนใจจริง ---
    threshold = goes_class_to_flux(cfg.flare.positive_goes_class)
    n_raw_pos = int((raw["peak_flux"] >= threshold).sum())
    n_kept_pos = int((unique_mapped["peak_flux"] >= threshold).sum())

    print("\n" + "=" * 74)
    print(f"positive class ของโปรเจค: flare >= {cfg.flare.positive_goes_class}")
    print("=" * 74)
    print(f"  ในรายการดิบ      : {n_raw_pos:,}")
    print(f"  จับคู่ HARP สำเร็จ : {n_kept_pos:,}")

    if n_raw_pos:
        retention = n_kept_pos / n_raw_pos
        print(f"  อัตราการเก็บได้   : {retention:.1%}")
        print()
        if retention >= 0.85:
            print("  [ OK ] ดีมาก — label noise จากการสูญเสีย positive อยู่ในระดับต่ำ")
        elif retention >= 0.6:
            print("  [WARN] เก็บ positive ได้ไม่ครบ — โมเดลจะเห็น positive จริงบางส่วนเป็น negative")
            print("         ยังเทรนได้ แต่ให้คาดหวัง recall ต่ำกว่าที่ควรเล็กน้อย")
        else:
            print("  [FAIL] สูญเสีย positive มากเกินไป — ต้องแก้ก่อนเทรน")
            print("         ทางเลือก: ใช้ --flare-source hek หรือจับคู่ด้วยพิกัดแทนเลข AR")

    return 0


def _report_by_class(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    if df.empty:
        print("  (ไม่มีข้อมูล)")
        return
    letters = df["goes_class"].astype("string").str[0].str.upper()
    counts = letters.value_counts().reindex(["A", "B", "C", "M", "X"]).fillna(0).astype(int)
    print("  " + "  ".join(f"{c}={counts[c]:,}" for c in ["A", "B", "C", "M", "X"]))
    print(f"  ช่วงเวลา: {df['peak_time'].min():%Y-%m-%d} ถึง {df['peak_time'].max():%Y-%m-%d}")


if __name__ == "__main__":
    raise SystemExit(main())

"""ด่านตรวจก่อนลงทุนดาวน์โหลด 7-9 วัน — ประตูของงานเปรียบเทียบ SHARP-only vs SHARP+intensity

    python backend/scripts/intensity_gate_check.py

รันทั้ง pipeline บนหน้าต่าง พ.ค. 2024 (62 เฟรม @ 12 ชม. — ต้องดาวน์โหลดซ้ำก่อนด้วย
``python backend/scripts/download_images.py --case-study`` เพื่อให้ได้แผนที่ระบุตัวตน
ของ HARP ติดมาด้วย ดู ``sunseg.data.build_masks.build_fulldisk_identity_map``)

วัด **เรื่องท่อ** สามข้อที่เป็นเกณฑ์ตัดสิน (ห้ามลงทุนดาวน์โหลด 2011-2017 ถ้าข้อใดข้อหนึ่งไม่ผ่าน):

    1. อัตราการจับคู่ AR (ที่ U-Net ตรวจพบ) เข้ากับ HARP ที่ควรมองเห็นในเฟรม
    2. quiet-Sun normalisation ลบ drift ข้ามปีได้จริงหรือไม่ (วัดจากเฟรมรายสัปดาห์เดิม
       ที่มีอยู่แล้วครอบคลุม 2011-2024 — ไม่ต้องรอดาวน์โหลดใหม่)
    3. จำนวน AR ต่อเฟรมและจำนวนแถวที่ได้สมเหตุสมผลหรือไม่

ข้อที่สี่ (ความเข้มแสงแยกดวงที่จะปะทุจากดวงที่ไม่ปะทุได้แค่ไหน) เป็นข้อมูลประกอบเท่านั้น
**ห้ามใช้ตัดสินว่าจะไปต่อหรือไม่** — มันคือสิ่งที่การทดลองทั้งหมดกำลังจะวัด

artifacts เขียนลง ``artifacts/intensity_study/gate_check/`` — path ของงานศึกษาเอง
ไม่ทับของเดิม
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.aia import AiaFrameStore, region_intensities  # noqa: E402
from sunseg.data.build_sequences import build_flare_lookup, clean_sharp_frame, label_times  # noqa: E402
from sunseg.data.frame_wcs import FrameWcsStore  # noqa: E402
from sunseg.data.intensity import extract_frame_intensities, quiet_sun_normalise  # noqa: E402
from sunseg.inference.segment import SegmentationService  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.tracking.detect import detect_regions  # noqa: E402

logger = logging.getLogger("intensity_gate_check")

STUDY_ROOT_NAME = "intensity_study"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--drift-sample-frames",
        type=int,
        default=60,
        help="จำนวนเฟรมรายสัปดาห์เดิมสูงสุดที่สุ่มมาใช้ตรวจการลบ drift (เร็วกว่าใช้ทั้ง 417 เฟรม)",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# [1] อัตราการจับคู่ + จำนวนแถว — ต้องใช้เฟรม พ.ค. 2024 ที่มีแผนที่ระบุตัวตนแล้ว
# --------------------------------------------------------------------------- #


def expected_harps_by_frame(sharp_clean: pd.DataFrame, frame_times: list[pd.Timestamp]) -> dict[str, set[int]]:
    """HARP ที่ควรถูกมองเห็นในแต่ละเฟรม (ผ่านตัวกรองของโปรเจคแล้ว) จับคู่ตามเวลาที่ใกล้ที่สุด

    SHARP keyword เป็นรายชั่วโมง ส่วนเฟรมอยู่พอดีบนกริดชั่วโมง (00:00/12:00) จึงหาค่าที่ตรง
    เป๊ะหรือใกล้ที่สุดภายใน 30 นาทีได้เสมอถ้าข้อมูลครบ
    """
    result: dict[str, set[int]] = {}
    sharp_sorted = sharp_clean.sort_values("t_rec")
    for moment in frame_times:
        tag = moment.strftime("%Y%m%d_%H%M%S")
        window = sharp_sorted[(sharp_sorted["t_rec"] - moment).abs() <= pd.Timedelta(minutes=30)]
        result[tag] = set(int(h) for h in window["HARPNUM"].unique())
    return result


def run_matching_pass(
    frames_dir: Path,
    case_study_stems: list[str],
    expected: dict[str, set[int]],
    segmentation: SegmentationService,
    aia_store: AiaFrameStore,
    wcs_store: FrameWcsStore,
    channel_keys: list[str],
    min_area_px: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """สกัดความเข้มแสงของทุกเฟรมในหน้าต่าง พ.ค. 2024 พร้อมรายงานผลการจับคู่รายเฟรม

    Returns
    -------
    ``(intensity_table, match_report)`` — ``match_report`` หนึ่งแถวต่อหนึ่งเฟรม
    """
    all_rows: list[dict] = []
    match_rows: list[dict] = []

    for index, stem in enumerate(case_study_stems, start=1):
        identity_path = frames_dir / "harp_ids" / f"{stem}.npy"
        image_path = frames_dir / "images" / f"{stem}.npy"

        if not identity_path.exists():
            match_rows.append(
                {"stem": stem, "status": "skipped_no_identity_map", "n_expected": len(expected.get(stem, set())),
                 "n_matched": 0, "n_extra": 0, "missed_harps": sorted(expected.get(stem, set())), "reasons": {}}
            )
            continue
        if not image_path.exists():
            match_rows.append(
                {"stem": stem, "status": "skipped_no_image", "n_expected": len(expected.get(stem, set())),
                 "n_matched": 0, "n_extra": 0, "missed_harps": sorted(expected.get(stem, set())), "reasons": {}}
            )
            continue

        magnetogram = np.load(image_path).astype(np.float32)
        identity_map = np.load(identity_path)

        if not segmentation.available:
            raise RuntimeError("ยังไม่มีโมเดล segmentation — รัน backend/scripts/train_unet.py ก่อน")
        probability = segmentation.segment(magnetogram)
        predicted_mask = (probability >= segmentation.threshold).astype(np.uint8)

        channel_arrays = {ch: aia_store.load(stem, ch) for ch in channel_keys}
        solar_map = wcs_store.solar_map(stem, magnetogram)

        rows = extract_frame_intensities(
            timestamp=stem,
            magnetogram=magnetogram,
            predicted_mask=predicted_mask,
            identity_map=identity_map,
            channel_arrays=channel_arrays,
            min_area_px=min_area_px,
            solar_map=solar_map,
        )
        all_rows.extend(rows)

        matched_harps = {row["HARPNUM"] for row in rows}
        expected_harps = expected.get(stem, set())
        missed = expected_harps - matched_harps
        extra = matched_harps - expected_harps

        reasons: dict[int, str] = {}
        for harp in missed:
            present_in_identity = bool((identity_map == harp).any())
            reasons[harp] = (
                "U-Net ไม่ตรวจพบ AR ในบริเวณนี้ (พิกเซลของ HARP นี้มีอยู่ในแผนที่ระบุตัวตน "
                "แต่ไม่ซ้อนทับกับ component ใดที่ U-Net ทำนาย)"
                if present_in_identity
                else "ไม่มีพิกเซลของ HARP นี้ในแผนที่ระบุตัวตนของเฟรมนี้เลย "
                "(อาจ reproject patch ไม่สำเร็จ หรือหลุด threshold ของ bitmap)"
            )

        match_rows.append(
            {
                "stem": stem,
                "status": "ok",
                "n_expected": len(expected_harps),
                "n_matched": len(matched_harps & expected_harps),
                "n_extra": len(extra),
                "missed_harps": sorted(missed),
                "reasons": reasons,
            }
        )

        if index % 10 == 0 or index == len(case_study_stems):
            logger.info("  [%d/%d] ประมวลผลแล้ว", index, len(case_study_stems))

    return pd.DataFrame(all_rows), pd.DataFrame(match_rows)


# --------------------------------------------------------------------------- #
# [2] การลบ drift ข้ามปี — ใช้เฟรมรายสัปดาห์เดิมที่ครอบคลุม 2011-2024 อยู่แล้ว
#     (ไม่ต้องมีแผนที่ระบุตัวตน เพราะไม่ได้จับคู่กับ HARP ในขั้นนี้)
# --------------------------------------------------------------------------- #


def collect_ar_intensity_samples(
    frames_dir: Path,
    stems: list[str],
    segmentation: SegmentationService,
    aia_store: AiaFrameStore,
    channel_key: str,
    min_area_px: int,
) -> pd.DataFrame:
    """ค่าความเข้มแสง (ดิบ + normalise แล้ว) ของทุก AR ที่ U-Net ตรวจพบ ในเฟรมที่ระบุ

    ไม่จับคู่กับ HARP — ใช้เฉพาะตรวจว่า normalisation ลบผลของปีที่ต่างกันออกได้จริงหรือไม่
    """
    rows: list[dict] = []
    for stem in stems:
        image_path = frames_dir / "images" / f"{stem}.npy"
        if not image_path.exists():
            continue
        channel = aia_store.load(stem, channel_key)
        if channel is None:
            continue

        magnetogram = np.load(image_path).astype(np.float32)
        if not segmentation.available:
            continue
        probability = segmentation.segment(magnetogram)
        predicted_mask = (probability >= segmentation.threshold).astype(np.uint8)

        detections = detect_regions(predicted_mask, magnetogram=magnetogram, min_area_px=min_area_px)
        if not detections:
            continue

        normalised, quiet_level = quiet_sun_normalise(channel, predicted_mask)
        if not np.isfinite(quiet_level):
            continue

        raw_stats = region_intensities(detections, channel)
        norm_stats = region_intensities(detections, normalised)
        year = int(stem[:4])
        for raw, norm in zip(raw_stats, norm_stats, strict=True):
            if raw["n_pixels"] == 0:
                continue
            rows.append(
                {"stem": stem, "year": year, "raw_median": raw["median"], "normalised_median": norm["median"]}
            )
    return pd.DataFrame(rows)


def drift_check(frames_dir: Path, segmentation, aia_store, channel_key: str, min_area_px: int, sample_limit: int) -> dict:
    """เทียบการกระจายของความเข้มแสงระหว่างปีต้นช่วงกับปีปลายช่วง ก่อนและหลัง normalise"""
    all_stems = sorted(p.stem for p in (frames_dir / "images").glob("*.npy"))
    # ตัดเฟรม พ.ค. 2024 ออก (คนละหน้าต่างกับชุด weekly เดิม จะทำให้ปีปลายช่วงเอียงไปทาง
    # เดือนเดียว) ใช้เฉพาะเฟรมรายสัปดาห์ที่กระจายทั้งช่วง 2011-2017/2024
    weekly_stems = [s for s in all_stems if s < "20240101_000000" or s > "20240601_000000"]
    if len(weekly_stems) > sample_limit:
        idx = np.linspace(0, len(weekly_stems) - 1, sample_limit).astype(int)
        weekly_stems = [weekly_stems[i] for i in idx]

    logger.info("  เก็บตัวอย่างความเข้มแสงจาก %d เฟรมรายสัปดาห์เดิม (ช่อง %s)", len(weekly_stems), channel_key)
    samples = collect_ar_intensity_samples(frames_dir, weekly_stems, segmentation, aia_store, channel_key, min_area_px)
    if samples.empty or samples["year"].nunique() < 2:
        return {"ok": False, "reason": "ตัวอย่างไม่พอสำหรับเทียบข้ามปี (ต้องมีอย่างน้อย 2 ปีที่มีข้อมูล)"}

    years = sorted(samples["year"].unique())
    early_years = set(years[: max(1, len(years) // 3)])
    late_years = set(years[-max(1, len(years) // 3):])
    early = samples[samples["year"].isin(early_years)]
    late = samples[samples["year"].isin(late_years)]

    def overlap(a: pd.Series, b: pd.Series) -> float:
        """สัดส่วนที่ช่วง [p25,p75] ของสองกลุ่มคาบเกี่ยวกัน (0 = ไม่เกี่ยวเลย, 1 = ซ้อนทับสมบูรณ์)"""
        lo = max(a.quantile(0.25), b.quantile(0.25))
        hi = min(a.quantile(0.75), b.quantile(0.75))
        span = max(hi - lo, 0.0)
        widest = max(a.quantile(0.75) - a.quantile(0.25), b.quantile(0.75) - b.quantile(0.25), 1e-9)
        return float(span / widest)

    raw_overlap = overlap(early["raw_median"], late["raw_median"])
    norm_overlap = overlap(early["normalised_median"], late["normalised_median"])

    return {
        "ok": True,
        "early_years": sorted(early_years),
        "late_years": sorted(late_years),
        "n_early": len(early),
        "n_late": len(late),
        "raw_early_median": float(early["raw_median"].median()),
        "raw_late_median": float(late["raw_median"].median()),
        "raw_overlap": raw_overlap,
        "norm_early_median": float(early["normalised_median"].median()),
        "norm_late_median": float(late["normalised_median"].median()),
        "norm_overlap": norm_overlap,
        "improved": norm_overlap > raw_overlap,
    }


# --------------------------------------------------------------------------- #
# [3] สัญญาณ (รายงานเฉยๆ — ห้ามใช้ตัดสิน)
# --------------------------------------------------------------------------- #


def signal_preview(intensity_table: pd.DataFrame, flares: pd.DataFrame, horizon_hours: int, positive_class: str) -> dict:
    if intensity_table.empty:
        return {"ok": False, "reason": "ไม่มีข้อมูลความเข้มแสง"}

    lookup = build_flare_lookup(flares, positive_class)
    times = pd.to_datetime(intensity_table["issue_time"], format="%Y%m%d_%H%M%S").to_numpy()
    labels = np.zeros(len(intensity_table), dtype=np.uint8)
    for harpnum, group_idx in intensity_table.groupby("HARPNUM").indices.items():
        flare_times = lookup.get(int(harpnum))
        labels[group_idx] = label_times(times[group_idx], flare_times, horizon_hours)

    col = "171_median" if "171_median" in intensity_table.columns else None
    if col is None:
        return {"ok": False, "reason": "ไม่มีคอลัมน์ 171_median"}

    pos = intensity_table.loc[labels == 1, col].dropna()
    neg = intensity_table.loc[labels == 0, col].dropna()
    return {
        "ok": True,
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "positive_median": float(pos.median()) if len(pos) else float("nan"),
        "negative_median": float(neg.median()) if len(neg) else float("nan"),
    }


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    study_dir = cfg.paths.artifacts / STUDY_ROOT_NAME / "gate_check"
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "intensity_gate_check.log")

    logger.info("=" * 70)
    logger.info("ด่านตรวจก่อนลงทุนดาวน์โหลด — งานเปรียบเทียบ SHARP-only vs SHARP+intensity")
    logger.info("=" * 70)

    frames_dir = cfg.paths.processed / "frames"
    segmentation = SegmentationService(
        cfg.paths.artifacts / "models" / "unet.pt", frames_dir=frames_dir, device="cpu"
    )
    if not segmentation.available:
        logger.error("ยังไม่มีโมเดล segmentation — รัน backend/scripts/train_unet.py ก่อน")
        return 1

    aia_store = AiaFrameStore(cfg.aia.root)
    wcs_store = FrameWcsStore(cfg.paths.interim / "frame_wcs.parquet")
    channel_keys = [ch.key for ch in cfg.aia.channels]

    case_study_stems = sorted(
        p.stem for p in (frames_dir / "images").glob("*.npy")
        if p.stem >= "20240501_000000" and p.stem <= "20240531_235959"
    )
    logger.info("[1/4] หน้าต่าง พ.ค. 2024: พบ %d เฟรมบนดิสก์", len(case_study_stems))
    n_with_identity = sum(1 for s in case_study_stems if (frames_dir / "harp_ids" / f"{s}.npy").exists())
    logger.info("      มีแผนที่ระบุตัวตนแล้ว %d จาก %d เฟรม", n_with_identity, len(case_study_stems))

    logger.info("[2/4] โหลดตาราง SHARP + กรองตามกติกาของโปรเจค")
    sharp_raw = pd.read_parquet(cfg.paths.interim / "sharp_keywords.parquet")
    sharp_clean = clean_sharp_frame(sharp_raw, cfg)
    frame_times = [pd.Timestamp(s, tz=None) for s in
                   pd.to_datetime([s for s in case_study_stems], format="%Y%m%d_%H%M%S")]
    expected = expected_harps_by_frame(sharp_clean, frame_times)

    logger.info("[3/4] สกัดความเข้มแสง + จับคู่กับ HARP ทุกเฟรมในหน้าต่าง พ.ค. 2024")
    intensity_table, match_report = run_matching_pass(
        frames_dir, case_study_stems, expected, segmentation, aia_store, wcs_store,
        channel_keys, cfg.fulldisk.min_ar_area_px,
    )

    logger.info("[4/4] ตรวจการลบ drift ข้ามปีด้วยเฟรมรายสัปดาห์เดิม (2011-2024)")
    drift = drift_check(
        frames_dir, segmentation, aia_store, channel_keys[0], cfg.fulldisk.min_ar_area_px,
        args.drift_sample_frames,
    )

    flares = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    signal = signal_preview(intensity_table, flares, cfg.flare.horizon_hours, cfg.flare.positive_goes_class)

    # ------------------------------------------------------------------ #
    study_dir.mkdir(parents=True, exist_ok=True)
    intensity_table.to_parquet(study_dir / "intensity_may2024.parquet", index=False)
    match_report.drop(columns=["reasons"]).to_parquet(study_dir / "match_report.parquet", index=False)

    ok_frames = match_report[match_report["status"] == "ok"]
    total_expected = int(ok_frames["n_expected"].sum())
    total_matched = int(ok_frames["n_matched"].sum())
    match_rate = total_matched / total_expected if total_expected else 0.0

    report = _render_report(
        case_study_stems, n_with_identity, match_report, total_expected, total_matched, match_rate,
        drift, signal, intensity_table,
    )
    (study_dir / "report.md").write_text(report, encoding="utf-8")

    logger.info("=" * 70)
    logger.info("บันทึกผลด่านตรวจที่ %s", study_dir)
    logger.info("  อัตราการจับคู่: %.1f%% (%d/%d)", 100 * match_rate, total_matched, total_expected)
    if drift.get("ok"):
        logger.info(
            "  overlap ก่อน normalise: %.2f  หลัง normalise: %.2f  (%s)",
            drift["raw_overlap"], drift["norm_overlap"],
            "ดีขึ้น" if drift["improved"] else "ไม่ดีขึ้น",
        )
    logger.info("  จำนวนแถวความเข้มแสงที่ได้: %d", len(intensity_table))
    logger.info("อ่านรายงานเต็มที่ %s", study_dir / "report.md")
    logger.info("=" * 70)
    return 0


def _render_report(
    case_study_stems, n_with_identity, match_report, total_expected, total_matched, match_rate,
    drift, signal, intensity_table,
) -> str:
    lines = [
        "# ด่านตรวจงานเปรียบเทียบ SHARP-only vs SHARP+intensity — หน้าต่าง พ.ค. 2024",
        "",
        f"เฟรมบนดิสก์: {len(case_study_stems)} · มีแผนที่ระบุตัวตนของ HARP: {n_with_identity}",
        "",
        "## 1. อัตราการจับคู่ AR เข้ากับ HARP (เกณฑ์ตัดสิน)",
        "",
        f"**{total_matched}/{total_expected} = {100*match_rate:.1f}%**",
        "",
    ]

    missed_lines = []
    for row in match_report.itertuples():
        if getattr(row, "status", "ok") != "ok" or not row.missed_harps:
            continue
        for harp in row.missed_harps:
            reason = row.reasons.get(harp, "?")
            missed_lines.append(f"- {row.stem} · HARP {harp}: {reason}")
    if missed_lines:
        lines.append("### รายการที่พลาด")
        lines.append("")
        lines.extend(missed_lines[:50])
        if len(missed_lines) > 50:
            lines.append(f"... และอีก {len(missed_lines) - 50} รายการ")
        lines.append("")

    skipped = match_report[match_report["status"] != "ok"]
    if len(skipped):
        lines.append(f"เฟรมที่ข้าม (ไม่มีแผนที่ระบุตัวตน/ภาพ): {len(skipped)} เฟรม")
        lines.append("")

    lines += [
        "## 2. การลบ drift ข้ามปี (เกณฑ์ตัดสิน)",
        "",
    ]
    if drift.get("ok"):
        lines += [
            f"เทียบปี {drift['early_years']} (n={drift['n_early']}) กับปี {drift['late_years']} (n={drift['n_late']})",
            "",
            f"- ก่อน normalise: median {drift['raw_early_median']:.2f} vs {drift['raw_late_median']:.2f} "
            f"· overlap ของ IQR = {drift['raw_overlap']:.2f}",
            f"- หลัง normalise: median {drift['norm_early_median']:.2f} vs {drift['norm_late_median']:.2f} "
            f"· overlap ของ IQR = {drift['norm_overlap']:.2f}",
            "",
            f"**{'ผ่าน' if drift['improved'] else 'ไม่ผ่าน'}** — overlap หลัง normalise "
            f"{'มากกว่า' if drift['improved'] else 'ไม่ได้มากกว่า'}ก่อน normalise",
        ]
    else:
        lines.append(f"ตรวจไม่ได้: {drift.get('reason')}")
    lines.append("")

    lines += [
        "## 3. ความสมเหตุสมผลของจำนวน (เกณฑ์ตัดสิน)",
        "",
        f"จำนวนแถวความเข้มแสงที่ได้ (matched แล้ว): {len(intensity_table)}",
        f"จำนวน HARP ที่ปรากฏ: {intensity_table['HARPNUM'].nunique() if not intensity_table.empty else 0}",
        f"เฉลี่ย AR ที่จับคู่ได้ต่อเฟรม: "
        f"{match_report.loc[match_report['status']=='ok','n_matched'].mean():.2f}" if (match_report['status']=='ok').any() else "n/a",
        "",
        "## 4. สัญญาณเบื้องต้น (ข้อมูลประกอบ — ห้ามใช้ตัดสินว่าจะไปต่อหรือไม่)",
        "",
    ]
    if signal.get("ok"):
        lines += [
            f"HARP-เวลาที่จะเกิด flare ในหน้าต่างนี้: {signal['n_positive']} · ไม่เกิด: {signal['n_negative']}",
            f"171Å median: positive={signal['positive_median']:.3f} เท่าของ quiet Sun, "
            f"negative={signal['negative_median']:.3f} เท่าของ quiet Sun",
            "",
            "**คำเตือน**: หน้าต่างนี้ถูกครอบงำโดย AR 13664 (พายุ Gannon) เดือนเดียว "
            "ตัวเลขนี้จึงไม่ใช่ตัวแทนของสัญญาณในช่วง 7 ปีเต็ม — ห้ามใช้ตัดสินว่าจะไปต่อหรือไม่",
        ]
    else:
        lines.append(f"ยังรายงานไม่ได้: {signal.get('reason')}")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

"""วิเคราะห์ผลงาน feature-evidence-cv ตาม spec ที่ล็อกไว้ก่อนเห็นผล (``.scratch/feature-evidence-cv/spec.md``)

    python backend/scripts/study/evidence_analysis.py

อ่าน ``artifacts/feature_evidence_cv/predictions.parquet`` + ``runs.parquet`` (จาก ``study/train.py --cv``)
แล้วเขียน ``evidence_report.md`` + ``evidence_summary.json`` ในโฟลเดอร์เดียวกัน

ตัวเลขหลักทั้งหมดมาจาก :mod:`sunseg.study.evidence` — สคริปต์นี้แค่จัดตาราง · ก่อนคำนวณตรวจว่า spec
ยังตรงกับ hash ที่บันทึกตอนล็อก ถ้าไม่ตรงจะเตือนในรายงาน (ไม่หยุด — ให้ผู้อ่านเห็นเอง)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import PROJECT_ROOT, load_data_config  # noqa: E402
from sunseg.data.study_dataset import load_variants  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.study.evidence import (  # noqa: E402
    bootstrap_deltas,
    cluster_counts,
    holm,
    pooled_tss,
    tss_from_counts,
    verdict,
)

logger = logging.getLogger("evidence_analysis")

CONTROL = "V0"
PRIMARY = {"VH": "H1", "VI": "H2"}
SECONDARY = ["V1", "V2", "VHI"]
ARCHITECTURE = "lstm"
SPEC = PROJECT_ROOT / ".scratch" / "feature-evidence-cv" / "spec.md"
VARIANTS_FILE = PROJECT_ROOT / "backend" / "configs" / "study" / "variants_evidence.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, help="ปริยาย artifacts/feature_evidence_cv")
    p.add_argument("--n-boot", type=int, default=10_000, help="จำนวนรอบ bootstrap (spec: 10,000)")
    p.add_argument("--seed", type=int, default=0, help="seed ของ bootstrap")
    return p.parse_args()


def spec_status() -> str:
    recorded_path = SPEC.with_name("spec.sha256")
    if not SPEC.exists() or not recorded_path.exists():
        return "ไม่พบ spec หรือ hash ที่บันทึกไว้"
    recorded = recorded_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    if actual == recorded:
        return f"spec ตรงกับ hash ที่บันทึกตอนล็อก (`{actual[:12]}…`)"
    return f"**คำเตือน: spec ถูกแก้หลังล็อก** (บันทึก `{recorded[:12]}…` ตอนนี้ `{actual[:12]}…`)"


def fold_table(test: pd.DataFrame) -> pd.DataFrame:
    """ขนาดของ test ต่อ fold — ใช้แถวของ control seed แรก (ทุกแบบ/seed มีแถวชุดเดียวกัน)"""
    one = test[(test["variant"] == CONTROL) & (test["seed"] == test["seed"].min())]
    rows = []
    for fold, group in one.groupby("fold"):
        times = pd.to_datetime(group["issue_time"])
        rows.append(
            {
                "fold": int(fold),
                "test": f"{times.min():%Y-%m-%d} – {times.max():%Y-%m-%d}",
                "sample": len(group),
                "positive": int(group["label"].sum()),
                "HARP": group["HARPNUM"].nunique(),
                "HARP ที่ปะทุ": group.loc[group["label"] == 1, "HARPNUM"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def per_fold_delta(cc, variant: str) -> pd.Series:
    """ΔTSS ราย fold (ค่าเฉลี่ยข้าม seed) — ตัวเลขประกอบ ไม่มีการอนุมาน"""
    out = {}
    for fold in np.unique(cc.cluster_fold):
        mask = cc.cluster_fold == fold
        v = tss_from_counts(cc.counts[variant][:, mask].sum(axis=1))
        c = tss_from_counts(cc.counts[CONTROL][:, mask].sum(axis=1))
        out[int(fold)] = float(np.nanmean(v - c))
    return pd.Series(out)


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def fmt_p(p: float, n_boot: int) -> str:
    """p = 0 จาก bootstrap แปลว่าเล็กกว่าที่จำนวนรอบแยกได้ ไม่ใช่ศูนย์จริง"""
    return f"< {2 / n_boot:.4f}" if p == 0 else f"{p:.4f}"


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_data_config()
    out_dir = args.out_dir or cfg.paths.artifacts / "feature_evidence_cv"

    predictions = pd.read_parquet(out_dir / "predictions.parquet")
    runs = pd.read_parquet(out_dir / "runs.parquet")
    labels = {name: spec["label"] for name, spec in load_variants(VARIANTS_FILE).items()}

    test = predictions[(predictions["split"] == "test") & (predictions["architecture"] == ARCHITECTURE)]
    if "fold" not in test.columns:
        logger.error("predictions ไม่มีคอลัมน์ fold — ต้องเทรนด้วย study/train.py --cv รุ่นที่ใส่ fold ลงไฟล์")
        return 1
    present = [v for v in [CONTROL, *PRIMARY, *SECONDARY] if v in set(test["variant"])]
    missing_primary = [v for v in [CONTROL, *PRIMARY] if v not in present]
    if missing_primary:
        logger.error("ไม่มีผลของแบบหลัก %s", missing_primary)
        return 1

    cc = cluster_counts(test, present)
    logger.info("cluster (fold, HARP) ของ test: %d · seed %s · bootstrap %d รอบ", len(cc.clusters), cc.seeds, args.n_boot)
    results = bootstrap_deltas(cc, CONTROL, n_boot=args.n_boot, seed=args.seed)
    adjusted = holm({v: results[v].p_two_sided for v in PRIMARY})

    folds = fold_table(test)
    tss_by_variant = {v: pooled_tss(cc, v) for v in present}

    lines = [
        "# หลักฐานว่า feature ใหม่ช่วยพยากรณ์ flare หรือไม่ — CV แบบ expanding window",
        "",
        f"วิเคราะห์ตาม `.scratch/feature-evidence-cv/spec.md` · {spec_status()} · "
        f"LSTM · seed {', '.join(map(str, cc.seeds))} · {len(folds)} fold · bootstrap {args.n_boot:,} รอบ "
        f"(สุ่ม HARP ภายใน fold + สุ่ม seed, seed ของ bootstrap = {args.seed})",
        "",
        "## fold ที่ผ่านเกณฑ์ (test)",
        "",
        "| fold | ช่วง test | sample | positive | HARP | HARP ที่ปะทุ |",
        "|---:|---|---:|---:|---:|---:|",
        *[
            f"| {r['fold']} | {r['test']} | {r['sample']} | {r['positive']} | {r['HARP']} | {r['HARP ที่ปะทุ']} |"
            for r in folds.to_dict(orient="records")
        ],
        f"| **รวม** | | **{folds['sample'].sum()}** | **{folds['positive'].sum()}** | **{folds['HARP'].sum()}** | "
        f"**{folds['HARP ที่ปะทุ'].sum()}** |",
        "",
        "## ผลหลัก (ตัดสินตาม spec หัวข้อ 7)",
        "",
        f"TSS รวมทุก fold ต่อ seed แล้วเฉลี่ยข้าม seed · Δ เทียบ {CONTROL} แบบจับคู่ · p สองทางจาก bootstrap ปรับ Holm สองสมมติฐาน",
        "",
        "| สมมติฐาน | แบบ | TSS (mean ± SD ข้าม seed) | ΔTSS | 95% CI | 90% CI | SE | p | p (Holm) | ผล |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---|",
        f"| control | {CONTROL} {labels[CONTROL]} | {np.mean(tss_by_variant[CONTROL]):.3f} ± "
        f"{np.std(tss_by_variant[CONTROL], ddof=1):.3f} | — | | | | | | |",
    ]
    summary: dict = {"spec": spec_status(), "folds": folds.to_dict(orient="records"), "primary": {}, "secondary": {}}
    for variant, hypothesis in PRIMARY.items():
        r = results[variant]
        decided = verdict(r, adjusted[variant])
        lines.append(
            f"| **{hypothesis}** | {variant} {labels[variant]} | {np.mean(tss_by_variant[variant]):.3f} ± "
            f"{np.std(tss_by_variant[variant], ddof=1):.3f} | {r.delta:+.4f} | {fmt_ci(r.ci95)} | {fmt_ci(r.ci90)} | "
            f"{r.se:.4f} | {fmt_p(r.p_two_sided, args.n_boot)} | {fmt_p(adjusted[variant], args.n_boot)} | **{decided}** |"
        )
        summary["primary"][variant] = {
            "hypothesis": hypothesis, "delta": r.delta, "se": r.se, "ci95": r.ci95, "ci90": r.ci90,
            "p": r.p_two_sided, "p_holm": adjusted[variant], "verdict": decided,
        }
    mean_se = float(np.mean([results[v].se for v in PRIMARY]))
    lines += [
        "",
        f"- Δ เล็กที่สุดที่การทดลองนี้ตรวจพบได้ (power 80%, α 0.05 สองทาง ≈ 2.8 × SE) ≈ **{2.8 * mean_se:.3f}** "
        f"(spec คาดไว้ก่อนรัน ≈ 0.02)",
        "- เกณฑ์ \"ไม่ต่างในทางปฏิบัติ\" = 90% CI อยู่ใน ±0.02 ทั้งช่วง",
        "",
        "## ผลรอง (สำรวจ — ไม่ปรับ Holm ไม่ใช้ตัดสิน)",
        "",
        "| แบบ | TSS (mean ± SD ข้าม seed) | ΔTSS | 95% CI | p (ไม่ปรับ) |",
        "|---|---:|---:|---|---:|",
    ]
    for variant in [v for v in SECONDARY if v in present]:
        r = results[variant]
        lines.append(
            f"| {variant} {labels[variant]} | {np.mean(tss_by_variant[variant]):.3f} ± "
            f"{np.std(tss_by_variant[variant], ddof=1):.3f} | {r.delta:+.4f} | {fmt_ci(r.ci95)} | {fmt_p(r.p_two_sided, args.n_boot)} |"
        )
        summary["secondary"][variant] = {"delta": r.delta, "se": r.se, "ci95": r.ci95, "p": r.p_two_sided}

    compared = [v for v in present if v != CONTROL]
    per_fold = pd.DataFrame({v: per_fold_delta(cc, v) for v in compared})
    lines += [
        "",
        "### ΔTSS ราย fold (เฉลี่ยข้าม seed — ประกอบการอ่าน ไม่มีการอนุมาน)",
        "",
        "| fold | " + " | ".join(compared) + " |",
        "|---:|" + "---:|" * len(compared),
        *[
            f"| {fold} | " + " | ".join(f"{per_fold.at[fold, v]:+.3f}" for v in compared) + " |"
            for fold in per_fold.index
        ],
        "",
    ]

    scoped = runs[(runs["scope"] == "test") & (runs["architecture"] == ARCHITECTURE) & runs["variant"].isin(present)]
    metric_names = [m for m in ("tss", "auc", "bss", "hss2", "recall", "precision") if m in scoped.columns]
    lines += [
        "### ตัวชี้วัดอื่นบน test (mean ± SD ข้าม fold × seed ของค่าราย fold)",
        "",
        "| แบบ | " + " | ".join(m.upper() for m in metric_names) + " |",
        "|---|" + "---:|" * len(metric_names),
    ]
    for variant in present:
        g = scoped[scoped["variant"] == variant]
        lines.append(
            f"| {variant} | " + " | ".join(f"{g[m].mean():+.3f} ± {g[m].std(ddof=1):.3f}" for m in metric_names) + " |"
        )
    lines += [
        "",
        "## อ่านผลนี้อย่างไร",
        "",
        "- ข้อสรุปมาจากตารางผลหลักเท่านั้น (H1, H2) — ผลรองและตาราง ราย fold ห้ามหยิบมาเล่าเป็นข้อค้นพบ",
        "- label มาจาก `flares_v2.parquet` (แก้เลข NOAA AR แล้ว) — ตัวเลขจึงเทียบตรง ๆ กับผลเดิมใน "
        "`model_comparison`/`lstm_feature_ablation` ไม่ได้ (label และชุด test ต่างกัน)",
        "- fold คือปีปฏิทินที่ผ่านเกณฑ์ positive ≥ 40 ทั้ง val และ test — ปีที่ไม่ผ่านไม่ได้ถูกเลือกทิ้งด้วยมือ",
        "",
    ]

    (out_dir / "evidence_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    logger.info("เขียน %s", out_dir / "evidence_report.md")
    for variant, info in summary["primary"].items():
        logger.info("  %s %s: Δ %+.4f · 95%% CI %s · p(Holm) %.4f -> %s", info["hypothesis"], variant,
                    info["delta"], fmt_ci(tuple(info["ci95"])), info["p_holm"], info["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

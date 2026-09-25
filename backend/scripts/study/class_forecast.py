"""รายงานของโมเดลหลัก (LSTM + V3) — คำพยากรณ์ระดับคลาส <M / M / X ใน 24 ชม.

    python backend/scripts/study/class_forecast.py

ต้องเทรนเซลล์ของโมเดลหลักไว้ทั้งสองระดับก่อน (ค่าตาม ``class_forecast`` ใน ``configs/forecast.yaml``)::

    python backend/scripts/study/train.py                                  # ระดับ ≥M1.0 -> artifacts/model_comparison
    python backend/scripts/study/build_dataset.py --positive-class X1.0 --out-dir data/processed/study_sequences_x
    python backend/scripts/study/train.py --data-dir data/processed/study_sequences_x \\
        --out-dir artifacts/model_comparison_x --variants V3 --architectures-file <yaml ที่มีแค่ lstm>

ตัวเลขทั้งหมดมาจาก :class:`sunseg.inference.class_forecast.ClassForecastService` ตัวเดียวกับที่ API ใช้
(ensemble ทุก seed ของสองระดับ รวมเป็นระดับเดียวด้วยจุดทำงาน ``sensitive``/``strict``) รายงานกับหน้าเว็บจึงตรงกัน
โดยโครงสร้าง · ค่าอ้างอิง (TSS ราย seed, อันดับในงานเปรียบเทียบ, persistence) อ่านจากไฟล์ที่สคริปต์อื่นเขียนไว้
ถ้าไม่มีไฟล์นั้นก็ข้ามหัวข้อนั้นไป

เขียนลง ``artifacts/class_forecast/``: ``report.md`` · ``summary.json`` · ``figures/confusion_test.png`` (+ CSV)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# report_style ต้องมาก่อน pyplot — ตั้ง backend Agg + ธีม seaborn (ดู docstring ของโมดูลนั้น)
from sunseg.report_style import SEQUENTIAL, save_figure  # noqa: E402, I001

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config, load_forecast_config  # noqa: E402
from sunseg.inference.class_forecast import LEVELS, MODES, ClassForecastService  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("class_forecast")

OUT_DIR_NAME = "class_forecast"
MODE_TITLES = {
    "sensitive": "เตือนไว (sensitive)",
    "strict": "ระมัดระวัง (strict)",
}
MODE_FIGURE_TITLES = {"sensitive": "Sensitive (TSS-optimal X)", "strict": "Strict (val-selected X)"}
LEVEL_TEXT = {"<M": "<M", "M": "M", "X": "X"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, help="ที่เก็บผล (ปริยาย: artifacts/class_forecast)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# ค่าอ้างอิงจากไฟล์ของสคริปต์อื่น
# --------------------------------------------------------------------------- #


def per_seed_tss(artifacts: Path, study_dir: str, architecture: str, variant: str) -> dict | None:
    """TSS ราย seed ของแบบจำลองระดับเดียว (ไม่ ensemble) จาก ``runs.parquet`` ของงานเปรียบเทียบ"""
    path = artifacts / study_dir / "runs.parquet"
    if not path.exists():
        return None
    runs = pd.read_parquet(path)
    runs = runs[(runs["architecture"] == architecture) & (runs["variant"] == variant) & runs["fold"].isna()]
    out = {}
    for scope, rows in runs.groupby("scope"):
        out[scope] = {"mean": float(rows["tss"].mean()), "sd": float(rows["tss"].std(ddof=1)), "n_seeds": len(rows)}
    return out or None


def ranking_entry(artifacts: Path, architecture: str, variant: str) -> dict | None:
    """อันดับของเซลล์โมเดลหลักในงานเปรียบเทียบ + จำนวนเซลล์ที่แยกจากอันดับ 1 ไม่ได้"""
    path = artifacts / "model_comparison" / "summary.json"
    if not path.exists():
        return None
    ranking = json.loads(path.read_text(encoding="utf-8")).get("ranking") or []
    mine = next((r for r in ranking if r["architecture"] == architecture and r["variant"] == variant), None)
    if mine is None:
        return None
    return {
        **mine,
        "n_cells": len(ranking),
        "n_tied": sum(1 for r in ranking if r.get("tied_with_best")),
        "n_val_ranked": sum(1 for r in ranking if r.get("val_rank") is not None),
    }


def persistence(artifacts: Path) -> dict | None:
    """กฎ persistence ต่อระดับ (``persistence_baseline.py``) — ตัวเทียบที่ไม่ใช้ machine learning"""
    path = artifacts / "model_comparison" / "persistence_baseline.json"
    if not path.exists():
        return None
    levels = json.loads(path.read_text(encoding="utf-8")).get("levels", {})
    out = {}
    for name, positive in (("M", "M1.0"), ("X", "X1.0")):
        level = levels.get(positive)
        if not level:
            continue
        tuned = (level.get("tuned_persistence") or {}).get("chosen") or {}
        out[name] = {
            "classic_test": level.get("persistence", {}).get("test"),
            "tuned_rule": {k: tuned.get(k) for k in ("trigger", "lookback_hours")} if tuned else None,
            "tuned_test": tuned.get("test"),
        }
    return out or None


# --------------------------------------------------------------------------- #
# รูป + รายงาน
# --------------------------------------------------------------------------- #


def confusion_figure(evaluation: dict, path: Path) -> None:
    """ตาราง 3×3 ของ test สองจุดทำงานเคียงกัน — สีตามสัดส่วนในแถว (แถวละระดับจริง) ตัวเลขคือจำนวนนับ"""
    fig, axes = plt.subplots(1, len(MODES), figsize=(4.2 * len(MODES), 3.9), constrained_layout=True)
    rows = []
    for ax, mode in zip(np.atleast_1d(axes), MODES, strict=True):
        cm = np.array(evaluation[mode]["test"]["confusion"])
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(LEVELS)):
            for j in range(len(LEVELS)):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                        color="white" if share[i, j] > 0.55 else "black")
                rows.append({"mode": mode, "true_level": LEVELS[i], "predicted_level": LEVELS[j],
                             "count": int(cm[i, j]), "row_share": float(share[i, j])})
        ax.set_xticks(range(len(LEVELS)), LEVELS)
        ax.set_yticks(range(len(LEVELS)), LEVELS)
        ax.set_xlabel("Predicted level")
        ax.set_ylabel("Actual level (next 24 h)")
        ax.set_title(MODE_FIGURE_TITLES[mode])
        ax.grid(False)
    fig.suptitle("LSTM + V3 class forecast — test set", fontweight="bold")
    save_figure(fig, path)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def render_report(info: dict, evaluation: dict, seeds: dict, ranking: dict | None, persist: dict | None,
                  n_test_x_harps: int, figure: str) -> str:
    label = info["label"]
    n_seeds = info["n_seeds"]
    lines = [
        f"# โมเดลหลัก: {label} — คำพยากรณ์ระดับคลาสของ flare ใน 24 ชม.",
        "",
        f"สร้างเมื่อ {datetime.now():%Y-%m-%d %H:%M} · สร้างด้วย `backend/scripts/study/class_forecast.py` · "
        f"ตัวเลขชุดเดียวกับที่หน้าเว็บแสดง (`/api/class-forecast`)",
        "",
        f"**{label}** (LSTM + 18 SHARP + intensity 3 ช่อง + X-ray, {len(info['features'])} feature × "
        f"{info['sequence_length']} timestep ทุก {info['cadence_hours']} ชม.) เป็นโมเดลหลักของโปรเจค "
        "ทุกคำพยากรณ์ออกเป็น **หนึ่งในสามระดับ** ของ flare ที่แรงที่สุดใน 24 ชม. ถัดไป:",
        "",
        "| ระดับ | ความหมาย |",
        "|---|---|",
        "| **X** | จะเกิด flare ≥ X1.0 |",
        "| **M** | จะเกิด flare ≥ M1.0 แต่ไม่ถึง X |",
        "| **<M** | ไม่มีระดับไหนเตือน — ไม่มี flare หรือมีแค่ B/C (ระบบไม่มีแบบจำลอง ≥C1.0 จึงแยก C ออกจากเงียบไม่ได้) |",
        "",
        "## วิธีทำ",
        "",
        f"- สองแบบจำลองทวิภาคของเซลล์เดียวกัน: ระดับ M เทรนด้วย label ≥M1.0 ({n_seeds.get('M')} seed) และระดับ X "
        f"เทรนด้วย label ≥X1.0 ({n_seeds.get('X')} seed) — แถวข้อมูล, split และ hyperparameter เหมือนกันทุกประการ ต่างกันแค่ label",
        "- ในแต่ละระดับ seed หนึ่งเตือนเมื่อความน่าจะเป็น ≥ threshold ของตัวเอง (เลือกให้ TSS บน val สูงสุด) · "
        "ensemble เตือนเมื่อ seed เกินครึ่งเตือน",
        "- ระดับที่ทำนาย = ระดับสูงสุดที่เตือน",
        "- **สองจุดทำงาน** ต่างกันแค่ระดับ X (ระดับ M เหมือนกัน):",
        "  - **เตือนไว (sensitive, ค่าปริยาย)** — ระดับ X ใช้เสียงข้างมากของ seed ตามข้างบน",
        "  - **ระมัดระวัง (strict)** — ระดับ X เตือนเมื่อความน่าจะเป็นเฉลี่ยข้าม seed ≥ "
        f"{_fmt(info.get('strict_threshold'))} ซึ่งเลือกบน val ให้ HSS แบบ 3 คลาสสูงสุด",
        "",
    ]

    if ranking:
        lines += [
            "## ทำไมเป็น LSTM + V3",
            "",
            f"เป็นอันดับ {ranking['rank']} จาก {ranking['n_cells']} เซลล์ของงานเปรียบเทียบ "
            f"(`artifacts/model_comparison/report.md`) ด้วย TSS บน test {ranking['mean']:.3f} ± {ranking['sd']:.3f} "
            f"({ranking['n_seeds']} seed) — **แต่มี {ranking['n_tied']} เซลล์ที่แยกจากอันดับ 1 ไม่ได้** และบน validation "
            f"เซลล์นี้อยู่อันดับ {ranking.get('val_rank', '—')} (TSS {_fmt(ranking.get('val_mean'))}) "
            "การเลือกเป็นโมเดลหลักจึงเป็นการตัดสินใจของโปรเจค ไม่ใช่ข้อสรุปว่ามันดีกว่าตัวอื่นอย่างมีนัยสำคัญ",
            "",
        ]

    lines += [
        "## ผลบน test",
        "",
        f"![confusion]({figure})",
        "",
        "| ตัวชี้วัด | " + " | ".join(MODE_TITLES[m] for m in MODES) + " |",
        "|---|" + "---|" * len(MODES),
    ]
    test = {m: evaluation[m]["test"] for m in MODES}
    val = {m: evaluation[m]["val"] for m in MODES}
    for level in ("M", "X"):
        for key, title in (("tss", "TSS"), ("hss2", "HSS2"), ("recall", "recall"), ("precision", "precision")):
            lines.append(f"| ≥{level} {title} | " + " | ".join(_fmt(test[m]["thresholds"][level][key]) for m in MODES) + " |")
        lines.append(
            f"| ≥{level} TP / FP / FN | "
            + " | ".join(
                f"{t['tp']} / {t['fp']} / {t['fn']}" for t in (test[m]["thresholds"][level] for m in MODES)
            )
            + " |"
        )
    lines += [
        "| HSS 3 คลาส | " + " | ".join(_fmt(test[m]["hss_multiclass"]) for m in MODES) + " |",
        "| ทายระดับถูกเป๊ะ (เฉพาะ event ≥M จริง) | " + " | ".join(_pct(test[m]["exact_on_events"]) for m in MODES) + " |",
        "| X เตือนแต่ M ไม่เตือน (นับเป็น X) | " + " | ".join(str(test[m]["n_inconsistent"]) for m in MODES) + " |",
        "| HSS 3 คลาสบน val (ใช้เลือก strict) | " + " | ".join(_fmt(val[m]["hss_multiclass"]) for m in MODES) + " |",
        "",
    ]
    for mode in MODES:
        cm = test[mode]["confusion"]
        lines += [
            f"**{MODE_TITLES[mode]}** — แถว = ระดับจริง, คอลัมน์ = ระดับที่ทำนาย",
            "",
            "| จริง \\ ทำนาย | " + " | ".join(LEVELS) + " |",
            "|---|" + "---|" * len(LEVELS),
            *(f"| {LEVELS[i]} | " + " | ".join(f"{v:,}" for v in row) + " |" for i, row in enumerate(cm)),
            "",
        ]

    lines += ["## อ่านผลอย่างไร", ""]
    s, t = test["sensitive"], test["strict"]
    lines += [
        f"- **ระดับ ≥M ใช้ได้จริง** — TSS {_fmt(s['thresholds']['M']['tss'])} / {_fmt(t['thresholds']['M']['tss'])} "
        f"(recall {_pct(s['thresholds']['M']['recall'])}, precision {_pct(s['thresholds']['M']['precision'])}) · "
        f"ต่างกันเล็กน้อยเพราะจุดเตือนไวนับ {s['n_inconsistent']} แถวที่ระดับ X เตือนแต่ระดับ M ไม่เตือนเป็นการเตือน ≥M ด้วย",
        f"- **ระดับ X แยก X ออกจาก M ได้ไม่ดี** — จุดเตือนไวจับ X ได้ {s['thresholds']['X']['tp']} จาก "
        f"{s['thresholds']['X']['n_positive']} แต่ flare ระดับ M จริง {s['confusion'][1][2]} จาก {sum(s['confusion'][1])} "
        f"ถูกเรียกว่า X ไปด้วย (ทายระดับถูกเป๊ะแค่ {_pct(s['exact_on_events'])}) · จุดระมัดระวังทายระดับถูก "
        f"{_pct(t['exact_on_events'])} แต่จับ X ได้แค่ {t['thresholds']['X']['tp']} จาก {t['thresholds']['X']['n_positive']}",
        f"- **X ใน test มีแค่ {s['thresholds']['X']['n_positive']} sample จาก {n_test_x_harps} HARP** "
        "— ตัวเลขทุกตัวของระดับ X แกว่งได้มากจากการเพิ่ม/ลด HARP เดียว",
        "- ไม่มีจุดทำงานไหน \"ถูก\" — เลือกตามต้นทุน: พลาด X แพงกว่าเตือนผิด → เตือนไว, ต้องการให้ระดับที่บอกเชื่อได้ → ระมัดระวัง",
        "",
    ]

    if seeds:
        lines += [
            "## เทียบกับแบบจำลองเดี่ยวราย seed",
            "",
            "TSS ของแบบจำลองระดับเดียว (ไม่ ensemble, ค่าเฉลี่ย ± SD ข้าม seed จาก `runs.parquet`) — ensemble "
            "จุดเตือนไวควรใกล้เคียงค่าเหล่านี้",
            "",
            "| ระดับ | val | test |",
            "|---|---|---|",
        ]
        for level, stats in seeds.items():
            cells = [
                f"{stats[s]['mean']:.3f} ± {stats[s]['sd']:.3f}" if s in stats else "—" for s in ("val", "test")
            ]
            lines.append(f"| ≥{level} | " + " | ".join(cells) + " |")
        lines.append("")

    if persist:
        lines += [
            "## เทียบกับ persistence (ไม่ใช้ machine learning)",
            "",
            "จาก `artifacts/model_comparison/persistence_baseline.md` — กฎ persistence ที่เลือกบน val (\"เคยเกิด flare "
            "≥ trigger ในช่วงย้อนหลัง\") **ต้องรายงานคู่กันเสมอ** เพราะเป็นตัวเทียบที่ไม่ต้องเทรนอะไรเลย",
            "",
            "| ระดับ | persistence ดั้งเดิม TSS | persistence ที่เลือกบน val | TSS | โมเดลหลัก (เตือนไว) TSS |",
            "|---|---|---|---|---|",
        ]
        for level, p in persist.items():
            rule = p.get("tuned_rule") or {}
            rule_text = f"≥{rule.get('trigger')} ใน {rule.get('lookback_hours')} ชม." if rule else "—"
            lines.append(
                f"| ≥{level} | {_fmt((p.get('classic_test') or {}).get('tss'))} | {rule_text} | "
                f"{_fmt((p.get('tuned_test') or {}).get('tss'))} | {_fmt(s['thresholds'][level]['tss'])} |"
            )
        beaten = [
            f"≥{level}" for level, p in persist.items()
            if (p.get("tuned_test") or {}).get("tss") is not None
            and p["tuned_test"]["tss"] > s["thresholds"][level]["tss"]
        ]
        lines += [
            "",
            (
                f"**persistence ที่เลือกบน val ให้ TSS บน test สูงกว่าโมเดลหลักที่ระดับ {' และ '.join(beaten)}**"
                if beaten
                else "โมเดลหลักให้ TSS บน test สูงกว่า persistence ที่เลือกบน val ทุกระดับ"
            ),
            "",
        ]

    lines += [
        "## ข้อจำกัดที่ต้องบอกผู้อ่าน",
        "",
        "- **การมีสองจุดทำงานถูกตัดสินหลังจากเห็นเส้น trade-off บน test** (2026-09-24) — threshold ของจุดระมัดระวัง"
        "เลือกบน val ล้วน แต่ความคิดที่จะเสนอจุดที่สองเกิดขึ้นหลังเห็นว่าจุดเตือนไวเรียก M เป็น X บน test",
        "- <M ไม่ใช่คำพยากรณ์ว่าจะเกิด C",
        "- คำพยากรณ์ของ HARP ใน split train เป็นข้อมูลที่โมเดลเคยเห็นตอนเทรน — หน้าเว็บแสดงได้ แต่ห้ามใช้วัดผล",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    cfg = load_forecast_config().class_forecast
    artifacts = data_cfg.paths.artifacts
    out_dir = args.out_dir or artifacts / OUT_DIR_NAME
    setup_logging(log_file=artifacts / "logs" / "class_forecast.log")

    service = ClassForecastService(cfg, artifacts, data_cfg.paths.processed)
    if not service.available:
        logger.error("โมเดลหลักยังไม่พร้อม — รัน %s", service.hint)
        return 1

    info = service.info()
    evaluation = {mode: {split: service.evaluation(split, mode) for split in ("val", "test")} for mode in MODES}
    seeds = {
        level: stats
        for level, source in cfg.levels.items()
        if (stats := per_seed_tss(artifacts, source.study_dir, cfg.architecture, cfg.variant))
    }
    ranking = ranking_entry(artifacts, cfg.architecture, cfg.variant)
    persist = persistence(artifacts)
    test_rows = service.table[service.table["split"] == "test"]
    n_test_x_harps = int(test_rows.loc[test_rows["true_level"] == 2, "HARPNUM"].nunique())

    figure = Path("figures") / "confusion_test.png"
    confusion_figure(evaluation, out_dir / figure)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **info,
        "evaluation": evaluation,
        "per_seed_tss": seeds,
        "ranking": ranking,
        "persistence": persist,
        "n_test_x_harps": n_test_x_harps,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(info, evaluation, seeds, ranking, persist, n_test_x_harps, figure.as_posix()), encoding="utf-8"
    )

    for mode in MODES:
        ev = evaluation[mode]["test"]
        logger.info(
            "%-9s test: TSS ≥M %.3f · ≥X %.3f · HSS 3 คลาส %.3f · ทายระดับถูก %s ของ event",
            mode, ev["thresholds"]["M"]["tss"], ev["thresholds"]["X"]["tss"], ev["hss_multiclass"],
            _pct(ev["exact_on_events"]),
        )
    logger.info("รายงาน: %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

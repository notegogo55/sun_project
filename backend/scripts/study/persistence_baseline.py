"""baseline แบบ persistence เทียบกับแบบจำลองของงานเปรียบเทียบ

    python backend/scripts/study/persistence_baseline.py

กติกาของ baseline: **ทายว่าจะเกิด flare ≥ ระดับนั้นใน 24 ชม. ข้างหน้า ถ้าบริเวณเดียวกันเพิ่งมี flare ≥ ระดับนั้นใน 24 ชม.
ที่ผ่านมา** — ไม่ต้องเทรน ไม่มี threshold ใช้แค่ข้อมูลที่รู้แล้ว ณ เวลาออกพยากรณ์ ตอบคำถามที่กรรมการมักถามว่าแบบจำลอง
ดีกว่า "จำว่าเมื่อวานเกิด" แค่ไหน

ความยุติธรรมของการเทียบ:

- ใช้ **แถวข้อมูลชุดเดียวกับแบบจำลอง** (``meta.parquet`` ของ pool งานเปรียบเทียบ) split เดียวกัน
- ใช้ **flare catalog และการจับคู่ flare-HARP ชุดเดียวกับ label** (``build_flare_lookup``) ช่วงอดีต ``(t-24ชม., t]`` ต่อกับ
  ช่วงของ label ``(t, t+24ชม.]`` พอดี (``persistence_times``) — สคริปต์สร้าง label ซ้ำจาก catalog แล้ว **error ถ้าไม่ตรง
  กับ label ใน pool ทุกแถว** เพื่อยืนยันว่านิยามเดียวกันจริง
- baseline ทายเป็น 0/1 จึงรายงานเฉพาะตัวชี้วัดที่ใช้คำตอบ 0/1 (TSS, HSS2, recall, precision) — AUC/BSS ของคำตอบ 0/1
  เทียบกับความน่าจะเป็นต่อเนื่องของแบบจำลองไม่ได้

**เทียบที่อัตราเตือนผิดเท่ากัน** — threshold ของแบบจำลองถูกเลือกให้ TSS สูงสุด จึงเตือนบ่อยกว่า persistence มาก (recall
สูง precision ต่ำ) ตัวชี้วัดคนละตัวจึงให้ผู้ชนะคนละฝั่ง สคริปต์จึงเลื่อน threshold ของแบบจำลองแต่ละ seed บน test ไปที่จุดที่
FPR ไม่เกิน FPR ของ persistence แล้ววัด recall/precision ณ จุดนั้น — ตอบว่า "ถ้าเตือนผิดเท่ากัน ใครจับได้มากกว่า" โดยไม่ขึ้น
กับการเลือก threshold (เป็นการวิเคราะห์ ไม่ใช่วิธีเลือก threshold ที่ใช้งานจริง เพราะใช้ test เลือกจุด)

**persistence ที่ปรับได้ (tuned persistence)** — persistence คลาสสิกมีจุดทำงานจุดเดียว (เตือนน้อย) ส่วนแบบจำลองเลือก
threshold ได้ จึงลองกฎตระกูลเดียวกันที่เตือนบ่อยขึ้น: flare ≥ ``trigger`` ใน ``lookback`` ชม. ที่ผ่านมา (``TRIGGERS`` ×
``LOOKBACKS``) **เลือกกฎบน validation ด้วย TSS สูงสุด** — กติกาเดียวกับที่แบบจำลองเลือก threshold — แล้วรายงานบน test
ห้ามหยิบกฎที่ดีที่สุดบน test (ตารางครบทุกกฎมีให้ดูใน json เพื่อความโปร่งใส แต่ตัวที่เทียบคือตัวที่ validation เลือก)

**ข้อจำกัดที่ต้องบอก**: ตระกูลกฎนี้ **ไม่ได้ประกาศล่วงหน้า** — กำหนดขึ้น (2026-09-24) หลังเห็น TSS บน test ของกฎ
C1.0/C5.0/M1.0 × 24-120 ชม. ไปแล้ว การเลือกกฎตัวสุดท้ายทำบน validation แต่ขอบเขตของตระกูลอาจได้อิทธิพลจาก test

แบบจำลองที่เทียบ: เซลล์ ``--cell`` (ปริยาย lstm:V3) จาก ``artifacts/model_comparison`` (ระดับ M) และ
``artifacts/model_comparison_x`` (ระดับ X ถ้ามี) — 25 seed mean ± SD และจำนวน seed ที่ TSS สูงกว่า baseline
เขียน ``artifacts/model_comparison/persistence_baseline.json`` และ ``.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.build_sequences import build_flare_lookup, label_times, persistence_times  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.metrics import confusion_counts, hss2, precision_recall_f1, tss  # noqa: E402

logger = logging.getLogger("persistence_baseline")

#: (ระดับ, pool ของ label ระดับนั้น, โฟลเดอร์ผลของแบบจำลองระดับนั้น)
LEVELS = (
    ("M1.0", "study_sequences", "model_comparison"),
    ("X1.0", "study_sequences_x", "model_comparison_x"),
)
METRICS = ("tss", "hss2", "recall", "precision", "tp", "fp", "fn")
#: ตระกูลกฎของ tuned persistence — กำหนดหลังสำรวจบน test แล้ว (ดู docstring) ห้ามขยายเพิ่มอีกตามผล
TRIGGERS = ("C1.0", "C5.0", "M1.0", "X1.0")
LOOKBACKS = (24, 48, 72, 120)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="lstm:V3", help="เซลล์ที่เทียบ <สถาปัตยกรรม>:<แบบ> (ปริยาย lstm:V3 — อันดับ 1)")
    p.add_argument("--lookback-hours", type=int, default=24, help="ช่วงอดีตที่ดู (ปริยาย 24 = เท่ากับหน้าต่างพยากรณ์)")
    return p.parse_args()


def binary_report(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    c = confusion_counts(y, pred)
    precision, recall, _ = precision_recall_f1(y, pred)
    return {"tss": tss(y, pred), "hss2": hss2(y, pred), "recall": recall, "precision": precision,
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "n": int(len(y)), "n_pos": int(y.sum())}


def persistence_for_pool(meta: pd.DataFrame, flares: pd.DataFrame, level: str, horizon: int,
                         lookback: int) -> pd.DataFrame:
    """คำตอบของ baseline ทุกแถวของ pool + label ที่สร้างซ้ำจาก catalog (ใช้ตรวจว่านิยามตรงกับ pool)"""
    lookup = build_flare_lookup(flares, level)
    out = meta[["HARPNUM", "issue_time", "split", "label"]].copy()
    out["issue_time"] = pd.to_datetime(out["issue_time"])
    out["persistence"] = 0
    out["label_rebuilt"] = 0
    for harpnum, idx in out.groupby("HARPNUM").groups.items():
        times = out.loc[idx, "issue_time"].to_numpy(dtype="datetime64[ns]")
        flare_times = lookup.get(int(harpnum))
        out.loc[idx, "persistence"] = persistence_times(times, flare_times, lookback)
        out.loc[idx, "label_rebuilt"] = label_times(times, flare_times, horizon)
    return out


def persistence_rule(meta: pd.DataFrame, flares: pd.DataFrame, trigger: str, lookback: int) -> np.ndarray:
    """คำตอบ 0/1 ของกฎ "มี flare ≥ trigger จาก HARP เดียวกันใน lookback ชม. ที่ผ่านมา" ทุกแถวของ ``meta``"""
    lookup = build_flare_lookup(flares, trigger)
    times_all = pd.to_datetime(meta["issue_time"]).to_numpy(dtype="datetime64[ns]")
    pred = np.zeros(len(meta), dtype=bool)
    for harpnum, rows in meta.groupby("HARPNUM").indices.items():
        pred[rows] = persistence_times(times_all[rows], lookup.get(int(harpnum)), lookback).astype(bool)
    return pred


def tuned_persistence(meta: pd.DataFrame, flares: pd.DataFrame, level: str) -> dict:
    """ลองทุกกฎในตระกูล เลือกตัวที่ TSS บน validation สูงสุด (เสมอกัน = กฎที่ประกาศก่อน) แล้วรายงานบน test

    กฎที่ trigger ต่ำกว่าเกณฑ์ของ label ไม่ได้ (เช่น trigger X1.0 สำหรับ label ≥M1.0 ยังใช้ได้ แต่ไม่ลอง trigger ต่ำ
    กว่า C1.0) — ตระกูลเดียวกันใช้กับทุกระดับเพื่อไม่ให้ปรับตามผล
    """
    split = meta["split"].to_numpy()
    y = meta["label"].to_numpy(dtype=bool)
    rows = []
    for trigger in TRIGGERS:
        for lookback in LOOKBACKS:
            pred = persistence_rule(meta, flares, trigger, lookback)
            rows.append({"trigger": trigger, "lookback_hours": lookback,
                         "val": binary_report(y[split == "val"], pred[split == "val"]),
                         "test": binary_report(y[split == "test"], pred[split == "test"])})
    best = max(rows, key=lambda r: r["val"]["tss"])
    logger.info("%s tuned persistence: validation เลือก ≥%s ใน %d ชม. (val TSS %.3f) -> test TSS %.3f", level,
                best["trigger"], best["lookback_hours"], best["val"]["tss"], best["test"]["tss"])
    return {"chosen": best, "all_rules": rows}


def model_summary(runs_path: Path, architecture: str, variant: str, baseline_tss: float) -> dict | None:
    if not runs_path.exists():
        return None
    runs = pd.read_parquet(runs_path)
    test = runs[(runs["architecture"] == architecture) & (runs["variant"] == variant) & (runs["scope"] == "test")]
    if test.empty:
        return None
    summary = {m: {"mean": float(test[m].mean()), "sd": float(test[m].std(ddof=1))} for m in METRICS}
    summary["n_seeds"] = int(len(test))
    summary["seeds_above_baseline_tss"] = int((test["tss"] > baseline_tss).sum())
    summary["tss_by_seed"] = [float(v) for v in test.sort_values("seed")["tss"]]
    return summary


def matched_fpr(predictions_path: Path, architecture: str, variant: str, fpr_target: float,
                recall_baseline: float) -> dict | None:
    """recall/precision ของแบบจำลองแต่ละ seed บน test ณ threshold ที่ FPR ไม่เกิน ``fpr_target`` (จุดที่ดีที่สุดบน ROC)"""
    if not predictions_path.exists():
        return None
    pred = pd.read_parquet(predictions_path, columns=["architecture", "variant", "seed", "split", "label", "prob"],
                           filters=[("architecture", "==", architecture), ("variant", "==", variant),
                                    ("split", "==", "test")])
    if pred.empty:
        return None
    recalls, precisions = [], []
    for _, g in pred.groupby("seed"):
        y = g["label"].to_numpy(dtype=bool)
        prob = g["prob"].to_numpy()
        order = np.argsort(-prob, kind="stable")
        y_sorted, p_sorted = y[order], prob[order]
        tp, fp = np.cumsum(y_sorted), np.cumsum(~y_sorted)
        # ตัดได้เฉพาะระหว่างค่าที่ต่างกัน — ค่าเท่ากันต้องเตือนพร้อมกันทั้งกลุ่ม
        cut = np.r_[p_sorted[1:] != p_sorted[:-1], True]
        tp, fp = tp[cut], fp[cut]
        ok = fp / (~y).sum() <= fpr_target
        best = int(np.argmax(np.where(ok, tp, -1))) if ok.any() else None
        tp_best = int(tp[best]) if best is not None else 0
        fp_best = int(fp[best]) if best is not None else 0
        recalls.append(tp_best / y.sum())
        precisions.append(tp_best / (tp_best + fp_best) if tp_best + fp_best else 0.0)
    recalls, precisions = np.array(recalls), np.array(precisions)
    return {"fpr_target": fpr_target, "n_seeds": int(len(recalls)),
            "recall": {"mean": float(recalls.mean()), "sd": float(recalls.std(ddof=1))},
            "precision": {"mean": float(precisions.mean()), "sd": float(precisions.std(ddof=1))},
            "seeds_above_baseline_recall": int((recalls > recall_baseline).sum()),
            "seeds_equal_baseline_recall": int(np.isclose(recalls, recall_baseline).sum())}


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "persistence_baseline.log")
    architecture, variant = args.cell.split(":")
    flares = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    horizon = cfg.flare.horizon_hours

    results = {"lookback_hours": args.lookback_hours, "horizon_hours": horizon, "cell": args.cell, "levels": {}}
    for level, pool, study_dir in LEVELS:
        meta_path = cfg.paths.processed / pool / "meta.parquet"
        if not meta_path.exists():
            logger.warning("ไม่พบ %s — ข้ามระดับ %s", meta_path, level)
            continue
        frame = persistence_for_pool(pd.read_parquet(meta_path), flares, level, horizon, args.lookback_hours)
        mismatch = int((frame["label"] != frame["label_rebuilt"]).sum())
        if mismatch:
            raise SystemExit(f"{level}: label ที่สร้างซ้ำจาก catalog ไม่ตรงกับ pool {mismatch} แถว — นิยามไม่ตรงกัน เทียบไม่ได้")
        logger.info("%s: label สร้างซ้ำตรงกับ pool ครบ %d แถว", level, len(frame))

        per_split = {
            split: binary_report(g["label"].to_numpy(dtype=bool), g["persistence"].to_numpy(dtype=bool))
            for split, g in frame.groupby("split")
        }
        model = model_summary(cfg.paths.artifacts / study_dir / "runs.parquet", architecture, variant,
                              per_split["test"]["tss"])
        test_p = per_split["test"]
        fpr_p = test_p["fp"] / (test_p["n"] - test_p["n_pos"])
        matched = matched_fpr(cfg.paths.artifacts / study_dir / "predictions.parquet", architecture, variant,
                              fpr_p, test_p["recall"])
        tuned = tuned_persistence(frame, flares, level)
        results["levels"][level] = {"persistence": per_split, "model_test": model, "matched_fpr": matched,
                                    "tuned_persistence": tuned}
        test = per_split["test"]
        logger.info("%s persistence test: TSS %.3f · HSS2 %.3f · recall %.3f · precision %.3f (TP %d FP %d FN %d)",
                    level, test["tss"], test["hss2"], test["recall"], test["precision"], test["tp"], test["fp"],
                    test["fn"])
        if model:
            logger.info("%s %s test: TSS %.3f ± %.3f · seed ที่ TSS สูงกว่า baseline %d/%d", level, args.cell,
                        model["tss"]["mean"], model["tss"]["sd"], model["seeds_above_baseline_tss"], model["n_seeds"])
        if matched:
            logger.info("%s ที่ FPR ≤ %.4f (เท่า persistence): recall %.3f ± %.3f vs %.3f · precision %.3f ± %.3f vs %.3f"
                        " · seed ที่ recall สูงกว่า %d/%d", level, matched["fpr_target"], matched["recall"]["mean"],
                        matched["recall"]["sd"], test["recall"], matched["precision"]["mean"],
                        matched["precision"]["sd"], test["precision"], matched["seeds_above_baseline_recall"],
                        matched["n_seeds"])

    out_dir = cfg.paths.artifacts / "model_comparison"
    (out_dir / "persistence_baseline.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    (out_dir / "persistence_baseline.md").write_text(render_markdown(results), encoding="utf-8")
    logger.info("เขียน %s", out_dir / "persistence_baseline.md")
    return 0


def _fmt(model: dict, metric: str, digits: int = 3) -> str:
    return f"{model[metric]['mean']:.{digits}f} ± {model[metric]['sd']:.{digits}f}"


def render_markdown(results: dict) -> str:
    cell = results["cell"]
    lines = [
        f"# baseline แบบ persistence เทียบกับ {cell}",
        "",
        f"persistence = ทายว่าจะเกิด flare ≥ ระดับนั้นใน {results['horizon_hours']} ชม. ข้างหน้า ถ้าบริเวณเดียวกันมี flare "
        f"≥ ระดับนั้นใน {results['lookback_hours']} ชม. ที่ผ่านมา · แถวข้อมูล split และ flare catalog ชุดเดียวกับแบบจำลอง "
        "(label ที่สร้างซ้ำตรงกับ pool ทุกแถว) · ผลบน **test** · แบบจำลองคือ mean ± SD ข้าม seed",
        "",
        "| ระดับ | วิธี | TSS | HSS2 | recall | precision | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, body in results["levels"].items():
        p = body["persistence"]["test"]
        lines.append(f"| ≥{level} | persistence | {p['tss']:.3f} | {p['hss2']:.3f} | {p['recall']:.3f} | "
                     f"{p['precision']:.3f} | {p['tp']} | {p['fp']} | {p['fn']} |")
        m = body["model_test"]
        if m:
            lines.append(f"| ≥{level} | {cell} ({m['n_seeds']} seed) | {_fmt(m, 'tss')} | {_fmt(m, 'hss2')} | "
                         f"{_fmt(m, 'recall')} | {_fmt(m, 'precision')} | {m['tp']['mean']:.1f} | "
                         f"{m['fp']['mean']:.1f} | {m['fn']['mean']:.1f} |")
    lines += ["", "seed ที่ TSS สูงกว่า persistence: " + " · ".join(
        f"≥{level} {b['model_test']['seeds_above_baseline_tss']}/{b['model_test']['n_seeds']}"
        for level, b in results["levels"].items() if b["model_test"]), ""]

    lines += [
        "## persistence ที่ปรับได้ (เลือกกฎบน validation)",
        "",
        "กฎ = มี flare ≥ trigger จากบริเวณเดียวกันใน lookback ชม. ที่ผ่านมา · ตระกูลกฎ trigger "
        f"{', '.join(TRIGGERS)} × lookback {', '.join(map(str, LOOKBACKS))} ชม. · เลือกด้วย TSS สูงสุดบน "
        "validation (กติกาเดียวกับที่แบบจำลองเลือก threshold) แล้ววัดบน test · **ตระกูลกฎกำหนดหลังเห็นผลบน test "
        "ของบางกฎแล้ว ไม่ได้ประกาศล่วงหน้า**",
        "",
        "| ระดับ | กฎที่ validation เลือก | TSS val | TSS test | HSS2 test | recall test | precision test | "
        "แบบจำลอง TSS test | seed ที่ TSS สูงกว่ากฎนี้ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, body in results["levels"].items():
        t, m = body.get("tuned_persistence"), body["model_test"]
        if not t:
            continue
        c = t["chosen"]
        above = "—"
        if m and "tss_by_seed" in m:
            above = f"{sum(v > c['test']['tss'] for v in m['tss_by_seed'])}/{m['n_seeds']}"
        lines.append(f"| ≥{level} | ≥{c['trigger']} ใน {c['lookback_hours']} ชม. | {c['val']['tss']:.3f} | "
                     f"{c['test']['tss']:.3f} | {c['test']['hss2']:.3f} | {c['test']['recall']:.3f} | "
                     f"{c['test']['precision']:.3f} | {_fmt(m, 'tss') if m else 'n/a'} | {above} |")
    lines += [""]

    lines += [
        "## เทียบที่อัตราเตือนผิดเท่ากัน",
        "",
        "threshold ของแบบจำลองที่ใช้ในตารางบนเลือกให้ TSS สูงสุด จึงเตือนบ่อยกว่า persistence มาก — TSS จึงเข้าข้างแบบจำลอง "
        "ส่วน HSS2/precision เข้าข้าง persistence ตารางนี้เลื่อน threshold ของแต่ละ seed ไปที่ FPR ไม่เกินของ persistence "
        "แล้ววัดว่าใครจับได้มากกว่า (**เป็นการวิเคราะห์บน test ไม่ใช่ threshold ที่ใช้งานจริง**)",
        "",
        "| ระดับ | FPR ของ persistence | recall: persistence | recall: แบบจำลอง | precision: persistence | "
        "precision: แบบจำลอง | seed ที่ recall สูงกว่า |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level, body in results["levels"].items():
        m, p = body.get("matched_fpr"), body["persistence"]["test"]
        if not m:
            continue
        lines.append(f"| ≥{level} | {m['fpr_target']:.4f} | {p['recall']:.3f} | {_fmt(m, 'recall')} | "
                     f"{p['precision']:.3f} | {_fmt(m, 'precision')} | {m['seeds_above_baseline_recall']}/{m['n_seeds']} "
                     f"(เท่ากัน {m['seeds_equal_baseline_recall']}) |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

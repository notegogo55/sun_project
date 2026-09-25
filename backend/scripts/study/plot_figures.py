"""สร้างรูปประกอบรายงานจากผลของงานเปรียบเทียบโมเดล (สถาปัตยกรรม x ชุด feature) (ticket 03 ของ
``.scratch/model-comparison/``)

    # หลัง study/train.py เสร็จ
    python backend/scripts/study/plot_figures.py

    # เลือกเฉพาะบางรูป / เขียนไปที่อื่น (ใช้ตอนลองกับข้อมูลสังเคราะห์)
    python backend/scripts/study/plot_figures.py --figures 1,5 --out-dir <tmp>

อ่าน ``runs.parquet`` + ``predictions.parquet`` ที่ ``study/train.py`` เขียนไว้ แล้วเขียน
PNG ลง ``figures/`` กับ **ตารางตัวเลขของทุกรูป** ลง ``figures/data/*.csv``

ตารางคู่รูปไม่ใช่ของแถม: รูปเป็น PNG ในรายงานกระดาษ ผู้อ่านที่แยกสีไม่ออกหรือได้ไฟล์
ขาวดำต้องอ่านตัวเลขชุดเดียวกันได้จากที่อื่น และตอนเขียนรายงานก็ต้องคัดตัวเลขไปใส่ในเนื้อหา
อยู่ดี

**รูปทั้งเจ็ด** (แต่ละรูปตอบคำถามคนละข้อ — ไม่มีรูปไหนซ้ำงานของรูปอื่น) · เลขรูปคงไว้ตามเดิมเพื่อไม่ให้
``--figures`` ที่อ้างถึงในรายงานเปลี่ยน รูปที่ 7 จึงเป็นผลหลักทั้งที่เลขมาท้ายสุด:

7. ``ranking``        ทุกเซลล์เรียงตาม TSS เฉลี่ยบน test — **คำตอบหลักของงาน "โมเดลไหนดีที่สุด"**
                      (คำถามหลักตั้งแต่ 2026-09-24)
1. ``delta_tss``      Δ ตามแบบ ของทุกสถาปัตยกรรมในรูปเดียว — ชุด feature เปลี่ยนผลของแต่ละสถาปัตยกรรมแค่ไหน
2. ``interaction``    Δ ตามแบบ ของสถาปัตยกรรมหนึ่ง ลบด้วยของสถาปัตยกรรมอ้างอิง — ชุด feature ให้ผลต่างกันตาม
                      สถาปัตยกรรมไหม (เดิมเป็นคำตอบหลัก ก่อนเปลี่ยนคำถามเป็นการหาโมเดลที่ดีที่สุด)
3. ``tss_by_seed``    TSS ดิบรายเซลล์ราย seed — ผลต่างระหว่างเซลล์ใหญ่กว่าความผันผวนของ seed ไหม
4. ``roc``            ROC แผงละแบบ เส้นละสถาปัตยกรรม — ภาพที่ไม่ขึ้นกับ threshold ที่เลือก
5. ``confusion``      confusion matrix ของทุกสถาปัตยกรรม ที่แบบแรกกับแบบสุดท้ายของตาราง (V0, V3) —
                      ผลต่างมาจาก TP ที่เพิ่มหรือ FP ที่ลด
6. ``ar_timeline``    ความน่าจะเป็นรายชั่วโมงของทุกสถาปัตยกรรม (แบบ V0) บน HARP ที่มี positive มากที่สุด
                      ใน test เทียบกับ threshold ของแต่ละเซลล์

รูปที่ 5 และ 6 แสดงทุกสถาปัตยกรรมเท่ากัน ไม่เลือกเฉพาะ anchor กับเซลล์ที่ชนะ — ผู้อ่านต้องเห็น
ผลจริงของทุกตัวโดยไม่ผ่านการคัดเลือก ส่วนตัวเลขของครบทั้ง 16 เซลล์อยู่ใน ``confusion.csv``

**การวาด** ใช้ของสำเร็จรูปของ library ไม่วาดเอง: seaborn (``barplot``/``heatmap``/``stripplot``/
``lineplot`` — แท่ง error bar และแถบ ±SD ข้าม seed คำนวณโดย seaborn จากค่าราย seed) และ scikit-learn
(``roc_curve`` + ``RocCurveDisplay``, ``ConfusionMatrixDisplay``) · ธีมและจานสีอยู่ที่ ``sunseg.report_style``
· ข้อความในรูปเป็นภาษาอังกฤษ
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# report_style ต้องมาก่อน pyplot — ตั้ง backend Agg + ธีม seaborn (ดู docstring ของโมดูลนั้น)
from sunseg.report_style import (  # noqa: E402, I001
    DIVERGING,
    GREY,
    PALETTE,
    SEQUENTIAL,
    THRESHOLD,
    save_figure,
    true_runs,
)

import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve  # noqa: E402

from sunseg.config import STUDY_VARIANTS_FILE, load_data_config, load_study_architectures  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.models.forecast import architecture_label  # noqa: E402
from sunseg.study.summary import delta_by_variant, interaction_summary, rank_cells  # noqa: E402

logger = logging.getLogger("plot_study_figures")

STUDY_DIR_NAME = "model_comparison"
VARIANTS_FILE = STUDY_VARIANTS_FILE

#: anchor ของตาราง — ต้องตรงกับ study/train.py (ดู CONTEXT.md หัวข้อ anchor)
ANCHOR_ARCHITECTURE = "lstm"
ANCHOR_VARIANT = "V0"

SCOPE_TITLES = {
    # ไม่ฝังปีไว้ในชื่อ — ช่วงของ test ขึ้นกับ split ใน data.yaml (ดูเหตุผลใน study/train.py)
    "test": "test",
    "val": "validation",
}
REPORT_SCOPES = ("test",)


# --------------------------------------------------------------------------- #
# การโหลดข้อมูล
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StudyLabels:
    """ชื่อและลำดับของทั้งสองแกน อ่านจาก config ไม่ประกาศซ้ำในไฟล์นี้

    ``matrix_*`` คือเซลล์ของตารางหลัก — แถวเสริม (V1/V4/V5 และสถาปัตยกรรมที่ตั้ง
    ``supplementary: true``) ถูกกันออกจากทุกรูป เพราะรูปของตารางหลักต้องเป็นการเทียบ
    ที่ควบคุมตัวแปรแล้ว
    """

    variants: dict[str, str]
    architectures: dict[str, str]
    matrix_architectures: list[str]
    matrix_variants: list[str]

    def cells(self) -> list[tuple[str, str]]:
        return [(a, v) for a in self.matrix_architectures for v in self.matrix_variants]

    def rivals(self) -> list[str]:
        return [a for a in self.matrix_architectures if a != ANCHOR_ARCHITECTURE]

    def compared(self) -> list[str]:
        return [v for v in self.matrix_variants if v != ANCHOR_VARIANT]

    def arch(self, name: str) -> str:
        return self.architectures.get(name, name)

    def variant(self, name: str) -> str:
        """``V2\\n18 SHARP + X-ray`` — ป้ายแกนของชุด feature"""
        return f"{name}\n{self.variants.get(name, name)}"


def load_labels() -> StudyLabels:
    spec = yaml.safe_load(VARIANTS_FILE.read_text(encoding="utf-8"))
    study = load_study_architectures()
    return StudyLabels(
        variants={name: body["label"] for name, body in spec["variants"].items()},
        architectures={
            name: entry.label or architecture_label(entry.model.kind)
            for name, entry in study.architectures.items()
        },
        matrix_architectures=[n for n, e in study.architectures.items() if not e.supplementary],
        matrix_variants=list(study.default_variants),
    )


def load_inputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, StudyLabels]:
    runs_path, pred_path = out_dir / "runs.parquet", out_dir / "predictions.parquet"
    for path in (runs_path, pred_path):
        if not path.exists():
            raise SystemExit(f"ไม่พบ {path} — รัน backend/scripts/study/train.py ให้เสร็จก่อน")
    runs = pd.read_parquet(runs_path)
    predictions = pd.read_parquet(pred_path)
    if "architecture" not in runs.columns:
        raise SystemExit(
            f"{runs_path} ไม่มีคอลัมน์ architecture — เป็นผลของงานเปรียบเทียบชุด feature รุ่นเก่า "
            "ให้ชี้ --out-dir ไปที่ผลของ model_comparison"
        )
    labels = load_labels()
    # เก็บเฉพาะเซลล์ที่ config ปัจจุบันประกาศไว้ และที่มีผลจริง
    present = {(a, v) for a, v in zip(runs["architecture"], runs["variant"], strict=True)}
    labels = StudyLabels(
        variants=labels.variants,
        architectures=labels.architectures,
        matrix_architectures=[a for a in labels.matrix_architectures
                              if any((a, v) in present for v in labels.matrix_variants)],
        matrix_variants=[v for v in labels.matrix_variants
                         if any((a, v) in present for a in labels.matrix_architectures)],
    )
    logger.info(
        "อ่านผล %d run · %d แถวทำนาย · ตารางหลัก %d สถาปัตยกรรม × %d แบบ",
        len(runs[["architecture", "variant", "seed"]].drop_duplicates()), len(predictions),
        len(labels.matrix_architectures), len(labels.matrix_variants),
    )
    return runs, predictions, labels


def matrix_only(frame: pd.DataFrame, labels: StudyLabels) -> pd.DataFrame:
    """ตัดแถวเสริมออก — การจับคู่ของตารางหลักต้องการเซลล์ครบทุกช่อง แถวเสริมมีไม่ครบ"""
    return frame[
        frame["architecture"].isin(labels.matrix_architectures)
        & frame["variant"].isin(labels.matrix_variants)
    ]


def arch_palette(labels: StudyLabels) -> dict[str, tuple]:
    """สีตามตัวตนของสถาปัตยกรรม — ลำดับใน config ไม่ใช่ลำดับผล (สีไม่เปลี่ยนเมื่อกรองเซลล์)"""
    return {labels.arch(a): PALETTE[i % len(PALETTE)] for i, a in enumerate(labels.matrix_architectures)}


def save(fig: plt.Figure, out_dir: Path, name: str, table: pd.DataFrame, dpi: int) -> None:
    fig_dir, data_dir = out_dir / "figures", out_dir / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    save_figure(fig, png, dpi)
    csv = data_dir / f"{name}.csv"
    table.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info("เขียน %s  (ตาราง: %s)", png, csv.name)


# --------------------------------------------------------------------------- #
# รูปที่ 1 — Δ ตามแบบ ของทุกสถาปัตยกรรม
# --------------------------------------------------------------------------- #
def per_seed_delta(runs: pd.DataFrame, labels: StudyLabels, metric: str = "tss") -> pd.DataFrame:
    """Δ ราย seed เทียบแบบอ้างอิงของสถาปัตยกรรมเดียวกัน (จับคู่ seed) — ข้อมูลดิบให้ seaborn รวมเอง

    ค่าเฉลี่ย/SD (ddof=1) ที่ ``sns.barplot(errorbar="sd")`` คำนวณจากตารางนี้เท่ากับ
    ``delta_by_variant`` ที่รายงานใช้
    """
    sub = matrix_only(runs[runs["scope"] == "test"], labels)
    wide = sub.pivot(index=["architecture", "seed"], columns="variant", values=metric).astype(float)
    delta = wide.sub(wide[ANCHOR_VARIANT], axis=0).drop(columns=ANCHOR_VARIANT)
    return delta.reset_index().melt(id_vars=["architecture", "seed"], var_name="variant", value_name="delta")


def figure_delta_tss(runs: pd.DataFrame, labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """Δ ตามแบบ ของทุกสถาปัตยกรรมในรูปเดียว — กลุ่มละแบบ แท่งละสถาปัตยกรรม

    สีเข้ารหัส **สถาปัตยกรรม** ไม่ใช่เครื่องหมายของ Δ: คำถามคือ "สถาปัตยกรรมไหนได้ประโยชน์
    จาก feature นี้" ส่วนเครื่องหมายอ่านได้จากตำแหน่งเทียบเส้นศูนย์อยู่แล้ว
    แบบอ้างอิงไม่อยู่ในรูป เพราะ Δ ของมันเป็นศูนย์ตามนิยาม ไม่ใช่ผลการวัด
    """
    summary = delta_by_variant(matrix_only(runs, labels), anchor_variant=ANCHOR_VARIANT, metric="tss")
    summary = summary[summary["scope"] == "test"].set_index(["architecture", "variant"])
    archs, groups = labels.matrix_architectures, labels.compared()
    if not groups:
        raise ValueError("ไม่มีแบบให้เทียบนอกจากแบบอ้างอิง — รูปนี้ต้องการอย่างน้อยสองแบบ")

    rows = [
        {"scope": "test", "architecture": a, "architecture_label": labels.arch(a),
         "variant": v, "variant_label": labels.variants.get(v, v),
         "delta_tss_mean": float(summary.loc[(a, v), "delta_mean"]) if (a, v) in summary.index else float("nan"),
         "delta_tss_sd": float(summary.loc[(a, v), "delta_sd"]) if (a, v) in summary.index else float("nan")}
        for a in archs for v in groups
    ]

    data = per_seed_delta(runs, labels)
    data["Architecture"] = data["architecture"].map(labels.arch)
    data["Feature set"] = data["variant"].map(labels.variant)
    n_seeds = int(data.groupby(["architecture", "variant"]).size().max())

    fig, ax = plt.subplots(figsize=(1.7 * len(groups) + 3.2, 4.6))
    sns.barplot(
        data=data, x="Feature set", y="delta", hue="Architecture",
        order=[labels.variant(v) for v in groups], hue_order=[labels.arch(a) for a in archs],
        palette=arch_palette(labels), errorbar="sd", capsize=0.15, err_kws={"linewidth": 1.1},
        ax=ax,
    )
    ax.axhline(0, color="0.25", linewidth=1.0)
    ax.set(xlabel=None, ylabel=f"ΔTSS vs {ANCHOR_VARIANT} (same architecture)")
    ax.set_title(f"Effect of added features on TSS (mean ± 1 SD over {n_seeds} seeds)", loc="left")
    sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(archs), title=None, frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 2 — interaction
# --------------------------------------------------------------------------- #
def figure_interaction(runs: pd.DataFrame, labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """Δ ตามแบบ ของสถาปัตยกรรมหนึ่ง ลบด้วย Δ ตามแบบ ของสถาปัตยกรรมอ้างอิง

    ใกล้ศูนย์ทุกช่อง = ชุด feature ให้ผลเท่ากันทุกสถาปัตยกรรม · เป็นบวกอย่างมีนัย = ตัวนั้นใช้ชุด feature นั้น
    ได้ดีกว่าตัวอ้างอิง · ใช้ colormap แบบ diverging (center=0) เพราะศูนย์มีความหมายจริง
    """
    table = interaction_summary(
        matrix_only(runs, labels),
        anchor_architecture=ANCHOR_ARCHITECTURE, anchor_variant=ANCHOR_VARIANT, metric="tss",
    )
    table = table[table["scope"] == "test"].set_index(["architecture", "variant"])
    archs, groups = labels.rivals(), labels.compared()
    if not archs or not groups:
        raise ValueError("ต้องมีอย่างน้อยหนึ่งสถาปัตยกรรมคู่แข่งและหนึ่งแบบที่ไม่ใช่แบบอ้างอิง")

    grid = np.full((len(archs), len(groups)), np.nan)
    sd_grid = np.full_like(grid, np.nan)
    rows: list[dict] = []
    for i, architecture in enumerate(archs):
        for j, variant in enumerate(groups):
            key = (architecture, variant)
            if key not in table.index:
                continue
            grid[i, j] = float(table.loc[key, "interaction_mean"])
            sd_grid[i, j] = float(table.loc[key, "interaction_sd"])
            rows.append({
                "scope": "test", "architecture": architecture, "architecture_label": labels.arch(architecture),
                "variant": variant, "variant_label": labels.variants.get(variant, variant),
                "interaction_mean": grid[i, j], "interaction_sd": sd_grid[i, j],
                "delta_tss_mean": float(table.loc[key, "delta_mean"]),
                "delta_tss_sd": float(table.loc[key, "delta_sd"]),
            })

    # **ลงสีเฉพาะช่องที่ค่าใหญ่กว่า SD ของตัวเอง** — กฎเดียวกับตัวหนาในตารางของรายงาน
    # ถ้าปล่อยให้สเกลสีมาจากค่าสูงสุดในตาราง ช่องที่บังเอิญมีค่ามากที่สุดจะได้สีเข้มสุดเสมอ
    # แม้ SD ของมันจะใหญ่กว่าตัวมันเอง · ตัวเลขในช่องยังเป็นค่าจริงเสมอ
    significant = np.abs(grid) > sd_grid
    shaded = np.where(significant, grid, 0.0)
    span = max(float(np.nanmax(np.abs(shaded))) if shaded.any() else 0.05, 0.01)
    annot = np.array([
        ["n/a" if not np.isfinite(grid[i, j]) else f"{grid[i, j]:+.3f}\n± {sd_grid[i, j]:.3f}"
         for j in range(len(groups))]
        for i in range(len(archs))
    ])

    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 3.4, 0.9 * len(archs) + 2.4))
    sns.heatmap(
        pd.DataFrame(shaded, index=[labels.arch(a) for a in archs], columns=[labels.variant(v) for v in groups]),
        cmap=DIVERGING, center=0, vmin=-span, vmax=span, annot=annot, fmt="",
        linewidths=1, linecolor="0.85", square=False, ax=ax,
        cbar_kws={"label": "Interaction ΔTSS (colored only if |mean| > SD)"},
    )
    ax.set(xlabel=None, ylabel=None)
    ax.tick_params(axis="y", rotation=0)
    anchor = labels.arch(ANCHOR_ARCHITECTURE)
    ax.set_title(f"Interaction: ΔTSS of each architecture minus ΔTSS of {anchor}\n"
                 "(seed-paired, mean ± SD)", loc="left")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 3 — TSS ราย seed
# --------------------------------------------------------------------------- #
def figure_tss_by_seed(runs: pd.DataFrame, labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """จุดหนึ่งจุดต่อหนึ่ง seed ของทุกเซลล์ — ให้เห็นว่าผลต่างในรูปที่ 1 โผล่พ้นความผันผวนของ seed หรือไม่"""
    sub = matrix_only(runs[runs["scope"] == "test"], labels)
    cells = [c for c in labels.cells() if not sub[(sub["architecture"] == c[0]) & (sub["variant"] == c[1])].empty]
    if not cells:
        raise ValueError("ไม่มีเซลล์ของตารางหลักที่มีผลบน test")

    data = sub[["architecture", "variant", "seed", "tss"]].copy()
    data["Architecture"] = data["architecture"].map(labels.arch)
    data["Cell"] = data["Architecture"] + " · " + data["variant"]
    order = [f"{labels.arch(a)} · {v}" for a, v in cells]

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(cells) + 1.6))
    sns.stripplot(data=data, x="tss", y="Cell", hue="Architecture", order=order,
                  hue_order=[labels.arch(a) for a in labels.matrix_architectures], palette=arch_palette(labels),
                  size=5, alpha=0.7, jitter=0.18, legend=False, ax=ax)
    sns.pointplot(data=data, x="tss", y="Cell", order=order, estimator="mean", errorbar=None,
                  linestyle="none", marker="|", markersize=16, markeredgewidth=2.2, color="0.1", ax=ax)
    ax.set(xlabel="TSS (test)", ylabel=None)
    ax.set_title("TSS of every run (dots = seeds, black tick = mean)", loc="left")
    fig.tight_layout()

    table = data[["architecture", "variant", "seed", "tss"]].assign(scope="test")
    table.insert(1, "architecture_label", table["architecture"].map(labels.arch))
    return fig, table[["scope", "architecture", "architecture_label", "variant", "seed", "tss"]]


# --------------------------------------------------------------------------- #
# รูปที่ 7 — อันดับของทุกเซลล์ (ผลหลัก)
# --------------------------------------------------------------------------- #
def figure_ranking(runs: pd.DataFrame, labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """ทุกเซลล์เรียงตาม TSS เฉลี่ยบน test — **คำตอบหลักของงาน** "โมเดลไหนดีที่สุด"

    จุด = ค่าเฉลี่ย · เส้น = ±1 SD ข้าม seed · สีคือสถาปัตยกรรม (สีเดียวกับทุกรูป) · จุดกลวง = เซลล์ที่
    ต่ำกว่าอันดับ 1 เกิน SD ของผลต่างจับคู่ (``rank_cells``) ส่วนจุดทึบคือเซลล์ที่ยังแยกจากอันดับ 1 ไม่ได้
    — การเข้ารหัสนี้ไม่พึ่งสีอย่างเดียว เส้นประแนวตั้งคือค่าเฉลี่ยของอันดับ 1
    """
    table = rank_cells(matrix_only(runs, labels), scope="test", metric="tss")
    if table.empty:
        raise ValueError("ไม่มีเซลล์ของตารางหลักที่มีผลบน test")
    table.insert(2, "architecture_label", table["architecture"].map(labels.arch))
    table.insert(4, "variant_label", table["variant"].map(lambda v: labels.variants.get(v, v)))
    table.insert(0, "scope", "test")

    palette = arch_palette(labels)
    names = [f"{r.rank}. {r.architecture_label} · {r.variant} ({r.variant_label})" for r in table.itertuples()]
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(table) + 1.9))
    for y, row in enumerate(table.itertuples()):
        color = palette[row.architecture_label]
        ax.errorbar(row.mean, y, xerr=row.sd, fmt="none", ecolor=color, elinewidth=2, capsize=0)
        ax.plot(row.mean, y, "o", markersize=8, markeredgewidth=2, color=color,
                markerfacecolor=color if row.tied_with_best else "white")
    ax.axvline(table.loc[0, "mean"], color=GREY, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(table)), names)
    ax.invert_yaxis()
    ax.set(xlabel="TSS (test, mean ± 1 SD over seeds)", ylabel=None)
    n_seeds = int(table["n_seeds"].max())
    ax.set_title(f"Ranking of all {len(table)} architecture × feature-set cells ({n_seeds} seeds each)", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=8, color=palette[labels.arch(a)],
                      label=labels.arch(a)) for a in labels.matrix_architectures]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=8, color="0.3", label="not separable from #1"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color="0.3", markerfacecolor="white",
               markeredgewidth=2, label="below #1 by more than paired SD"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
    fig.tight_layout()
    return fig, table


# --------------------------------------------------------------------------- #
# รูปที่ 4 — ROC แผงละแบบ
# --------------------------------------------------------------------------- #
def _mean_roc(frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray | None:
    """TPR เฉลี่ยข้าม seed บนกริด FPR ร่วม (``sklearn.metrics.roc_curve`` ราย seed แล้ว interpolate)

    เฉลี่ยเส้นราย seed ไม่ใช่รวม prob ของทุก seed เข้าด้วยกัน — การรวมจะได้เส้นของ
    ensemble ซึ่งเป็นคนละโมเดลกับที่ตาราง TSS รายงาน
    """
    curves = []
    for _, group in frame.groupby("seed", sort=True):
        y = group["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, group["prob"].to_numpy(dtype=float))
        curves.append(np.interp(grid, fpr, tpr))
    return np.mean(curves, axis=0) if curves else None


def figure_roc(runs: pd.DataFrame, predictions: pd.DataFrame,
               labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """หนึ่งแผงต่อหนึ่งแบบ เส้นละสถาปัตยกรรม — ตอบ "สถาปัตยกรรมไหนดีกว่าเมื่อได้ feature ชุดเดียวกัน"
    ภายในแผงเดียว · AUC ในป้ายคือค่าเฉลี่ยราย seed จาก ``runs`` (ค่าเดียวกับตารางในรายงาน)
    """
    grid = np.linspace(0, 1, 201)
    sub = predictions[predictions["split"] == "test"]
    auc_means = (
        matrix_only(runs[runs["scope"] == "test"], labels)
        .groupby(["architecture", "variant"])["auc"].mean()
    )
    colours = arch_palette(labels)
    panels = labels.matrix_variants
    n_cols = min(2, len(panels))
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.4 * n_rows + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_axis_off()
    rows: list[dict] = []

    for ax, variant in zip(axes.flat, panels, strict=False):
        for i, architecture in enumerate(labels.matrix_architectures):
            frame = sub[(sub["architecture"] == architecture) & (sub["variant"] == variant)]
            curve = _mean_roc(frame, grid)
            if curve is None:
                continue
            auc = float(auc_means.get((architecture, variant), float("nan")))
            # AUC ใส่ในชื่อเอง 3 ตำแหน่ง — ป้ายปริยายของ RocCurveDisplay ปัดเหลือ 2 ตำแหน่ง ซึ่งเท่ากันทุกเซลล์
            RocCurveDisplay(fpr=grid, tpr=curve, name=f"{labels.arch(architecture)} (AUC = {auc:.3f})").plot(
                ax=ax, curve_kwargs={"color": colours[labels.arch(architecture)], "linewidth": 1.8},
                plot_chance_level=i == len(labels.matrix_architectures) - 1,
                chance_level_kw={"color": GREY, "linestyle": "--", "linewidth": 1},
            )
            rows += [
                {"scope": "test", "architecture": architecture, "variant": variant,
                 "auc_mean": auc, "fpr": float(f), "tpr": float(t)}
                for f, t in zip(grid, curve, strict=True)
            ]
        ax.set_title(f"{variant}: {labels.variants.get(variant, variant)}", loc="left")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
        ax.label_outer()
    fig.suptitle("ROC curves on the test set (mean TPR over seeds)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 5 — confusion matrix ของทุกสถาปัตยกรรม
# --------------------------------------------------------------------------- #
def panel_variants(labels: StudyLabels) -> list[str]:
    """แบบที่รูปที่ 5 วาด: แบบแรกของตาราง (ไม่มี feature เสริม) กับแบบสุดท้าย (มีครบทุกกลุ่ม)

    16 แผงบนหน้ากระดาษเดียวอ่านไม่ออก จึงวาดแค่สองแบบที่อยู่ปลายสุดของแกนชุด feature แต่วาด
    **ทุกสถาปัตยกรรม** — ตัวเลขของครบทุกเซลล์อยู่ในตารางคู่รูปอยู่แล้ว
    """
    first, last = labels.matrix_variants[0], labels.matrix_variants[-1]
    return [first] if first == last else [first, last]


def mean_confusion(runs: pd.DataFrame, labels: StudyLabels, scope: str = "test") -> pd.DataFrame:
    """ค่าเฉลี่ยข้าม seed ของ TP/FN/FP/TN และ TSS — ทุกเซลล์ของตารางหลัก เรียงตามลำดับใน config"""
    sub = matrix_only(runs[runs["scope"] == scope], labels)
    table = sub.groupby(["architecture", "variant"])[["tp", "fn", "fp", "tn", "tss"]].mean().reset_index()
    table["scope"] = scope
    table["a_order"] = table["architecture"].map(labels.matrix_architectures.index)
    table["v_order"] = table["variant"].map(labels.matrix_variants.index)
    table = table.sort_values(["a_order", "v_order"]).drop(columns=["a_order", "v_order"])
    return table[["scope", "architecture", "variant", "tp", "fn", "fp", "tn", "tss"]].reset_index(drop=True)


def figure_confusion(runs: pd.DataFrame, labels: StudyLabels,
                     scope: str = "test") -> tuple[plt.Figure, pd.DataFrame]:
    """ผลต่างของ TSS มาจากการจับ positive ได้เพิ่ม หรือจากการเตือนผิดที่ลดลง — รูปนี้แยกให้เห็น

    แถว = แบบ (V0, V3) · คอลัมน์ = สถาปัตยกรรมทุกตัวของตารางหลัก · เรียงช่องแบบ scikit-learn
    (แถว = ค่าจริง, คอลัมน์ = ที่ทำนาย, ลำดับ [No flare, Flare]) · สี = สัดส่วนภายในแถว เพราะจำนวนดิบ
    ของ negative มากกว่า positive หลายสิบเท่า
    """
    table = mean_confusion(runs, labels, scope)
    variants, architectures = panel_variants(labels), labels.matrix_architectures
    fig, axes = plt.subplots(len(variants), len(architectures),
                             figsize=(2.7 * len(architectures) + 0.6, 3.1 * len(variants) + 0.5), squeeze=False,
                             layout="constrained")

    for r, variant in enumerate(variants):
        for c, architecture in enumerate(architectures):
            ax = axes[r, c]
            cell = table[(table["architecture"] == architecture) & (table["variant"] == variant)]
            if cell.empty:
                ax.set_axis_off()
                continue
            counts = cell.iloc[0]
            matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
            shares = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-9, None)
            display = ConfusionMatrixDisplay(confusion_matrix=shares, display_labels=["No flare", "M1+ flare"])
            display.plot(ax=ax, cmap=SEQUENTIAL, colorbar=False, im_kw={"vmin": 0, "vmax": 1})
            for (i, j), text in np.ndenumerate(display.text_):
                text.set_text(f"{matrix[i, j]:,.0f}\n({shares[i, j]:.0%})")
            ax.set_title(f"{labels.arch(architecture)} · {variant}\nTSS = {float(counts['tss']):.3f}")
            ax.grid(False)
            if c:
                ax.set(ylabel=None, yticklabels=[])
            if r < len(variants) - 1:
                ax.set(xlabel=None, xticklabels=[])
    fig.suptitle(f"Confusion matrices on the {SCOPE_TITLES[scope]} set (mean counts over seeds; "
                 "color = share within each true class)", x=0.02, ha="left", fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.12)
    return fig, table


# --------------------------------------------------------------------------- #
# รูปที่ 6 — ไทม์ไลน์ของ HARP ที่มี positive มากที่สุดใน test
# --------------------------------------------------------------------------- #
def _busiest_harp(frame: pd.DataFrame) -> int | None:
    """HARP ที่มีชั่วโมง positive มากที่สุดใน test — ตัดสินจากข้อมูลล้วน ไม่ผูกกับเหตุการณ์ใดเหตุการณ์หนึ่ง"""
    positives = frame[frame["label"] == 1]
    pool = positives if len(positives) else frame
    return int(pool["HARPNUM"].mode().iat[0]) if len(pool) else None


def figure_ar_timeline(runs: pd.DataFrame, predictions: pd.DataFrame,
                       labels: StudyLabels) -> tuple[plt.Figure, pd.DataFrame]:
    """ทุกสถาปัตยกรรมของตารางหลักที่แบบแรก (V0) — แผงละสถาปัตยกรรม เพราะ threshold ของแต่ละเซลล์ไม่เท่ากัน

    เส้น = ค่าเฉลี่ยข้าม seed และแถบ = ±1 SD ข้าม seed (``sns.lineplot(errorbar="sd")``)
    """
    cells = [(architecture, labels.matrix_variants[0]) for architecture in labels.matrix_architectures]
    # เฉพาะแถวของ test — รูปนี้ต้องมาจากข้อมูลที่ held-out จริงเท่านั้น
    sub = predictions[predictions["split"] == "test"].copy()
    if sub.empty:
        raise SystemExit("ไม่มีแถว test ใน predictions.parquet")
    sub["issue_time"] = pd.to_datetime(sub["issue_time"])

    harp = _busiest_harp(sub)
    sub = sub[sub["HARPNUM"] == harp]
    fig, axes = plt.subplots(len(cells), 1, figsize=(8.4, 1.55 * len(cells) + 1.4), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colours = arch_palette(labels)
    rows: list[dict] = []

    for ax, (architecture, variant) in zip(axes, cells, strict=True):
        frame = sub[(sub["architecture"] == architecture) & (sub["variant"] == variant)]
        curve = frame.groupby("issue_time").agg(prob=("prob", "mean"), threshold=("threshold", "mean"),
                                                label=("label", "max")).reset_index()
        threshold = float(curve["threshold"].mean())

        # แรเงาชั่วโมงที่เกิด M1+ จริงภายใน 24 ชม. ข้างหน้า — คือสิ่งที่โมเดลควรเตือน
        times = curve["issue_time"].to_numpy()
        for start, end in true_runs(curve["label"].to_numpy() == 1):
            ax.axvspan(times[start], times[end], color="0.88", zorder=0)
        sns.lineplot(data=frame, x="issue_time", y="prob", errorbar="sd", color=colours[labels.arch(architecture)],
                     linewidth=1.6, ax=ax)
        ax.axhline(threshold, color=THRESHOLD, linewidth=1.1, linestyle="--")
        ax.set(ylim=(0, 1.02), yticks=[0, 0.5, 1], ylabel="P(M1+)", xlabel=None)
        ax.set_title(f"{labels.arch(architecture)} ({variant}) · threshold = {threshold:.2f}", loc="left", fontsize=10)
        rows += [
            {"architecture": architecture, "variant": variant, "HARPNUM": harp,
             "issue_time": time, "prob": prob, "threshold": threshold, "label": int(flag)}
            for time, prob, flag in zip(curve["issue_time"], curve["prob"], curve["label"], strict=True)
        ]

    last = axes[-1]
    last.set_xlabel("Forecast issue time (UTC)")
    last.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=3, maxticks=7))
    last.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %Y"))
    handles = [
        Line2D([], [], color="0.3", linewidth=1.6, label="Mean probability over seeds (band = ±1 SD)"),
        Line2D([], [], color=THRESHOLD, linestyle="--", label="Decision threshold"),
        Patch(facecolor="0.88", label="M1+ flare within next 24 h"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    span = f"{sub['issue_time'].min():%d %b %Y} – {sub['issue_time'].max():%d %b %Y}"
    fig.suptitle(f"Hourly flare probability for HARP {harp} ({span}), test set",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
FIGURES = {
    "1": ("delta_tss", lambda run, pred, lab: figure_delta_tss(run, lab)),
    "2": ("interaction", lambda run, pred, lab: figure_interaction(run, lab)),
    "3": ("tss_by_seed", lambda run, pred, lab: figure_tss_by_seed(run, lab)),
    "4": ("roc", lambda run, pred, lab: figure_roc(run, pred, lab)),
    "5": ("confusion", lambda run, pred, lab: figure_confusion(run, lab)),
    "6": ("ar_timeline", lambda run, pred, lab: figure_ar_timeline(run, pred, lab)),
    "7": ("ranking", lambda run, pred, lab: figure_ranking(run, lab)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, help="โฟลเดอร์ผลของงานศึกษา (ปริยาย: artifacts/model_comparison)")
    p.add_argument("--figures", default=",".join(FIGURES), help="เลือกเฉพาะบางรูป เช่น 1,5 (ปริยาย: ทุกรูป)")
    p.add_argument("--dpi", type=int, default=200, help="ความละเอียด PNG (ปริยาย 200 — พอสำหรับงานพิมพ์)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    out_dir = args.out_dir or (cfg.paths.artifacts / STUDY_DIR_NAME)
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "plot_study_figures.log")

    runs, predictions, labels = load_inputs(out_dir)
    wanted = [key.strip() for key in args.figures.split(",") if key.strip()]
    if unknown := [k for k in wanted if k not in FIGURES]:
        raise SystemExit(f"ไม่รู้จักรูป {unknown} — มีให้เลือก {sorted(FIGURES)}")

    failures: list[str] = []
    for key in wanted:
        name, build = FIGURES[key]
        try:
            fig, table = build(runs, predictions, labels)
        except Exception as exc:  # รูปหนึ่งพังไม่ควรทำให้รูปที่เหลือไม่ได้ออก
            logger.error("รูป %s (%s) ล้มเหลว: %s", key, name, exc)
            failures.append(name)
            continue
        save(fig, out_dir, name, table, args.dpi)

    logger.info("=" * 70)
    logger.info("รูปทั้งหมดอยู่ที่ %s", out_dir / "figures")
    logger.info("ตารางตัวเลขคู่รูปอยู่ที่ %s", out_dir / "figures" / "data")
    if failures:
        logger.warning("รูปที่ยังไม่ได้: %s", ", ".join(failures))
    logger.info("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

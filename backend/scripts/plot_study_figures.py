"""สร้างรูปประกอบรายงานจากผลของงานเปรียบเทียบ 5 แบบ (ticket 07 ของ
``.scratch/lstm-feature-ablation/``)

    # หลัง train_study.py เสร็จ
    python backend/scripts/plot_study_figures.py

    # เลือกเฉพาะบางรูป / เขียนไปที่อื่น (ใช้ตอนลองกับข้อมูลสังเคราะห์)
    python backend/scripts/plot_study_figures.py --figures 1,5 --out-dir <tmp>

อ่าน ``runs.parquet`` + ``predictions.parquet`` ที่ ``train_study.py`` เขียนไว้ แล้วเขียน
PNG ลง ``figures/`` กับ **ตารางตัวเลขของทุกรูป** ลง ``figures/data/*.csv``

ตารางคู่รูปไม่ใช่ของแถม: รูปเป็น PNG ในรายงานกระดาษ ผู้อ่านที่แยกสีไม่ออกหรือได้ไฟล์
ขาวดำต้องอ่านตัวเลขชุดเดียวกันได้จากที่อื่น และตอนเขียนรายงานก็ต้องคัดตัวเลขไปใส่ในเนื้อหา
อยู่ดี

**รูปทั้งห้า** (แต่ละรูปตอบคำถามคนละข้อ — ไม่มีรูปไหนซ้ำงานของรูปอื่น):

1. ``delta_tss``      ΔTSS ของแต่ละแบบเทียบ control จับคู่ seed — **คำตอบหลักของงานวิจัย**
2. ``tss_by_seed``    TSS ดิบราย seed — ผลต่างในรูปที่ 1 ใหญ่กว่าความผันผวนของ seed ไหม
3. ``roc``            ROC ของแต่ละแบบ — ภาพที่ไม่ขึ้นกับ threshold ที่เลือก
4. ``confusion``      confusion matrix รายแบบ — ผลต่างมาจาก TP ที่เพิ่มหรือ FP ที่ลด
5. ``ar_timeline``    ความน่าจะเป็นรายชั่วโมงของ HARP ที่มี positive มากที่สุดใน test เทียบกับ threshold ของแบบนั้น

**เรื่องสี** — จานสีมาจาก dataviz reference palette และผ่าน ``validate_palette.js`` แล้ว:
categorical 5 ช่อง (รูปที่ 3) ผ่านทุกด่านบน light surface โดยมีคำเตือน contrast 3 ช่อง ซึ่ง
ชดเชยด้วย direct label ที่ปลายเส้น + ตาราง CSV ตามกติกา relief · คู่ diverging blue/red
(รูปที่ 1) ผ่านครบรวม contrast

รูปพวกนี้ตั้งใจทำโหมดสว่างอย่างเดียว — ปลายทางคือรายงานที่พิมพ์ลงกระดาษ ไม่ใช่หน้าเว็บที่
ผู้อ่านสลับธีมได้
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

import matplotlib  # noqa: E402
import matplotlib.font_manager  # noqa: E402

matplotlib.use("Agg")

# เหตุผลเดียวกับ plot_masks.py: ฟอนต์ปริยายไม่มี glyph ไทย หัวข้อจะกลายเป็นกล่องสี่เหลี่ยม
_THAI_FONTS = ["Leelawadee UI", "Tahoma", "Angsana New", "Noto Sans Thai"]
_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
matplotlib.rcParams["font.family"] = [*[f for f in _THAI_FONTS if f in _INSTALLED], "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.study_summary import paired_summary  # noqa: E402

logger = logging.getLogger("plot_study_figures")

STUDY_DIR_NAME = "lstm_feature_ablation"
VARIANTS_FILE = SCRIPTS_DIR.parent / "configs" / "study_variants.yaml"
CONTROL = "V0"

# --- จานสี (dataviz reference palette, light surface) ---------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

POS = "#2a78d6"  # ขั้วบวกของ diverging (และ hue เดี่ยวของรูปที่ 2/4/5)
NEG = "#e34948"  # ขั้วลบ

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

SCOPE_TITLES = {
    # ไม่ฝังปีไว้ในชื่อ — ช่วงของ test ขึ้นกับ split ใน data.yaml (ดูเหตุผลใน train_study.py)
    "test": "test",
    "val": "validation",
}
REPORT_SCOPES = ("test",)


# --------------------------------------------------------------------------- #
# การโหลดข้อมูล
# --------------------------------------------------------------------------- #
def load_variant_labels() -> dict[str, str]:
    spec = yaml.safe_load(VARIANTS_FILE.read_text(encoding="utf-8"))
    return {name: body["label"] for name, body in spec["variants"].items()}


def load_inputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    runs_path, pred_path = out_dir / "runs.parquet", out_dir / "predictions.parquet"
    for path in (runs_path, pred_path):
        if not path.exists():
            raise SystemExit(f"ไม่พบ {path} — รัน backend/scripts/train_study.py ให้เสร็จก่อน")
    runs = pd.read_parquet(runs_path)
    predictions = pd.read_parquet(pred_path)
    labels = load_variant_labels()
    present = [v for v in labels if v in set(runs["variant"])]
    logger.info(
        "อ่านผล %d run · %d แถวทำนาย · แบบที่มีผล %s",
        len(runs[["variant", "seed"]].drop_duplicates()),
        len(predictions),
        ", ".join(present),
    )
    return runs, predictions, labels


def variant_order(runs: pd.DataFrame, labels: dict[str, str]) -> list[str]:
    """เรียงตามลำดับใน study_variants.yaml เสมอ ไม่ใช่ตามลำดับที่บังเอิญเทรนเสร็จ"""
    return [v for v in labels if v in set(runs["variant"])]


def style_axes(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(INK_2)
    if xgrid or ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="x" if xgrid and not ygrid else "y" if ygrid and not xgrid else "both",
                color=GRID, linewidth=0.8, linestyle="-")


def new_figure(width: float, height: float) -> plt.Figure:
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    return fig


HEADER_IN = 1.0  # ความสูงที่กันไว้ให้หัวเรื่อง (นิ้ว)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> tuple[float, float, float, float]:
    """วางหัวเรื่องด้วยระยะเป็น "นิ้ว" แล้วคืน rect ให้ tight_layout

    วางด้วยเศษส่วนของความสูงตรง ๆ ไม่ได้ เพราะรูปในชุดนี้สูงไม่เท่ากัน (รูปที่ 5 สูงตาม
    จำนวนแบบ) ระยะที่ดูพอดีบนรูปเตี้ยจะกลายเป็นช่องว่างครึ่งหน้าบนรูปสูง
    """
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.30 / height, title, fontsize=12, color=INK, ha="left", va="center")
    fig.text(0.02, 1 - 0.60 / height, subtitle, fontsize=8.5, color=INK_2, ha="left", va="center")
    return (0, 0, 1, 1 - HEADER_IN / height)


def save(fig: plt.Figure, out_dir: Path, name: str, table: pd.DataFrame, dpi: int) -> None:
    fig_dir, data_dir = out_dir / "figures", out_dir / "figures" / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    fig.savefig(png, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    csv = data_dir / f"{name}.csv"
    table.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info("เขียน %s  (ตาราง: %s)", png.relative_to(out_dir.parent.parent) if out_dir.is_absolute() else png, csv.name)


# --------------------------------------------------------------------------- #
# รูปที่ 1 — ΔTSS เทียบ control
# --------------------------------------------------------------------------- #
def figure_delta_tss(runs: pd.DataFrame, labels: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    """แท่ง diverging รอบเส้นศูนย์ = control

    รูปนี้คือคำตอบของคำถามวิจัย จึงเป็นรูปเดียวที่ใช้สีเชิงขั้ว (บวก/ลบ) ไม่ใช่สีเชิงตัวตน
    """
    summary = paired_summary(runs, control=CONTROL, metric="tss")
    order = [v for v in variant_order(runs, labels) if v != CONTROL]
    panels = {scope: summary[summary["scope"] == scope].set_index("variant")
              for scope in REPORT_SCOPES}
    panels = {scope: sub for scope, sub in panels.items() if not sub.empty}

    reach = [float(np.nanmax(np.abs(sub.loc[order, "delta_mean"].to_numpy(float))
                             + np.nan_to_num(sub.loc[order, "delta_sd"].to_numpy(float))))
             for sub in panels.values()]
    span = max([r for r in reach if np.isfinite(r)] or [0.05]) or 0.05

    fig = new_figure(8.6, 0.62 * len(order) + 2.4)
    axes = np.atleast_1d(fig.subplots(1, len(panels), sharey=True, sharex=True))
    rows: list[dict] = []

    for ax, (scope, sub) in zip(axes, panels.items(), strict=True):
        means = sub.loc[order, "delta_mean"].to_numpy(float)
        sds = sub.loc[order, "delta_sd"].to_numpy(float)
        ypos = np.arange(len(order))[::-1]

        ax.barh(ypos, means, height=0.55, color=[POS if m >= 0 else NEG for m in means],
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.errorbar(means, ypos, xerr=np.nan_to_num(sds), fmt="none",
                    ecolor=INK_2, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
        ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
        ax.set_xlim(-span * 2.05, span * 2.05)

        for y, mean, sd in zip(ypos, means, sds, strict=True):
            # เกาะปลาย "หนวด" ไม่ใช่ปลายแท่ง ไม่งั้นตัวเลขทับ cap ของ error bar
            reach_end = mean + (1 if mean >= 0 else -1) * (0.0 if not np.isfinite(sd) else sd)
            ax.text(reach_end + (1 if mean >= 0 else -1) * span * 0.12, y,
                    f"{mean:+.3f} ± {0.0 if not np.isfinite(sd) else sd:.3f}",
                    va="center", ha="left" if mean >= 0 else "right",
                    fontsize=8, color=INK, zorder=5)

        base = float(sub.loc[CONTROL, "mean"]) if CONTROL in sub.index else float("nan")
        ax.set_title(f"{SCOPE_TITLES[scope]}\ncontrol TSS = {base:+.3f}",
                     fontsize=9.5, color=INK, pad=10, loc="left")
        ax.set_yticks(ypos, [f"{v}  {labels[v]}" for v in order])
        ax.set_xlabel("ΔTSS เทียบ control (จับคู่ seed)", fontsize=8.5, color=INK_2)
        style_axes(ax, xgrid=True)
        rows += [{"scope": scope, "variant": v, "label": labels[v],
                  "delta_tss_mean": m, "delta_tss_sd": s, "control_tss_mean": base}
                 for v, m, s in zip(order, means, sds, strict=True)]

    rect = add_header(
        fig,
        "ผลต่างของทักษะการพยากรณ์เทียบชุด 18 SHARP (control)",
        f"แท่ง = ค่าเฉลี่ยของ Δ ราย seed · หนวด = ±1 SD (n={int(summary['n_seeds'].max())} seed) · "
        "ขวาของเส้น = ดีกว่า control",
    )
    fig.tight_layout(rect=rect)
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 2 — TSS ดิบราย seed
# --------------------------------------------------------------------------- #
def figure_tss_by_seed(runs: pd.DataFrame, labels: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    """จุดหนึ่งจุดต่อหนึ่ง seed — ให้เห็นว่าผลต่างในรูปที่ 1 โผล่พ้นความผันผวนของ seed หรือไม่

    สีเดียวทั้งรูปโดยตั้งใจ: ตัวตนของแบบอยู่บนแกน y แล้ว การใส่สีรายแบบจะเป็นการ
    เข้ารหัสซ้ำโดยไม่เพิ่มข้อมูล
    """
    order = variant_order(runs, labels)
    fig = new_figure(8.6, 0.52 * len(order) + 2.4)
    axes = fig.subplots(1, len(REPORT_SCOPES), sharey=True, sharex=True)
    rows: list[dict] = []

    for ax, scope in zip(np.atleast_1d(axes), REPORT_SCOPES, strict=True):
        sub = runs[runs["scope"] == scope]
        ypos = np.arange(len(order))[::-1]
        for y, variant in zip(ypos, order, strict=True):
            values = sub.loc[sub["variant"] == variant, "tss"].astype(float).to_numpy()
            if not len(values):
                continue
            ax.plot(values, np.full(len(values), y), "o", markersize=7,
                    color=POS, markeredgecolor=SURFACE, markeredgewidth=2, alpha=0.85, zorder=3)
            mean = float(np.nanmean(values))
            ax.plot([mean, mean], [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
            rows += [{"scope": scope, "variant": variant, "label": labels[variant],
                      "seed": int(s), "tss": float(t)}
                     for s, t in zip(sub.loc[sub["variant"] == variant, "seed"], values, strict=True)]

        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
        ax.set_yticks(ypos, [f"{v}  {labels[v]}" for v in order])
        ax.set_xlabel("TSS", fontsize=8.5, color=INK_2)
        ax.set_title(SCOPE_TITLES[scope], fontsize=9.5, color=INK, pad=10, loc="left")
        style_axes(ax, xgrid=True)

    rect = add_header(fig, "TSS ดิบของทุกแบบ แยกราย seed",
                      "จุด = หนึ่ง seed · ขีดตั้งทึบ = ค่าเฉลี่ยของแบบนั้น · TSS = 0 คือทายมั่ว")
    fig.tight_layout(rect=rect)
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 3 — ROC
# --------------------------------------------------------------------------- #
def _mean_roc(frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray | None:
    """TPR เฉลี่ยข้าม seed บนกริด FPR ร่วม

    เฉลี่ยเส้นราย seed ไม่ใช่รวม prob ของทุก seed เข้าด้วยกัน — การรวมจะได้เส้นของ
    ensemble ซึ่งเป็นคนละโมเดลกับที่ตาราง TSS รายงาน
    """
    curves = []
    for _, group in frame.groupby("seed", sort=True):
        y = group["label"].to_numpy(dtype=float)
        p = group["prob"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        order = np.argsort(-p)
        y = y[order]
        tps, fps = np.cumsum(y), np.cumsum(1 - y)
        tpr = np.concatenate([[0.0], tps / tps[-1]])
        fpr = np.concatenate([[0.0], fps / fps[-1]])
        curves.append(np.interp(grid, fpr, tpr))
    return np.mean(curves, axis=0) if curves else None


def figure_roc(runs: pd.DataFrame, predictions: pd.DataFrame,
               labels: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    order = variant_order(runs, labels)
    grid = np.linspace(0, 1, 201)
    fig = new_figure(8.4, 4.6)
    axes = fig.subplots(1, len(REPORT_SCOPES))
    rows: list[dict] = []

    for ax, scope in zip(np.atleast_1d(axes), REPORT_SCOPES, strict=True):
        sub = predictions[predictions["split"] == "test"]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, zorder=2)
        auc_means = runs[runs["scope"] == scope].groupby("variant")["auc"].mean()

        for slot, variant in enumerate(order):
            curve = _mean_roc(sub[sub["variant"] == variant], grid)
            if curve is None:
                continue
            colour = CATEGORICAL[slot % len(CATEGORICAL)]
            auc = float(auc_means.get(variant, float("nan")))
            # ไม่ใส่ direct label ที่ปลายเส้น: ROC ทุกเส้นบรรจบที่ (1,1) เสมอ ป้ายห้าอันจึง
            # กองทับกันเป็นก้อนอ่านไม่ออก — relief ของคำเตือน contrast มาจาก legend
            # (ชื่อแบบ + AUC ติดกับตัวอย่างสี) และตาราง CSV คู่รูปแทน
            ax.plot(grid, curve, color=colour, linewidth=2.0, zorder=3,
                    label=f"{variant}  AUC {auc:.3f}  {labels[variant]}")
            rows += [{"scope": scope, "variant": variant, "label": labels[variant],
                      "auc_mean": auc, "fpr": float(f), "tpr": float(t)}
                     for f, t in zip(grid, curve, strict=True)]

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("อัตราเตือนผิด (FPR)", fontsize=8.5, color=INK_2)
        ax.set_ylabel("อัตราจับได้ (TPR)", fontsize=8.5, color=INK_2)
        ax.set_title(SCOPE_TITLES[scope], fontsize=9.5, color=INK, pad=10, loc="left")
        style_axes(ax, xgrid=True, ygrid=True)
        legend = ax.legend(fontsize=7.5, frameon=False, loc="lower right", handlelength=1.6)
        for text in legend.get_texts():
            text.set_color(INK_2)

    rect = add_header(fig, "ROC — ภาพที่ไม่ขึ้นกับ threshold ที่แต่ละแบบเลือก",
                      "เส้น = TPR เฉลี่ยข้าม seed บนกริด FPR ร่วม · เส้นทแยงสีเทา = ทายมั่ว")
    fig.tight_layout(rect=rect)
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 4 — confusion matrix
# --------------------------------------------------------------------------- #
def figure_confusion(runs: pd.DataFrame, labels: dict[str, str],
                     scope: str = "test") -> tuple[plt.Figure, pd.DataFrame]:
    """ผลต่างของ TSS มาจากการจับ positive ได้เพิ่ม หรือจากการเตือนผิดที่ลดลง — รูปนี้แยกให้เห็น"""
    order = variant_order(runs, labels)
    sub = runs[runs["scope"] == scope]
    fig = new_figure(2.05 * len(order) + 0.6, 3.5)
    axes = fig.subplots(1, len(order))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    rows: list[dict] = []

    for ax, variant in zip(np.atleast_1d(axes), order, strict=True):
        counts = sub.loc[sub["variant"] == variant, ["tp", "fp", "fn", "tn"]].astype(float).mean()
        matrix = np.array([[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]])
        # normalise ตามแถว: จำนวนดิบคนละสเกลกันมาก (negative เยอะกว่า positive หลายสิบเท่า)
        # ถ้าลงสีตามจำนวนดิบ ทุกแบบจะได้ภาพเดียวกันคือมุมขวาล่างเข้มช่องเดียว
        shares = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-9, None)
        ax.imshow(shares, cmap=ramp, vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,.0f}\n{shares[i, j]:.0%}",
                        ha="center", va="center", fontsize=8.5,
                        color=SURFACE if shares[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["เตือน", "ไม่เตือน"], fontsize=8)
        ax.set_yticks([0, 1], ["เกิดจริง", "ไม่เกิด"], fontsize=8)
        ax.set_title(f"{variant}\n{labels[variant]}", fontsize=8.5, color=INK, pad=8)
        ax.tick_params(colors=MUTED, length=0)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_color(INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if ax is np.atleast_1d(axes)[0]:
            # บอกว่าแกนไหนคือความจริง แกนไหนคือคำทำนาย — แผงเดียวพอ ซ้ำห้ารอบคือ noise
            ax.set_ylabel("สิ่งที่เกิดขึ้นจริง", fontsize=8.5, color=INK_2, labelpad=8)
            ax.set_xlabel("สิ่งที่โมเดลบอก", fontsize=8.5, color=INK_2, labelpad=8)
        rows.append({"scope": scope, "variant": variant, "label": labels[variant],
                     "tp": matrix[0, 0], "fn": matrix[0, 1], "fp": matrix[1, 0], "tn": matrix[1, 1]})

    rect = add_header(fig, f"Confusion matrix รายแบบ — {SCOPE_TITLES[scope]}",
                      "ค่าเฉลี่ยข้าม seed ที่ threshold ซึ่งแต่ละแบบเลือกไว้บน validation · "
                      "ความเข้ม = สัดส่วนภายในแถว ไม่ใช่จำนวนดิบ")
    fig.tight_layout(rect=rect)
    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# รูปที่ 5 — ไทม์ไลน์ของ HARP ที่มี positive มากที่สุดใน test
# --------------------------------------------------------------------------- #
def _busiest_harp(frame: pd.DataFrame) -> int | None:
    """HARP ที่มีชั่วโมง positive มากที่สุดใน test — ตัดสินจากข้อมูลล้วน ไม่ผูกกับเหตุการณ์ใดเหตุการณ์หนึ่ง"""
    positives = frame[frame["label"] == 1]
    pool = positives if len(positives) else frame
    return int(pool["HARPNUM"].mode().iat[0]) if len(pool) else None


def figure_ar_timeline(runs: pd.DataFrame, predictions: pd.DataFrame,
                       labels: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    """แต่ละแบบได้แผงของตัวเอง ไม่ใช่ห้าเส้นซ้อนกันในแผงเดียว

    เพราะ threshold ของแต่ละแบบไม่เท่ากัน (เลือกบน validation ของใครของมัน) การวางซ้อน
    ต้องลากเส้น threshold ห้าเส้น ซึ่งอ่านไม่ออก — แผงละแบบทำให้เทียบ "เส้นโผล่พ้น
    threshold ตอนไหน" ได้ตรง ๆ โดยแกนเวลาเดียวกัน
    """
    order = variant_order(runs, labels)
    # เฉพาะแถวของ test — รูปนี้ต้องมาจากข้อมูลที่ held-out จริงเท่านั้น ไม่ใช่แถวที่
    # โมเดลเคยเห็นตอนเทรน/เลือก threshold มิฉะนั้นเส้นที่วาดจะไม่ใช่ "ผลทำนาย" จริง
    sub = predictions[predictions["split"] == "test"].copy()
    if sub.empty:
        raise SystemExit("ไม่มีแถว test ใน predictions.parquet")
    sub["issue_time"] = pd.to_datetime(sub["issue_time"])

    harp = _busiest_harp(sub)
    sub = sub[sub["HARPNUM"] == harp]
    fig = new_figure(8.4, 1.25 * len(order) + 1.6)
    axes = fig.subplots(len(order), 1, sharex=True)
    rows: list[dict] = []

    for ax, variant in zip(np.atleast_1d(axes), order, strict=True):
        frame = sub[sub["variant"] == variant]
        curve = frame.groupby("issue_time").agg(prob=("prob", "mean"),
                                                threshold=("threshold", "mean"),
                                                label=("label", "max")).reset_index()
        ax.plot(curve["issue_time"], curve["prob"], color=POS, linewidth=2.0, zorder=4)
        threshold = float(curve["threshold"].mean())
        ax.axhline(threshold, color=NEG, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)

        # แรเงาชั่วโมงที่เกิด M1+ จริงภายใน 24 ชม. ข้างหน้า — คือสิ่งที่โมเดลควรเตือน
        flags = curve["label"].to_numpy() == 1
        times = curve["issue_time"].to_numpy()
        for start, end in _true_runs(flags):
            ax.axvspan(times[start], times[end], color=GRID, zorder=1)

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        # ค่า threshold อยู่ที่ป้ายแกน ไม่ใช่ข้อความลอยในแผง — ของเดิมทับเส้นกราฟของ
        # แบบที่ threshold สูง (V5) จนอ่านไม่ออกทั้งสองอย่าง
        ax.set_ylabel(f"{variant}\nthr {threshold:.2f}", fontsize=9, color=INK,
                      rotation=0, ha="right", va="center", labelpad=14, linespacing=1.6)
        ax.text(1.01, 0.5, labels[variant], transform=ax.transAxes, fontsize=7.5,
                color=MUTED, va="center", ha="left")
        style_axes(ax, ygrid=True)
        rows += [{"variant": variant, "label_text": labels[variant], "HARPNUM": harp,
                  "issue_time": time, "prob": prob, "threshold": threshold, "label": int(flag)}
                 for time, prob, flag in zip(curve["issue_time"], curve["prob"],
                                             curve["label"], strict=True)]

    last = np.atleast_1d(axes)[-1]
    last.set_xlabel("เวลาที่ออกคำพยากรณ์ (UTC)", fontsize=8.5, color=INK_2)
    # AutoDateLocator: อายุของ HARP หนึ่งดวงยาวไม่เท่ากัน (ไม่กี่วันถึงสองสัปดาห์)
    # locator ที่ตรึงวันที่ไว้ตายตัวจะได้ tick เดียวบ้าง ชนกันบ้าง แล้วแต่ดวงที่ถูกเลือก
    last.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=3, maxticks=7))
    last.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %y"))

    span = f"{sub['issue_time'].min():%d %b %Y} – {sub['issue_time'].max():%d %b %Y}"
    rect = add_header(fig, f"ความน่าจะเป็นที่แต่ละแบบให้ — HARP {harp} ({span})",
                      "เส้นน้ำเงิน = ความน่าจะเป็นเฉลี่ยข้าม seed · เส้นแดงประ = threshold ของแบบนั้น · "
                      "แถบเทา = ช่วงที่เกิด M1+ จริงใน 24 ชม. ข้างหน้า")
    fig.tight_layout(rect=rect)
    return fig, pd.DataFrame(rows)


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """ช่วง index ที่ต่อเนื่องกันของค่า True — ใช้แรเงาเป็นแถบเดียวแทนแถบละจุด"""
    spans, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans


# --------------------------------------------------------------------------- #
FIGURES = {
    "1": ("delta_tss", lambda run, pred, lab: figure_delta_tss(run, lab)),
    "2": ("tss_by_seed", lambda run, pred, lab: figure_tss_by_seed(run, lab)),
    "3": ("roc", lambda run, pred, lab: figure_roc(run, pred, lab)),
    "4": ("confusion", lambda run, pred, lab: figure_confusion(run, lab)),
    "5": ("ar_timeline", lambda run, pred, lab: figure_ar_timeline(run, pred, lab)),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, help="โฟลเดอร์ผลของงานศึกษา (ปริยาย: artifacts/lstm_feature_ablation)")
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

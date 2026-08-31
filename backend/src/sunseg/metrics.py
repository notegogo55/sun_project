"""ตัวชี้วัดสำหรับ flare forecasting และ segmentation

**ทำไมไม่ใช้ accuracy**: ในการพยากรณ์ flare ระดับ M+ ภายใน 24 ชม. มี positive จริง
เพียง ~1-3% ของ sample โมเดลที่ตอบว่า "ไม่เกิด" ทุกครั้งจะได้ accuracy 97-99%
ทั้งที่ไร้ประโยชน์โดยสิ้นเชิง วงการ space weather จึงใช้ **TSS** เป็นตัวชี้วัดหลัก
เพราะโมเดลที่ทายด้านเดียวจะได้ TSS = 0 เสมอ ไม่ว่าข้อมูลจะเบ้แค่ไหน
"""

from __future__ import annotations

import numpy as np


class ConfusionCounts(dict):
    """ผลนับ 4 ช่องของ confusion matrix พร้อมตัวชี้วัดที่คำนวณจากมัน"""

    @property
    def tp(self) -> int:
        return int(self["tp"])

    @property
    def fp(self) -> int:
        return int(self["fp"])

    @property
    def tn(self) -> int:
        return int(self["tn"])

    @property
    def fn(self) -> int:
        return int(self["fn"])


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    """นับ TP/FP/TN/FN จากป้ายกำกับและคำทำนายแบบไบนารี"""
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_pred = np.asarray(y_pred).astype(bool).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"ขนาดไม่ตรงกัน: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    return ConfusionCounts(
        tp=int(np.sum(y_true & y_pred)),
        fp=int(np.sum(~y_true & y_pred)),
        tn=int(np.sum(~y_true & ~y_pred)),
        fn=int(np.sum(y_true & ~y_pred)),
    )


def tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Skill Statistic (Hanssen-Kuipers) — ตัวชี้วัดหลักของโปรเจคนี้

    ``TSS = TP/(TP+FN) - FP/(FP+TN) = recall - false_positive_rate``

    อยู่ในช่วง [-1, 1] โดย 0 = ไม่ต่างจากการเดาสุ่ม, 1 = สมบูรณ์แบบ
    คุณสมบัติสำคัญคือ **ไม่ขึ้นกับสัดส่วน class** ทำให้เทียบข้ามชุดข้อมูลได้
    งานตีพิมพ์ที่ดีสำหรับ M+ 24 ชม. อยู่ราว 0.7-0.8
    """
    c = confusion_counts(y_true, y_pred)
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    fpr = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    return recall - fpr


def hss2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Heidke Skill Score — วัดความเก่งเทียบกับการเดาแบบสุ่มที่รักษาสัดส่วน class

    ต่างจาก TSS ตรงที่ HSS2 **ขึ้นกับสัดส่วน class** จึงลงโทษ false alarm หนักกว่า
    เมื่อ positive หายาก รายงานคู่กับ TSS เสมอเพื่อให้เห็นภาพครบ
    """
    c = confusion_counts(y_true, y_pred)
    numerator = 2.0 * (c.tp * c.tn - c.fn * c.fp)
    denominator = (c.tp + c.fn) * (c.fn + c.tn) + (c.tp + c.fp) * (c.fp + c.tn)
    return numerator / denominator if denominator else 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ค่าเฉลี่ยกำลังสองของส่วนต่างระหว่างความน่าจะเป็นที่ทำนายกับผลจริง"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BSS เทียบกับการพยากรณ์แบบ climatology (ทายค่าเฉลี่ยระยะยาวตลอด)

    วัดว่า *ความน่าจะเป็น* ที่โมเดลให้มา calibrate ดีแค่ไหน ไม่ใช่แค่จัดอันดับถูก
    ค่าติดลบแปลว่าแย่กว่าการทายค่าเฉลี่ยเฉยๆ
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    climatology = float(y_true.mean())
    bs_ref = float(np.mean((climatology - y_true) ** 2))
    if bs_ref == 0:
        return 0.0
    return 1.0 - brier_score(y_true, y_prob) / bs_ref


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    c = confusion_counts(y_true, y_pred)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC คำนวณด้วยสถิติ Mann-Whitney U (ไม่ต้องพึ่ง scikit-learn)"""
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # เฉลี่ยอันดับของค่าที่เท่ากัน เพื่อให้ค่าที่ผูกกันไม่ทำให้ AUC เบ้
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_prob) + 1)

    sorted_prob = y_prob[order]
    idx = 0
    while idx < len(sorted_prob):
        end = idx
        while end + 1 < len(sorted_prob) and sorted_prob[end + 1] == sorted_prob[idx]:
            end += 1
        if end > idx:
            ranks[order[idx : end + 1]] = (idx + 1 + end + 1) / 2.0
        idx = end + 1

    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "tss",
    n_steps: int = 200,
) -> tuple[float, float]:
    """หา threshold ที่ทำให้ตัวชี้วัดสูงสุด

    **ต้องเรียกด้วย validation set เท่านั้น** แล้ว freeze ค่าที่ได้ก่อนวัดผลบน test
    การเลือก threshold บน test คือการมองคำตอบ ซึ่งทำให้ตัวเลขสูงเกินจริง

    Returns
    -------
    (threshold, ค่าตัวชี้วัดที่ threshold นั้น)
    """
    scorers = {"tss": tss, "hss2": hss2, "f1": lambda t, p: precision_recall_f1(t, p)[2]}
    if metric not in scorers:
        raise ValueError(f"ไม่รู้จักตัวชี้วัด {metric!r} (รองรับ: {sorted(scorers)})")
    scorer = scorers[metric]

    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    candidates = np.linspace(0.01, 0.99, n_steps)

    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        score = scorer(y_true, y_prob >= threshold)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """นับ TP/FP/TN/FN ที่ทุกจุดของกริด threshold ในครั้งเดียว

    ผลต้องเท่ากับการเรียก :func:`confusion_counts` ทีละจุดด้วย
    ``y_prob >= threshold`` แต่เร็วกว่ามากเมื่อกริดมีหลายร้อยจุด — ใช้เรียงความ
    น่าจะเป็นของฝั่ง positive/negative แยกกันครั้งเดียว แล้วหาจำนวนที่ >= แต่ละ
    threshold ด้วย binary search (``np.searchsorted``) แทนการวนเทียบทั้ง array ซ้ำ

    รองรับเฉพาะการอ่านอย่างเดียว (ไม่แก้ label/prob) — เป็นฟังก์ชันเดียวที่ endpoint
    สรุปของแผง confusion matrix เรียกใช้ เพื่อรับประกันว่าไม่มีตรรกะการนับซ้ำอยู่
    นอกโมดูลนี้ (ดู `.scratch/lstm-confusion-matrix/spec.md`)

    Returns
    -------
    dict คีย์ ``thresholds``, ``tp``, ``fp``, ``tn``, ``fn`` แต่ละค่ายาวเท่ากับ
    ``thresholds`` และเรียงลำดับตรงกัน ผลรวมทั้งสี่ค่าที่ทุก index เท่ากับ
    ``len(y_true)`` เสมอ
    """
    y_true = np.asarray(y_true).astype(bool).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"ขนาดไม่ตรงกัน: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    pos_sorted = np.sort(y_prob[y_true])
    neg_sorted = np.sort(y_prob[~y_true])

    # searchsorted(..., side="left") คืนจำนวนค่าที่ "น้อยกว่า" threshold อย่างเคร่งครัด
    # n_pos ลบด้วยจำนวนนั้นจึงเท่ากับจำนวนค่าที่ ">= threshold" ตรงกับนิยามของ
    # confusion_counts ที่ใช้ y_prob >= threshold ทุกประการ
    tp = n_pos - np.searchsorted(pos_sorted, thresholds, side="left")
    fp = n_neg - np.searchsorted(neg_sorted, thresholds, side="left")
    fn = n_pos - tp
    tn = n_neg - fp

    return {
        "thresholds": thresholds,
        "tp": tp.astype(np.int64),
        "fp": fp.astype(np.int64),
        "tn": tn.astype(np.int64),
        "fn": fn.astype(np.int64),
    }


def classification_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """รวมทุกตัวชี้วัดของ forecasting ไว้ในที่เดียว"""
    y_pred = np.asarray(y_prob).ravel() >= threshold
    c = confusion_counts(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    total = c.tp + c.fp + c.tn + c.fn

    return {
        "threshold": float(threshold),
        "tss": tss(y_true, y_pred),
        "hss2": hss2(y_true, y_pred),
        "bss": brier_skill_score(y_true, y_prob),
        "auc": roc_auc(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (c.tp + c.tn) / total if total else 0.0,
        "tp": c.tp,
        "fp": c.fp,
        "tn": c.tn,
        "fn": c.fn,
    }


def format_report(report: dict[str, float]) -> str:
    """จัดรูปผลลัพธ์ให้อ่านง่ายบนคอนโซล"""
    return "\n".join(
        [
            f"  threshold = {report['threshold']:.3f}",
            f"  TSS       = {report['tss']:+.4f}   <-- ตัวชี้วัดหลัก",
            f"  HSS2      = {report['hss2']:+.4f}",
            f"  BSS       = {report['bss']:+.4f}",
            f"  AUC       = {report['auc']:.4f}",
            f"  precision = {report['precision']:.4f}   recall = {report['recall']:.4f}"
            f"   F1 = {report['f1']:.4f}",
            f"  confusion : TP={report['tp']}  FP={report['fp']}  "
            f"TN={report['tn']}  FN={report['fn']}",
        ]
    )


# --------------------------------------------------------------------------- #
# ตัวชี้วัดของ segmentation
# --------------------------------------------------------------------------- #


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Dice = 2|A∩B| / (|A|+|B|) — เท่ากับ F1 ที่ระดับ pixel"""
    pred = np.asarray(pred).astype(bool).ravel()
    target = np.asarray(target).astype(bool).ravel()
    intersection = float(np.sum(pred & target))
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection over Union (Jaccard index)"""
    pred = np.asarray(pred).astype(bool).ravel()
    target = np.asarray(target).astype(bool).ravel()
    intersection = float(np.sum(pred & target))
    union = float(np.sum(pred | target))
    return (intersection + eps) / (union + eps)


def segmentation_report(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """ตัวชี้วัดระดับ pixel สำหรับ segmentation"""
    binary = np.asarray(pred) >= threshold
    target = np.asarray(target).astype(bool)
    precision, recall, _ = precision_recall_f1(target, binary)

    return {
        "dice": dice_coefficient(binary, target),
        "iou": iou_score(binary, target),
        "precision": precision,
        "recall": recall,
        "pred_area_frac": float(binary.mean()),
        "true_area_frac": float(target.mean()),
    }

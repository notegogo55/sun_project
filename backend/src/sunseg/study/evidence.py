"""การวิเคราะห์หลักของงาน feature-evidence-cv — ตาม ``.scratch/feature-evidence-cv/spec.md`` หัวข้อ 7

เขียนและทดสอบด้วยข้อมูลสังเคราะห์ **ก่อน** เห็นผลจริง (spec ล็อก 2026-09-23)

ตัวประมาณหลัก: ต่อ seed รวม confusion count ของ test ทุก fold → TSS ของแต่ละแบบ แล้ว Δ = ค่าเฉลี่ยข้าม seed
ของ (TSS แบบ − TSS control) · ความไม่แน่นอนจาก bootstrap ที่สุ่ม **HARP ภายในแต่ละ fold** (HARP คือ cluster —
แถวของ HARP เดียวกันสัมพันธ์กันแรงมาก) พร้อมสุ่ม seed ใหม่ ทั้งสองอย่างแบบใส่คืน และจับคู่กันระหว่างแบบ
(ทุกแบบใช้ HARP และ seed ชุดเดียวกันในรอบ bootstrap เดียวกัน)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: ลำดับคอลัมน์ในแกนสุดท้ายของอาร์เรย์ count
COUNTS = ("tp", "fn", "fp", "tn")


@dataclass
class ClusterCounts:
    """confusion count ต่อ (แบบ, seed, cluster) — cluster = (fold, HARPNUM) ของ test

    ``counts[variant]`` รูปทรง ``(n_seeds, n_clusters, 4)`` เรียงตาม :data:`COUNTS`
    ``cluster_fold`` บอกว่า cluster แต่ละตัวอยู่ fold ไหน (ใช้ stratify ตอน bootstrap)
    """

    variants: list[str]
    seeds: list[int]
    clusters: pd.DataFrame  # คอลัมน์ fold, HARPNUM เรียงตามแกน cluster
    counts: dict[str, np.ndarray]

    @property
    def cluster_fold(self) -> np.ndarray:
        return self.clusters["fold"].to_numpy()


def cluster_counts(predictions: pd.DataFrame, variants: list[str]) -> ClusterCounts:
    """แปลง predictions รูปแบบยาว (เฉพาะ test) เป็น count ต่อ (แบบ, seed, fold, HARP)

    ``predictions`` ต้องมี ``variant, seed, fold, HARPNUM, label, prob, threshold`` — threshold ถูก freeze
    จาก val ของแต่ละ (fold, แบบ, seed) มาแล้ว · ทุกแบบต้องมี seed และ (fold, HARP) ชุดเดียวกันเป๊ะ
    """
    test = predictions[predictions["variant"].isin(variants)].copy()
    pred = test["prob"].to_numpy() >= test["threshold"].to_numpy()
    label = test["label"].to_numpy().astype(bool)
    test["tp"] = pred & label
    test["fn"] = ~pred & label
    test["fp"] = pred & ~label
    test["tn"] = ~pred & ~label

    grouped = test.groupby(["variant", "seed", "fold", "HARPNUM"])[list(COUNTS)].sum()
    clusters = (
        test[["fold", "HARPNUM"]].drop_duplicates().sort_values(["fold", "HARPNUM"]).reset_index(drop=True)
    )
    seeds = sorted(test["seed"].unique().tolist())

    counts: dict[str, np.ndarray] = {}
    index = pd.MultiIndex.from_frame(clusters)
    for variant in variants:
        per_seed = []
        for seed in seeds:
            try:
                block = grouped.loc[(variant, seed)]
            except KeyError as exc:
                raise ValueError(f"ไม่มีผลของ ({variant}, seed {seed})") from exc
            aligned = block.reindex(index)
            if aligned.isna().any().any():
                # ทุกแบบใช้แถวร่วมชุดเดียวกัน (spec หัวข้อ 3) — ขาดแปลว่าผลไม่ครบ ไม่ใช่เรื่องปกติ
                missing = aligned.index[aligned.isna().any(axis=1)]
                raise ValueError(f"({variant}, seed {seed}) ขาด (fold, HARP) {list(missing[:5])}")
            per_seed.append(aligned.to_numpy(dtype=np.float64))
        counts[variant] = np.stack(per_seed)
    return ClusterCounts(variants=list(variants), seeds=seeds, clusters=clusters, counts=counts)


def tss_from_counts(summed: np.ndarray) -> np.ndarray:
    """TSS จาก count ที่รวมแล้ว (แกนสุดท้ายยาว 4) — ไม่มี positive หรือไม่มี negative ให้ NaN"""
    tp, fn, fp, tn = (summed[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fn) - fp / (fp + tn)


def pooled_tss(cc: ClusterCounts, variant: str) -> np.ndarray:
    """TSS ต่อ seed ของแบบหนึ่ง รวม count ของ test ทุก fold — รูปทรง ``(n_seeds,)``"""
    return tss_from_counts(cc.counts[variant].sum(axis=1))


def _stratified_weights(cluster_fold: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """น้ำหนัก multinomial ``(n_boot, n_clusters)`` — สุ่ม cluster ใหม่แบบใส่คืน **ภายใน fold เดียวกัน**"""
    weights = np.zeros((n_boot, len(cluster_fold)), dtype=np.float64)
    for fold in np.unique(cluster_fold):
        members = np.flatnonzero(cluster_fold == fold)
        draws = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), size=n_boot)
        weights[:, members] = draws
    return weights


@dataclass
class DeltaResult:
    variant: str
    control: str
    delta: float  # ค่าจริง (ไม่ bootstrap)
    se: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    p_two_sided: float
    boot: np.ndarray  # Δ ของทุกรอบ bootstrap


def bootstrap_deltas(
    cc: ClusterCounts, control: str, n_boot: int = 10_000, seed: int = 0, chunk: int = 500
) -> dict[str, DeltaResult]:
    """Δ ของทุกแบบเทียบ ``control`` พร้อม bootstrap ที่จับคู่กัน (HARP + seed ชุดเดียวกันทุกแบบในรอบเดียวกัน)"""
    rng = np.random.default_rng(seed)
    n_seeds = len(cc.seeds)
    boots: dict[str, list[np.ndarray]] = {v: [] for v in cc.variants}

    remaining = n_boot
    while remaining > 0:
        b = min(chunk, remaining)
        weights = _stratified_weights(cc.cluster_fold, b, rng)
        seed_idx = rng.integers(0, n_seeds, size=(b, n_seeds))
        for variant in cc.variants:
            summed = np.einsum("bc,sck->bsk", weights, cc.counts[variant])  # (b, seed, 4)
            summed = np.take_along_axis(summed, seed_idx[:, :, None], axis=1)
            boots[variant].append(tss_from_counts(summed))  # (b, seed)
        remaining -= b

    boot_tss = {v: np.concatenate(chunks) for v, chunks in boots.items()}
    control_point = pooled_tss(cc, control)
    results: dict[str, DeltaResult] = {}
    for variant in cc.variants:
        if variant == control:
            continue
        delta_boot = np.nanmean(boot_tss[variant] - boot_tss[control], axis=1)
        point = float(np.mean(pooled_tss(cc, variant) - control_point))
        p = 2 * min(float(np.mean(delta_boot <= 0)), float(np.mean(delta_boot >= 0)))
        results[variant] = DeltaResult(
            variant=variant,
            control=control,
            delta=point,
            se=float(np.std(delta_boot, ddof=1)),
            ci95=(float(np.percentile(delta_boot, 2.5)), float(np.percentile(delta_boot, 97.5))),
            ci90=(float(np.percentile(delta_boot, 5)), float(np.percentile(delta_boot, 95))),
            p_two_sided=min(1.0, p),
            boot=delta_boot,
        )
    return results


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """p ที่ปรับด้วย Holm (step-down) — เทียบกับ α เดิมได้ตรง ๆ"""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def verdict(result: DeltaResult, p_adjusted: float, alpha: float = 0.05, margin: float = 0.02) -> str:
    """เกณฑ์ตัดสินของ spec หัวข้อ 7 — ลำดับการเช็คตามที่เขียนใน spec"""
    if p_adjusted < alpha:
        return "ช่วย" if result.delta > 0 else "แย่ลง"
    if -margin <= result.ci90[0] and result.ci90[1] <= margin:
        return "ไม่ต่างในทางปฏิบัติ"
    return "สรุปไม่ได้"

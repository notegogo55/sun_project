"""สรุปผลงานเปรียบเทียบชุด feature หลายแบบ × หลาย seed (ticket 07 ของ
``.scratch/lstm-feature-ablation/``)

หลักที่ spec กำหนด (หัวข้อ "วิธีสรุปผล") และโมดูลนี้บังคับไว้:

- **ผลต่างเทียบ control คิดแบบจับคู่ seed** — ``Δ_s = metric(แบบ, s) − metric(control, s)``
  แล้วรายงาน mean±SD ของ ``Δ_s`` ค่าเฉลี่ยเท่ากับ ``mean(แบบ) − mean(control)`` แต่ SD ไม่
  เท่ากัน: SD แบบจับคู่หักความผันผวนร่วมที่มาจาก seed เดียวกันออก ซึ่งคือเหตุผลที่ใช้ seed
  ชุดเดียวกันทุกแบบ
- **SD เป็น sample SD (ddof=1)** — seed ที่ใช้เป็นตัวอย่างจากค่าเริ่มต้นที่เป็นไปได้ทั้งหมด
- **seed ต้องครบทุกแบบ** — คู่ที่ขาดทำให้ได้ error บอกว่าขาดอะไร ไม่ใช่เฉลี่ยบนชุด seed
  คนละชุดเงียบ ๆ
- **ค่าที่เป็น NaN ไม่ถูกข้าม** (เช่น TSS ของขอบเขตที่ไม่มี positive) — ผลรวมเป็น NaN ให้เห็น
  ไม่ใช่เฉลี่ยเฉพาะ seed ที่มีค่า
- ไม่คำนวณค่า p โดยตั้งใจ — seed ระดับหลักหน่วยให้กำลังทางสถิติไม่พอ
"""

from __future__ import annotations

import pandas as pd

SUMMARY_COLUMNS: tuple[str, ...] = ("scope", "variant", "n_seeds", "mean", "sd", "delta_mean", "delta_sd")


def _sd(values: pd.Series) -> float:
    return float(values.std(ddof=1, skipna=False)) if len(values) > 1 else float("nan")


def paired_summary(runs: pd.DataFrame, control: str = "V0", metric: str = "tss") -> pd.DataFrame:
    """สรุปผลต่อ (ขอบเขต, แบบ) จากตารางผลรายรอบ

    Parameters
    ----------
    runs
        หนึ่งแถวต่อ ``(variant, seed, scope)`` มีคอลัมน์ ``metric``
    control
        แบบอ้างอิงของ Δ — ทุกแบบถูกเทียบกับตัวนี้แบบจับคู่ seed

    Returns
    -------
    หนึ่งแถวต่อ ``(scope, variant)`` คอลัมน์ตาม :data:`SUMMARY_COLUMNS` เรียงแบบตามลำดับที่
    ปรากฏครั้งแรกใน ``runs`` — แถวของ control มี ``delta_mean = delta_sd = 0``
    """
    required = {"variant", "seed", "scope", metric}
    if missing := required - set(runs.columns):
        raise ValueError(f"ตารางผลขาดคอลัมน์ {sorted(missing)}")
    if control not in set(runs["variant"]):
        raise ValueError(f"ไม่มีผลของแบบอ้างอิง {control!r} — เทียบ Δ ไม่ได้")

    duplicated = runs.duplicated(["variant", "seed", "scope"], keep=False)
    if duplicated.any():
        dupes = runs.loc[duplicated, ["variant", "seed", "scope"]].drop_duplicates().to_dict("records")
        raise ValueError(f"มีผลซ้ำของ (variant, seed, scope): {dupes}")

    variant_order = list(dict.fromkeys(runs["variant"]))
    rows: list[dict] = []
    for scope, group in runs.groupby("scope", sort=False):
        seeds = sorted(set(group["seed"]))
        present = set(zip(group["variant"], group["seed"], strict=True))
        absent = [(v, s) for v in variant_order for s in seeds if (v, s) not in present]
        if absent:
            raise ValueError(
                f"ขอบเขต {scope!r}: seed ไม่ครบทุกแบบ จับคู่ไม่ได้ — ขาด (variant, seed) {absent}"
            )

        pivot = group.pivot(index="seed", columns="variant", values=metric).astype(float)
        reference = pivot[control]
        for variant in variant_order:
            values = pivot[variant]
            delta = values - reference
            rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "n_seeds": int(len(values)),
                    "mean": float(values.mean(skipna=False)),
                    "sd": _sd(values),
                    "delta_mean": float(delta.mean(skipna=False)),
                    "delta_sd": _sd(delta),
                }
            )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def seed_table(runs: pd.DataFrame, scope: str, metric: str = "tss") -> pd.DataFrame:
    """ตัวเลขดิบรายแบบ × ราย seed ของขอบเขตหนึ่ง — ให้ผู้อ่านคำนวณ Δ ตรวจเองได้"""
    subset = runs[runs["scope"] == scope]
    variant_order = list(dict.fromkeys(subset["variant"]))
    table = subset.pivot(index="variant", columns="seed", values=metric)
    return table.reindex(variant_order).astype(float) if len(table) else table

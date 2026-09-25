"""สรุปผลงานเปรียบเทียบโมเดล (สถาปัตยกรรม × ชุด feature) หลาย seed

คำถามหลักตั้งแต่ 2026-09-24 คือ "เซลล์ไหนให้ผลดีที่สุด" (:func:`rank_cells`) — ส่วน Δ ตามแบบ,
Δ ตามสถาปัตยกรรม และ interaction เป็นการวิเคราะห์ประกอบ

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

import re
from collections.abc import Iterable

import pandas as pd

SUMMARY_COLUMNS: tuple[str, ...] = ("scope", "variant", "n_seeds", "mean", "sd", "delta_mean", "delta_sd")

#: คอลัมน์ของตาราง interaction — ``delta`` คือ Δ ตามแบบ ของสถาปัตยกรรมนั้น และ
#: ``interaction`` คือส่วนที่ต่างจาก Δ ตามแบบ ของสถาปัตยกรรมอ้างอิง
INTERACTION_COLUMNS: tuple[str, ...] = (
    "scope",
    "architecture",
    "variant",
    "n_seeds",
    "delta_mean",
    "delta_sd",
    "interaction_mean",
    "interaction_sd",
)

#: คอลัมน์ของตารางอันดับเซลล์ (:func:`rank_cells`)
RANKING_COLUMNS: tuple[str, ...] = (
    "rank",
    "architecture",
    "variant",
    "n_seeds",
    "mean",
    "sd",
    "gap_mean",
    "gap_sd",
    "tied_with_best",
)

TIMING_COLUMNS: tuple[str, ...] = (
    "n_runs",
    "prepare_seconds",
    "total_mean",
    "total_sd",
    "train_mean",
    "other_mean",
    "epochs_mean",
    "sec_per_epoch",
    "start_to_result_seconds",
    "total_ratio",
    "sec_per_epoch_ratio",
)

END_TO_END_COLUMNS: tuple[str, ...] = (
    "shared_hours",
    "intensity_hours",
    "xray_hours",
    "train_hours",
    "total_hours",
    "total_ratio",
)

# ช่องของ intensity ขึ้นต้นด้วยเลขความยาวคลื่น/ช่อง เช่น 4500_p95, 304_median — SHARP ขึ้นต้นด้วยตัวอักษร
_INTENSITY_COLUMN = re.compile(r"^\d+_\w+$")


def _sd(values: pd.Series) -> float:
    return float(values.std(ddof=1, skipna=False)) if len(values) > 1 else float("nan")


def _pivot_key(pair_cols: tuple[str, ...]) -> str | list[str]:
    """pandas pivot ให้ index/columns แบบเรียบ (ไม่ห่อ MultiIndex) เมื่อมีคีย์เดียว"""
    return list(pair_cols) if len(pair_cols) > 1 else pair_cols[0]


def paired_summary(
    runs: pd.DataFrame,
    control: str = "V0",
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
    unit_col: str = "variant",
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """สรุปผลต่อ (ขอบเขต, *group_cols, หน่วยที่เทียบ) จากตารางผลรายรอบ

    Parameters
    ----------
    runs
        หนึ่งแถวต่อ ``(*group_cols, unit_col, *pair_cols, scope)`` มีคอลัมน์ ``metric``
    control
        ค่าอ้างอิงของ Δ ในคอลัมน์ ``unit_col`` — ทุกหน่วยถูกเทียบกับตัวนี้แบบจับคู่ ``pair_cols``
    pair_cols
        คอลัมน์ที่ใช้จับคู่รอบ (ปริยาย ``("seed",)``) — ส่ง ``("fold", "seed")`` ตอนสรุปผล
        cross-validation เพื่อจับคู่ทั้ง fold และ seed พร้อมกัน (ตัดความผันผวนร่วมของทั้งสอง
        มิติออกจาก SD ของ Δ เหมือนที่การจับคู่ seed เดี่ยว ๆ ทำ)
    unit_col
        แกนที่กำลังเทียบ: ``"variant"`` ได้ Δ ตามแบบ, ``"architecture"`` ได้ Δ ตามสถาปัตยกรรม
    group_cols
        แกนที่ถูก **ตรึง** ระหว่างเทียบ — ส่ง ``("architecture",)`` คู่กับ ``unit_col="variant"``
        เพื่อให้แต่ละสถาปัตยกรรมเทียบกับแบบอ้างอิง **ของตัวเอง** ไม่ใช่ข้ามสถาปัตยกรรม
        (ดู CONTEXT.md: Δ ตามแบบ / Δ ตามสถาปัตยกรรม)

    Returns
    -------
    หนึ่งแถวต่อ ``(scope, *group_cols, unit_col)`` เรียงตามลำดับที่ปรากฏครั้งแรกใน ``runs`` —
    แถวของ control มี ``delta_mean = delta_sd = 0``
    """
    required = {unit_col, "scope", metric, *pair_cols, *group_cols}
    if missing := required - set(runs.columns):
        raise ValueError(f"ตารางผลขาดคอลัมน์ {sorted(missing)}")
    if control not in set(runs[unit_col]):
        raise ValueError(f"ไม่มีผลของ {unit_col} อ้างอิง {control!r} — เทียบ Δ ไม่ได้")

    pair_label = ", ".join(pair_cols)
    key_cols = [unit_col, *group_cols, *pair_cols, "scope"]
    duplicated = runs.duplicated(key_cols, keep=False)
    if duplicated.any():
        dupes = runs.loc[duplicated, key_cols].drop_duplicates().to_dict("records")
        raise ValueError(f"มีผลซ้ำของ ({', '.join(key_cols)}): {dupes}")

    rows: list[dict] = []
    for scope_key, group in runs.groupby(["scope", *group_cols], sort=False):
        scope_values = scope_key if isinstance(scope_key, tuple) else (scope_key,)
        context = dict(zip(["scope", *group_cols], scope_values, strict=True))
        unit_order = list(dict.fromkeys(group[unit_col]))
        if control not in unit_order:
            raise ValueError(f"{context}: ไม่มี {unit_col} อ้างอิง {control!r} — เทียบ Δ ไม่ได้")

        keys = sorted(set(zip(*[group[c] for c in pair_cols], strict=True)))
        present = set(zip(group[unit_col], *[group[c] for c in pair_cols], strict=True))
        absent = [(u, *k) for u in unit_order for k in keys if (u, *k) not in present]
        if absent:
            raise ValueError(
                f"{context}: {pair_label} ไม่ครบทุก {unit_col} จับคู่ไม่ได้ — "
                f"ขาด ({unit_col}, {pair_label}) {absent}"
            )

        pivot = group.pivot(index=_pivot_key(pair_cols), columns=unit_col, values=metric).astype(float)
        reference = pivot[control]
        for unit in unit_order:
            values = pivot[unit]
            delta = values - reference
            rows.append(
                {
                    **context,
                    unit_col: unit,
                    "n_seeds": int(len(values)),
                    "mean": float(values.mean(skipna=False)),
                    "sd": _sd(values),
                    "delta_mean": float(delta.mean(skipna=False)),
                    "delta_sd": _sd(delta),
                }
            )
    columns = ["scope", *group_cols, unit_col, "n_seeds", "mean", "sd", "delta_mean", "delta_sd"]
    return pd.DataFrame(rows, columns=columns)


def delta_by_variant(
    runs: pd.DataFrame,
    anchor_variant: str = "V0",
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
) -> pd.DataFrame:
    """Δ ตามแบบ — ในแต่ละสถาปัตยกรรม เทียบกับแบบอ้างอิง **ของตัวมันเอง**

    ตอบว่า feature ที่เพิ่มเข้ามาช่วยอะไร โดยไม่ปนกับคำถามว่าสถาปัตยกรรมไหนดีกว่า
    """
    return paired_summary(
        runs, control=anchor_variant, metric=metric, pair_cols=pair_cols,
        unit_col="variant", group_cols=("architecture",),
    )


def delta_by_architecture(
    runs: pd.DataFrame,
    anchor_architecture: str = "lstm",
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
) -> pd.DataFrame:
    """Δ ตามสถาปัตยกรรม — ในแต่ละแบบ เทียบกับสถาปัตยกรรมอ้างอิงของแบบเดียวกัน"""
    return paired_summary(
        runs, control=anchor_architecture, metric=metric, pair_cols=pair_cols,
        unit_col="architecture", group_cols=("variant",),
    )


def interaction_summary(
    runs: pd.DataFrame,
    anchor_architecture: str = "lstm",
    anchor_variant: str = "V0",
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
) -> pd.DataFrame:
    """ส่วนที่ Δ ตามแบบ ของสถาปัตยกรรมหนึ่ง ต่างจาก Δ ตามแบบ ของสถาปัตยกรรมอ้างอิง

    นี่คือตัวเลขเดียวที่ตารางแกนเดียวให้ไม่ได้ — ตอบว่า feature กลุ่มนั้น **ไม่มีข้อมูล**
    (interaction ใกล้ศูนย์ทุกสถาปัตยกรรม จึงไม่มีใครดึงอะไรออกมาได้) หรือ **LSTM อ่านไม่ออก**
    (interaction เป็นบวกอย่างมีนัยในสถาปัตยกรรมอื่น)

    คำนวณแบบจับคู่ระดับ seed ทุกขั้น ไม่ใช่ลบค่าเฉลี่ยกัน — SD ที่ได้จึงหักความผันผวนร่วม
    ของ seed ออกทั้งสองชั้น (ชั้น Δ ตามแบบ และชั้นผลต่างของ Δ)

    Returns
    -------
    หนึ่งแถวต่อ ``(scope, architecture, variant)`` ตาม :data:`INTERACTION_COLUMNS` —
    แถวของสถาปัตยกรรมอ้างอิงมี interaction เป็น 0 และแถวของแบบอ้างอิงมีทั้งสองค่าเป็น 0
    """
    required = {"architecture", "variant", "scope", metric, *pair_cols}
    if missing := required - set(runs.columns):
        raise ValueError(f"ตารางผลขาดคอลัมน์ {sorted(missing)}")

    arch_order = list(dict.fromkeys(runs["architecture"]))
    variant_order = list(dict.fromkeys(runs["variant"]))
    for name, order, column in (
        (anchor_architecture, arch_order, "architecture"),
        (anchor_variant, variant_order, "variant"),
    ):
        if name not in order:
            raise ValueError(f"ไม่มีผลของ {column} อ้างอิง {name!r} — คำนวณ interaction ไม่ได้")

    pair_label = ", ".join(pair_cols)
    rows: list[dict] = []
    for scope, group in runs.groupby("scope", sort=False):
        keys = sorted(set(zip(*[group[c] for c in pair_cols], strict=True)))
        present = set(zip(group["architecture"], group["variant"], *[group[c] for c in pair_cols], strict=True))
        absent = [(a, v, *k) for a in arch_order for v in variant_order for k in keys if (a, v, *k) not in present]
        if absent:
            raise ValueError(
                f"ขอบเขต {scope!r}: เซลล์ไม่ครบทุก {pair_label} จับคู่ไม่ได้ — "
                f"ขาด (architecture, variant, {pair_label}) {absent}"
            )

        wide = group.pivot(
            index=["architecture", *pair_cols], columns="variant", values=metric
        ).astype(float)
        delta = wide.sub(wide[anchor_variant], axis=0)          # Δ ตามแบบ ของทุก (สถาปัตยกรรม, รอบ)
        anchor_delta = delta.xs(anchor_architecture, level="architecture")

        for architecture in arch_order:
            own = delta.xs(architecture, level="architecture")
            interaction = own - anchor_delta.reindex(own.index)
            for variant in variant_order:
                rows.append(
                    {
                        "scope": scope,
                        "architecture": architecture,
                        "variant": variant,
                        "n_seeds": int(len(own)),
                        "delta_mean": float(own[variant].mean(skipna=False)),
                        "delta_sd": _sd(own[variant]),
                        "interaction_mean": float(interaction[variant].mean(skipna=False)),
                        "interaction_sd": _sd(interaction[variant]),
                    }
                )
    return pd.DataFrame(rows, columns=list(INTERACTION_COLUMNS))


def rank_cells(
    runs: pd.DataFrame,
    scope: str = "test",
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
) -> pd.DataFrame:
    """จัดอันดับทุกเซลล์ (สถาปัตยกรรม, แบบ) ของขอบเขตหนึ่งด้วยค่าเฉลี่ยของ ``metric`` — คำถามหลักของงาน
    "โมเดลไหนให้ผลดีที่สุด" (ดู CONTEXT.md: เซลล์ที่ดีที่สุด)

    อันดับ 1 คือเซลล์ที่ค่าเฉลี่ยสูงสุด ทุกเซลล์ถูกเทียบกับมันแบบจับคู่ ``pair_cols`` —
    ``tied_with_best`` เป็นจริงเมื่อระยะห่างจากอันดับ 1 ไม่เกิน SD ของผลต่างนั้นเอง คือแยกจาก
    อันดับ 1 ไม่ออกที่จำนวน seed เท่านี้ (กติกาเดียวกับตัวหนาในรายงาน ไม่ใช่การทดสอบนัยสำคัญ)

    Returns
    -------
    หนึ่งแถวต่อเซลล์ เรียงตามอันดับ: ``rank``, ``architecture``, ``variant``, ``n_seeds``, ``mean``, ``sd``,
    ``gap_mean``/``gap_sd`` (ค่าของเซลล์ลบค่าของอันดับ 1 จับคู่ — ≤ 0 เสมอในค่าเฉลี่ย) และ ``tied_with_best``
    · เซลล์ที่ค่าเฉลี่ยเป็น NaN อยู่ท้ายสุดและไม่ถือว่าเสมอ
    """
    required = {"architecture", "variant", "scope", metric, *pair_cols}
    if missing := required - set(runs.columns):
        raise ValueError(f"ตารางผลขาดคอลัมน์ {sorted(missing)}")
    subset = runs[runs["scope"] == scope]
    if subset.empty:
        raise ValueError(f"ไม่มีผลของขอบเขต {scope!r} ให้จัดอันดับ")

    cells = list(dict.fromkeys(zip(subset["architecture"], subset["variant"], strict=True)))
    keys = sorted(set(zip(*[subset[c] for c in pair_cols], strict=True)))
    present = set(zip(subset["architecture"], subset["variant"], *[subset[c] for c in pair_cols], strict=True))
    absent = [(*cell, *k) for cell in cells for k in keys if (*cell, *k) not in present]
    if absent:
        pair_label = ", ".join(pair_cols)
        raise ValueError(
            f"ขอบเขต {scope!r}: {pair_label} ไม่ครบทุกเซลล์ จับคู่ไม่ได้ — "
            f"ขาด (architecture, variant, {pair_label}) {absent}"
        )

    wide = subset.pivot(index=_pivot_key(pair_cols), columns=["architecture", "variant"], values=metric).astype(float)
    means = {cell: float(wide[cell].mean(skipna=False)) for cell in cells}
    # เรียงด้วยค่าเฉลี่ยจากมากไปน้อย NaN ท้ายสุด — เท่ากันพอดีให้ลำดับใน runs ตัดสิน (sorted เสถียร)
    ordered = sorted(cells, key=lambda c: (not pd.notna(means[c]), -means[c] if pd.notna(means[c]) else 0.0))
    best = ordered[0]

    rows = []
    for rank, cell in enumerate(ordered, start=1):
        gap = wide[cell] - wide[best]
        gap_mean, gap_sd = float(gap.mean(skipna=False)), _sd(gap)
        tied = cell == best or (pd.notna(gap_mean) and pd.notna(gap_sd) and -gap_mean <= gap_sd)
        rows.append(
            {
                "rank": rank,
                "architecture": cell[0],
                "variant": cell[1],
                "n_seeds": int(len(wide)),
                "mean": means[cell],
                "sd": _sd(wide[cell]),
                "gap_mean": gap_mean,
                "gap_sd": gap_sd if cell != best else 0.0,
                "tied_with_best": bool(tied),
            }
        )
    return pd.DataFrame(rows, columns=list(RANKING_COLUMNS))


def seed_table(
    runs: pd.DataFrame,
    scope: str,
    metric: str = "tss",
    pair_cols: tuple[str, ...] = ("seed",),
    unit_cols: tuple[str, ...] = ("variant",),
) -> pd.DataFrame:
    """ตัวเลขดิบรายหน่วย × รายรอบของขอบเขตหนึ่ง — ให้ผู้อ่านคำนวณ Δ ตรวจเองได้

    ``pair_cols`` เหมือนใน :func:`paired_summary` — ปริยายคือคอลัมน์เดียว (``seed``) ให้ผลลัพธ์
    แบบเดิม ส่ง ``("fold", "seed")`` เพื่อได้ตารางที่คอลัมน์เป็น MultiIndex (fold, seed)
    ส่วน ``unit_cols`` เลือกว่าแถวคือแบบ หรือเซลล์ (``("architecture", "variant")``)
    """
    subset = runs[runs["scope"] == scope]
    if len(unit_cols) > 1:
        order = list(dict.fromkeys(map(tuple, subset[list(unit_cols)].itertuples(index=False, name=None))))
    else:
        order = list(dict.fromkeys(subset[unit_cols[0]]))
    table = subset.pivot(index=_pivot_key(unit_cols), columns=_pivot_key(pair_cols), values=metric)
    return table.reindex(order).astype(float) if len(table) else table


def timing_summary(
    runs: pd.DataFrame,
    control: str | tuple[str, ...] = "V0",
    unit_cols: tuple[str, ...] = ("variant",),
) -> pd.DataFrame:
    """เวลาที่ใช้ตั้งแต่เริ่มจนได้ผล แยกตาม ``unit_cols`` จากตารางรายรัน

    ``unit_cols`` เลือกได้ว่าจะสรุปต่อแบบ (``("variant",)``) ต่อสถาปัตยกรรม
    (``("architecture",)`` — ตอบว่าสถาปัตยกรรมไหนแพงกว่ากี่เท่า) หรือต่อเซลล์
    (``("architecture", "variant")``) โดย ``control`` ต้องมีจำนวนค่าตรงกับ ``unit_cols``

    ``runs`` หนึ่งแถวต่อรัน คอลัมน์: ``variant``, ``fold`` (ว่างในโหมด single split),
    ``epochs_run``, ``train_seconds`` (เฉพาะลูป epoch), ``total_seconds`` (ทั้งรัน: สร้างโมเดล →
    เทรน → ประเมิน → เขียนไฟล์) และ ``prepare_seconds`` (เตรียมข้อมูลของแบบ)

    - ``prepare_seconds`` นับ **ครั้งเดียวต่อ (แบบ, fold)** ไม่ใช่ต่อรัน — ทุก seed ของแบบเดียวกัน
      ใช้ข้อมูลที่เตรียมไว้ก้อนเดียว ค่านั้นถูกบันทึกซ้ำในไฟล์ของทุกรัน (ให้แต่ละไฟล์ครบในตัวและ
      resume ได้) ถ้าบวกทุกแถวจะนับซ้ำเท่าจำนวน seed
    - ``sec_per_epoch`` = Σเวลาเทรน / Σepoch ไม่ใช่เฉลี่ยอัตราต่อรัน — เวลาต่อรันขึ้นกับจำนวน epoch
      ก่อน early stopping ซึ่งแต่ละ seed หยุดไม่เท่ากัน จึงต้องแยกอัตราต่อ epoch ไว้เทียบต้นทุนของแต่ละแบบ
    - ``start_to_result_seconds`` = prepare + Σ ``total_seconds`` คือเวลารวมที่ใช้ผลิตผลทั้งแถวของแบบนั้น
    - ``total_ratio`` / ``sec_per_epoch_ratio`` เทียบกับ ``control`` (ตัวมันเองเท่ากับ 1)

    Returns
    -------
    หนึ่งแถวต่อแบบ คอลัมน์ตาม :data:`TIMING_COLUMNS` เรียงตามลำดับที่ปรากฏครั้งแรกใน ``runs``
    """
    required = {"variant", "fold", "epochs_run", "train_seconds", "total_seconds", "prepare_seconds", *unit_cols}
    if missing := required - set(runs.columns):
        raise ValueError(f"ตารางเวลาขาดคอลัมน์ {sorted(missing)}")

    reference_key = (control,) if isinstance(control, str) else tuple(control)
    if len(reference_key) != len(unit_cols):
        raise ValueError(f"control {reference_key} มี {len(reference_key)} ค่า แต่ unit_cols มี {len(unit_cols)}")

    # groupby ทิ้งคีย์ที่ว่างเงียบ ๆ — single split มี fold ว่างทุกแถว ต้องแทนค่าก่อนไม่งั้นได้ 0 กลุ่ม
    runs = runs.assign(fold=runs["fold"].astype(float).fillna(-1))

    rows: list[dict] = []
    for key, group in runs.groupby(list(unit_cols), sort=False):
        values = key if isinstance(key, tuple) else (key,)
        train, total = group["train_seconds"], group["total_seconds"]
        # prepare เป็นของ (แบบ, fold) เสมอ ไม่ว่าจะสรุปด้วยแกนไหน — จัดกลุ่มด้วยคู่นั้นตรง ๆ
        # เพื่อไม่ให้นับซ้ำเมื่อกลุ่มหนึ่งกินหลายแบบ (เช่นตอนสรุปต่อสถาปัตยกรรม)
        prepare = float(group.groupby(["variant", "fold"])["prepare_seconds"].mean().sum())
        rows.append(
            {
                **dict(zip(unit_cols, values, strict=True)),
                "n_runs": int(len(group)),
                "prepare_seconds": prepare,
                "total_mean": float(total.mean()),
                "total_sd": _sd(total),
                "train_mean": float(train.mean()),
                "other_mean": float((total - train).mean()),
                "epochs_mean": float(group["epochs_run"].mean()),
                "sec_per_epoch": float(train.sum() / group["epochs_run"].sum()),
                "start_to_result_seconds": prepare + float(total.sum()),
            }
        )

    table = pd.DataFrame(rows)
    indexed = table.set_index(list(unit_cols))
    lookup = reference_key[0] if len(reference_key) == 1 else reference_key
    if lookup not in indexed.index:
        raise ValueError(f"ไม่มีเวลาของตัวอ้างอิง {reference_key} — เทียบสัดส่วนไม่ได้")
    reference = indexed.loc[lookup]
    table["total_ratio"] = table["total_mean"] / reference["total_mean"]
    table["sec_per_epoch_ratio"] = table["sec_per_epoch"] / reference["sec_per_epoch"]
    return table[[*unit_cols, *TIMING_COLUMNS]]


def feature_group(column: str) -> str:
    """กลุ่มต้นทางของ feature หนึ่งคอลัมน์: ``"xray"`` (ขึ้นต้น ``xray_``), ``"intensity"`` (ช่องที่ขึ้นต้น
    ด้วยเลข เช่น ``4500_p95``) นอกนั้นคือ ``"sharp"``

    แยกจากชื่อคอลัมน์ ไม่ต้องกำหนดตารางคู่ขนานไว้ที่อื่น — เพิ่ม feature ใน ``study_variants.yaml`` แล้ว
    ต้นทางที่แบบนั้นต้องใช้ตามมาเอง
    """
    if column.startswith("xray_"):
        return "xray"
    if _INTENSITY_COLUMN.match(column):
        return "intensity"
    return "sharp"


def feature_groups(columns: Iterable[str]) -> frozenset[str]:
    """กลุ่มต้นทางทั้งหมดที่คอลัมน์ของแบบหนึ่งมาจาก"""
    return frozenset(feature_group(column) for column in columns)


def end_to_end_hours(
    timing: pd.DataFrame,
    components: Iterable[dict],
    variant_groups: dict[str, frozenset[str] | set[str]],
    control: str | tuple[str, ...] = "V0",
    unit_cols: tuple[str, ...] = ("variant",),
) -> pd.DataFrame:
    """เวลารวมตั้งแต่ต้นทางจนได้ผลรายแบบ (ชั่วโมง) = ต้นทางที่ใช้ร่วมทุกแบบ + ต้นทางเฉพาะกลุ่ม feature
    ที่แบบนั้นใช้ + เวลาเทรน/ประเมินทั้งแถว

    Parameters
    ----------
    timing
        ผลของ :func:`timing_summary` — ใช้ ``start_to_result_seconds`` เป็นเวลาเทรนทั้งแถวของแบบ
    components
        แต่ละขั้นต้นทางเป็น dict มี ``group`` (``"shared"`` = ทุกแบบต้องผ่าน เช่น SHARP กำหนดแถวและ label,
        ``"intensity"`` หรือ ``"xray"`` = เฉพาะแบบที่ใช้ feature กลุ่มนั้น) และ ``hours``
    variant_groups
        กลุ่ม feature ที่แต่ละแบบใช้ (จาก :func:`feature_groups`) — แบบที่ไม่มี intensity ไม่จ่ายต้นทุนของ intensity

    ต้นทุน ``shared`` ถูกนับให้ทุกแบบแม้แบบนั้นไม่ใช้ SHARP เป็น feature (V4, V5) เพราะแถวข้อมูลและ label ของ
    ทุกแบบมาจาก SHARP เหมือนกัน (ดู spec: ทุกแบบใช้แถวชุดเดียวกัน)
    """
    hours = {"shared": 0.0, "intensity": 0.0, "xray": 0.0}
    for component in components:
        group = component["group"]
        if group not in hours:
            raise ValueError(f"กลุ่มต้นทาง {group!r} ไม่รู้จัก (ต้องเป็น {sorted(hours)})")
        hours[group] += float(component["hours"])

    if "variant" not in timing.columns:
        raise ValueError("ตารางเวลาต้องมีคอลัมน์ variant — ต้นทุนต้นทางผูกกับชุด feature ไม่ใช่สถาปัตยกรรม")

    rows = []
    for row in timing.to_dict("records"):
        variant = row["variant"]
        if variant not in variant_groups:
            raise ValueError(f"ไม่มีข้อมูลกลุ่ม feature ของแบบ {variant!r}")
        groups = variant_groups[variant]
        intensity = hours["intensity"] if "intensity" in groups else 0.0
        xray = hours["xray"] if "xray" in groups else 0.0
        train = row["start_to_result_seconds"] / 3600
        rows.append(
            {
                **{col: row[col] for col in unit_cols},
                "shared_hours": hours["shared"],
                "intensity_hours": intensity,
                "xray_hours": xray,
                "train_hours": train,
                "total_hours": hours["shared"] + intensity + xray + train,
            }
        )

    table = pd.DataFrame(rows)
    reference_key = (control,) if isinstance(control, str) else tuple(control)
    lookup = reference_key[0] if len(reference_key) == 1 else reference_key
    indexed = table.set_index(list(unit_cols))
    if lookup not in indexed.index:
        raise ValueError(f"ไม่มีตัวอ้างอิง {reference_key} — เทียบสัดส่วนไม่ได้")
    table["total_ratio"] = table["total_hours"] / indexed.loc[lookup, "total_hours"]
    return table[[*unit_cols, *END_TO_END_COLUMNS]]

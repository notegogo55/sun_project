"""ทดสอบ baseline แบบ persistence — ภาพสะท้อนของ label_times (ช่วง (t-24ชม., t] กับ (t, t+24ชม.])"""

from __future__ import annotations

import numpy as np

from sunseg.data.build_sequences import label_times, persistence_times


def _t(*stamps: str) -> np.ndarray:
    return np.array(stamps, dtype="datetime64[ns]")


FLARES = _t("2025-01-02T06:00")


def test_flare_in_previous_24h_predicts_positive():
    times = _t("2025-01-02T06:30", "2025-01-03T05:59", "2025-01-03T06:00", "2025-01-03T06:01")
    # 06:00 วันที่ 3 = ครบ 24 ชม. พอดี -> ยังนับ (ปลายช่วงของ lookback ตัด "พอดี t - lookback" ทิ้ง)
    assert persistence_times(times, FLARES, 24).tolist() == [1, 1, 0, 0]


def test_flare_exactly_at_issue_time_is_past_not_label():
    times = _t("2025-01-02T06:00")
    assert persistence_times(times, FLARES, 24).tolist() == [1]
    assert label_times(times, FLARES, 24).tolist() == [0]


def test_windows_meet_without_gap_or_overlap():
    # flare หนึ่งลูกต้องตกในช่วงอดีต หรือช่วงอนาคต ของเวลาออกพยากรณ์หนึ่ง ไม่ใช่ทั้งสองหรือไม่ใช่เลย
    times = _t(*[f"2025-01-0{d}T{h:02d}:00" for d in (1, 2, 3) for h in (0, 6, 12, 18)])
    past = persistence_times(times, FLARES, 24)
    future = label_times(times, FLARES, 24)
    assert not (past & future).any()
    # อดีต: t ∈ [F, F+24) · อนาคต: t ∈ [F-24, F) -> รวมกันเป็น [F-24, F+24) ต่อกันพอดีที่ F
    day = np.timedelta64(24, "h")
    near = (times >= FLARES[0] - day) & (times < FLARES[0] + day)
    assert ((past | future) == near).all()


def test_no_flares_gives_all_zero():
    times = _t("2025-01-02T06:30")
    assert persistence_times(times, None, 24).tolist() == [0]
    assert persistence_times(times, _t(), 24).tolist() == [0]

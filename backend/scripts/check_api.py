"""ตรวจว่า API ของ webapp ที่กำลังรันอยู่ตอบสนองถูกต้อง

รันเซิร์ฟเวอร์ในอีกหน้าต่างก่อน (``uvicorn app.main:app``) แล้วสั่ง::

    python backend/scripts/check_api.py

ต่างจาก ``tests/test_app_smoke.py`` ตรงที่ชุดนั้นใช้ TestClient (เรียก ASGI ตรงๆ)
ส่วนสคริปต์นี้ยิงผ่าน HTTP จริง จึงยืนยันได้ว่าเซิร์ฟเวอร์ใช้งานได้จริงจากเบราว์เซอร์
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sunseg.logging_utils import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

BASE = "http://127.0.0.1:8000"
OK, BAD, SKIP = "[ OK ]", "[FAIL]", "[SKIP]"


def get(path: str, timeout: int = 90):
    request = urllib.request.Request(f"{BASE}{path}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        return None, {"detail": str(exc)}


def main() -> int:
    print("=" * 72)
    print(f"ตรวจสอบ API ที่ {BASE}")
    print("=" * 72)

    status, health = get("/api/health")
    if status != 200:
        print(f"{BAD} เชื่อมต่อไม่ได้ — เซิร์ฟเวอร์รันอยู่หรือไม่?")
        print(f"       {health.get('detail', '')}")
        print("\n       เริ่มเซิร์ฟเวอร์ด้วย: uvicorn app.main:app")
        return 1

    print(f"\n{OK} /api/health")
    for key, label in [
        ("forecast_model", "โมเดลพยากรณ์"),
        ("segmentation_model", "โมเดล segmentation"),
        ("sequence_store", "ข้อมูล sequence"),
        ("aia_images", "ภาพ AIA สามชั้นบรรยากาศ"),
    ]:
        print(f"       {'พร้อม' if health[key] else 'ยังไม่พร้อม':<12} {label}")
    print(f"       เฟรมภาพ {health['n_frames']} เฟรม "
          f"(มีภาพ AIA {health.get('n_aia_frames', 0)} เฟรม)")

    failures = 0

    # ------------------------------------------------------------------ #
    status, info = get("/api/info")
    if status == 200:
        tss = info["forecast"].get("metrics", {}).get("test", {}).get("tss")
        print(f"\n{OK} /api/info")
        print(f"       features {info['forecast']['n_features']} ตัว, "
              f"threshold {info['forecast']['threshold']}")
        if tss is not None:
            print(f"       test TSS = {tss:+.4f}")
    else:
        print(f"\n{BAD} /api/info -> HTTP {status}")
        failures += 1

    # ------------------------------------------------------------------ #
    if not health["sequence_store"]:
        print(f"\n{SKIP} /api/harps และ /api/forecast (ยังไม่มีข้อมูล sequence)")
    else:
        status, harps = get("/api/harps?limit=5&only_flaring=true")
        if status == 200 and harps:
            print(f"\n{OK} /api/harps -> {len(harps)} รายการ")
            top = harps[0]
            print(f"       เด่นสุด: HARP {top['harpnum']} "
                  f"(NOAA {top['noaa_ar']}) มีหน้าต่าง flare {top['n_positive']} ครั้ง")

            status, series = get(f"/api/forecast?harpnum={top['harpnum']}")
            if status == 200:
                latest = series["latest"]
                print(f"\n{OK} /api/forecast?harpnum={top['harpnum']}")
                print(f"       {series['n_points']} จุดเวลา, "
                      f"ล่าสุด {latest['probability']:.1%} ({latest['risk_level']})")
                print(f"       attention {len(latest['attention'])} ค่า, "
                      f"features {len(latest['features'])} ตัว")
                if len(latest["attention"]) != 24:
                    print(f"{BAD}   คาดหวัง attention 24 ค่า แต่ได้ {len(latest['attention'])}")
                    failures += 1
            else:
                print(f"\n{BAD} /api/forecast -> HTTP {status}: {series.get('detail')}")
                failures += 1

            # ตรวจ endpoint ที่ใช้ดูภาพรวมทั้งดวงอาทิตย์ ณ เวลาหนึ่ง
            probe_time = series["times"][len(series["times"]) // 2] if status == 200 else None
            if probe_time:
                status, at_time = get(f"/api/forecast/at?time={probe_time}")
                if status == 200:
                    print(f"\n{OK} /api/forecast/at -> {len(at_time)} active region")
                    if at_time:
                        print(f"       เสี่ยงสุด: HARP {at_time[0]['harpnum']} "
                              f"{at_time[0]['probability']:.1%}")
                else:
                    print(f"\n{BAD} /api/forecast/at -> HTTP {status}")
                    failures += 1
        else:
            print(f"\n{BAD} /api/harps -> HTTP {status}")
            failures += 1

    # ------------------------------------------------------------------ #
    status, goes = get("/api/goes?start=2014-01-01&end=2014-02-01&min_class=C1.0")
    if status == 200:
        print(f"\n{OK} /api/goes -> {goes['n_events']} flare ในเดือน ม.ค. 2014")
        print("       " + "  ".join(f"{c}={goes['class_counts'][c]}" for c in "BCMX"))
    else:
        print(f"\n{BAD} /api/goes -> HTTP {status}: {goes.get('detail')}")
        failures += 1

    # ------------------------------------------------------------------ #
    status, frames = get("/api/frames")
    if status == 200:
        print(f"\n{OK} /api/frames")
    elif status == 503:
        print(f"\n{SKIP} /api/frames (ยังไม่มีเฟรมภาพ — ปกติถ้ายังไม่ได้ดาวน์โหลด)")
        frames = []
    else:
        print(f"\n{BAD} /api/frames -> HTTP {status}")
        failures += 1
        frames = []

    # ------------------------------------------------------------------ #
    # เลเยอร์ภาพ: คีย์ที่ไม่รู้จักต้องได้ 422 ส่วนเฟรมที่ยังไม่มีข้อมูลต้องถอยเป็น mag
    # พร้อมสถานะ 200 ไม่ใช่ error — สองกรณีนี้ต่างกันโดยเจตนา
    if not isinstance(frames, list) or not frames:
        print(f"\n{SKIP} /api/segment?layer=... (ยังไม่มีเฟรมภาพ)")
    else:
        stem = frames[0]
        status, _ = get(f"/api/segment?timestamp={stem}&layer=not-a-layer")
        if status == 422:
            print(f"\n{OK} /api/segment ปฏิเสธเลเยอร์ที่ไม่รู้จัก (422)")
        else:
            print(f"\n{BAD} /api/segment?layer=not-a-layer -> HTTP {status} (คาดหวัง 422)")
            failures += 1

        status, seg = get(f"/api/segment?timestamp={stem}&use_ground_truth=true&layer=171")
        if status == 200:
            names = ", ".join(
                f"{layer['key']}{'' if layer['available'] else '(ไม่มีข้อมูล)'}"
                for layer in seg.get("layers", [])
            )
            print(f"{OK} /api/segment?layer=171 -> ได้ layer={seg.get('layer')}")
            print(f"       เลเยอร์: {names}")

            first = (seg.get("detections") or [{}])[0]
            if first.get("intensities"):
                values = "  ".join(
                    f"{i['channel']}={i['mean']:.1f}" for i in first["intensities"]
                )
                print(f"       ความเข้มแสงของ AR อันดับ 1 (DN/s): {values}")
            elif health.get("aia_images"):
                print("       (เฟรมนี้ยังไม่มีภาพ AIA — รัน download_aia.py ให้ครอบคลุมเฟรมนี้)")
        elif status == 503:
            print(f"{SKIP} /api/segment?layer=171 (เฟรมแรกไม่มี mask จริงกำกับ)")
        else:
            print(f"{BAD} /api/segment?layer=171 -> HTTP {status}: {seg.get('detail')}")
            failures += 1

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    if failures:
        print(f"{BAD} พบปัญหา {failures} จุด")
    else:
        print(f"{OK} API ทำงานถูกต้องทุกจุดที่ตรวจได้")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

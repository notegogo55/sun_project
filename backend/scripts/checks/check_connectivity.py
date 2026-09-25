"""ตรวจว่าเข้าถึงแหล่งข้อมูลภายนอกที่โปรเจคต้องใช้ได้หรือไม่

โปรเจคนี้ดึงข้อมูลจากหลายโดเมน บางเครือข่าย (proxy องค์กร, firewall) เปิดบางโดเมน
แต่ปิดบางโดเมน สคริปต์นี้ช่วยแยกแยะว่าปัญหาอยู่ที่ DNS, การเชื่อมต่อ, หรือตัวโค้ด::

    python backend/scripts/checks/check_connectivity.py
"""

from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sunseg.logging_utils import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

# (โฮสต์, ใช้ทำอะไร, URL สำหรับทดสอบดึงข้อมูลจริง)
TARGETS = [
    (
        "jsoc.stanford.edu",
        "SHARP keywords + ภาพ HMI (แหล่งหลัก)",
        "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt",
    ),
    (
        "jsoc1.stanford.edu",
        "JSOC mirror สำหรับ export",
        None,
    ),
    (
        "www.lmsal.com",
        "HEK — รายการ flare (แหล่งหลักของ label)",
        "https://www.lmsal.com/hek/her?cosec=2&cmd=search&type=column&event_type=fl&"
        "event_starttime=2014-01-01T00:00:00&event_endtime=2014-01-02T00:00:00&"
        "event_coordsys=helioprojective&x1=-1200&x2=1200&y1=-1200&y2=1200&return=fl_goescls",
    ),
    (
        "www.ngdc.noaa.gov",
        "NGDC GOES XRS flare reports — แหล่ง label สำรอง (ครอบคลุม 1975-2017)",
        "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
        "solar-flares/x-rays/goes/xrs/goes-xrs-report_2014.txt",
    ),
    (
        "services.swpc.noaa.gov",
        "NOAA SWPC — GOES X-ray flux ปัจจุบัน (ใช้ในหน้า webapp)",
        "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
    ),
]

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def check_dns(host: str) -> str | None:
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return None


def check_fetch(url: str, timeout: int = 20) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "sunseg/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(400)
            return True, f"HTTP {resp.status}, {len(body)}+ ไบต์"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def main() -> int:
    print("=" * 72)
    print("ตรวจสอบการเชื่อมต่อแหล่งข้อมูลภายนอก")
    print("=" * 72)

    reachable: dict[str, bool] = {}

    for host, purpose, test_url in TARGETS:
        print(f"\n{host}")
        print(f"  ใช้ทำอะไร: {purpose}")

        ip = check_dns(host)
        if ip is None:
            print(f"  {BAD} DNS resolve ไม่ได้ — โดเมนนี้น่าจะถูกบล็อกโดยเครือข่าย/firewall")
            reachable[host] = False
            continue
        print(f"  {OK} DNS -> {ip}")

        if test_url is None:
            reachable[host] = True
            continue

        ok, detail = check_fetch(test_url)
        if ok:
            print(f"  {OK} ดึงข้อมูลได้ ({detail})")
        else:
            print(f"  {BAD} ดึงข้อมูลไม่ได้: {detail}")
        reachable[host] = ok

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("สรุป")
    print("=" * 72)

    jsoc_ok = reachable.get("jsoc.stanford.edu", False)
    hek_ok = reachable.get("www.lmsal.com", False)
    ngdc_ok = reachable.get("www.ngdc.noaa.gov", False)

    if jsoc_ok:
        print(f"{OK} JSOC ใช้ได้ — ดึง SHARP parameters และภาพ HMI ได้")
    else:
        print(f"{BAD} JSOC เข้าไม่ถึง — โปรเจคนี้ทำงานไม่ได้เลยหากขาด JSOC")

    if hek_ok and ngdc_ok:
        print(f"{OK} เข้าถึงแหล่ง flare ได้ทั้ง HEK และ NGDC")
        print("       -> ใช้ค่าเริ่มต้น --flare-source auto ได้เลย")
    elif ngdc_ok:
        print(f"{WARN} HEK เข้าไม่ถึง แต่ NGDC ใช้ได้ (ครอบคลุม 1975-2017 ซึ่งพอสำหรับโปรเจคนี้)")
        print("       -> --flare-source auto จะเลือก NGDC ให้อัตโนมัติ")
    elif hek_ok:
        print(f"{WARN} NGDC เข้าไม่ถึง แต่ HEK ใช้ได้")
    else:
        print(f"{BAD} เข้าไม่ถึงทั้ง HEK และ NGDC — สร้าง label ของ flare ไม่ได้")

    return 0 if jsoc_ok and (hek_ok or ngdc_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

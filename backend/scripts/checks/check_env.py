"""ตรวจสอบว่า environment พร้อมใช้งาน: config โหลดได้, ไลบรารีครบ, CUDA ใช้ได้

รันก่อนเริ่มดาวน์โหลดข้อมูล เพื่อจับปัญหาการติดตั้งตั้งแต่เนิ่นๆ::

    python backend/scripts/checks/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sunseg.logging_utils import force_utf8_stdio  # noqa: E402

force_utf8_stdio()

OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"


def main() -> int:
    failures = 0

    print("=" * 68)
    print("ตรวจสอบ environment ของ sunseg")
    print("=" * 68)

    # ------------------------------------------------------------------ #
    print("\n-- Python --")
    v = sys.version_info
    if (v.major, v.minor) == (3, 12):
        print(f"{OK} Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"{BAD} Python {v.major}.{v.minor} — โปรเจคนี้ต้องใช้ 3.12 (PyTorch ยังไม่รองรับ 3.13+)")
        failures += 1

    # ------------------------------------------------------------------ #
    print("\n-- ไลบรารี --")
    for module, label in [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("drms", "drms (JSOC)"),
        ("sunpy", "SunPy"),
        ("astropy", "Astropy"),
        ("cv2", "OpenCV"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("fastapi", "FastAPI"),
    ]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            print(f"{OK} {label:<16} {version}")
        except ImportError as exc:
            print(f"{BAD} {label:<16} import ไม่ได้: {exc}")
            failures += 1

    # ------------------------------------------------------------------ #
    print("\n-- GPU --")
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cap = torch.cuda.get_device_capability(0)
            print(f"{OK} {name} — VRAM {vram:.1f} GB, compute capability {cap[0]}.{cap[1]}")
            if vram < 6:
                print(f"{WARN} VRAM น้อยกว่า 6 GB — config ถูกปรับมาให้พอดีแล้ว (batch เล็ก + AMP)")
        else:
            print(f"{WARN} ไม่พบ CUDA — เทรนได้บน CPU แต่ U-Net จะช้ามาก")
    except ImportError:
        print(f"{BAD} import torch ไม่ได้")
        failures += 1

    # ------------------------------------------------------------------ #
    print("\n-- ไฟล์ config --")
    try:
        from sunseg.config import (
            load_data_config,
            load_forecast_config,
            load_study_architectures,
            load_tracking_config,
            load_unet_config,
        )

        data = load_data_config()
        print(f"{OK} data.yaml     — {data.sharp.n_features} features, "
              f"{data.time_range.start} ถึง {data.time_range.end}")
        print(f"       ขอ keyword จาก JSOC ทั้งหมด {len(data.sharp.all_keys)} ตัว")

        unet = load_unet_config()
        print(f"{OK} unet.yaml     — {unet.data.image_size}px, "
              f"base_channels={unet.model.base_channels}, batch={unet.train.batch_size}")

        forecast = load_forecast_config()
        print(f"{OK} forecast.yaml — โมเดล {', '.join(forecast.names)} (ปริยาย {forecast.default_model}), "
              f"loss={forecast.loss.type}")
        for name in forecast.names:
            forecast.resolve(name)  # ค่าทับรายโมเดลต้องรวมกับค่าร่วมได้

        study = load_study_architectures()
        print(f"{OK} study/architectures.yaml — {len(study.architectures)} แถว, "
              f"แบบในตารางหลัก {', '.join(study.default_variants)}")

        track = load_tracking_config()
        print(f"{OK} tracking.yaml — Snodgrass A={track.rotation.snodgrass_A} deg/day")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} โหลด config ไม่สำเร็จ: {exc}")
        failures += 1
        data = None

    # ------------------------------------------------------------------ #
    print("\n-- อีเมล JSOC --")
    if data is not None:
        try:
            print(f"{OK} SUNSEG_JSOC_EMAIL = {data.jsoc.email}")
        except RuntimeError as exc:
            print(f"{WARN} {exc}")
            print("       (query keyword ทำได้โดยไม่ต้องมีอีเมล — จำเป็นเฉพาะตอนดาวน์โหลดภาพ)")

    # ------------------------------------------------------------------ #
    print("\n-- ไดเรกทอรีข้อมูล --")
    if data is not None:
        data.paths.mkdirs()
        for label, path in data.paths.model_dump().items():
            print(f"{OK} {label:<10} {path}")

    print("\n" + "=" * 68)
    if failures:
        print(f"{BAD} พบปัญหา {failures} รายการ — แก้ก่อนดำเนินการต่อ")
    else:
        print(f"{OK} environment พร้อมใช้งาน")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

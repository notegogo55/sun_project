"""ตัวช่วยที่ทุกไฟล์ทดสอบใช้ร่วมกัน"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(relative_path: str) -> ModuleType:
    """import สคริปต์ใต้ ``backend/scripts/`` ด้วย path เช่น ``"study/tune.py"``

    สคริปต์แยกอยู่ในโฟลเดอร์ย่อยและหลายตัวชื่อซ้ำกัน (``forecast/train.py``,
    ``study/train.py``) จึง import ด้วยชื่อโมดูลไม่ได้ — ตั้งชื่อโมดูลจาก path แทน
    (``scripts_study_tune``) แล้ว cache ไว้ใน ``sys.modules`` เหมือน import ปกติ
    """
    path = SCRIPTS_DIR / relative_path
    name = "scripts_" + "_".join(Path(relative_path).with_suffix("").parts)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

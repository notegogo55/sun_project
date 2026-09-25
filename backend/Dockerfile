# Dockerfile สำหรับ backend เท่านั้น (API — ไม่เสิร์ฟไฟล์ frontend อีกต่อไป)
#
# ใช้ PyTorch เวอร์ชัน CPU เพราะการ inference ของโปรเจคนี้เบามาก:
# โมเดลพยากรณ์ทั้งสี่ตัวมีเพียงหลักพันถึงหมื่นพารามิเตอร์ และ U-Net รันภาพ 512x512 ทีละภาพ
# การใส่ CUDA runtime เข้ามาจะทำให้ image ใหญ่ขึ้นราว 2 GB โดยแทบไม่ได้ประโยชน์
#
# การเทรนยังคงทำบนเครื่องที่มี GPU ตามขั้นตอนใน README
#
# build จาก root ของ repo เสมอ (ต้องการ README.md ที่อยู่นอก backend/):
#     docker build -f backend/Dockerfile -t sunseg-backend .

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1

# OpenCV ต้องการไลบรารีระบบเหล่านี้แม้ในเวอร์ชัน headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# WORKDIR คือ /app/backend เพื่อให้ path resolution ใน sunseg/config.py และ app/main.py
# (ซึ่งคำนวณ PROJECT_ROOT จาก parent ของ backend/) ตรงกับตำแหน่ง mount ของ data/, artifacts/ ที่ /app
WORKDIR /app/backend

# ติดตั้ง PyTorch เวอร์ชัน CPU ก่อน เพื่อไม่ให้ pip ดึงเวอร์ชัน CUDA มาให้
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# คัดลอกเฉพาะไฟล์ที่ระบุ dependency ก่อน เพื่อให้ layer cache ทำงานได้ดี
# README.md อยู่นอก backend/ (pyproject.toml อ้างอิงเป็น ../README.md) — hatchling
# รุ่นใหม่ๆ ปฏิเสธ readme path ที่ชี้ออกนอก project directory แม้ไฟล์จะมีอยู่จริงก็ตาม
# จึงต้อง pin ไว้ที่รุ่นที่ยังไม่บังคับกฎนี้ (ยืนยันแล้วว่า build ผ่านที่ 1.18.0)
COPY README.md /app/README.md
COPY backend/pyproject.toml ./
COPY backend/src/ ./src/
RUN pip install "hatchling==1.18.0" && pip install --no-build-isolation .

COPY backend/app/ ./app/
COPY backend/configs/ ./configs/

# artifacts (โมเดลที่เทรนแล้ว) และ data ถูก mount เข้ามาตอนรัน ไม่ได้ฝังใน image
VOLUME ["/app/artifacts", "/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

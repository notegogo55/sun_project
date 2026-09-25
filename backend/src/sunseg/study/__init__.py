"""งานเปรียบเทียบ (สถาปัตยกรรม × ชุด feature) — ตรรกะที่สคริปต์ใน ``scripts/study/`` ใช้

- :mod:`.summary` — สรุปผลแบบจับคู่ seed: Δ ตามแบบ, Δ ตามสถาปัตยกรรม, interaction, เวลา
- :mod:`.acquisition_time` — ประเมินเวลาดาวน์โหลด/สกัด feature ต้นทางจาก log

dataset ของงานนี้อยู่ที่ ``sunseg.data.study_dataset`` (สร้าง) และ ``sunseg.datasets.study``
(โหลดเป็น split ของแต่ละแบบ) ส่วนลูปเทรนใช้ตัวเดียวกับ production ที่ ``sunseg.training.forecast``
"""

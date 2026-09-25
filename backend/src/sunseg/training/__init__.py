"""ส่วนประกอบการเทรนที่ใช้ร่วมกันทุกโมเดล

- :mod:`.forecast` — ลูปเทรนโมเดลพยากรณ์ (ทุกสถาปัตยกรรม) + logistic baseline
- :mod:`.losses` — focal loss (พยากรณ์) และ BCE+Dice / Tversky (segmentation)
- :mod:`.utils` — seed, device, scheduler, early stopping, การบันทึก checkpoint/metrics
"""

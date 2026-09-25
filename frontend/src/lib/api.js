/** ตัวช่วยเรียก backend — คืน JSON เมื่อสำเร็จ, throw Error ที่มีข้อความอ่านได้เมื่อไม่สำเร็จ
 *
 *  `.catch(() => ({}))` กัน response ที่ไม่ใช่ JSON เลย (เช่น 502 จาก proxy) ไม่ให้ throw
 *  ซ้อน throw จนข้อความ error ที่แท้จริงหายไป — ตกลงมาที่ payload ว่างแล้วใช้ HTTP status
 *  แทน ส่วน `payload.detail` คือรูปแบบ error message ของ FastAPI (`HTTPException(detail=...)`) */
export async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `คำขอล้มเหลว (HTTP ${response.status})`);
  }
  return payload;
}

/** ต่อท้าย `&model=<ชื่อ>` เมื่อระบุ — ไม่ระบุคือให้ backend ใช้โมเดลปริยาย (configs/forecast.yaml) */
const modelParam = (model) => (model ? `&model=${encodeURIComponent(model)}` : "");

export const api = {
  health: () => fetchJson("/api/health"),
  info: () => fetchJson("/api/info"),
  harps: () => fetchJson("/api/harps?limit=3000&only_flaring=false"),
  /** โมเดลพยากรณ์ทุกตัว (LSTM/TCN/Transformer/DA-RNN) พร้อมความพร้อมและผลบน test */
  forecastModels: () => fetchJson("/api/forecast/models"),
  forecast: (harpnum, model) =>
    fetchJson(`/api/forecast?harpnum=${harpnum}&limit=2000${modelParam(model)}`),
  forecastAt: (isoDate, model) =>
    fetchJson(`/api/forecast/at?time=${isoDate}T00:00:00&tolerance_hours=24${modelParam(model)}`),
  /** โมเดลหลัก (LSTM + V3) แยกระดับ <M/M/X — สรุป + ผล val/test ทุกจุดทำงาน (ตอบ 200 เสมอ พร้อม available/hint) */
  classForecastSummary: () => fetchJson("/api/class-forecast/summary"),
  classForecast: (harpnum, mode) =>
    fetchJson(`/api/class-forecast?harpnum=${harpnum}&mode=${encodeURIComponent(mode)}`),
  goes: (start, end, minClass) =>
    fetchJson(`/api/goes?start=${start}&end=${end}&min_class=${minClass}`),
  xray: (start, end) => fetchJson(`/api/xray?start=${start}&end=${end}`),
  proton: (peakTime, goesClass) => {
    const query = new URLSearchParams({ t: peakTime });
    if (goesClass) query.set("goes_class", goesClass);
    return fetchJson(`/api/proton?${query}`);
  },
  frames: () => fetchJson("/api/frames?limit=5000"),
  flarePositions: () => fetchJson("/api/flare-positions"),
  segment: (timestamp, useGroundTruth, layer) =>
    fetchJson(`/api/segment?timestamp=${timestamp}&use_ground_truth=${useGroundTruth}&layer=${layer}`),
  intensitySeries: (start, end, useGroundTruth) =>
    fetchJson(`/api/intensity-series?start=${start}&end=${end}&use_ground_truth=${useGroundTruth}&top=6`),
  confusionMatrix: (model, split) =>
    fetchJson(`/api/forecast/confusion-matrix?model=${model}&split=${split}`),
  confusionSamples: (model, split, threshold, cell) =>
    fetchJson(
      `/api/forecast/confusion-matrix/samples?model=${model}&split=${split}` +
      `&threshold=${threshold}&cell=${cell}&limit=200`,
    ),
};

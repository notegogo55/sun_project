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

export const api = {
  health: () => fetchJson("/api/health"),
  info: () => fetchJson("/api/info"),
  harps: () => fetchJson("/api/harps?limit=3000&only_flaring=false"),
  forecast: (harpnum) => fetchJson(`/api/forecast?harpnum=${harpnum}&limit=2000`),
  forecastAt: (isoDate) =>
    fetchJson(`/api/forecast/at?time=${isoDate}T00:00:00&tolerance_hours=24`),
  goes: (start, end, minClass) =>
    fetchJson(`/api/goes?start=${start}&end=${end}&min_class=${minClass}`),
  xray: (start, end) => fetchJson(`/api/xray?start=${start}&end=${end}`),
  proton: (peakTime, goesClass) => {
    const query = new URLSearchParams({ t: peakTime });
    if (goesClass) query.set("goes_class", goesClass);
    return fetchJson(`/api/proton?${query}`);
  },
  frames: () => fetchJson("/api/frames?limit=5000"),
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

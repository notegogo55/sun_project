import { useEffect, useMemo, useRef } from "react";
import { useApp } from "../../state/AppContext.js";
import { THEME } from "../../lib/theme.js";
import { basePlotLayout, PLOT_CONFIG } from "../../lib/plotConstants.js";
import { formatDateTime, formatHeliographic, markerSize, seqColor } from "../../lib/format.js";
import { scrollToDashboard } from "../../lib/motion.js";

function projectToDisk(lat, lon) {
  const toRad = Math.PI / 180;
  return { x: Math.cos(lat * toRad) * Math.sin(lon * toRad), y: Math.sin(lat * toRad) };
}

/** เส้นกริด heliographic บนจาน — วาดเป็น trace เส้นบางที่ hover ไม่ได้ */
function diskGraticule() {
  const traces = [];
  const line = { color: THEME.gridline, width: 1 };
  for (const lat of [-60, -30, 0, 30, 60]) {
    const points = [];
    for (let lon = -90; lon <= 90; lon += 5) points.push(projectToDisk(lat, lon));
    traces.push({
      x: points.map((p) => p.x), y: points.map((p) => p.y),
      type: "scatter", mode: "lines", hoverinfo: "skip",
      line: lat === 0 ? { ...line, color: THEME.baseline } : line,
    });
  }
  for (const lon of [-60, -30, 0, 30, 60]) {
    const points = [];
    for (let lat = -90; lat <= 90; lat += 5) points.push(projectToDisk(lat, lon));
    traces.push({
      x: points.map((p) => p.x), y: points.map((p) => p.y),
      type: "scatter", mode: "lines", hoverinfo: "skip",
      line: lon === 0 ? { ...line, color: THEME.baseline } : line,
    });
  }
  return traces;
}

/** จานสุริยะใน hero — วาด/ผูก event ของ Plotly เองแบบ imperative เหมือนต้นฉบับ
 *  (renderFlareDisk เดิม) เพราะต้องคุม restyle/animate ของวงแหวน hover/highlight
 *  และ tooltip ที่วาดเองแบบละเอียด ซึ่งไม่เหมาะจะไปวิ่งผ่าน re-render ของ React ทุกครั้ง */
export default function DiskChart({ located, diskClasses }) {
  const { health, pinnedFlare, chartedFlare, selectFlareEvent, previewFlareEvent, clearProtonPreview, selectHarp } = useApp();
  const nodeRef = useRef(null);
  const tipRef = useRef(null);
  const pinnedRef = useRef(pinnedFlare);
  pinnedRef.current = pinnedFlare;
  const healthRef = useRef(health);
  healthRef.current = health;

  const shown = useMemo(
    () => located.filter((e) => diskClasses.has(e.goes_class[0])),
    [located, diskClasses],
  );
  const projected = useMemo(() => shown.map((e) => projectToDisk(e.lat, e.lon)), [shown]);

  function hideDiskTip() {
    const tip = tipRef.current;
    if (!tip) return;
    tip.classList.remove("is-on");
    tip.setAttribute("aria-hidden", "true");
  }

  function showDiskTip(flare, bbox) {
    const tip = tipRef.current;
    const host = nodeRef.current;
    if (!tip || !host) return;
    const rows = [
      ["เวลาที่พีค", `${formatDateTime(flare.peak_time)} UT`],
      ["flux 1-8 Å", `${flare.peak_flux.toExponential(2)} W/m²`],
      ["ตำแหน่ง", formatHeliographic(flare.lat, flare.lon)],
      ["lat / lon", `${flare.lat.toFixed(1)}° / ${flare.lon.toFixed(1)}°`],
      ["active region", flare.noaa_ar ? `AR ${flare.noaa_ar}` : "—"],
      ["HARP", flare.harpnum ? String(flare.harpnum) : "ไม่มีคู่"],
    ];
    const actions = [];
    if (pinnedRef.current?.peak_time === flare.peak_time) {
      actions.push("คลิกซ้ำเพื่อปลดตรึง");
    } else {
      if (healthRef.current?.proton_flux) actions.push("คลิกเพื่อตรึงกราฟโปรตอน");
      if (flare.harpnum) actions.push("เปิด HARP นี้ใน dashboard");
    }

    tip.innerHTML =
      `<div class="disktip__head">`
      + `<span class="disktip__dot" style="background:${seqColor(flare.goes_class)}"></span>`
      + `${flare.goes_class}</div>`
      + `<table><tbody>`
      + rows.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join("")
      + `</tbody></table>`
      + (actions.length ? `<p class="disktip__foot">${actions.join(" · ")}</p>` : "");

    tip.classList.add("is-on");
    tip.setAttribute("aria-hidden", "false");

    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    let left = bbox.x1 + 14;
    if (left + width > host.clientWidth) left = bbox.x0 - width - 14;
    const top = (bbox.y0 + bbox.y1) / 2 - height / 2;
    tip.style.left = `${Math.max(0, Math.min(host.clientWidth - width, left))}px`;
    tip.style.top = `${Math.max(0, Math.min(host.clientHeight - height, top))}px`;
  }

  useEffect(() => {
    const node = nodeRef.current;
    if (!node) return;
    if (!located.length) {
      window.Plotly.purge(node);
      hideDiskTip();
      return;
    }

    const flares = {
      x: projected.map((p) => p.x),
      y: projected.map((p) => p.y),
      customdata: shown.map((_, index) => index),
      type: "scatter",
      mode: "markers",
      marker: {
        size: shown.map((e) => markerSize(e.peak_flux)),
        color: shown.map((e) => seqColor(e.goes_class)),
        line: { width: 2, color: THEME.surface1 },
      },
      hoverinfo: "none",
    };
    const hoverRing = {
      x: [], y: [],
      type: "scatter", mode: "markers",
      marker: { size: 30, color: "rgba(0,0,0,0)", line: { width: 1.4, color: THEME.textSecondary } },
      hoverinfo: "skip",
    };
    const highlight = {
      x: [], y: [],
      type: "scatter", mode: "markers",
      marker: { size: 22, color: "rgba(0,0,0,0)", line: { width: 2, color: THEME.brand } },
      hoverinfo: "skip",
    };

    const axis = {
      range: [-1.12, 1.12],
      showgrid: false, zeroline: false, showticklabels: false,
      ticks: "", fixedrange: true,
    };

    const traces = [...diskGraticule(), flares, hoverRing, highlight];
    const hoverIndex = traces.length - 2;

    window.Plotly.react(node, traces, {
      ...basePlotLayout(),
      margin: { l: 8, r: 8, t: 8, b: 8 },
      hovermode: "closest",
      xaxis: { ...axis },
      yaxis: { ...axis, scaleanchor: "x", scaleratio: 1 },
      shapes: [{
        type: "circle", xref: "x", yref: "y",
        x0: -1, y0: -1, x1: 1, y1: 1,
        line: { color: "rgba(255, 176, 84, 0.45)", width: 1.5 },
        fillcolor: "rgba(255, 176, 84, 0.05)",
        layer: "below",
      }],
      annotations: [
        { xref: "x", yref: "y", x: -1.03, y: 0, text: "E", showarrow: false, font: { size: 10, color: THEME.textMuted }, xanchor: "right" },
        { xref: "x", yref: "y", x: 1.03, y: 0, text: "W", showarrow: false, font: { size: 10, color: THEME.textMuted }, xanchor: "left" },
        { xref: "x", yref: "y", x: 0, y: 1.03, text: "N", showarrow: false, font: { size: 10, color: THEME.textMuted }, yanchor: "bottom" },
      ],
    }, PLOT_CONFIG);

    node.removeAllListeners?.("plotly_hover");
    node.removeAllListeners?.("plotly_unhover");
    node.removeAllListeners?.("plotly_click");

    node.on("plotly_hover", (hover) => {
      const point = hover.points?.[0];
      if (typeof point?.customdata !== "number") return;
      window.Plotly.restyle(node, { x: [[projected[point.customdata].x]], y: [[projected[point.customdata].y]] }, [hoverIndex]);
      showDiskTip(shown[point.customdata], point.bbox);
      previewFlareEvent(shown[point.customdata]);
    });
    node.on("plotly_unhover", () => {
      clearProtonPreview();
      hideDiskTip();
      window.Plotly.restyle(node, { x: [[]], y: [[]] }, [hoverIndex]);
    });
    node.on("plotly_click", (click) => {
      const point = click.points?.[0];
      if (typeof point?.customdata !== "number") return;
      const flare = shown[point.customdata];
      if (!flare.harpnum && !healthRef.current?.proton_flux) return;
      selectFlareEvent(flare);
      showDiskTip(flare, point.bbox);
      if (flare.harpnum) {
        selectHarp(Number(flare.harpnum));
        scrollToDashboard();
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [located, shown, projected]);

  // เลื่อนวงแหวนไฮไลต์ไปยัง flare ที่กราฟโปรตอนกำลังแสดงอยู่ทุกครั้งที่เปลี่ยน
  useEffect(() => {
    const node = nodeRef.current;
    if (!node?.data) return;
    const highlightIndex = node.data.length - 1;
    const match = shown.findIndex((e) => e.peak_time === chartedFlare?.peak_time);
    const [x, y] = match < 0 ? [[], []] : [[projected[match].x], [projected[match].y]];
    window.Plotly.animate(node, { data: [{ x, y }], traces: [highlightIndex] }, {
      transition: { duration: 260, easing: "cubic-in-out" },
      frame: { duration: 260, redraw: false },
    });
  }, [chartedFlare, shown, projected]);

  return (
    <div className="disk">
      <div id="diskChart" className="chart chart--disk" ref={nodeRef}>
        {!located.length && (
          <p className="empty">flare ในช่วงนี้ไม่มีพิกัดยืนยัน — NOAA ระบุตำแหน่งให้ราว 2 ใน 3 ของรายการเท่านั้น</p>
        )}
      </div>
      <div id="diskTip" className="disktip" role="tooltip" aria-hidden="true" ref={tipRef} />
    </div>
  );
}

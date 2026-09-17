import { useMemo, useState } from "react";
import { useApp } from "../../state/AppContext.js";
import { useReveal } from "../../hooks/useReveal.js";

const COLUMNS = [
  { key: "harpnum", label: "HARP", num: false },
  { key: "noaa_ar", label: "NOAA", num: false },
  { key: "n_samples", label: "ตัวอย่าง", num: true },
  { key: "n_positive", label: "flare", num: true },
];

export default function HarpListCard() {
  const { harps, harpsError, health, selectedHarp, selectHarp } = useApp();
  const revealRef = useReveal();
  const [onlyFlaring, setOnlyFlaring] = useState(true);
  const [query, setQuery] = useState("");
  const [sort, setSortState] = useState({ key: "n_positive", dir: "desc" });

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = harps.filter((h) => {
      if (onlyFlaring && !h.n_positive) return false;
      if (!q) return true;
      return String(h.harpnum).includes(q) || (h.noaa_ar && String(h.noaa_ar).includes(q));
    });
    const { key, dir } = sort;
    const sign = dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      // HARP ที่ยังไม่มีเลข NOAA ให้ตกไปท้ายตารางเสมอ ไม่ว่าจะเรียงทางไหน
      const left = a[key] ?? -Infinity;
      const right = b[key] ?? -Infinity;
      return left === right ? a.harpnum - b.harpnum : (left < right ? -sign : sign);
    });
  }, [harps, onlyFlaring, query, sort]);

  function handleSort(key) {
    setSortState((current) => (
      key === current.key
        ? { key, dir: current.dir === "desc" ? "asc" : "desc" }
        : { key, dir: key === "harpnum" || key === "noaa_ar" ? "asc" : "desc" }
    ));
  }

  function handleRowKeyDown(event, harpnum, rowIndex) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectHarp(harpnum, { syncRange: true });
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const target = event.currentTarget.parentElement?.children[rowIndex + (event.key === "ArrowDown" ? 1 : -1)];
      target?.focus();
    }
  }

  let body;
  if (!health?.sequence_store) {
    body = <tr><td colSpan={4} className="empty">ยังไม่มีข้อมูล sequence</td></tr>;
  } else if (harpsError) {
    body = <tr><td colSpan={4} className="empty">{harpsError}</td></tr>;
  } else if (!harps.length) {
    body = <tr><td colSpan={4} className="empty">กำลังโหลด…</td></tr>;
  } else if (!rows.length) {
    body = <tr><td colSpan={4} className="empty">ไม่พบ active region ที่ตรงกับคำค้น</td></tr>;
  } else {
    body = rows.map((h, index) => (
      <tr
        key={h.harpnum}
        tabIndex={0}
        className={h.harpnum === selectedHarp ? "selected" : ""}
        onClick={() => selectHarp(h.harpnum, { syncRange: true })}
        onKeyDown={(e) => handleRowKeyDown(e, h.harpnum, index)}
      >
        <td>{h.harpnum}</td>
        <td>{h.noaa_ar ?? "—"}</td>
        <td className="num">{h.n_samples.toLocaleString()}</td>
        <td className={`num ${h.n_positive ? "flare-count" : ""}`}>{h.n_positive.toLocaleString()}</td>
      </tr>
    ));
  }

  return (
    <section ref={revealRef} id="harps" className="card card--list" aria-labelledby="h-list" data-reveal>
      <div className="card__head">
        <h2 id="h-list">Active Regions</h2>
        <label className="switch">
          <input type="checkbox" checked={onlyFlaring} onChange={(e) => setOnlyFlaring(e.target.checked)} />
          <span>เฉพาะที่เคยเกิด flare</span>
        </label>
      </div>
      <input
        type="search"
        className="search"
        placeholder="ค้นหาหมายเลข HARP หรือ NOAA AR…"
        aria-label="ค้นหา active region"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="table-wrap">
        <table className="harp-table">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th key={col.key} className={col.num ? "num" : undefined}>
                  <button
                    type="button"
                    className="sort"
                    data-active={sort.key === col.key ? "" : undefined}
                    data-dir={sort.key === col.key ? sort.dir : undefined}
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody id="harpList">{body}</tbody>
        </table>
      </div>
      <p className="hint" id="harpCount">
        {harps.length ? `${rows.length.toLocaleString()} จาก ${harps.length.toLocaleString()} active region` : ""}
      </p>
    </section>
  );
}

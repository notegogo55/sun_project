/** หัวข้อสองแบบของธีม — ใช้ซ้ำทุกหน้าให้จังหวะตัวอักษรตรงกันเสมอ */

/** หัวหน้า (แบบ CURRENT CONDITIONS ของต้นแบบ): kicker mono + ชื่อ Orbitron + คำอธิบาย */
export function PageHeader({ kicker, title, sub, children }) {
  return (
    <header className="page-header">
      <div>
        {kicker && <p className="page-header__kicker">{kicker}</p>}
        <h1 className="page-header__title">{title}</h1>
        {sub && <p className="page-header__sub">{sub}</p>}
      </div>
      {children && <div className="page-header__aside">{children}</div>}
    </header>
  );
}

/** หัว section "// SECTION_01 / ..." ตามด้วยหัวข้อใหญ่ (ใส่ <em> เพื่อเน้นคำเป็นสีน้ำเงิน) */
export function SectionHead({ index, label, title, lead }) {
  return (
    <header className="section-head">
      <p className="section-head__kicker">// SECTION_{index} / {label}</p>
      <h2 className="section-head__title">{title}</h2>
      {lead && <p className="section-head__lead">{lead}</p>}
    </header>
  );
}

/** แถบตัวเลขสรุปมีขอบซ้ายเป็นสี — `tone`: undefined (ฟ้า) | "green" | "violet" | "sun" */
export function StatTile({ label, value, unit, tone, title }) {
  return (
    <div className={`stat-tile${tone ? ` stat-tile--${tone}` : ""}`} title={title}>
      <p className="stat-tile__label">{label}</p>
      <p className="stat-tile__row">
        <span className="stat-tile__value">{value}</span>
        {unit && <span className="stat-tile__unit">{unit}</span>}
      </p>
    </div>
  );
}

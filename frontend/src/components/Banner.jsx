import { useApp } from "../state/AppContext.js";

export default function Banner() {
  const { bootError, missingSetup } = useApp();

  if (bootError) {
    return <div className="banner" role="status">{bootError}</div>;
  }
  if (!missingSetup.length) return null;

  return (
    <div className="banner" role="status">
      ยังใช้งานได้ไม่ครบทุกส่วน — รันขั้นตอนต่อไปนี้เพื่อเปิดใช้:{" "}
      {missingSetup.map((commands, index) => (
        <span key={commands[0]}>
          {index > 0 ? " · " : ""}
          {commands.map((cmd, i) => (
            <span key={cmd}>
              {i > 0 ? " แล้ว " : ""}
              <code>{cmd}</code>
            </span>
          ))}
        </span>
      ))}
    </div>
  );
}

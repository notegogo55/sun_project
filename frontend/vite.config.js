import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// frontend เป็น container/origin ของตัวเอง (เสิร์ฟด้วย nginx) แยกจาก backend เต็มตัว
// จึง build ที่ root ตรงๆ ไม่ต้องมี prefix แบบตอนที่ยังฝากอยู่ใต้ FastAPI /static
export default defineConfig({
  base: "/",
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/docs": "http://localhost:8000",
      "/openapi.json": "http://localhost:8000",
    },
  },
});

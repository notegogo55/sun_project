/** sunseg — Solar Flare Forecasting Hub
 *
 * โครงแบบหลายหน้าตามธีม space-weather-monitoring: navbar ติดบนสุดคงที่ แล้วเปลี่ยนเนื้อหา
 * ตามเส้นทาง
 *   /           HUB — แผนที่ตำแหน่ง flare ทั้งคลัง (แนวงาน PositionFlare) + ตาราง flare, การ์ด pipeline
 *   /dashboard  แผงข้อมูลเต็ม (ภาพ, ฟลักซ์, พยากรณ์) — ลิงก์ย่อยด้วย hash เช่น #xray
 *   /model      ผลการทดลอง + confusion matrix
 *   /about      วิธีการและแหล่งข้อมูล
 *
 * AppProvider ครอบทุกหน้า state (ช่วงเวลา, HARP ที่เลือก, ผล GOES) จึงคงอยู่ข้ามการเปลี่ยน
 * หน้า — เลือก flare บนจานใน HUB แล้วไป dashboard ก็เห็น HARP เดียวกันทันที
 *
 * ทุกส่วนออกแบบให้ degrade ได้: ถ้าโมเดลใดยังไม่ถูกเทรน แผงนั้นจะบอกวิธีสร้าง
 * (ดู Banner.jsx และ health flags ใน AppProvider) แทนที่จะพังทั้งหน้า
 */
import { Navigate, Route, Routes } from "react-router-dom";
import { AppProvider } from "./state/AppProvider.jsx";
import Navbar from "./components/layout/Navbar.jsx";
import Footer from "./components/layout/Footer.jsx";
import ScrollManager from "./components/layout/ScrollManager.jsx";
import Banner from "./components/Banner.jsx";
import HubPage from "./pages/HubPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import ModelPage from "./pages/ModelPage.jsx";
import AboutPage from "./pages/AboutPage.jsx";

export default function App() {
  return (
    <AppProvider>
      <ScrollManager />
      <Navbar />
      <main className="app-main">
        <Banner />
        <Routes>
          <Route path="/" element={<HubPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </AppProvider>
  );
}

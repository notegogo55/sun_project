/** sunseg dashboard
 *
 * หน้าเว็บแบ่งเป็นสามฉาก: hero (ตัวเลขสรุป + จานสุริยะ) → dashboard (แผงข้อมูล)
 * → about (วิธีการและผลการทดลอง) ทั้งสามฉากดึงจาก API ชุดเดียวกันผ่าน AppProvider
 *
 * ทุกส่วนออกแบบให้ degrade ได้: ถ้าโมเดลใดยังไม่ถูกเทรน แผงนั้นจะบอกวิธีสร้าง
 * (ดู Banner.jsx และ health flags ใน AppProvider) แทนที่จะพังทั้งหน้า
 */
import { AppProvider } from "./state/AppProvider.jsx";
import { useChrome } from "./hooks/useChrome.js";
import Topbar from "./components/Topbar.jsx";
import Banner from "./components/Banner.jsx";
import Hero from "./components/Hero/Hero.jsx";
import Dashboard from "./components/Dashboard/Dashboard.jsx";
import AboutSection from "./components/About/AboutSection.jsx";
import Footer from "./components/Footer.jsx";

function Page() {
  useChrome();
  return (
    <>
      <Topbar />
      <Banner />
      <Hero />
      <Dashboard />
      <AboutSection />
      <Footer />
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Page />
    </AppProvider>
  );
}

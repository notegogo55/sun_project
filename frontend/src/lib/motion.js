export const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function scrollToDashboard() {
  document.getElementById("dashboard")?.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth" });
}

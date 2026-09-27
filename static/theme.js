const themeToggle = document.getElementById("theme-toggle");
const themeLabel = document.getElementById("theme-label");
const themeColor = document.querySelector('meta[name="theme-color"]');

function setTheme(theme) {
  const light = theme === "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  themeToggle.setAttribute("aria-pressed", String(light));
  themeToggle.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode");
  themeToggle.title = light ? "Switch to dark mode" : "Switch to light mode";
  themeLabel.textContent = light ? "Dark" : "Light";
  themeColor.content = light ? "#edf1f7" : "#030615";
  try {
    localStorage.setItem("sonolab-theme", light ? "light" : "dark");
  } catch (error) {
    // Keep the current page usable when storage is unavailable.
  }
}

setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
});

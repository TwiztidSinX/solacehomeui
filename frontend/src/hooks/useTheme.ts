import { useEffect, useState } from "react";

const DEFAULT_THEME = {
  primaryColor: "#4a90e2",
  backgroundColor: "#1a1a2e",
  chatBackgroundColor: "#22223b",
  chatInputBackgroundColor: "#333355",
  userMessageColor: "#4a236b",
  aiMessageColor: "#333355",
  textColor: "#e0e0e0",
};

export type Theme = typeof DEFAULT_THEME;

export const useTheme = () => {
  const [theme, setTheme] = useState<Theme>(DEFAULT_THEME);

  useEffect(() => {
    const savedTheme = localStorage.getItem("nova_theme");
    if (savedTheme) {
      setTheme(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("nova_theme", JSON.stringify(theme));
    Object.keys(theme).forEach((key) => {
      document.documentElement.style.setProperty(
        `--${key.replace(/([A-Z])/g, "-$1").toLowerCase()}`,
        (theme as any)[key],
      );
    });
  }, [theme]);

  const handleThemeChange = (property: string, value: string) => {
    setTheme((prevTheme) => ({ ...prevTheme, [property]: value }));
  };

  return { theme, setTheme, handleThemeChange };
};

export default useTheme;

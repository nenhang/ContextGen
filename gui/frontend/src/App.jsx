// frontend/src/App.jsx

import React from "react";
import { ThemeProvider, CssBaseline } from "@mui/material";
import useMediaQuery from "@mui/material/useMediaQuery";
import CanvasEditor from "./components/CanvasEditor";
import { lightTheme, darkTheme } from "./theme/themes";
import { useThemeDetector } from "./hooks/useThemeDetector";

function App() {
  const { isDark } = useThemeDetector();

  // 根据系统偏好选择主题对象
  const theme = isDark ? darkTheme : lightTheme;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <CanvasEditor />
    </ThemeProvider>
  );
}

export default App;

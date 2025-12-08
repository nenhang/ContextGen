// src/theme/themes.js

import { createTheme } from "@mui/material/styles";

// 自定义主色调 (Primary Color)
const customPrimary = {
  light: "#FFD740", // 暗色模式下的主黄色
  main: "#3787FF", // 亮色模式下的主蓝色
  // dark: "#005AC1",
};

// --- 1. 亮色主题 (Light Theme) ---
export const lightTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: customPrimary.main,
    },
    secondary: {
      main: "#6B7280",
    },
    background: {
      default: "#F9FAFB", // 页面背景
      paper: "#FFFFFF", // 卡片/面板背景 (Paper)
    },
    divider: "#E5E7EB", // 分割线/边框
  },
  typography: {
    // 你可以在这里定义所有字体样式
  },
  // 覆盖 MUI 组件的默认样式
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          // 确保 Paper 组件的背景色和阴影在亮色模式下符合预期
          boxShadow: "0px 3px 6px rgba(0,0,0,0.05)",
        },
      },
    },
  },
});

// --- 2. 暗色主题 (Dark Theme) ---
export const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: customPrimary.light, // 暗色模式下使用更亮的 Primary 色
    },
    secondary: {
      main: "#B0B0B0",
    },
    background: {
      default: "#121212", // 页面背景 (深黑)
      paper: "#1E1E1E", // 卡片/面板背景 (深灰)
    },
    text: {
      primary: "#E0E0E0",
      secondary: "#B0B0B0",
    },
    divider: "#424242", // 暗色模式下的分割线
  },
  typography: {
    // ...
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          // 确保 Paper 组件的阴影在暗色模式下不那么突兀
          boxShadow: "0px 1px 3px rgba(0,0,0,0.5)",
        },
      },
    },
  },
});

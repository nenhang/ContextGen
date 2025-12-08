// src/hooks/useThemeDetector.js

import { useState, useEffect } from "react";
import useMediaQuery from "@mui/material/useMediaQuery";

export const useThemeDetector = () => {
  // 1. 使用 MUI 的 useMediaQuery Hook 来检测 prefers-color-scheme
  // 初始值设置为 true，表示默认主题是暗色（如果系统检测不到，则会默认为亮色）
  const prefersDarkMode = useMediaQuery("(prefers-color-scheme: dark)");

  // 2. 存储当前是否是暗色模式的状态
  const [isDark, setIsDark] = useState(prefersDarkMode);

  // 3. 监听 prefersDarkMode 的变化并更新状态
  useEffect(() => {
    setIsDark(prefersDarkMode);
  }, [prefersDarkMode]);

  // 返回当前是否是暗色模式，以及一个切换函数（方便用户手动切换，但这里我们先不实现手动切换的 UI）
  return {
    isDark,
    toggleTheme: () => setIsDark((prev) => !prev),
  };
};

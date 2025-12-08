// frontend/src/components/CanvasEditor.jsx

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Stage, Layer } from "react-konva";
import ImageComponent from "./ImageComponent";
import { cutoutImage, predictModel, generateAssetModel } from "../api/imageApi";

import {
  Box,
  Paper,
  Grid,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Switch,
  FormControlLabel,
  Alert,
  Snackbar,
} from "@mui/material";

const MIN_DIMENSION = 256;
const MAX_DIMENSION = 1024;
const MAX_SEED = 2147483647; // 2^31 - 1
const MIN_STEPS = 15;
const MAX_STEPS = 30;

// 初始化图层数据结构
const initialLayer = (id, src) => ({
  id: String(id),
  src: src,
  x: 0,
  y: 0,
  scaleX: 1,
  scaleY: 1,
  rotation: 0,
  width: 0,
  height: 0,
  zIndex: id,
});

function CanvasEditor() {
  const [layers, setLayers] = useState([]);
  const [selectedId, selectShape] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const stageRef = useRef(null);
  let layerCounter = useRef(0);

  const [assetLibrary, setAssetLibrary] = useState([]);
  let assetCounter = useRef(0);

  const [prompt, setPrompt] = useState("");

  // 尺寸状态
  const [outputWidth, setOutputWidth] = useState(768);
  const [outputHeight, setOutputHeight] = useState(768);
  const [tempWidth, setTempWidth] = useState(768);
  const [tempHeight, setTempHeight] = useState(768);

  // >>> Seed 状态 <<<
  const [seed, setSeed] = useState(42);
  const [isRandomSeed, setIsRandomSeed] = useState(true);

  const [generatedImageUrl, setGeneratedImageUrl] = useState(null);
  const innerBoxRef = useRef(null);

  const [canvasAreaWidth, setCanvasAreaWidth] = useState(null); // 画布显示区域宽度
  const canvasContainerRef = useRef(null); // 指向画布外层 Box 的 Ref

  const [steps, setSteps] = useState(28);

  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success"); // success, error, warning, info

  const [newAssetPrompt, setNewAssetPrompt] = useState("");

  //  通用的提示函数
  const showSnackbar = useCallback((message, severity = "success") => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  }, []);

  //  关闭提示的函数
  const handleSnackbarClose = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setSnackbarOpen(false);
  };

  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      // 容器 Box 实际宽度是 clientWidth
      setCanvasAreaWidth(entries[0].contentRect.width);
    });

    observer.observe(container);

    return () => {
      observer.unobserve(container);
    };
  }, []); // 仅在组件挂载时执行

  // --- 尺寸输入确认逻辑 ---
  const applyDimensions = () => {
    // 确保值是有效的数字且不小于最小值
    const newW = Math.max(MIN_DIMENSION, parseInt(tempWidth) || MIN_DIMENSION);
    const newH = Math.max(MIN_DIMENSION, parseInt(tempHeight) || MIN_DIMENSION);

    setOutputWidth(newW);
    setOutputHeight(newH);
    setTempWidth(newW); // 同步临时状态
    setTempHeight(newH);
  };

  // --- 下载按钮逻辑 (已修复：强制下载) ---
  const handleDownload = useCallback(async () => {
    if (generatedImageUrl) {
      try {
        // 1. Fetch the image as a Blob
        const response = await fetch(generatedImageUrl);
        if (!response.ok) throw new Error("Failed to fetch image for download");

        const blob = await response.blob();

        // 2. Create a temporary URL
        const url = window.URL.createObjectURL(blob);

        // 3. Create a link and click it
        const link = document.createElement("a");
        link.href = url;
        // 设置 download 属性，强制浏览器下载文件
        link.setAttribute("download", `predicted_image_${Date.now()}.png`);
        document.body.appendChild(link);
        link.click();

        // 4. Clean up
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } catch (error) {
        console.error("Download failed:", error);
        showSnackbar(
          "Download failed: Please check if the image URL is valid.",
          "error"
        );
      }
    }
  }, [generatedImageUrl]);

  // --- 1. 文件上传和抠图结果获取 ---
  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (files.length === 0) return;

    setIsLoading(true);

    try {
      const fileArray = Array.from(files);

      // 1. 在前端生成原图的 Data URL 列表
      const originalSrcs = fileArray.map((file) => URL.createObjectURL(file));

      // 2. 调用批量 API 获取抠图 URL 列表
      const cutoutUrls = await cutoutImage(fileArray);

      if (cutoutUrls.length !== fileArray.length) {
        throw new Error(
          "Backend returned an unexpected number of cutout URLs."
        );
      }

      const newAssetsToAdd = [];
      fileArray.forEach((file, index) => {
        assetCounter.current += 1;
        const newAsset = {
          assetId: String(assetCounter.current),
          originalSrc: originalSrcs[index],
          cutoutSrc: cutoutUrls[index],
        };
        newAssetsToAdd.push(newAsset);
      });

      setAssetLibrary((prev) => {
        return [...newAssetsToAdd, ...prev];
      });

      let addedCount = 0;
      newAssetsToAdd.forEach((asset) => {
        addLayerFromAsset(asset.assetId, asset.cutoutSrc);
        addedCount++;
      });

      showSnackbar(
        `Successfully uploaded and cut out ${addedCount} image(s)!`,
        "success"
      );
    } catch (error) {
      showSnackbar(`Failed to upload images: ${error.message}`, "error");
    } finally {
      setIsLoading(false);
      event.target.value = null; // 清空 input 确保下次选择相同文件也能触发
    }
  };

  // ===========================================
  // >>> 2.素材库和图层管理函数 <<<
  // ===========================================

  // 从素材库添加图层到画布
  const addLayerFromAsset = (assetId, src) => {
    // 每次添加都需要一个新的图层ID
    layerCounter.current += 1;
    const newLayer = {
      ...initialLayer(layerCounter.current, src),
      assetId: assetId,
      zIndex: layerCounter.current - 1, // 初始 ZIndex
    };

    setLayers((prevLayers) => [...prevLayers, newLayer]);
    selectShape(newLayer.id);
  };

  // 仅从画布删除图层 (保持素材库不变)
  const removeLayerFromCanvas = (layerId) => {
    setLayers((prevLayers) => {
      let newLayers = prevLayers.filter((l) => l.id !== layerId);

      // 1. 按照原始的 zIndex 顺序排序（从底到顶）
      newLayers.sort((a, b) => a.zIndex - b.zIndex);
      // 2. 重新赋值连续的 zIndex (0, 1, 2, ...)
      newLayers = newLayers.map((l, index) => ({ ...l, zIndex: index }));

      if (selectedId === layerId) {
        selectShape(null);
      }
      return newLayers;
    });
  };

  // 从素材库删除素材 (同时删除画布上所有关联图层)
  const removeAssetFromLibrary = (assetId) => {
    // 1. 从素材库中移除
    setAssetLibrary((prev) => prev.filter((a) => a.assetId !== assetId));

    // 2. 从画布中移除所有关联的图层
    setLayers((prevLayers) => {
      let remainingLayers = prevLayers.filter((l) => l.assetId !== assetId);

      // 1. 按照原始的 zIndex 顺序排序（从底到顶）
      remainingLayers.sort((a, b) => a.zIndex - b.zIndex);
      // 2. 重新赋值连续的 zIndex (0, 1, 2, ...)
      const newLayers = remainingLayers.map((l, index) => ({
        ...l,
        zIndex: index,
      }));

      // 如果被删除的图层中包含当前选中项，则取消选中
      if (
        selectedId &&
        prevLayers.some((l) => l.assetId === assetId && l.id === selectedId)
      ) {
        selectShape(null);
      }

      return newLayers;
    });
  };

  const handleGenerateAsset = async () => {
    if (!newAssetPrompt) {
      showSnackbar("Please enter a prompt to generate a new asset.", "warning");
      return;
    }

    setIsLoading(true);
    setNewAssetPrompt(""); // 清空输入框

    try {
      const result = await generateAssetModel({
        prompt: newAssetPrompt,
        width: outputWidth > 768 ? 768 : outputWidth > 512 ? 512 : 384,
        height: outputHeight > 768 ? 768 : outputHeight > 512 ? 512 : 384,
      });
      const { original_url, cutout_url } = result;

      // 1. 将结果添加到素材库
      assetCounter.current += 1;
      const newAsset = {
        assetId: String(assetCounter.current),
        originalSrc: original_url, //  存储原图 URL
        cutoutSrc: cutout_url, //  存储抠图 URL
      };

      setAssetLibrary((prev) => [newAsset, ...prev]);

      // 2. 自动添加到画布 (默认使用抠图)
      addLayerFromAsset(newAsset.assetId, newAsset.cutoutSrc);
      showSnackbar("New asset generated and added successfully!", "success");
    } catch (error) {
      showSnackbar(`Failed to generate asset: ${error.message}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  // ===========================================
  // --- 最终拼合与发送 (已加入 Seed) ---
  const handleMergeAndSend = async () => {
    if (!stageRef.current || layers.length === 0) {
      showSnackbar("Please upload at least one asset.", "warning");
      return;
    }

    setIsLoading(true);

    if (selectedId !== null) {
      selectShape(null);
    }
    await new Promise((resolve) => setTimeout(resolve, 0));

    const currentStage = stageRef.current;
    if (!currentStage) return;

    const clonedStage = currentStage.clone();
    clonedStage.scaleX(1);
    clonedStage.scaleY(1);
    clonedStage.width(outputWidth);
    clonedStage.height(outputHeight);

    const mergedImageBase64 = clonedStage.toDataURL({
      mimeType: "image/png",
      quality: 1,
      pixelRatio: 1,
    });

    const boundingBoxData = layers.map((l) => ({
      id: l.id,
      asset_src: l.src,
      x: l.x,
      y: l.y,
      scaleX: l.scaleX,
      scaleY: l.scaleY,
      rotation: l.rotation,
      originalWidth: l.width,
      originalHeight: l.height,
    }));

    // 构造发送给后端的数据包
    const finalPayload = {
      prompt: prompt,
      target_width: outputWidth,
      target_height: outputHeight,
      merged_image: mergedImageBase64,
      layer_data: boundingBoxData,
      seed: isRandomSeed ? null : parseInt(seed),
      steps: steps,
    };

    try {
      const result = await predictModel(finalPayload);
      if (result.status === "success") {
        setGeneratedImageUrl(result.result_image_url);
        showSnackbar("Model prediction succeeded!", "success");
      } else {
        showSnackbar(`Model prediction failed: ${result.message}`, "error");
      }
    } catch (error) {
      showSnackbar(`Model prediction failed: ${error.message}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  // --- 3. 辅助函数 (保持不变) ---

  const handleLayerChange = (newAttrs) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) => (layer.id === newAttrs.id ? newAttrs : layer))
    );
  };

  // 图片加载完成后的初始化尺寸更新 (由 ImageComponent 调用)
  const handleImageLoad = (id, initialProps) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === id ? { ...layer, ...initialProps } : layer
      )
    );
  };

  // 调节图层上下位置
  const handleZIndexChange = (id, direction) => {
    const layerIndex = layers.findIndex((l) => l.id === id);
    if (layerIndex === -1) return;

    let newLayers = [...layers];

    if (direction === "up" && layerIndex < newLayers.length - 1) {
      [newLayers[layerIndex], newLayers[layerIndex + 1]] = [
        newLayers[layerIndex + 1],
        newLayers[layerIndex],
      ];
    } else if (direction === "down" && layerIndex > 0) {
      [newLayers[layerIndex], newLayers[layerIndex - 1]] = [
        newLayers[layerIndex - 1],
        newLayers[layerIndex],
      ];
    }

    // 重新设置 zIndex 确保 Konva 渲染顺序正确
    newLayers = newLayers.map((l, index) => ({ ...l, zIndex: index }));
    setLayers(newLayers);
  };
  const displayScale = canvasAreaWidth / outputWidth;
  const canvasDisplayHeight = outputHeight * displayScale;
  const resultPlaceholderHeight = `${(outputHeight / outputWidth) * 100}%`;

  return (
    <Box
      sx={{
        bgcolor: "background.default",
        minHeight: "100vh",
        width: "100%", // 确保背景色覆盖整个宽度
      }}
    >
      <Box
        sx={{
          margin: "0 auto", // 整体居中
          padding: "30px 20px",
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          align="center"
          gutterBottom
          sx={{ color: "text.primary", fontWeight: 600, mb: 4 }}
        >
          🎨 ContextGen GUI
        </Typography>

        {/* 整体布局：左右两栏 (Grid 实现) */}
        <Grid container spacing={4} ref={innerBoxRef} justifyContent="center">
          {/* --------------------------------------- */}
          {/* 左侧：画布编辑区 (Grid item 占比 6/12) */}
          {/* --------------------------------------- */}
          {/* 一般宽度为 600 左右比较合适，再小了就看不清了 */}
          <Grid item size={{ xs: 10, sm: 10, md: 8, lg: 4 }}>
            <Box display="flex" flexDirection="column" gap={3}>
              {/* 1. 素材库 (Asset Library) - 横向滚动 (保持不变) */}
              <Paper
                elevation={3}
                sx={{ p: 2, borderRadius: 2, flex: "1 0 auto" }}
              >
                <Typography
                  variant="h6"
                  sx={{
                    borderBottom: "1px solid",
                    borderColor: "divider",
                    pb: 1,
                    mb: 2,
                    color: "text.primary",
                    fontWeight: "600",
                  }}
                >
                  Asset Library ({assetLibrary.length})
                </Typography>

                <Box mb={2}>
                  <Button
                    variant="contained"
                    component="label" // 关键：让 Button 充当标签
                    disabled={isLoading}
                    fullWidth
                    sx={{ py: 1, fontWeight: 700 }}
                  >
                    Upload Image (Auto Cutout)
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      multiple
                      style={{ display: "none" }}
                    />
                  </Button>
                </Box>

                <Box display="flex" gap={1} mb={2}>
                  {/* 输入框 */}
                  <TextField
                    fullWidth
                    size="small"
                    placeholder="Prompt for new asset (e.g., 'a blue futuristic car')"
                    value={newAssetPrompt}
                    onChange={(e) => setNewAssetPrompt(e.target.value)}
                    disabled={isLoading}
                  />
                  {/* 按钮 */}
                  <Button
                    onClick={handleGenerateAsset}
                    disabled={isLoading || !newAssetPrompt}
                    variant="outlined"
                    sx={{
                      // 字体稍微小一点，让按钮短一点
                      fontSize: "0.75rem",
                      whiteSpace: "nowrap",
                      minWidth: "auto", // 允许宽度自适应
                      p: "8px 12px", // 调整内边距使按钮更短
                    }}
                  >
                    {isLoading ? (
                      <CircularProgress size={16} color="inherit" />
                    ) : (
                      "Generate New Asset"
                    )}
                  </Button>
                </Box>

                <Box
                  sx={{ display: "flex", gap: 1.5, overflowX: "auto", pb: 1.5 }}
                >
                  {assetLibrary.map((asset) => (
                    <Box
                      key={asset.assetId}
                      sx={{
                        minWidth: 100,
                        maxWidth: 100,
                        p: 1,
                        border: "1px solid",
                        borderColor: "divider",
                        borderRadius: 1,
                        bgcolor: "background.default",
                        flexShrink: 0,
                        textAlign: "center",
                      }}
                    >
                      <Box
                        component="img"
                        src={asset.cutoutSrc}
                        alt={`Asset ${asset.assetId}`}
                        sx={{
                          width: "100%",
                          height: 70,
                          objectFit: "contain",
                          mb: 1,
                          cursor: "pointer",
                          border: "1px dashed",
                          borderColor: "divider",
                          borderRadius: 1,
                        }}
                      />
                      <Box
                        mt={1}
                        display="flex"
                        flexDirection="column"
                        gap={0.5}
                      >
                        <Button
                          size="small"
                          variant="contained"
                          color="success"
                          onClick={() =>
                            addLayerFromAsset(asset.assetId, asset.cutoutSrc)
                          } // 默认添加抠图
                          sx={{ flexGrow: 1 }}
                        >
                          +cutout
                        </Button>
                        <Button
                          size="small"
                          variant="outlined" // 使用 outlined 区分
                          onClick={() =>
                            addLayerFromAsset(asset.assetId, asset.originalSrc)
                          } // 添加原图
                          sx={{ flexGrow: 1 }}
                        >
                          +original
                        </Button>
                      </Box>
                      <Button
                        size="small"
                        variant="contained"
                        color="error"
                        onClick={() => removeAssetFromLibrary(asset.assetId)}
                        sx={{ mt: 0.5, width: "100%" }} // 调整样式，占满宽度
                      >
                        Delete
                      </Button>
                    </Box>
                  ))}
                </Box>
              </Paper>

              {/* 2. 画布区域 (Stage) (保持不变) */}
              <Paper
                elevation={3}
                sx={{
                  p: 2.5,
                  borderRadius: 2,
                  flexGrow: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  minHeight: "400px",
                }}
              >
                <Box
                  ref={canvasContainerRef}
                  sx={{
                    border: "2px solid",
                    borderColor: "primary.main",
                    boxShadow: (theme) =>
                      `0 0 0 1px ${theme.palette.primary.light}`,
                    position: "relative",
                    width: "100%",
                    height: canvasDisplayHeight,
                    borderRadius: 1.5,
                    overflow: "hidden",
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: outputWidth,
                      height: outputHeight,
                      transform: `scale(${displayScale})`,
                      transformOrigin: "top left",
                    }}
                  >
                    <Stage
                      width={outputWidth}
                      height={outputHeight}
                      ref={stageRef}
                      onMouseDown={(e) => {
                        const clickedOnEmpty = e.target === e.target.getStage();
                        if (clickedOnEmpty) {
                          selectShape(null);
                        }
                      }}
                    >
                      <Layer>
                        {layers.map((layer) => (
                          <ImageComponent
                            key={layer.id}
                            layer={layer}
                            isSelected={layer.id === selectedId}
                            onSelect={() => selectShape(layer.id)}
                            onChange={handleLayerChange}
                            onLoad={handleImageLoad}
                            canvasWidth={outputWidth}
                            canvasHeight={outputHeight}
                          />
                        ))}
                      </Layer>
                    </Stage>
                  </Box>
                </Box>
              </Paper>

              {/* 3. 图层设置 (Layer List) */}
              <Paper elevation={3} sx={{ p: 2, borderRadius: 2 }}>
                <Typography
                  variant="h6"
                  sx={{
                    borderBottom: "1px solid",
                    borderColor: "divider",
                    pb: 1,
                    mb: 2,
                    color: "text.primary",
                    fontWeight: "600",
                  }}
                >
                  Layer Management ({layers.length})
                </Typography>
                <Box sx={{ maxHeight: "200px", overflowY: "auto", pr: 1 }}>
                  {layers
                    .slice()
                    .sort((a, b) => b.zIndex - a.zIndex) // 顶部图层 (Z-index 最高) 排在最前面
                    .map((layer, index) => (
                      <Box
                        key={layer.id}
                        onClick={() => selectShape(layer.id)}
                        sx={{
                          p: 1.5,
                          mb: 1,
                          borderRadius: 1.5,
                          border:
                            layer.id === selectedId ? `2px solid` : "1px solid",
                          borderColor:
                            layer.id === selectedId
                              ? "primary.main"
                              : "divider",
                          cursor: "pointer",
                          bgcolor: "background.paper",
                          transition: "all 0.15s",
                        }}
                      >
                        <Typography variant="subtitle1" fontWeight="bold">
                          Layer {index + 1} (z: {layer.zIndex})
                        </Typography>
                        <Box
                          display="flex"
                          justifyContent="space-between"
                          mt={1}
                          gap={1}
                        >
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleZIndexChange(layer.id, "up");
                            }}
                            disabled={layer.zIndex === layers.length - 1}
                          >
                            up ▲
                          </Button>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleZIndexChange(layer.id, "down");
                            }}
                            disabled={layer.zIndex === 0}
                          >
                            down ▼
                          </Button>
                          <Button
                            size="small"
                            variant="contained"
                            color="error"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeLayerFromCanvas(layer.id);
                            }}
                          >
                            remove ✕
                          </Button>
                        </Box>
                      </Box>
                    ))}
                </Box>
              </Paper>
            </Box>
          </Grid>

          {/* --------------------------------------- */}
          {/* 右侧：控制/预测区 (Grid item 占比 6/12) */}
          {/* --------------------------------------- */}
          <Grid item size={{ xs: 10, sm: 10, md: 8, lg: 4 }}>
            <Box display="flex" flexDirection="column" gap={3}>
              {/* 1. 配置设置 (Prompt, 尺寸, Seed) */}
              <Paper elevation={3} sx={{ p: 2.5, borderRadius: 2 }}>
                <Typography
                  variant="h6"
                  sx={{
                    borderBottom: "1px solid",
                    borderColor: "divider",
                    pb: 1,
                    mb: 2,
                    color: "text.primary",
                    fontWeight: "600",
                  }}
                >
                  Inference Settings
                </Typography>

                <Typography
                  variant="subtitle1"
                  component="label"
                  sx={{ display: "block", mb: 0.5, fontWeight: "600" }}
                >
                  Target Size (px):
                </Typography>
                <Box display="flex" gap={1.5} alignItems="center" mb={1}>
                  <TextField
                    type="number"
                    value={tempWidth}
                    onChange={(e) => setTempWidth(e.target.value)}
                    onBlur={applyDimensions} // 失去焦点时应用
                    slotProps={{
                      htmlInput: {
                        min: MIN_DIMENSION,
                        max: MAX_DIMENSION,
                        style: { padding: "10px" },
                      },
                    }}
                    sx={{ width: 100 }}
                  />
                  <Typography>x</Typography>
                  <TextField
                    type="number"
                    value={tempHeight}
                    onChange={(e) => setTempHeight(e.target.value)}
                    onBlur={applyDimensions} // 失去焦点时应用
                    slotProps={{
                      htmlInput: {
                        min: MIN_DIMENSION,
                        max: MAX_DIMENSION,
                        style: { padding: "10px" },
                      },
                    }}
                    sx={{ width: 100 }}
                  />
                </Box>

                <Typography
                  variant="subtitle1"
                  component="label"
                  sx={{ display: "block", mb: 0.5, fontWeight: "600" }}
                >
                  Prompt:
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  value={prompt}
                  placeholder="Enrich your prompt with more details..."
                  onChange={(e) => setPrompt(e.target.value)}
                  variant="outlined"
                  sx={{ mb: 1 }}
                />

                {/* >>> Seed 输入 <<< */}
                <Grid
                  container
                  spacing={3}
                  mb={3}
                  justifyContent={"space-between"}
                >
                  {/* --- 1. 左列: Steps 输入 --- */}
                  <Grid
                    item
                    sx={{
                      flex: "1 0 auto",
                    }}
                  >
                    <Typography
                      variant="subtitle1"
                      component="label"
                      sx={{ display: "block", mb: 1, fontWeight: "600" }}
                    >
                      Number of Steps:
                    </Typography>
                    <TextField
                      fullWidth // 确保在 Grid item 中占满宽度
                      type="number"
                      label={`range ${MIN_STEPS} - ${MAX_STEPS}`}
                      value={steps}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        if (val >= MIN_STEPS && val <= MAX_STEPS) setSteps(val);
                      }}
                      onBlur={(e) => {
                        const val = parseInt(e.target.value);
                        if (val < MIN_STEPS || isNaN(val)) setSteps(MIN_STEPS);
                        else setSteps(Math.min(val, MAX_STEPS));
                      }}
                      slotProps={{
                        htmlInput: {
                          min: MIN_STEPS,
                          max: MAX_STEPS,
                          style: { padding: "10px" },
                        },
                      }}
                    />
                  </Grid>

                  {/* --- 2. 右列: Seed 输入和 Switch --- */}
                  <Grid item sx={{ flex: "1 0 auto", alignItems: "center" }}>
                    <Typography
                      variant="subtitle1"
                      component="label"
                      sx={{ display: "block", mb: 1, fontWeight: "600" }}
                    >
                      Seed:
                    </Typography>
                    <Box display="flex" gap={1.5} alignItems="flex-start">
                      <TextField
                        type="number"
                        label={`${
                          isRandomSeed
                            ? "Set to Random"
                            : `range 0 - ${MAX_SEED}`
                        }`}
                        value={seed}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (val >= 0 && val <= MAX_SEED) setSeed(val);
                        }}
                        onBlur={(e) => {
                          const val = parseInt(e.target.value);
                          if (val < 0 || isNaN(val)) setSeed(0);
                          else setSeed(Math.min(val, MAX_SEED));
                        }}
                        disabled={isRandomSeed}
                        slotProps={{
                          htmlInput: {
                            min: 0,
                            max: MAX_SEED,
                            style: { padding: "10px" },
                          },
                        }}
                        sx={{ flexGrow: 1 }} // 确保 TextField 占据剩余空间
                      />
                      {/* Switch 保持在 TextField 旁边，添加微小 margin 辅助垂直对齐 */}
                      <FormControlLabel
                        control={
                          <Switch
                            checked={isRandomSeed}
                            onChange={(e) => setIsRandomSeed(e.target.checked)}
                            color="primary"
                          />
                        }
                        label="Random"
                      />
                    </Box>
                  </Grid>
                </Grid>

                {/* 预测按钮 */}
                <Button
                  onClick={handleMergeAndSend}
                  disabled={isLoading || layers.length === 0}
                  variant="contained"
                  color="primary"
                  fullWidth
                  size="large"
                  sx={{ py: 1, fontWeight: 700 }}
                >
                  {isLoading ? (
                    <Box display="flex" alignItems="center" gap={1}>
                      <CircularProgress size={20} color="inherit" />
                      Waiting for Result...
                    </Box>
                  ) : (
                    "Generate Image"
                  )}
                </Button>
              </Paper>

              {/* 2. 预测结果放置区 */}
              <Paper
                elevation={3}
                sx={{
                  p: 2.5,
                  borderRadius: 2,
                  flexGrow: 1,
                  minHeight: canvasAreaWidth * 0.75,
                }}
              >
                <Box
                  sx={{
                    width: canvasAreaWidth,
                    paddingTop: generatedImageUrl ? resultPlaceholderHeight : 0,
                    minHeight: generatedImageUrl ? 0 : canvasAreaWidth * 0.75,
                    maxWidth: generatedImageUrl ? "100%" : canvasAreaWidth,
                    position: "relative",
                    bgcolor: "background.default",
                    borderRadius: 1.5,
                    border: "1px dashed",
                    borderColor: "divider",
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {generatedImageUrl ? (
                      <Box
                        component="img"
                        src={generatedImageUrl}
                        alt="Generated Image Result"
                        sx={{
                          maxWidth: "100%",
                          maxHeight: "100%",
                          objectFit: "contain",
                          borderRadius: 1,
                        }}
                      />
                    ) : (
                      <Typography
                        color="text.secondary"
                        align="center"
                        variant="body1"
                      >
                        {isLoading
                          ? "Generating image..."
                          : "Result image will be displayed here"}
                      </Typography>
                    )}
                  </Box>
                </Box>

                {/* 下载按钮 (已修复下载行为) */}
                {generatedImageUrl && (
                  <Button
                    onClick={handleDownload}
                    variant="contained"
                    color="success"
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    Download Result Image
                  </Button>
                )}
              </Paper>
            </Box>
          </Grid>
        </Grid>
      </Box>
      {/* 消息提示组件 */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000} // 3秒后自动关闭
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbarSeverity}
          // variant="filled" // 使用实心样式
          sx={{ width: "100%" }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default CanvasEditor;

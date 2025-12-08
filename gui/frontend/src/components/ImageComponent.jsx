// frontend/src/components/ImageComponent.jsx

import React, { useRef, useEffect } from "react";
import { Image, Transformer } from "react-konva";
import useImage from "use-image";

const ImageComponent = ({
  layer,
  onSelect,
  isSelected,
  onChange,
  onLoad,
  canvasWidth,
  canvasHeight,
}) => {
  // use-image 简化了图片加载过程
  const [image] = useImage(layer.src, "anonymous");
  const shapeRef = useRef(null);
  const trRef = useRef(null);

  // 1. 自动启用/禁用 Transformer
  useEffect(() => {
    // 确保组件已经渲染（image已加载，尺寸已初始化）
    if (!image || layer.width === 0) {
      return;
    }

    // 只有当 isSelected 为 true 时，才尝试绑定
    if (isSelected) {
      // 检查 Refs 是否存在
      if (trRef.current && shapeRef.current) {
        trRef.current.nodes([shapeRef.current]);

        // 确保 Transformer 在 Image 节点上方绘制
        trRef.current.moveToTop();

        // 强制重新绘制图层
        trRef.current.getLayer().batchDraw();

        // console.log(
        //   `[Konva Debug] Binding successful. Transformer ready for layer ${layer.id}.`
        // );
      } else {
        // 如果 ImageComponent 刚被渲染出来，但用户立刻点击了它，可能会出现这种情况
        console.warn(
          `[Konva Debug] Refs are missing for layer ${layer.id}. Retrying on next render.`
        );
      }
    } else if (trRef.current) {
      // 取消选中时，解除绑定
      trRef.current.nodes([]);
      trRef.current.getLayer().batchDraw();
    }

    // 依赖项：isSelected 变化时运行
  }, [isSelected, image, layer.width, layer.id]);

  // 2. 图片加载完成后，初始化尺寸和位置
  useEffect(() => {
    if (image && onLoad && layer.width === 0) {
      // 仅在未初始化尺寸时执行

      // 使用传入的实际画布尺寸来计算初始缩放
      let scale =
        Math.min(canvasWidth / image.width, canvasHeight / image.height) * 0.5;

      const initialProps = {
        width: image.width,
        height: image.height,
        scaleX: scale,
        scaleY: scale,
        // 初始位置也基于实际画布尺寸计算
        x: (canvasWidth - image.width * scale) / 2,
        y: (canvasHeight - image.height * scale) / 2,
      };

      onLoad(layer.id, initialProps);
    }
  }, [image, layer.id, layer.width, onLoad, canvasWidth, canvasHeight]);

  const handleTransformEnd = () => {
    // 变换结束后，获取新的变换属性并更新状态 (Bounding Box 信息)
    const node = shapeRef.current;

    const newProps = {
      x: node.x(),
      y: node.y(),
      scaleX: node.scaleX(),
      scaleY: node.scaleY(),
      rotation: node.rotation(),
      // width/height 保持原始尺寸，变换通过 scaleX/Y 体现
    };

    onChange({ ...layer, ...newProps });
  };

  const handleDragEnd = (e) => {
    // 拖拽结束后，获取新的 x, y 坐标并更新状态
    onChange({
      ...layer,
      x: e.target.x(),
      y: e.target.y(),
    });
  };

  if (!image || layer.width === 0) return null;

  return (
    <React.Fragment>
      <Image
        image={image}
        ref={shapeRef}
        // ********************************************
        // 关键点：将所有的变换属性通过 spread operator 传入 Image 组件
        // 确保 Konva 知道如何渲染它，并让 Transformer 可以修改它们。
        // ********************************************
        {...layer}
        draggable
        onClick={onSelect}
        onTap={onSelect}
        onDragEnd={handleDragEnd}
        onTransformEnd={handleTransformEnd}
      />

      {/* 只有在选中状态下才渲染 Transformer，它提供了缩放/旋转的手柄 */}
      {isSelected && (
        <Transformer
          ref={trRef}
          keepRatio={true}
          // ----------------------------------------
          // >>>>>> 增强可见性设置 <<<<<<
          // ----------------------------------------
          // anchorSize={15} // 增大锚点尺寸，更容易点击和看到
          // borderStrokeWidth={3} // 加粗边框
          // borderDash={[3, 3]} // 使用虚线边框
          // ----------------------------------------
          anchorStroke="#3787FF"
          anchorFill="#3787FF"
          borderStroke="#3787FF"
        />
      )}
    </React.Fragment>
  );
};

export default ImageComponent;

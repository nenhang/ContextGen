# app.py

import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import torch
from ben2 import BEN_Base
from diffusers import FluxPipeline
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).parents[2].resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(str(PROJECT_ROOT))
from src.model.generate import generate, load_model
from src.utils.image_process import annotate

if os.path.exists(dotenv_path := PROJECT_ROOT / ".env"):
    load_dotenv(dotenv_path=dotenv_path)

# --- 配置 ---
app = Flask(__name__)
CORS(app)  # 启用 CORS，允许前端（如 localhost:3000）访问
ASSETS_FOLDER = PROJECT_ROOT / "gui" / "backend" / "assets"
os.makedirs(ASSETS_FOLDER, exist_ok=True)
app.config["ASSETS_FOLDER"] = ASSETS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 限制最大 16MB


BEN_CKPT_PATH = os.getenv("BEN_CKPT_PATH", "PramaLLC/BEN2")  # 从环境变量获取模型权重路径
KONTEXT_MODEL_PATH = os.getenv("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev")
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH", "/root/autodl-tmp/mig-flux/runs/plus_lora512_middle/plus_lora512_middle_20251203-202115/ckpt"
)
FLUX_MODEL_PATH = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")

cutout_model = BEN_Base.from_pretrained(BEN_CKPT_PATH)
cutout_model.to(device).eval()

# TODO: If your single GPU memory is limited, consider loading the following two models on different GPUs.
# Load prediction model
predict_model = load_model(KONTEXT_MODEL_PATH, adapter_path=ADAPTER_PATH, device=device)

# Load generation model
# TODO: Generate model is used for new asset generation from text prompt, it's an optional feature. If not needed or the GPU memory is limited, you can comment it out.
generate_model = FluxPipeline.from_pretrained(FLUX_MODEL_PATH, torch_dtype=torch.bfloat16).to(device)


# app.py (辅助函数部分)
def calculate_aabb(
    layer_data,
    max_width,
    max_height,
):
    """
    根据前端传来的 Konva 变换参数，计算旋转后素材的最小外接矩形 (AABB)。

    参数:
        layer_data (dict): 前端单个素材的变换数据，包含 x, y, scaleX, rotation, originalWidth 等。

    返回:
        dict: 包含 AABB 边界信息的字典 (min_x, max_x, min_y, max_y)。
    """
    x = layer_data["x"]
    y = layer_data["y"]
    scale_x = layer_data["scaleX"]
    scale_y = layer_data["scaleY"]
    rotation_deg = layer_data["rotation"]

    # 原始尺寸
    w_orig = layer_data["originalWidth"]
    h_orig = layer_data["originalHeight"]

    # 1. 转换为弧度
    rotation_rad = math.radians(rotation_deg)

    # 2. 计算最终的缩放后尺寸
    w_scaled = w_orig * scale_x
    h_scaled = h_orig * scale_y

    # 3. 确定素材的四个角点在画布上的相对坐标（以 (x, y) 为原点）
    # Konva 坐标：(x, y) 是左上角

    # 角点相对于 (x, y) 的偏移量
    corners_offset = [
        (0, 0),  # P0: 左上角
        (w_scaled, 0),  # P1: 右上角
        (w_scaled, h_scaled),  # P2: 右下角
        (0, h_scaled),  # P3: 左下角
    ]

    # 4. 旋转和平移计算
    x0, x1 = float("inf"), float("-inf")
    y0, y1 = float("inf"), float("-inf")

    # 预计算旋转值
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)

    for offset_x, offset_y in corners_offset:
        # Konva 的旋转是绕素材左上角（或可配置的中心点，但默认是左上角）
        # 这里使用简化模型：先旋转，再平移 (如果您的 Konva 配置没有改变旋转中心)

        # 旋转变换（相对左上角 (x, y)）
        # 注意：Konva 的复杂变换涉及到旋转中心，这里使用最简的旋转，
        # 如果需要绝对精确，需要使用 Konva 变换矩阵，但以下计算AABB对于大部分场景是足够的：

        # 旋转后的新坐标 (P')
        # Konva 的变换矩阵通常更复杂，但对于 AABB 计算，我们可以采用以下方式：

        # 将点 (offset_x, offset_y) 绕 (0, 0) 旋转
        p_x_rotated = offset_x * cos_theta - offset_y * sin_theta
        p_y_rotated = offset_x * sin_theta + offset_y * cos_theta

        # 加上素材在画布上的实际起始坐标 (x, y)
        p_x_final = p_x_rotated + x
        p_y_final = p_y_rotated + y

        # 更新 AABB 边界
        x0 = min(x0, p_x_final)
        x1 = max(x1, p_x_final)
        y0 = min(y0, p_y_final)
        y1 = max(y1, p_y_final)

    # 5. 确保 AABB 在画布范围内
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(max_width, x1)
    y1 = min(max_height, y1)

    return {
        "aabb_min_x": int(round(x0)),
        "aabb_max_x": int(round(x1)),
        "aabb_min_y": int(round(y0)),
        "aabb_max_y": int(round(y1)),
        "aabb_width": int(round(x1 - x0)),
        "aabb_height": int(round(y1 - y0)),
    }


def process_image_for_cutout(image_files):
    """
    【步骤 1】接收原始图片文件，执行抠图操作。

    参数:
        image_file: werkzeug.datastructures.FileStorage 对象，即用户上传的原始文件
    返回:
        Image.Image 对象，抠图后的透明 PNG 图像
    """
    try:
        save_paths = []
        for image_file in image_files:
            original_img = Image.open(image_file.stream).convert("RGBA")
            cutout_img = cutout_model.inference(original_img)
            cutout_img_name = secure_filename(f"cutout_{os.urandom(8).hex()}.png")
            cutout_img.save(os.path.join(app.config["ASSETS_FOLDER"], cutout_img_name), format="PNG")
            save_paths.append(cutout_img_name)
        return save_paths
    except Exception as e:
        print(f"Error during image cutout: {e}")
        raise ValueError("Image processing failed, please ensure it is a valid image file.")


def run_prediction_model(prompt, image_width, image_height, layout_image, layer_data, seed, num_inference_steps):
    """
    【步骤 2】接收拼合后的图像和图层数据，执行模型预测。

    参数:
        merged_img: PIL Image.Image 对象，拼合后的画布图像
        layer_data: list，前端传递的 bounding box 数据列表
    返回:
        dict，模型预测的结果
    """
    print("Calling backend model prediction...")
    reference_info = [
        {
            "image": os.path.join(app.config["ASSETS_FOLDER"], layer["asset_src"].split("/")[-1]),
            "bbox": torch.tensor(
                [
                    layer["aabb_min_x"],
                    layer["aabb_min_y"],
                    layer["aabb_max_x"],
                    layer["aabb_max_y"],
                ]
            ),
        }
        for layer in layer_data
    ]
    status = "success"
    message = ""
    try:
        result_image = generate(
            flux_pipe=predict_model,
            prompts=[prompt],
            height=image_height,
            width=image_width,
            layout_image=[layout_image],
            reference_info=[reference_info],
            seed=[seed] if seed is not None else None,
            num_inference_steps=num_inference_steps,
        )[0]
        file_name = secure_filename(f"prediction_{os.urandom(8).hex()}.png")
        result_image.save(os.path.join(app.config["ASSETS_FOLDER"], file_name))
        bboxes = [info["bbox"] for info in reference_info]
        annotated_image = annotate(result_image, bboxes, phrases=[f"Object {i + 1}" for i in range(len(bboxes))])
        annotated_file_name = secure_filename(f"annotated_{os.urandom(8).hex()}.png")
        annotated_image.save(os.path.join(app.config["ASSETS_FOLDER"], annotated_file_name))

    except Exception as e:
        status = "error"
        message = json.dumps(str(e))

    prediction_result = {
        "status": status,
        "message": message,
        "input_layers_count": len(layer_data),
        "result_image_url": f"/assets/{file_name}",
    }

    return prediction_result


# --- 1. 抠图 API 路由: /api/cutout ---
@app.route("/api/cutout", methods=["POST"])
def handle_cutout_upload():
    """
    处理前端上传的原始图片文件，执行抠图，并返回抠图后的图片 URL。
    前端使用 FormData 方式发送文件。
    """
    if "files" not in request.files:
        return jsonify({"error": "Missing file field 'files' in request"}), 400

    files = request.files.getlist("files")
    if not files or any(file.filename == "" for file in files):
        return jsonify({"error": "No file selected"}), 400
    try:
        # 1. 调用抠图处理函数
        cutout_img_paths = process_image_for_cutout(files)
        # 2. 返回抠图结果 URL 列表
        cutout_urls = [f"/assets/{img_path}" for img_path in cutout_img_paths]
        return jsonify({"cutout_urls": cutout_urls}), 200

    except ValueError as e:
        return jsonify({"error": json.dumps(str(e))}), 400
    except Exception as e:
        # 捕获其他意外错误
        return jsonify({"error": json.dumps(str(e))}), 500


@app.route("/api/generate_asset", methods=["POST"])
def handle_asset_generation():
    """
    处理前端发送的生成素材请求，执行文本到图像的生成。
    前端使用 application/json 方式发送。
    """
    if not request.is_json:
        return jsonify({"error": "Post content-type must be application/json"}), 400

    data = request.get_json()

    prompt = data["prompt"]
    target_width = data["width"]
    target_height = data["height"]
    num_inference_steps = data.get("steps", 28)

    try:
        # 调用生成模型
        generated_image = generate_model(
            prompt=prompt,
            height=target_height,
            width=target_width,
            num_inference_steps=num_inference_steps,
        ).images[0]

        # 保存生成结果
        filename = secure_filename(f"generated_{os.urandom(8).hex()}.png")
        save_path = os.path.join(app.config["ASSETS_FOLDER"], filename)
        generated_image.save(save_path)

        # 调用抠图模型进行透明背景处理
        cutout_image = cutout_model.inference(generated_image)
        cutout_filename = secure_filename(f"generated_cutout_{os.urandom(8).hex()}.png")
        cutout_save_path = os.path.join(app.config["ASSETS_FOLDER"], cutout_filename)
        cutout_image.save(cutout_save_path, format="PNG")

        return jsonify(
            {
                "original_url": f"/assets/{filename}",
                "cutout_url": f"/assets/{cutout_filename}",
                "message": "generation and cutout success",
            }
        ), 200

    except Exception as e:
        return jsonify({"error": json.dumps(str(e))}), 500


# 静态文件路由：用于提供上传文件夹中的图片
@app.route("/assets/<path:filename>")
def uploaded_file(filename):
    """提供 /assets/ 路径下的静态文件，供前端渲染 Konva 画布"""
    return send_from_directory(app.config["ASSETS_FOLDER"], filename)


# --- 2. 模型预测 API 路由: /api/predict ---
@app.route("/api/predict", methods=["POST"])
def handle_model_prediction():
    """
    处理前端发送的 JSON 数据包，其中包含 Base64 编码的拼合图像和图层数据。
    前端使用 application/json 方式发送。
    """
    if not request.is_json:
        return jsonify({"error": "Post content-type must be application/json"}), 400

    data = request.get_json()

    prompt = data["prompt"]
    target_width = data["target_width"]
    target_height = data["target_height"]
    base64_img_string = data["merged_image"]
    raw_layer_data = data["layer_data"]

    seed = data.get("seed", None)
    num_inference_steps = data["steps"]

    # 1. 计算每个素材的 AABB 并更新数据
    processed_layer_data = []
    for layer in raw_layer_data:
        # 调用 AABB 计算函数
        aabb_info = calculate_aabb(layer, target_width, target_height)
        processed_layer = {**layer, **aabb_info}
        processed_layer_data.append(processed_layer)

    try:
        # 1. 解析 Base64 图像
        # 格式通常是 'data:image/png;base64,...'，需要去除前缀
        header, encoded = base64_img_string.split(",", 1)
        decoded_img_data = base64.b64decode(encoded)

        # 2. 将字节流转换为 PIL 图像对象
        merged_img = Image.open(io.BytesIO(decoded_img_data)).convert("RGBA")

        # 3. 调用模型预测函数
        prediction_result = run_prediction_model(
            prompt=prompt,
            image_width=target_width,
            image_height=target_height,
            layout_image=merged_img,
            layer_data=processed_layer_data,
            seed=seed,
            num_inference_steps=num_inference_steps,
        )

        # 4. 返回模型结果
        return jsonify(prediction_result), 200 if prediction_result["status"] == "success" else 500

    except Exception as e:
        return jsonify({"error": json.dumps(str(e))}), 500


# --- Run Application ---
if __name__ == "__main__":
    # Note: In production, please use a WSGI server like Gunicorn or uWSGI
    app.run(debug=False, port=5000)

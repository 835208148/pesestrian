"""
行人目标检测系统 — Flask Web 应用
启动: python app.py
访问: http://127.0.0.1:5000
"""

import os
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from utils.detect import (
    load_model,
    detect_image_file,
    detect_video,
    detect_webcam,
)

# ========== 配置 ==========
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "images"
VIDEO_DIR = BASE_DIR / "data" / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB 上限

# ========== 加载模型（启动时一次性加载） ==========
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
model = None  # 延迟加载，首次请求时加载


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_NAME)
    return model


def allowed_file(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ========== 页面路由 ==========

@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def output_file(filename):
    """访问检测结果文件"""
    return send_from_directory(str(OUTPUT_DIR), filename)


# ========== 图片检测 API ==========

@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """上传图片 → 返回检测结果"""
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_IMAGE):
        return jsonify({"error": "文件格式不支持，请上传 jpg/png/bmp/webp"}), 400

    # 保存上传图片
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    file.save(str(filepath))

    # 检测
    try:
        annotated, num_people = detect_image_file(get_model(), filepath)
    except Exception as e:
        return jsonify({"error": f"检测失败: {str(e)}"}), 500

    # 保存结果图
    result_name = f"result_{filename}"
    result_path = OUTPUT_DIR / result_name
    import cv2

    cv2.imwrite(str(result_path), annotated)

    return jsonify({
        "success": True,
        "num_people": num_people,
        "result_image": f"/outputs/{result_name}",
    })


# ========== 视频检测 API ==========

@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    """上传视频 → 返回标注后的视频"""
    if "file" not in request.files:
        return jsonify({"error": "未收到文件"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "文件格式不支持，请上传 mp4/avi/mov/mkv/webm"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = VIDEO_DIR / filename
    file.save(str(filepath))

    result_name = f"result_{filename}"
    result_path = OUTPUT_DIR / result_name

    try:
        output_path, stats = detect_video(get_model(), filepath, result_path)
    except Exception as e:
        return jsonify({"error": f"检测失败: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "result_video": f"/outputs/{result_name}",
        "stats": stats,
    })


# ========== 摄像头实时检测 ==========

@app.route("/api/detect/webcam")
def api_detect_webcam():
    """MJPEG 流，前端 <img> 直接显示"""
    return Response(
        detect_webcam(get_model()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ========== 启动 ==========

if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════╗
    ║    行人目标检测系统                        ║
    ║    Pedestrian Detection System            ║
    ╠══════════════════════════════════════════╣
    ║    访问: http://127.0.0.1:5000            ║
    ║    模型: {MODEL_NAME:<31s} ║
    ╚══════════════════════════════════════════╝
    """)
    app.run(debug=True, host="0.0.0.0", port=5000)

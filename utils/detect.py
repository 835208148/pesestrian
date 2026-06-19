"""
行人检测核心模块
使用 YOLOv8 进行图片、视频、摄像头的行人检测
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

# 模型存放目录
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    """
    加载 YOLO 模型
    - 如果 models/ 下有本地模型就用本地的
    - 否则自动下载 ultralytics 预训练模型
    """
    local_path = MODEL_DIR / model_name
    if local_path.exists():
        return YOLO(str(local_path))
    return YOLO(model_name)


# ---------- 图片检测 ----------

def detect_image(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.35,
) -> np.ndarray:
    """
    对 numpy 图片数组做行人检测，返回画好框的图片
    classes=[0] 限定了只检测 person（COCO 类别 0）
    """
    results = model(image, conf=conf, classes=[0], verbose=False)
    return results[0].plot()


def detect_image_file(
    model: YOLO,
    image_path: Union[str, Path],
    conf: float = 0.35,
) -> tuple[np.ndarray, int]:
    """
    读取图片文件 → 检测 → 返回 (标注图, 人数)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    results = model(image, conf=conf, classes=[0], verbose=False)
    annotated = results[0].plot()

    num_people = len(results[0].boxes) if results[0].boxes is not None else 0
    return annotated, num_people


# ---------- 视频检测 ----------

def detect_video(
    model: YOLO,
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    conf: float = 0.35,
    progress_callback=None,
) -> tuple[str, dict]:
    """
    逐帧检测视频，输出标注后的 mp4

    Returns:
        (输出路径, 统计信息字典)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_detections = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=conf, classes=[0], verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

        num = len(results[0].boxes) if results[0].boxes is not None else 0
        total_detections += num

        if progress_callback:
            progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()

    stats = {
        "total_frames": total_frames,
        "total_detections": total_detections,
        "avg_per_frame": round(total_detections / total_frames, 2) if total_frames > 0 else 0,
    }
    return str(output_path), stats


# ---------- 摄像头实时检测 ----------

def detect_webcam(
    model: YOLO,
    camera_id: int = 0,
    conf: float = 0.35,
):
    """
    打开摄像头，逐帧检测，yield JPEG 字节流
    用于 Flask 的流式响应
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 (id={camera_id})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf, classes=[0], verbose=False)
            num = len(results[0].boxes) if results[0].boxes is not None else 0

            annotated = results[0].plot()
            cv2.putText(
                annotated,
                f"People: {num}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )

            _, buffer = cv2.imencode(".jpg", annotated)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes()
                   + b"\r\n")
    finally:
        cap.release()

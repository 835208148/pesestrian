"""
================================================================================
行人目标检测 — YOLOv8 训练脚本
================================================================================

使用方式：在 Google Colab 中运行
  1. 打开 https://colab.research.google.com
  2. 文件 → 上传笔记本 → 选此文件
  3. 运行时 → 更改运行时类型 → 选 T4 GPU
  4. 从上到下逐个单元格执行

注意：此脚本设计为在 Colab 上运行，本地无 GPU 无法训练
================================================================================
"""

# ============================================================================
# 第1步：安装依赖
# ============================================================================
# Colab 已经装了 PyTorch，只需额外装 ultralytics
!pip install ultralytics roboflow 2>&1 | tail -3


# ============================================================================
# 第2步：挂载 Google Drive（存放数据集和训练结果）
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# 确保 Google Drive 上有如下目录结构：
#   datasets/pedestrian/
#   ├── train/
#   │   ├── images/       ← 训练图片
#   │   └── labels/       ← YOLO 格式标注
#   └── val/
#       ├── images/       ← 验证图片
#       └── labels/       ← YOLO 格式标注


# ============================================================================
# 第3步（可选）：从 Roboflow 下载公开行人数据集
# ============================================================================
# 如果你没有自己的标注数据，可以先用 Roboflow 的公开数据集跑通流程
# 去 roboflow.com 免费注册，在 Settings 里获取 API Key

import os
if not os.path.exists('/content/drive/MyDrive/datasets/pedestrian'):
    from roboflow import Roboflow
    rf = Roboflow(api_key="你的API_KEY")  # 替换为你的 key
    project = rf.workspace("people-detection").project("people-detection-general")
    dataset = project.version(6).download("yolov8")
    print(f"数据集下载到: {dataset.location}")
    # 把它移动到 Google Drive 的固定路径
    import shutil
    shutil.move(dataset.location, '/content/drive/MyDrive/datasets/pedestrian')
else:
    print("已有数据集，跳过下载")

# 复制数据集到 Colab 本地磁盘（比从 Drive 读取快）
!mkdir -p /content/datasets/
!cp -r /content/drive/MyDrive/datasets/pedestrian /content/datasets/pedestrian


# ============================================================================
# 第4步：创建数据配置文件
# ============================================================================
yaml_content = """
path: /content/datasets/pedestrian
train: train/images
val: val/images

nc: 1
names: ['person']
"""

with open('/content/pedestrian.yaml', 'w') as f:
    f.write(yaml_content)

print("✅ 配置文件内容:")
!cat /content/pedestrian.yaml


# ============================================================================
# 第5步：开始训练
# ============================================================================
from ultralytics import YOLO

# 加载 YOLOv8 预训练权重
# yolov8n.pt = nano（最小，训练最快，适合跑通流程）
# yolov8s.pt = small（稍大，效果更好）
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(
    data='/content/pedestrian.yaml',  # 数据配置
    epochs=50,                        # 训练轮数，建议 50~100
    imgsz=640,                        # 图片尺寸
    batch=16,                         # batch size，T4 GPU 用 16
    name='pedestrian-det',            # 实验名称
    patience=10,                      # 10 个 epoch 指标不涨就提前停止
    lr0=0.001,                        # 初始学习率
    augment=True,                     # 开启数据增强
    device=0,                         # 使用 GPU
)

# 训练完成后，最优模型在 runs/detect/pedestrian-det/weights/best.pt


# ============================================================================
# 第6步：评估模型效果
# ============================================================================
metrics = model.val()

print("=" * 50)
print("📊 验证集评估结果")
print(f"  mAP@0.5   = {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95 = {metrics.box.map:.4f}")
print(f"  Precision = {metrics.box.p[0]:.4f}")
print(f"  Recall    = {metrics.box.r[0]:.4f}")
print("=" * 50)


# ============================================================================
# 第7步：训练曲线
# ============================================================================
from IPython.display import Image, display
print("📈 训练曲线:")
display(Image(filename='runs/detect/pedestrian-det/results.png'))
# 这个图包含：loss 下降曲线、mAP 上升曲线、precision/recall 变化


# ============================================================================
# 第8步：拿张图测试看看效果
# ============================================================================
results = model.predict(
    source='/content/datasets/pedestrian/val/images/',
    conf=0.35,   # 置信度阈值
    save=True,
)
print("测试结果保存在 runs/detect/predict/ 下")
# 显示第一张结果
import glob
imgs = glob.glob('runs/detect/predict/*.jpg')
if imgs:
    display(Image(filename=imgs[0]))


# ============================================================================
# 第9步：导出模型 → 保存到 Google Drive → 下载到本地
# ============================================================================
import shutil

best_pt = 'runs/detect/pedestrian-det/weights/best.pt'
dest = '/content/drive/MyDrive/pedestrian_best.pt'
shutil.copy(best_pt, dest)

print(f"✅ 模型已保存到 Google Drive: {dest}")
print()
print("📥 接下来：")
print("   1. 打开 drive.google.com，找到 pedestrian_best.pt")
print("   2. 下载到本地")
print("   3. 放到项目的 models/ 目录下")
print("   4. 设置环境变量 MODEL_NAME=pedestrian_best.pt")
print("   5. python app.py 启动系统")

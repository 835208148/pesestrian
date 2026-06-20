# 行人目标检测系统

基于 YOLOv8 深度学习的行人目标检测 Web 系统，支持图片检测、视频检测和摄像头实时检测。

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 目标检测模型 | YOLOv8（Ultralytics） |
| Web 后端 | Flask |
| 前端 | HTML5 + CSS3 + JavaScript |
| 图像处理 | OpenCV |
| 训练平台 | Google Colab（T4 GPU） |

## 项目结构

```
pedestrian-detection-system/
├── app.py                 # Flask Web 服务入口
├── requirements.txt       # Python 依赖清单
├── train_colab.py         # Colab 训练脚本（备用）
├── utils/
│   └── detect.py          # 检测核心：图片/视频/摄像头
├── models/
│   └── pedestrian_best.pt # 自训模型（训练后放置）
├── templates/
│   └── index.html         # Web 前端界面
├── data/
│   ├── images/            # 上传的测试图片
│   └── videos/            # 上传的测试视频
├── outputs/               # 检测结果输出
└── notebooks/
    └── train.ipynb        # Colab 训练笔记本
```

## 快速开始

### 1. 环境配置

```bash
# 创建并激活 conda 环境
conda create --prefix D:\Software\Anaconda\envs\pedestrian python=3.10 -y
conda activate D:\Software\Anaconda\envs\pedestrian

# 安装依赖
pip install torch torchvision ultralytics flask opencv-python numpy pillow matplotlib tqdm
```

### 2. 启动系统

```bash
cd pedestrian-detection-system
python app.py
```

浏览器打开 `http://127.0.0.1:5000`

> **模型加载策略**：系统优先加载 `models/pedestrian_best.pt`（自训模型），若不存在则自动使用官方预训练 `yolov8n.pt`。无需手动配置。

### 3. 功能说明

| 功能 | 操作 | 说明 |
|------|------|------|
| 🖼️ 图片检测 | 上传图片 → 开始检测 | 返回标注框 + 人数统计 |
| 🎬 视频检测 | 上传视频 → 开始检测 | 逐帧处理，输出标注后视频 |
| 📷 摄像头实时 | 点击开启摄像头 | 实时检测画面中的行人 |

## 模型训练

### 训练环境

使用 Google Colab 免费 T4 GPU，避免本地无 GPU 无法训练的问题。

### 数据集

[Roboflow - People Detection v11](https://universe.roboflow.com/chris-kydks/people-detection-2csbw/dataset/11)
- 训练集：3036 张
- 验证集：336 张
- 类别：Person（单类）

### 训练步骤

1. 打开 [Google Colab](https://colab.research.google.com)
2. 文件 → 上传笔记本 → 选择 `notebooks/train.ipynb`
3. 运行时 → 更改运行时类型 → **T4 GPU**
4. 从上到下逐个单元格执行
5. 训练完成后，模型自动保存到 Google Drive
6. 下载 `pedestrian_best.pt` 到本地 `models/` 目录

### 训练结果

| 指标 | 数值 |
|------|:---:|
| mAP@0.5 | **85.22%** |
| mAP@0.5:0.95 | **54.22%** |
| Precision | **85.1%** |
| Recall | **77.0%** |
| 模型参数量 | 3,005,843 |
| GPU | Tesla T4 |

## 指标说明

| 指标 | 含义 |
|------|------|
| mAP@0.5 | 交并比 50% 时的平均精度（常用检测指标） |
| mAP@0.5:0.95 | 0.5 到 0.95 多阈值下的综合精度（更严苛） |
| Precision | 检测结果中有多少是真正的人 |
| Recall | 实际行人中有多少被检测到 |

## Author

Guo Jian — 陕西师范大学计算机科学与技术

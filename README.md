# MHSD-Net

MHSD-Net 是一个用于远程光电容积脉搏波描记法（remote photoplethysmography, rPPG）的科研代码仓库。当前仓库整理了 MHSD-Net 的模型主体、无监督训练损失、优化器配置，以及从人脸视频帧生成 STMap 序列的预处理工具。

该仓库适合作为后续实验复现、模型训练脚本扩展和数据处理流程交接的基础版本。

## 仓库结构

```text
.
├── mhsd_net.py                         # MHSDNet 封装入口
├── training.py                         # 训练配置、损失组合和优化器构建
├── losses.py                           # 频域约束、稀疏性、方差和 Pearson 损失
├── model/
│   ├── __init__.py
│   └── physnet2d.py                    # PhysNet2D 信号恢复网络
└── preprocessing/
    └── rppg_preprocessing.py           # 人脸帧到 STMap 的预处理与运动增强
```

## 主要功能

- 基于 `PhysNet2D` 的 rPPG 信号恢复网络。
- 支持原始 STMap 与运动增强 STMap 的同源样本前向传播。
- 提供 MHSD 训练所需的组合损失：
  - 生理频段带宽约束
  - 频域稀疏性约束
  - 频谱方差约束
  - 时域一致性约束
  - 频域一致性约束
- 提供视频人脸帧预处理工具：
  - 68 点 landmark 生成人脸框
  - RGB 到 YUV 的网格 STMap 提取
  - 滑动裁剪运动增强
  - DIS optical flow 幅值 STMap 提取
  - 固定窗口切片

## 环境依赖

建议使用 Python 3.10 或更新版本。

核心依赖：

```bash
pip install torch numpy opencv-python
```

如果使用 GPU 版本 PyTorch，请根据本机 CUDA 版本从 PyTorch 官方安装命令安装。

## 快速使用

### 1. 构建模型

```python
import torch

from mhsd_net import MHSDNet

model = MHSDNet()

# 输入形状: [batch, channels, spatial_grid, time]
# 当前预处理默认 spatial_grid = 8 * 8 = 64, time = 256
stmap = torch.randn(4, 3, 64, 256)
signal = model(stmap)

print(signal.shape)  # [4, 256]
```

### 2. 计算 MHSD 损失

```python
import torch

from mhsd_net import MHSDNet
from training import MHSDConfig, MHSDLoss, build_optimizer, compute_loss

config = MHSDConfig(fps=30.0, fft_size=256)
model = MHSDNet()
loss_fn = MHSDLoss(config)
optimizer = build_optimizer(model, config)

original_stmap = torch.randn(4, 3, 64, 256)
motion_stmap = torch.randn(4, 3, 64, 256)

loss, terms = compute_loss(model, original_stmap, motion_stmap, loss_fn)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 3. 从人脸帧生成 STMap

`preprocessing/rppg_preprocessing.py` 以已经裁剪好的人脸 RGB 帧序列为输入，输入形状为 `[T, H, W, 3]`。

```python
import numpy as np

from preprocessing.rppg_preprocessing import build_sequences, iter_windows

# face_frames_rgb: numpy array, shape [T, H, W, 3], RGB, uint8 或可转为 uint8
rng = np.random.default_rng(42)
sequences = build_sequences(face_frames_rgb, rng)

for window in iter_windows(sequences, window_size=256, stride=10):
    stmap = window["stmap"]              # [64, 256, 3]
    motion_stmap = window["motion_stmap"]  # [64, 256, 3]

    # 转为模型输入: [channels, spatial_grid, time]
    model_input = np.transpose(stmap, (2, 0, 1))
```

## 数据格式约定

预处理模块默认使用：

- `GRID_SIZE = 8`
- 每帧人脸区域被划分为 `8 x 8 = 64` 个空间网格
- 每个网格提取 YUV 三通道均值
- STMap 形状为 `[64, T, 3]`
- 模型输入前需要转置为 `[3, 64, T]`
- 批量输入模型时形状为 `[B, 3, 64, T]`

默认训练窗口：

- `WINDOW_SIZE = 256`
- `WINDOW_STRIDE = 10`
- 默认帧率 `fps = 30.0`
- 生理心率频段为 `0.66 Hz` 到 `3.0 Hz`

## 当前代码边界

当前仓库提供的是模型、损失和预处理核心代码，不包含完整的数据集读取器、训练循环、验证评估脚本和模型权重文件。使用时需要根据具体数据集补充：

- 视频读取与人脸检测/landmark 提取流程
- 数据集 `Dataset` / `DataLoader`
- 完整训练循环
- 心率估计与评价指标
- checkpoint 保存与加载逻辑

## 后续建议

后续可以继续补充：

- `requirements.txt`
- 数据集适配脚本
- 训练入口脚本
- 推理与心率评估脚本
- 示例 notebook 或最小可运行 demo


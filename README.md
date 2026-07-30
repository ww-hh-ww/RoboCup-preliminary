# Robocup 预赛视觉识别程序

同一画面中检测所有人脸（姓名+性别/Unknown）和 1-8 号物品。

## 当前状态

- ✅ 人脸识别管线已就绪（模拟特征向量，可直接测试）
- ⏳ 物品检测：YOLO 基础模型就绪，需真实标注数据微调
- ⏳ FrameSource 图片/摄像头抽象：待编写

拿到组委会正式规则和真实数据后，按下方步骤替换即可。

## 环境

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 用真实照片替换人脸库

当前 `data/processed/face_gallery/` 存的是随机向量，供管线联调用。

### 拿到真实照片后

1. 每人 1-3 张正面/半侧面照，按以下结构放入：
```
data/raw/faces/person_001/
├── metadata.json    ← {"person_id":"person_001","name":"张三","gender":"男"}
└── front.jpg
data/raw/faces/person_002/
├── metadata.json
└── front.jpg
...
```

2. 重新编码入库：
```bash
python src/face/build_gallery.py real data/raw/faces/
```

`data/raw/faces/` 不进 Git，`data/processed/face_gallery/` 进 Git。

## 用真实物品数据训练 YOLO

当前使用 `yolo11n.pt` 作为基础模型，**未微调**，检测结果为 COCO 80 类。

### 拿到物品照片后

1. 拍摄 8 类物品照片（每类至少 50 张，建议 100+）。包括单物品和混合摆放场景。
```
data/raw/objects/item_01/  ~  item_08/
data/raw/objects/mixed_scenes/
```

2. 用 labelImg 标注 YOLO 格式（class_id, x_center, y_center, width, height 归一化）。

3. 标注好的图片和标签按场景划分：
```
data/processed/item_dataset/images/train/
data/processed/item_dataset/images/val/
data/processed/item_dataset/images/test/
data/processed/item_dataset/labels/train/
data/processed/item_dataset/labels/val/
data/processed/item_dataset/labels/test/
```
**必须按场景划分**（不能把同一段视频的相邻帧分到 train 和 val）。

4. 创建 `data/processed/item_dataset/data.yaml`：
```yaml
train: images/train
val: images/val
test: images/test
nc: 8
names: ["item_01", "item_02", "item_03", "item_04", "item_05", "item_06", "item_07", "item_08"]
```

5. 微调训练：
```bash
yolo train data=data/processed/item_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
```

6. 拷贝权重到程序目录：
```bash
cp runs/train/weights/best.pt models/object/best.pt
```

## 运行推理

```bash
# 图片模式
python src/pipeline/run.py --image path/to/scene.jpg

# 摄像头模式
python src/pipeline/run.py --camera 0
```

## 数据替换路径（不改代码）

| 组件 | 替换什么 | 备注 |
|------|---------|------|
| 人脸识别 | 重新跑 `build_gallery.py real` 覆盖 `face_gallery/` | 换照片 + 换 metadata |
| 物品检测 | 替换 `models/object/best.pt` + 更新 `data.yaml` | 换训练数据后重新训练 |

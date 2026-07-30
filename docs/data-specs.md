# 数据规范

## 人员数据

### 目录结构

```
data/raw/faces/
├── person_001/
│   ├── metadata.json
│   └── images/
│       ├── official.jpg       # 组委会提供
│       ├── front.jpg          # 自采正面
│       └── side_left.jpg      # 自采侧面（可选）
├── person_002/
└── unknown_test/              # 干扰人测试照片
```

### metadata.json 格式

```json
{
  "person_id": "person_001",
  "name": "张三",
  "gender": "男",
  "source": "organizer",
  "consent_confirmed": false
}
```

### 注意事项

- 人脸照片和特征向量**不得**提交到公开 Git 仓库
- 身份证号、联系方式等无关隐私数据**不要**录入

## 物品数据集

### 目录结构

```
data/processed/item_dataset/
├── images/
│   ├── train/       # 训练集图片
│   ├── val/         # 验证集图片
│   └── test/        # 测试集图片
└── labels/
    ├── train/       # 训练集 YOLO 标签
    ├── val/         # 验证集 YOLO 标签
    └── test/        # 测试集标签
```

### 标注格式

- YOLO 标准格式：`<class_id> <x_center> <y_center> <width> <height>`
- 坐标归一化到 [0, 1]
- class_id: 0-7 对应物品 1-8 号

### 数据集划分原则

- **必须按视频或场景划分**，不能把同一段视频的相邻帧随机分到训练集和
  验证集（否则验证结果会虚高）
- 例如：场景 A 的全部帧 → 训练集；场景 B 的全部帧 → 验证集

### 原始物品照片

```
data/raw/objects/
├── item_01/         # 每个物品独立目录
├── item_02/
├── ...
└── mixed_scenes/    # 多个物品混合摆放的场景照
```

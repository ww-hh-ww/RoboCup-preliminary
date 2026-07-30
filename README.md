# RoboCup 预赛视觉识别程序

用于 RoboCup 预赛线上视觉识别任务：
- 检测画面中的所有人脸
- 识别已知人员的姓名和性别
- 将未录入人员标记为 `Unknown`
- 检测并分类 1～8 号指定物品
- 批量处理测试图片
- 输出带标注图片和结构化识别结果

> 当前项目框架已经完成。正式识别效果依赖组委会提供的人脸照片、物品数据和最终提交规则。

---

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 人脸检测与特征提取 | 已完成 | 使用 InsightFace |
| 人脸图库构建 | 已完成 | 支持同一人员多张照片 |
| Unknown 判断 | 已完成 | 阈值需使用真实数据调优 |
| 通用物品检测 | 已完成 | 当前使用 YOLO COCO 预训练模型 |
| 自定义物品识别 | 待数据 | 需要训练 1～8 号物品模型 |
| 单张图片推理 | 已完成 | 输出识别结果 |
| 图片文件夹批处理 | 已完成 | 批量遍历并汇总 |
| JSON 和汇总文件输出 | 已完成 | 每图 JSON + summary / failures CSV |
| 数据集检查脚本 | 已完成 | 校验 YOLO 标注格式和分布 |
| 最终提交格式 | 待规则 | 等组委会公布后适配 |

---

## 项目流程

```text
组委会人脸照片
    ↓
构建人脸特征库
    ↓
测试图片 ──→ 人脸检测与身份匹配
         └─→ YOLO 物品检测
                    ↓
          带框图片 + JSON 结果
                    ↓
             转换为最终提交格式
```

---

## 环境安装

推荐使用 Python 3.11。

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

安装完成后，所有命令都应在项目根目录执行。

---

## 人脸图库

### 1. 准备人员照片

每个人建立一个独立目录：

```
data/raw/faces/
├── person_001/
│   ├── metadata.json
│   ├── front.jpg
│   └── side.jpg
├── person_002/
│   ├── metadata.json
│   └── front.jpg
└── ...
```

`metadata.json` 示例：

```json
{
  "person_id": "person_001",
  "name": "张三",
  "gender": "男"
}
```

照片要求：

- 每人建议准备 1～3 张照片
- 优先使用正脸和轻微侧脸
- 图像应清晰、无遮挡
- 每张图库照片中只能出现一张人脸
- 检测不到人脸或出现多张人脸时，程序应报错

### 2. 构建真实图库

```bash
python -m src.face.build_gallery real data/raw/faces/
```

构建过程会：

- 检查人员目录和 metadata.json
- 检测每张照片中的人脸
- 为每张有效照片提取 embedding
- 保存同一人员的多条 embedding
- 识别时与所有模板比较，取最高相似度

### 3. 构建模拟图库

```bash
python -m src.face.build_gallery simulate
```

模拟图库只用于验证：

- 图库文件能否加载
- 人脸匹配流程能否执行
- 结果结构能否正常输出

模拟 embedding 是随机生成的，不能用于验证真实识别准确率。

---

## 人脸识别阈值

人脸识别使用相似度阈值判断已知人员和 Unknown：

```bash
python -m src.pipeline.run --image test.jpg --face-threshold 0.4
```

默认值仅用于程序调试。

拿到真实人员数据后，需要使用：

- 已录入人员测试照片
- 未录入的干扰人员照片

共同调整阈值。

识别结果保留：

- `name`
- `gender`
- `top1_similarity`
- `top2_name`
- `top2_similarity`
- `match_status`

---

## 物品数据集

当前通用模型为 `yolo11n.pt`，只能识别 COCO 通用类别。

正式比赛需要使用组委会提供的 1～8 号物品数据训练自定义 YOLO 模型。

### 推荐目录结构

```
data/processed/item_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

`data.yaml` 示例：

```yaml
path: data/processed/item_dataset
train: images/train
val: images/val
test: images/test
nc: 8
names:
  0: item_01
  1: item_02
  2: item_03
  3: item_04
  4: item_05
  5: item_06
  6: item_07
  7: item_08
```

训练集、验证集和测试集应按拍摄场景或视频划分。不要把同一段视频中的相邻帧随机分到不同数据集，否则验证结果会虚高。

---

## 数据集检查

```bash
python -m scripts.check_dataset data/processed/item_dataset/data.yaml
```

检查内容：

- data.yaml 是否正确
- 图片和标签文件是否匹配
- 标签是否为 5 列 YOLO 格式
- 坐标是否位于 [0, 1]
- class_id 是否越界
- 是否存在漏标、空标签或异常标签
- 每个类别的样本数量
- 训练集和验证集是否存在重复文件

---

## 训练物品模型

```bash
# Apple Silicon
yolo train data=data/processed/item_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps

# NVIDIA GPU
yolo train data=data/processed/item_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0

# CPU
yolo train data=data/processed/item_dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=cpu
```

训练完成后复制最佳权重：

```bash
python -c "import shutil; shutil.copy('runs/detect/train/weights/best.pt', 'models/object/best.pt')"
```

实际的 `runs` 路径以 Ultralytics 训练输出为准。

程序直接读取自定义模型保存的类别名称，不在代码中重复硬编码类别顺序。

---

## 使用方式总览

项目只有两类操作。

### 1. 构建人脸图库

```bash
# 模拟图库，仅测试程序流程
python -m src.face.build_gallery simulate

# 真实图库，用于正式识别
python -m src.face.build_gallery real data/raw/faces/
```

启用人脸识别前，必须先构建图库。图库为空时程序会报错。

### 2. 运行识别

```bash
# 单张图片
python -m src.pipeline.run --image test.jpg --output results/

# 文件夹批处理
python -m src.pipeline.run --image test_images/ --output results/

# 摄像头调试
python -m src.pipeline.run --camera 0
```

单张图片和文件夹共用同一套批处理代码——单张图片等价于只有一张文件的批处理。

### 功能开关

默认同时运行人脸识别和物品检测：

```bash
# 只运行人脸（关闭物品检测）
python -m src.pipeline.run --image test.jpg --no-object

# 只运行物品（关闭人脸识别）
python -m src.pipeline.run --image test.jpg --no-face
```

### YOLO 安全开关

```bash
python -m src.pipeline.run --image test_images/ --output results/ --require-custom-model
```

- 不加：缺少 `models/object/best.pt` 时允许使用通用 COCO 模型
- 加上：缺少自定义模型时直接报错

正式处理比赛数据时加上此参数，开发调试时可以不加。

### 其他参数

`--face-threshold`、`--obj-threshold`、`--output`、`--no-display` 都是配置参数，不是独立运行模式。

### 一句话概括

```text
先构建人脸库，再选择图片、文件夹或摄像头运行识别。
```

### 输出目录

```
results/
├── images/
│   ├── scene_001.jpg
│   └── scene_002.jpg
├── results/
│   ├── scene_001.json
│   └── scene_002.json
├── summary.csv
└── failures.csv
```

### 单张 JSON 示例

```json
{
  "image": "scene_001.jpg",
  "faces": [
    {
      "bbox": [102, 84, 218, 241],
      "name": "张三",
      "gender": "男",
      "confidence": 0.56,
      "face_confidence": 0.98,
      "top1_name": "张三",
      "top1_similarity": 0.56,
      "top2_name": "李四",
      "top2_similarity": 0.31,
      "match_status": "known"
    }
  ],
  "objects": [
    {
      "bbox": [352, 271, 516, 468],
      "class_id": 2,
      "class_name": "item_03",
      "confidence": 0.91
    }
  ]
}
```

该 JSON 是项目内部统一格式。组委会公布正式提交格式后，再增加转换程序，不修改人脸和物品检测核心逻辑。

---

## 数据和模型替换

| 组件 | 操作 |
|---|---|
| 人脸人员变化 | 更新 `data/raw/faces/` 后重新构建图库 |
| 人脸识别阈值 | 使用真实测试数据调整 `--face-threshold` |
| 物品类别或数据变化 | 重新训练 YOLO |
| 物品模型更新 | 替换 `models/object/best.pt` |
| 提交格式变化 | 修改或新增提交格式转换模块 |

---

## 数据到位后的工作

1. 导入组委会提供的人脸照片
2. 构建真实人脸图库
3. 使用已知人员和干扰人员调整阈值
4. 校验物品标注数据
5. 训练自定义物品检测模型
6. 使用独立测试集进行批量验证
7. 检查漏检、误检和低置信度结果
8. 端到端验收测试

---

## 待规则公布后确认

- 输入是图片、视频还是压缩包
- 文件命名规则
- 最终提交文件格式
- 是否需要提交带框图片
- 是否需要提交姓名、性别和置信度
- 坐标格式及顺序
- 是否允许人工检查或修改结果
- 是否提交代码和模型
- 最终提交文件格式转换

---

## 当前限制

- 模拟人脸图库不能代表真实识别效果
- 默认人脸阈值尚未经过真实数据验证
- 通用 YOLO 模型不能识别比赛指定物品
- 最终输出格式尚未适配组委会规则

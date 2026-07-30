# 技术决策

## 人脸识别：InsightFace

- **模型**：buffalo_l（检测 + 识别一体）
- **流程**：人脸检测 → 人脸对齐 → 特征提取（512-dim embedding）→ 余弦相似度匹配
- **拒识机制**：相似度低于阈值标为 Unknown，阈值需实测调优（初始参考值 0.36）
- **运行时**：macOS 上使用 ONNX Runtime CPU

## 物品检测：Ultralytics YOLO

- **模型**：YOLO11n（微调前），自定义 8 类训练
- **训练后端**：PyTorch MPS（Apple Silicon GPU）
- **推理**：标准 YOLO predict 流程，NMS 内置
- **标注**：彩色空心矩形框，每类分配固定颜色

## 性别

- 不单独训练性别分类模型
- 人脸库 metadata 中手动标注，匹配后直接读取

## 输入抽象

- 统一 `InputSource` 接口，支持图片文件和摄像头流
- 初始化时决定来源，后续管线一致

## 推迟的技术路线

- TensorRT / Core ML 加速（预赛不需要）
- 多目标跟踪（决赛考虑）
- 活体检测 / 防照片冒充（除非规则明确要求）

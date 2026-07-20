# BLIP-2 辅助的图文多模态宠物品种识别实验报告

## 1. 论文信息

### 1.1 BLIP-2

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597
- 官方代码：https://github.com/salesforce/LAVIS/tree/main/projects/blip2

### 1.2 图文融合参考

- 论文名称：Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features
- 论文地址：https://arxiv.org/abs/2001.04732
- 官方代码：https://github.com/AndresPMD/Fine_Grained_Clf

## 2. 任务说明

本实验使用 BLIP-2 为宠物图像生成文本描述，然后将图像特征和文本特征拼接，用于宠物品种分类。

```text
Image -> ResNet-18 -> Image Feature --------┐
                                            ├-> Concatenate -> Classifier
Caption -> Text Encoder -> Text Feature ----┘
```

本实验需要比较：

```text
1. 只使用图像的分类结果；
2. 同时使用图像和 BLIP-2 caption 的分类结果。
```

## 3. 数据集

- 数据集名称：Oxford-IIIT Pet Dataset
- 数据集地址：https://www.robots.ox.ac.uk/~vgg/data/pets/
- 实际使用类别数：
- 实际使用类别名称：
- 实际使用图像总数：
- 训练图像数：
- 验证图像数：
- 测试图像数：
- 使用设备：GPU
- 总训练耗时：

请说明如何选择小型数据子集：

```text
（在这里填写）
```

## 4. BLIP-2 Caption 生成

- 使用模型：
- 使用 prompt：
- 是否使用教师提供的 caption：是 / 否
- 实际生成 caption 数量：
- Caption 保存格式：JSON / JSONL / CSV / 其他
- Caption 生成耗时：

至少展示 5 个 caption 样例：

| 图片编号 | 真实类别 | BLIP-2 Caption |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |

请简要说明 caption 是否能够描述图像中的宠物：

```text
（在这里填写）
```

## 5. 数据预处理

### 5.1 图像增强

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop |  |
| RandomHorizontalFlip |  |
| ColorJitter（可选） |  |
| Normalize |  |

### 5.2 文本处理

- 文本编码模型：BLIP-2
- 模型是否冻结：是
- 输入内容：BLIP-2 生成的完整 caption
- 输出特征维度：
- 文本特征是否提前缓存：是 / 否


## 6. 模型结构

### 6.1 Image-only 模型

- Image Encoder：ResNet-18 / 其他
- 是否使用预训练权重：
- 图像特征维度：
- 输出类别数：

模型结构：

```text
Image -> ResNet-18 -> Image Feature -> Linear Classifier
```

### 6.2 Text Encoder

- 实现方式：平均词向量 / GRU / 其他
- Embedding dimension：
- Text feature dimension：

模型结构：

```text
Caption -> Tokenize -> Embedding -> Mean Pooling / GRU -> Text Feature
```

### 6.3 图文拼接模型

- Image feature dimension：
- Text feature dimension：
- 拼接后的维度：
- MLP hidden dimension：
- 输出类别数：

```text
Image Feature + Text Feature -> Concatenate -> MLP -> Class Prediction
```

可以粘贴关键代码或伪代码：

```python
# 在这里填写
```

## 7. 训练设置

### 7.1 Image-only

| 配置 | 数值 |
|---|---:|
| epochs |  |
| batch size |  |
| optimizer |  |
| learning rate |  |
| loss | CrossEntropyLoss |

### 7.2 Image-Text Fusion

| 配置 | 数值 |
|---|---:|
| epochs |  |
| batch size |  |
| optimizer |  |
| learning rate |  |
| loss | CrossEntropyLoss |

## 8. 训练过程

### 8.1 Image-only

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| ... |  |  |

请粘贴 loss 曲线、accuracy 曲线或日志截图。

### 8.2 Image-Text Fusion

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| ... |  |  |

请粘贴 loss 曲线、accuracy 曲线或日志截图。

请简要描述 loss 是否下降，以及训练是否稳定：

```text
（在这里填写）
```

## 9. 测试结果

| 模型 | Test Accuracy |
|---|---:|
| Image-only ResNet-18 |  |
| Image + Caption Fusion |  |

请分析多模态模型是否优于 image-only 模型：

```text
（在这里填写）
```

如果多模态模型没有提升，请分析可能原因：

```text
（例如 caption 不准确、文本信息太少、数据量小、训练 epoch 少等）
```

## 10. 预测结果展示

至少展示 5 个测试样例。

| 图片编号 | Caption | 真实类别 | Image-only 预测 | 多模态预测 |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

请简单说明文本描述在哪些样例中提供了帮助，在哪些样例中可能产生了干扰：

```text
（在这里填写）
```

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 图像和 caption 是否出现过对应错误；
- 如果继续改进，可以从哪些方面入手，例如增加数据、调整 prompt、增加 epoch、微调 ResNet 或修改文本编码器。

```text
（在这里填写）
```

## 12. AI 对话过程记录

- 录制工具：
- 对话链接：
- 使用的 AI 模型：
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节提供帮助，以及哪些内容由自己检查或修改：

```text
（在这里填写）
```

## 13. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```



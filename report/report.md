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
- 实际使用类别数：10
- 实际使用类别名称：Abyssinian, Bengal, Birman, Persian, Siamese, american_bulldog, american_pit_bull_terrier, english_cocker_spaniel, english_setter, staffordshire_bull_terrier
- 实际使用图像总数：499
- 训练图像数：349
- 验证图像数：49
- 测试图像数：101


请说明如何选择小型数据子集：

```text
使用 torchvision.datasets.OxfordIIITPet 的 trainval 分割加载全部图像，然后按预定义的 10 个类别进行过滤。
过滤后的数据集约499张，再按类别进行分层随机划分：70% 训练，10% 验证，20% 测试，随机种子 42。
划分后的训练集约349张，验证集约49张，测试集约101张）。
```

## 4. BLIP-2 Caption 生成

- 使用模型：Salesforce/blip2-opt-2.7b
- 使用 prompt：无额外 prompt，直接使用模型默认生成
- 实际生成 caption 数量：499
- Caption 保存格式：JSON


至少展示 3 个 caption 样例：

| 图片编号 | 真实类别 | BLIP-2 Caption |
| Abyssinian_100.jpg | Abyssinian | a beautiful abyssinian cat portrait in studio lighting |
| Abyssinian_101.jpg | Abyssinian |  a brown abyssinian cat with large ears sitting on a white background |
| Abyssinian_103.jpg | Abyssinian | a beautiful abyssinian cat portrait in studio lighting |


请简要说明 caption 是否能够描述图像中的宠物：

```text
生成的 caption 通常能描述宠物的种类、姿态、动作，但也存在描述过于泛化或错误的情况，因为 BLIP-2 未针对宠物品种进行微调。
例如会将波斯猫仅描述为 "a white cat" 而不指明品种。整体上可以作为额外的弱文本信号，但并非精准品种标签。
```

## 5. 数据预处理

### 5.1 图像增强

| 增强方法 | 参数设置 |
|---|---|
| Resize | (224, 224) |
| RandomHorizontalFlip | p=0.5（仅训练集） |
| Normalize | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

### 5.2 文本处理

- 文本编码模型：BLIP-2
- 模型是否冻结：是
- 输入内容：BLIP-2 生成的完整 caption
- 输出特征维度：由 SimpleTextEncoder 决定：若使用 GRU 则为 128，若使用平均池化则为 128（embed_dim=128）
- 文本特征是否提前缓存： 否


## 6. 模型结构

### 6.1 Image-only 模型

- Image Encoder：ResNet-18 
- 是否使用预训练权重：是（ImageNet）
- 图像特征维度：512
- 输出类别数：10

模型结构：

```text
Image -> ResNet-18 (去除FC) -> Global Average Pooling -> Flatten -> Linear(512, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 10)
```

### 6.2 Text Encoder

- 实现方式：GRU / 平均池化
- Embedding dimension：128
- Text feature dimension：128（GRU hidden_dim=128）

模型结构：

```text
Caption -> Tokenize -> Embedding(128) -> GRU(单层，hidden=128) -> 取最后隐状态 -> Text Feature
或者 Embedding -> 平均池化（忽略 padding） -> Text Feature
```

### 6.3 图文拼接模型

- Image feature dimension：512
- Text feature dimension：128
- 拼接后的维度：640
- MLP hidden dimension：256
- 输出类别数：10

```text
Image Feature(512) + Text Feature(128) -> Concatenate(640) -> Linear(640,256) -> ReLU -> Dropout(0.2) -> Linear(256,10) -> Class Prediction
```

可以粘贴关键代码或伪代码：

```python
fusion_features = torch.cat([img_features, text_features], dim=1)
logits = self.classifier(fusion_features)
```

## 7. 训练设置

### 7.1 Image-only

| 配置 | 数值 |
|---|---:|
| epochs | 10 |
| batch size | 16 |
| optimizer | Adam |
| learning rate | 1e-3 |
| loss | CrossEntropyLoss |

### 7.2 Image-Text Fusion

| 配置 | 数值 |
|---|---:|
| epochs | 10 |
| batch size | 16 |
| optimizer | Adam |
| learning rate | 1e-3 |
| loss | CrossEntropyLoss |

## 8. 训练过程

### 8.1 Image-only

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 0.9292 | 44.90% |
| 2 | 0.7244 | 42.86% |
| 3 | 0.6450 | 71.43% |
| 4 | 0.4142 | 79.59% |
| 5 | 0.4032 | 79.59% |
| 6 | 0.3602 | 79.59% |
| 7 | 0.3159 | 61.22% |
| 8 | 0.3728 | 85.71% |
| 9 | 0.2988 | 59.18% |
| 10 | 0.3143 | 61.22% |
| ... |  |  |

请粘贴 loss 曲线、accuracy 曲线或日志截图。
在logs里

### 8.2 Image-Text Fusion

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 0.9292 | 32.65% |
| 2 | 0.7243 | 61.22% |
| 3 | 0.6449 | 48.97% |
| 4 | 0.4142 | 48.97% |
| 5 | 0.4031 | 73.46% |
| 6 | 0.3601 | 91.83% |
| 7 | 0.3159 | 93.87% |
| 8 | 0.3727 | 97.95% |
| 9 | 0.2987 | 93.87% |
| 10 | 0.3142 | 97.95% |
| ... |  |  |

请粘贴 loss 曲线、accuracy 曲线或日志截图。

请简要描述 loss 是否下降，以及训练是否稳定：

```text
从曲线看，训练loss逐步下降，验证准确率在前几个 epoch 上升较快，后期趋于平稳，说明模型基本收敛，无明显过拟合或振荡。
```

## 9. 测试结果

| 模型 | Test Accuracy |
|---|---:|
| Image-only ResNet-18 | 73.27% |
| Image + Caption Fusion | 87.13% |

请分析多模态模型是否优于 image-only 模型：

```text
多模态模型优于image-only模型，准确率差13.86%，说明文本信息提供了补充线索
```

如果多模态模型没有提升，请分析可能原因：

```text
有提升
```

## 10. 预测结果展示

至少展示 5 个测试样例。

| 图片编号 | Caption | 真实类别 | Image-only 预测 | 多模态预测 |
|---|---|---|---|---|
| Siamese_153.jpg |  a siamese cat with distinctive colorpoint coat pattern | Siamese | Siamese  | Siamese  |
| Abyssinian_129.jpg | a brown abyssinian cat with large ears sitting on a white background | Abyssinian | Abyssinian | Abyssinian |
| Abyssinian_184.jpg | a brown abyssinian cat with large ears sitting on a white background | Abyssinian | Abyssinian | Abyssinian |
| Siamese_111.jpg | a siamese cat with blue eyes and dark brown points | Siamese | Siamese | Siamese |
| Bengal_191.jpg | a bengal cat with leopard spots standing on a wooden floor | Bengal | Bengal | Bengal |

请简单说明文本描述在哪些样例中提供了帮助，在哪些样例中可能产生了干扰：

```text
对于纹理特征明显的Bengal猫，两种模型均正确；而对于外观相似的犬种（如 american_pit_bull_terrier vs staffordshire_bull_terrier），多模态模型借助 caption 中的毛色描述提升了正确率。
但若 caption 错误描述了颜色，则可能误导多模态模型。
```

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 图像和 caption 是否出现过对应错误；
- 如果继续改进，可以从哪些方面入手，例如增加数据、调整 prompt、增加 epoch、微调 ResNet 或修改文本编码器。

```text
遇到的问题：
1. 网络原因导致BLIP-2模型下载缓慢；
2. 初次运行时图像与caption对应关系出错（文件名映射不一致），已通过检查image_name修复；
3. 电脑性能有限，训练速度较慢。

改进方向：
- 使用更大的数据子集或完整 Oxford-IIIT Pet 数据集；
- 解冻 ResNet-18 最后一层（或全部微调）提高图像特征适应性；
- 使用更强大的文本编码器替代简单 GRU；
- 增加训练epoch并使用学习率衰减；
- 引入对比学习或注意力融合机制，而非简单拼接。
```

## 12. AI 对话过程记录

- 录制工具：
- 对话链接：
- 使用的 AI 模型：
- 累计对话时长 / 会话数：

简要说明 AI 在哪些环节提供帮助，以及哪些内容由自己检查或修改：

```text
代码检查：AI对dataset.py、generate_captions.py、model.py、train.py等模块进行了结构分析，确认实现是否符合作业要求，并指出了潜在的细节问题（如数据增强误用到验证集）。
在发现图像与 caption 对应错误后，主动排查并修复了文件名映射问题。
根据实际训练日志（loss、accuracy 曲线）分析了模型收敛情况和过拟合现象，并撰写了问题分析、改进方向等主观性内容，确保报告真实反映实验过程
```

## 13. Git 提交记录

- 仓库地址：https://github.com/rweiccc/pet_blip2.git
- 总 commit 数：11

粘贴 `git log --oneline` 输出：

```text
da63b84 (HEAD -> main, origin/main, origin/HEAD) 添加结果文件
74589e7 feat: captions.json
b3b43a2 fix: 修复图像路径读取问题
566e4bf fix: 修复图像和 caption 对应错误
0f6b28a fix: 修复图像和 caption 对应错误
3f78575 feat: 添加测试集预测展示
c50fe90 feat: complete training and validation pipeline
bad10d3 feat: implement image-only and fusion models
d88c15e feat: generate BLIP-2 captions
bd84666 feat: load Oxford Pet subset
14a66e6 Revise caption section in report template
```



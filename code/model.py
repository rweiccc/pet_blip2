import torch
import torch.nn as nn
import torchvision.models as models


class ImageOnlyClassifier(nn.Module):
   
    def __init__(self, num_classes, freeze_backbone=False, hidden_dim=256):
        super().__init__()
        
        # 加载预训练ResNet-18
        resnet = models.resnet18(pretrained=True)
        # 去掉最后的全连接层，保留到平均池化
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet-18输出特征维度
        
        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
      
        features = self.backbone(x)  # [B, 512, 1, 1]
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """提取图像特征"""
        features = self.backbone(x)
        return features.flatten(1)


class SimpleTextEncoder(nn.Module):
   
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, use_gru=True, pad_idx=0):
        super().__init__()
        
        self.use_gru = use_gru
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        if use_gru:
            # 单层GRU
            self.gru = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            self.output_dim = hidden_dim
        else:
            # 平均池化
            self.output_dim = embed_dim
    
    def forward(self, x):
        
        embedded = self.embedding(x)  # [B, seq_len, embed_dim]
        
        if self.use_gru:
            # GRU编码，取最后一个时间步的隐藏状态
            _, hidden = self.gru(embedded)  # hidden: [1, B, hidden_dim]
            text_features = hidden.squeeze(0)  # [B, hidden_dim]
        else:
            # 平均池化（忽略padding）
            mask = (x != self.pad_idx).float().unsqueeze(-1)  # [B, seq_len, 1]
            masked_embed = embedded * mask
            sum_embed = masked_embed.sum(dim=1)  # [B, embed_dim]
            lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            text_features = sum_embed / lengths
        
        return text_features


class ImageTextFusionModel(nn.Module):
  
    def __init__(self, num_classes, vocab_size, text_embed_dim=128, 
                 text_hidden_dim=128, use_gru=True, pad_idx=0, 
                 freeze_image_backbone=False, fusion_hidden_dim=256):
        super().__init__()
        
        # 图像编码器
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feature_dim = 512
        
        if freeze_image_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # 文本编码器
        self.text_encoder = SimpleTextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            use_gru=use_gru,
            pad_idx=pad_idx
        )
        self.text_feature_dim = self.text_encoder.output_dim
        
        # 融合后的分类头
        fusion_dim = self.image_feature_dim + self.text_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, num_classes)
        )
    
    def forward(self, images, captions):
      
        # 提取图像特征
        img_features = self.image_encoder(images)  # [B, 512, 1, 1]
        img_features = img_features.flatten(1)  # [B, 512]
        
        # 提取文本特征
        text_features = self.text_encoder(captions)  # [B, text_hidden_dim]
        
        # 特征拼接
        fusion_features = torch.cat([img_features, text_features], dim=1)  # [B, 512+text_dim]
        
        # 分类
        logits = self.classifier(fusion_features)
        return logits
    
    def get_image_features(self, images):
        """提取图像特征"""
        features = self.image_encoder(images)
        return features.flatten(1)
    
    def get_text_features(self, captions):
        """提取文本特征"""
        return self.text_encoder(captions)


def build_image_only_model(num_classes, freeze_backbone=False):
    """构建image-only模型"""
    model = ImageOnlyClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        hidden_dim=256
    )
    return model


def build_fusion_model(num_classes, vocab_size, use_gru=True, freeze_image_backbone=False, pad_idx=0):
    """构建图文融合模型"""
    model = ImageTextFusionModel(
        num_classes=num_classes,
        vocab_size=vocab_size,
        text_embed_dim=128,
        text_hidden_dim=128,
        use_gru=use_gru,
        pad_idx=pad_idx,
        freeze_image_backbone=freeze_image_backbone,
        fusion_hidden_dim=256
    )
    return model

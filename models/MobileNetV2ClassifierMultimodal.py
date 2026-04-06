from models.Model import Model
from torchvision import models
import torch
import torch.nn as nn

class MobileNetV2ClassifierMultimodal(Model):
    def __init__(self, num_classes=2, density_vocab_size=4, view_vocab_size=2):
        super().__init__()

        # Backbone
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features

        # Congelar casi todo y descongelar últimos bloques
        for param in self.features.parameters():
            param.requires_grad = False

        for i in range(16, len(self.features)):
            for param in self.features[i].parameters():
                param.requires_grad = True

        # Atención SE
        in_channels = 1280
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Pooling rama visual
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Embeddings para metadatos
        # breast_density: índices 0..3
        # image_view: índices 0..1
        self.density_emb = nn.Embedding(density_vocab_size, 8)
        self.view_emb = nn.Embedding(view_vocab_size, 4)

        # Rama metadata
        self.meta_head = nn.Sequential(
            nn.Linear(8 + 4, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        # Rama visual
        self.image_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        # Clasificador final tras fusión
        self.classifier = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, meta=None):
        # Rama visual
        x = self.features(x)
        attention = self.attention(x)
        x = x * attention
        x = self.pool(x)
        x = self.image_head(x)

        # Este modelo sí requiere meta
        if meta is None:
            raise ValueError(
                "MobileNetV2ClassifierMultimodal requiere meta con "
                "[breast_density, image_view]."
            )

        # Rama metadata
        density = meta[:, 0]   # valores 0..3
        view = meta[:, 1]      # valores 0..1

        density_feat = self.density_emb(density)
        view_feat = self.view_emb(view)

        meta_feat = torch.cat([density_feat, view_feat], dim=1)
        meta_feat = self.meta_head(meta_feat)

        # Fusión
        fused = torch.cat([x, meta_feat], dim=1)

        # Clasificación final
        out = self.classifier(fused)
        return out

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)
import torch
import torch.nn as nn
from torchvision import models
from models.Model import Model

class EfficientNetB0ClassifierMultimodal(Model):
    def __init__(self, num_classes=2, density_vocab_size=4, view_vocab_size=2):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Congelamos casi todo
        for p in backbone.parameters():
            p.requires_grad = False

        # Descongelamos últimos bloques
        for p in backbone.features[-2:].parameters():
            p.requires_grad = True

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        img_dim = backbone.classifier[1].in_features  # 1280

        # Embeddings para metadatos
        # breast_density: 4 categorías -> índices 0..3
        # image_view: 2 categorías -> índices 0..1
        self.density_emb = nn.Embedding(density_vocab_size, 8)
        self.view_emb = nn.Embedding(view_vocab_size, 8)

        self.meta_mlp = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(img_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta=None):
        # Rama imagen
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # Este modelo sí requiere meta
        if meta is None:
            raise ValueError(
                "EfficientNetB0ClassifierMultimodal requiere meta con "
                "[breast_density, image_view]."
            )

        # Rama metadata
        density = meta[:, 0]
        view = meta[:, 1]

        density_feat = self.density_emb(density)
        view_feat = self.view_emb(view)

        meta_feat = torch.cat([density_feat, view_feat], dim=1)
        meta_feat = self.meta_mlp(meta_feat)

        # Fusión
        fused = torch.cat([x, meta_feat], dim=1)
        logits = self.classifier(fused)
        return logits

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)
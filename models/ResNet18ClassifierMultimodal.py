import torch
import torch.nn as nn
from torchvision import models
from models.Model import Model


class ResNet18ClassifierMultimodal(Model):
    def __init__(self, num_classes=2, density_vocab_size=4, view_vocab_size=2):
        super().__init__()

        # Backbone preentrenado
        base_model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # ResNet18 sin la capa fully connected final
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # [B, 512, 1, 1]
        in_features = base_model.fc.in_features  # 512

        # Congela el extractor visual
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Ajuste fino del último bloque
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        # Embeddings para variables categóricas
        self.density_emb = nn.Embedding(density_vocab_size, 8)
        self.view_emb = nn.Embedding(view_vocab_size, 4)

        # Rama de metadatos
        self.meta_head = nn.Sequential(
            nn.Linear(8 + 4, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        # Rama visual
        self.image_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        # Clasificador tras fusionar imagen y metadatos
        self.classifier = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, meta=None):
        # Rama visual
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.image_head(x)

        # La versión multimodal necesita metadatos
        if meta is None:
            raise ValueError(
                "ResNet18ClassifierMultimodal requiere meta con "
                "[breast_density, image_view]."
            )

        # Rama de metadatos
        density = meta[:, 0]
        view = meta[:, 1]

        density_feat = self.density_emb(density)
        view_feat = self.view_emb(view)

        meta_feat = torch.cat([density_feat, view_feat], dim=1)
        meta_feat = self.meta_head(meta_feat)

        # Fusión de ambas ramas
        fused = torch.cat([x, meta_feat], dim=1)

        out = self.classifier(fused)
        return out

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)
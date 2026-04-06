import torch
import torch.nn as nn
from torchvision import models
from models.Model import Model

class ResNet18ClassifierMultimodal(Model):
    def __init__(self, num_classes=2, density_vocab_size=4, view_vocab_size=2):
        super().__init__()

        base_model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Backbone sin la fully connected final
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # [B, 512, 1, 1]
        in_features = base_model.fc.in_features  # 512

        # Congelar todo el backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar el último bloque real de ResNet (layer4)
        for param in self.backbone[-1].parameters():
            param.requires_grad = True

        # Si quieres afinar más, puedes abrir también layer3
        # for param in self.backbone[-2].parameters():
        #     param.requires_grad = True

        # Embeddings para metadatos
        # breast_density -> índices 0..3
        # image_view -> índices 0..1
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
            nn.Linear(in_features, 256),
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
        x = self.backbone(x)     # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.image_head(x)   # [B, 256]

        # Este modelo sí requiere meta
        if meta is None:
            raise ValueError(
                "ResNet18ClassifierMultimodal requiere meta con "
                "[breast_density, image_view]."
            )

        # Rama metadata
        density = meta[:, 0]   # valores 0..3
        view = meta[:, 1]      # valores 0..1

        density_feat = self.density_emb(density)   # [B, 8]
        view_feat = self.view_emb(view)            # [B, 4]

        meta_feat = torch.cat([density_feat, view_feat], dim=1)  # [B, 12]
        meta_feat = self.meta_head(meta_feat)                    # [B, 16]

        # Fusión
        fused = torch.cat([x, meta_feat], dim=1)   # [B, 272]

        # Clasificación final
        out = self.classifier(fused)
        return out

    def getModel(self):
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)
import timm
import torch
import torch.nn as nn
from models.Model import Model

class ViTSmallClassifierMultimodal(Model):
    def __init__(self, num_classes=2, density_vocab_size=4, view_vocab_size=2):
        super().__init__()

        # Backbone ViT sin cabeza final
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # Congelar todo
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar los últimos bloques transformer
        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True

        # Descongelar también la normalización final
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # Embeddings para metadatos
        # breast_density -> índices 0..3
        # image_view -> índices 0..1
        self.density_emb = nn.Embedding(density_vocab_size, 8)
        self.view_emb = nn.Embedding(view_vocab_size, 4)

        # Rama metadata
        self.meta_head = nn.Sequential(
            nn.Linear(8 + 4, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Rama visual
        self.image_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Clasificador final tras fusión
        self.classifier = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, meta=None):
        # Rama visual
        x = self.backbone.forward_features(x)

        # Según la versión de timm, puede devolver [B, C] o [B, N, C]
        if x.ndim == 3:
            x = x[:, 0]  # token CLS

        x = self.image_head(x)

        # Este modelo sí requiere meta
        if meta is None:
            raise ValueError(
                "ViTSmallClassifierMultimodal requiere meta con "
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
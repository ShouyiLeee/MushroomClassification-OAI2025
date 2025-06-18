import timm
import torch.nn as nn


class ViT_MushroomClassifier(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', num_classes=4):
        super(ViT_MushroomClassifier, self).__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()  # Bỏ classification head của ViT

        self.embedding_dim = self.vit.num_features  # Thường là 768

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 4, C, H, W]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        embeddings = self.vit(x)  # [B*4, D]
        embeddings = embeddings.view(B, N, -1)  # [B, 4, D]

        # Mean pooling over 4 embeddings
        pooled = embeddings.mean(dim=1)  # [B, D]
        out = self.classifier(pooled)
        return out

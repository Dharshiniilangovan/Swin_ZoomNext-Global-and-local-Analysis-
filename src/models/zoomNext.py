import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class ZoomNextLocalAnalysis(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, local_size=384, pretrained=True):
        super().__init__()
        self.local_size = int(local_size)

        self.backbone = create_model("pvt_v2_b2", pretrained=pretrained, num_classes=0)
        self.feat_dim = int(getattr(self.backbone, "num_features", 512))

        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, regions: torch.Tensor):
        if regions.shape[-1] != self.local_size:
            regions = F.interpolate(
                regions, size=(self.local_size, self.local_size),
                mode="bilinear", align_corners=False
            )

        feats = self.backbone.forward_features(regions)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)  
        else:  
            feats = feats.mean(dim=1)

        logits = self.head(feats)                
        probs  = torch.softmax(logits, dim=1)   

        return {
            "local_features": feats,
            "local_logits": logits,
            "local_probs": probs,
        }


__all__ = ["ZoomNextLocalAnalysis"]

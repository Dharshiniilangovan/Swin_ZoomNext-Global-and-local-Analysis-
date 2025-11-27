import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class SwinGlobalAnalysis(nn.Module):

    def __init__(self, num_classes=2, img_size=512, pretrained=True):
        super().__init__()
        self.img_size = int(img_size)
        self.num_classes = int(num_classes)

        self.backbone = create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0,
            img_size=self.img_size,
        )

        pe = getattr(self.backbone, 'patch_embed', None)
        if pe is not None:
            pe.img_size = (self.img_size, self.img_size)
            try:
                patch = pe.patch_size if isinstance(pe.patch_size, int) else pe.patch_size[0]
                pe.grid_size = (self.img_size // patch, self.img_size // patch)
            except Exception:
                pass

        self.feature_dim = int(getattr(self.backbone, "num_features", 1024))
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def _ensure_classifier(self, c: int, device):
        if not hasattr(self, "classifier") or self.classifier is None:
            self.classifier = nn.Linear(c, self.num_classes).to(device)
            self.feature_dim = c

    def _pool_and_fmap(self, feats: torch.Tensor):
        if feats.dim() == 4 and feats.shape[1] >= 256:
            B, C, H, W = feats.shape
            pooled = self.global_pool(feats).flatten(1)
            return pooled, feats, C

        if feats.dim() == 4 and feats.shape[-1] >= 256:
            B, H, W, C = feats.shape
            feats = feats.permute(0, 3, 1, 2).contiguous()
            pooled = self.global_pool(feats).flatten(1)
            return pooled, feats, C

        if feats.dim() == 3:
            B, L, C = feats.shape
            pooled = feats.mean(dim=1)
            s = int(math.sqrt(L))
            fmap = feats.transpose(1, 2).contiguous().view(B, C, s, s) if s * s == L else feats.transpose(1, 2).contiguous().view(B, C, 1, L)
            return pooled, fmap, C

        if feats.dim() == 2:
            B, C = feats.shape
            pooled = feats
            fmap = feats.unsqueeze(-1).unsqueeze(-1)
            return pooled, fmap, C

        raise RuntimeError(f"Unsupported feature shape: {tuple(feats.shape)}")

    def _make_cam_32(self, fmap: torch.Tensor, class_ix=None):
        B, C, H, W = fmap.shape
        if class_ix is None:
            class_ix = 1 if self.num_classes > 1 else 0
        class_ix = int(max(0, min(self.num_classes - 1, class_ix)))

        if hasattr(self.classifier, '__getitem__') and isinstance(self.classifier[0], nn.Linear):
            w = self.classifier[0].weight
        else:
            w = self.classifier.weight

        w = w[class_ix % w.shape[0]].view(1, -1, 1, 1)

        if w.shape[1] != C:
            w = F.interpolate(w.unsqueeze(0), size=(C, 1, 1), mode='nearest').squeeze(0)

        cam = (fmap * w).sum(dim=1)
        cam = F.relu(cam)
        cam_32 = F.interpolate(cam.unsqueeze(1), size=(32, 32),
                               mode='bilinear', align_corners=False).squeeze(1)

        cam_min = cam_32.amin(dim=(1, 2), keepdim=True)
        cam_max = cam_32.amax(dim=(1, 2), keepdim=True)
        cam_32 = (cam_32 - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_32

    def _make_uncertainty_map(self, fmap: torch.Tensor):
        B, C, H, W = fmap.shape

        if hasattr(self.classifier, '__getitem__') and isinstance(self.classifier[0], nn.Linear):
            Wcls = self.classifier[0].weight.unsqueeze(-1).unsqueeze(-1)
            bcls = self.classifier[0].bias
        else:
            Wcls = self.classifier.weight.unsqueeze(-1).unsqueeze(-1)
            bcls = self.classifier.bias

        if Wcls.shape[1] != C:
            Wcls = F.interpolate(Wcls, size=(C, 1, 1), mode='nearest')

        logits_map = F.conv2d(fmap, Wcls, bcls)
        probs_map = F.softmax(logits_map, dim=1)
        max_probs, _ = probs_map.max(dim=1, keepdim=True)
        uncertainty = 1.0 - max_probs

        umin = uncertainty.amin(dim=(2, 3), keepdim=True)
        umax = uncertainty.amax(dim=(2, 3), keepdim=True)
        uncertainty_n = (uncertainty - umin) / (umax - umin + 1e-8)
        return uncertainty_n

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)
        pooled, fmap, C = self._pool_and_fmap(feats)
        self._ensure_classifier(C, device=fmap.device)

        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=1)
        confidence = probabilities.max(dim=1).values
        cam_32x32 = self._make_cam_32(fmap, class_ix=1 if self.num_classes > 1 else 0)
        uncertainty_map = self._make_uncertainty_map(fmap)

        return {
            'global_features': pooled,
            'logits': logits,
            'probabilities': probabilities,
            'confidence': confidence,
            'features': fmap,
            'cam_32x32': cam_32x32,
            'uncertainty_map': uncertainty_map
        }


__all__ = ["SwinGlobalAnalysis"]

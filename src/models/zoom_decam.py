import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.region_selection import RegionSelection
from models.zoomNext import ZoomNextLocalAnalysis
from models.swinTransformer import SwinGlobalAnalysis


class ZoomNextDECAM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 512,
        num_regions: int = 3,
        region_size: int = 384,
        pretrained: bool = True,
        zoom_entropy_threshold: float = 0.75, 
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_regions = num_regions
        self.region_size = region_size
        self.zoom_entropy_threshold = zoom_entropy_threshold  

        self.global_model = SwinGlobalAnalysis(
            num_classes=num_classes,
            img_size=img_size,
            pretrained=False,
        )

        self.global_model.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.region_selector = RegionSelection(
            num_regions=num_regions,
            region_size=region_size,
            feature_dim=1024,
            img_size=img_size,
        )

        self.zoomnext_local = ZoomNextLocalAnalysis(
            num_classes=num_classes,
            input_channels=3,
            local_size=region_size,
            pretrained=pretrained,
        )
        self.zoomnext_local.zoom_entropy_threshold = self.zoom_entropy_threshold

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images: torch.Tensor, debug: bool = False):
        device = images.device
        B = images.size(0)

        feats = self.global_model.backbone.forward_features(images)
        pooled, fmap, C = self.global_model._pool_and_fmap(feats)
        self.global_model._ensure_classifier(C, device=pooled.device)

        global_logits = self.global_model.classifier(pooled)
        global_probs  = F.softmax(global_logits, dim=1)
        global_conf, global_pred = global_probs.max(dim=1)

        cam_32x32 = self.global_model._make_cam_32(fmap, class_ix=1 if self.num_classes > 1 else 0)
        uncertainty_map = self.global_model._make_uncertainty_map(fmap)

        entropy = -(global_probs * torch.log(global_probs + 1e-8)).sum(dim=1)
        unc_mean = uncertainty_map.mean(dim=(1, 2, 3))

        combined_score = 0.7 * entropy + 0.3 * unc_mean
        threshold = getattr(self.zoomnext_local, "zoom_entropy_threshold", 0.25)  
        run_zoom_mask: torch.Tensor = (combined_score > threshold)

        K = self.num_regions

        rs = self.region_selector(
            features=fmap,
            cam_32x32=cam_32x32,
            original=images,
            run_zoom_mask=run_zoom_mask,
            uncertainty_map=uncertainty_map,
        )
        regions: torch.Tensor = rs["regions"]

        local_mean_logits = torch.zeros(B, self.num_classes, device=device)
        local_mean_probs  = torch.full((B, self.num_classes), float('nan'), device=device)
        local_conf        = torch.full((B,), float('nan'), device=device)
        local_pred        = torch.full((B,), -1, dtype=torch.long, device=device)

        idx = torch.nonzero(run_zoom_mask).flatten()
        if idx.numel() > 0:
            chunks = []
            for i in idx.tolist():
                s, e = i * K, min((i + 1) * K, regions.shape[0])
                chunks.append(regions[s:e])
            zoom_regions = torch.cat(chunks, dim=0)

            l = self.zoomnext_local(zoom_regions)
            local_logits = l["local_logits"]
            local_probs  = l["local_probs"]

            total_regions = local_logits.shape[0]
            valid_K = max(1, total_regions // max(idx.numel(), 1))

            try:
                mean_logits = local_logits.view(idx.numel(), valid_K, -1).mean(dim=1)
                mean_probs  = local_probs.view(idx.numel(), valid_K, -1).mean(dim=1)
            except RuntimeError:
                print(f"[WARN] Reshape mismatch: logits={local_logits.shape}, idx={idx.numel()}, valid_K={valid_K}")
                trim_size = (idx.numel() * valid_K)
                mean_logits = local_logits[:trim_size].view(idx.numel(), valid_K, -1).mean(dim=1)
                mean_probs  = local_probs[:trim_size].view(idx.numel(), valid_K, -1).mean(dim=1)

            local_mean_logits[idx] = mean_logits
            local_mean_probs[idx]  = mean_probs
            lc, lp = mean_probs.max(dim=1)
            local_conf[idx] = lc
            local_pred[idx] = lp

        final_logits = global_logits.clone()
        final_logits[run_zoom_mask] = local_mean_logits[run_zoom_mask]

        final_probs = F.softmax(final_logits, dim=1)
        final_conf, final_pred = final_probs.max(dim=1)

        return {
            "logits_global": global_logits,
            "logits_final":  final_logits,

            "global_prediction":  global_pred.detach(),
            "global_confidence":  global_conf.detach(),
            "global_probs":       global_probs.detach(),

            "zoom_prediction":    local_pred.detach(),
            "zoom_confidence":    local_conf.detach(),
            "zoom_probs":         local_mean_probs.detach(),

            "final_prediction":   final_pred.detach(),
            "final_confidence":   final_conf.detach(),

            "zoom_applied":       run_zoom_mask.detach(),
            "attention_map":      rs.get("attention_map", None),
            "coordinates":        rs.get("coordinates", None),
        }

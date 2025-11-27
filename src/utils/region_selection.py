import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionSelection(nn.Module):
    def __init__(self, num_regions=3, region_size=384, feature_dim=1024, img_size=512):
        super().__init__()
        self.num_regions = int(num_regions)
        self.region_size = int(region_size)
        self.img_size = int(img_size)

        self.attention_conv = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.peak_window = 5 

    @torch.no_grad()
    def _pick_peaks(self, att_1x1: torch.Tensor, K: int):
        _, _, Hf, Wf = att_1x1.shape
        scores = att_1x1.clone()
        peaks = []
        for _ in range(K):
            val, idx = scores.view(1, -1).max(dim=1)
            if val.item() <= 0.0:
                break
            flat = idx.item()
            fy = flat // Wf
            fx = flat %  Wf
            peaks.append((int(fy), int(fx), float(val)))

            y1 = max(0, fy - self.peak_window)
            y2 = min(Hf, fy + self.peak_window + 1)
            x1 = max(0, fx - self.peak_window)
            x2 = min(Wf, fx + self.peak_window + 1)
            scores[:, :, y1:y2, x1:x2] = 0.0

        if not peaks:
            peaks.append((Hf // 2, Wf // 2, 1.0))
        return peaks

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,          
        cam_32x32: torch.Tensor,         
        original: torch.Tensor,         
        run_zoom_mask: torch.Tensor=None,
        uncertainty_map: torch.Tensor=None, 
        tau_uncert: float = 0.25         
    ):
        B, C, Hf, Wf = features.shape
        Hi = Wi = self.img_size

        att = self.attention_conv(features)  

        if cam_32x32 is not None:
            cam_f = F.interpolate(cam_32x32.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False)
            att = (att + cam_f) / 2.0

        if uncertainty_map is not None:
            unc_f = F.interpolate(uncertainty_map, size=(Hf, Wf), mode="bilinear", align_corners=False)  
            att = att * (0.5 + unc_f)

        sx = Wi / float(Wf)
        sy = Hi / float(Hf)

        all_regions = []
        all_coords  = []

        for i in range(B):
            coords_img = []
            regs = []

            need_zoom = True
            if run_zoom_mask is not None:
                need_zoom = bool(run_zoom_mask[i].item())

            if need_zoom:
                if uncertainty_map is not None:
                    unc_f_i = F.interpolate(uncertainty_map[i:i+1], size=(Hf, Wf),
                                            mode="bilinear", align_corners=False)[0, 0]
                    mask = (unc_f_i > tau_uncert).float()  
                    att_i = att[i:i+1].clone()             
                    if mask.sum() > 0:
                        att_i *= mask.unsqueeze(0).unsqueeze(0)
                    peaks = self._pick_peaks(att_i, self.num_regions)
                else:
                    peaks = self._pick_peaks(att[i:i+1], self.num_regions)
            else:
                Hc, Wc = Hf, Wf
                peaks = [(Hc // 2, Wc // 2, 1.0)] * self.num_regions

            for (fy, fx, _) in peaks:
                cx = int(fx * sx)
                cy = int(fy * sy)

                half = self.region_size // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(Wi, cx + half)
                y2 = min(Hi, cy + half)

                if (x2 - x1) < self.region_size:
                    if x1 == 0:
                        x2 = min(Wi, self.region_size)
                    else:
                        x1 = max(0, Wi - self.region_size)
                if (y2 - y1) < self.region_size:
                    if y1 == 0:
                        y2 = min(Hi, self.region_size)
                    else:
                        y1 = max(0, Hi - self.region_size)

                crop = original[i:i+1, :, y1:y2, x1:x2]
                if crop.shape[-1] != self.region_size or crop.shape[-2] != self.region_size:
                    crop = F.interpolate(crop, size=(self.region_size, self.region_size),
                                         mode="bilinear", align_corners=False)

                regs.append(crop)
                coords_img.append([int(x1), int(y1), int(x2), int(y2)])

            regs = torch.cat(regs, dim=0)  
            all_regions.append(regs)
            all_coords.append(coords_img)

        regions = torch.cat(all_regions, dim=0) if len(all_regions) > 0 else original.new_zeros(0, 3, self.region_size, self.region_size)

        return {
            "regions": regions,        
            "coordinates": all_coords, 
            "attention_map": att,        
        }


__all__ = ["RegionSelection"]

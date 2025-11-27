import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

def normalize_cam(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam

def average_drop_increase(model, image, cam, true_score):

    B, C, H, W = image.shape

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    cam_up = normalize_cam(cam_up)

    masked = image * (1 - cam_up.unsqueeze(0).unsqueeze(0))

    with torch.no_grad():
        out_masked = model(masked)
        masked_score = out_masked["confidence"][0].item()   

    drop = max(0, true_score - masked_score)
    increase = max(0, masked_score - true_score)

    return drop, increase

def deletion_insertion(model, image, cam, steps=50):

    cam = normalize_cam(cam)
    H_cam, W_cam = cam.shape[-2:]

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image.shape[-2:],
        mode='bilinear',
        align_corners=False
    ).squeeze()

    cam_up = cam_up.flatten()
    idx = torch.argsort(cam_up, descending=True)

    img_flat = image.view(1, 3, -1)

    deletion_scores = []
    insertion_scores = []

    img_del = img_flat.clone()
    img_ins = torch.zeros_like(img_flat)

    with torch.no_grad():
        for step in range(steps):
            k = int((step / steps) * cam_up.numel())

            img_del[:, :, idx[:k]] = 0
            out_del = model(img_del.view_as(image))
            deletion_scores.append(out_del["confidence"][0].item())   

            img_ins[:, :, idx[:k]] = img_flat[:, :, idx[:k]]
            out_ins = model(img_ins.view_as(image))
            insertion_scores.append(out_ins["confidence"][0].item()) 

    deletion_auc = np.trapz(deletion_scores)
    insertion_auc = np.trapz(insertion_scores)

    return deletion_auc, insertion_auc

def cam_complexity(cam):
    cam = normalize_cam(cam).detach().cpu().numpy().flatten() + 1e-8
    entropy = -np.sum(cam * np.log(cam))
    return entropy

def cam_coherency(cam):
    if torch.is_tensor(cam):
        cam_np = cam.detach().cpu().numpy()
    else:
        cam_np = cam

    smooth = gaussian_filter(cam_np, sigma=2)
    diff = np.mean(np.abs(cam_np - smooth))
    return 1 - diff

def compute_adcc(drop, increase, coh, ins_auc):
    return (increase + coh + ins_auc) / (1 + drop + 1e-8)

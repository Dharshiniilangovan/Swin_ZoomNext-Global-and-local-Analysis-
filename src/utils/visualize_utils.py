
import cv2
import numpy as np
import os

COLUMN_TITLES = [
    "Original",
    "Preprocessed",
    "Swin CAM",
    "Uncertainty",
    "Attention",
    "Region Patch"
]

def add_title(img, title):
    img = img.copy()
    cv2.putText(img, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def denormalize_tensor(img_t, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    import torch
    if isinstance(img_t, torch.Tensor):
        img = img_t.clone().cpu().numpy()
    else:
        img = np.array(img_t)
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    img = (img * np.array(std) + np.array(mean))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def heatmap_from_2d(arr, size=None):
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[0] in (1,):
        arr = arr.squeeze(0)
    else:
        raise ValueError("heatmap_from_2d expects HxW or 1xHxW")
    if size is not None:
        arr = cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + 1e-8)
    map_ = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(map_, cv2.COLORMAP_JET)

def overlay_heatmap(img_rgb, heatmap_rgb, alpha=0.45):
    img = img_rgb.astype(np.float32)
    hm = heatmap_rgb.astype(np.float32)
    out = cv2.addWeighted(hm, alpha, img, 1-alpha, 0)
    return out.astype(np.uint8)

def assemble_row(images, pad=4, bg=(255,255,255)):
    heights = [im.shape[0] for im in images]
    widths  = [im.shape[1] for im in images]
    H = max(heights)
    W = sum(widths) + pad * (len(images)-1)
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    x = 0
    for im in images:
        h,w = im.shape[:2]
        canvas[0:h, x:x+w] = im
        x += w + pad
    return canvas

def save_pipeline_panel(out_path, column_images):
    row = assemble_row(column_images)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, row)
    print("Saved:", out_path)

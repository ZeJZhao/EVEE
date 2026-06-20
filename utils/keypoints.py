# utils/keypoints.py
import numpy as np
import cv2

def extract_keypoints(heatmap, threshold=0.03, nms_size=5, top_k=1000):
    # heatmap: (H,W) float32 0~1
    kernel = np.ones((nms_size, nms_size), np.uint8)
    dilated = cv2.dilate(heatmap, kernel)
    local_max = (heatmap == dilated)
    mask = (heatmap > threshold) & local_max
    ys, xs = np.where(mask)
    scores = heatmap[ys, xs]
    if len(scores) > top_k:
        idx = np.argsort(scores)[::-1][:top_k]
        xs, ys = xs[idx], ys[idx]
    return list(zip(xs, ys))

def enhance_heatmap_contrast(hm):
    hm = np.clip(hm, 1e-6, 1.0)
    log_hm = np.log1p(hm * 10)
    hm = (log_hm - log_hm.min()) / (log_hm.max() - log_hm.min() + 1e-6)
    return (hm * 255).astype(np.uint8)
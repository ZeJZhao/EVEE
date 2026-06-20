# utils/heatmap.py
import torch
import torch.nn.functional as F

def event_to_heatmap(evt_img, sigma=3, thresh=0.2):
    """
    evt_img: (B,1,H,W) 或 (1,H,W) float32 0~1
    返回: (B,1,H,W) 0~1
    """
    if evt_img.dim() == 3:
        evt_img = evt_img.unsqueeze(0)
    mask = (evt_img > thresh).float()

    # 伪高斯：对二值 mask 做模糊，作为软标签
    k = int(2 * sigma + 1)
    heat = F.avg_pool2d(mask, (k, k), stride=1, padding=int(sigma))

    heat = heat / (heat.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-6))
    return heat

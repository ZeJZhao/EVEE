# -*- coding: utf-8 -*-
"""
reader/reader.py
集中管理：
- 帧扫描：scan_frames()
- 读取：load_rgb() / load_evt() / load_mask_binary()
- 事件目录自动选择：choose_event_dir()
- 动态分辨率：get_dynamic_hw()
- 默认路径与路径工具：DEFAULT_*、default_matches_dir()、ensure_dir()

说明：
- 与主程序解耦，只承担 “load/read/路径设置” 相关职责。
- load_rgb/load_evt 接收来自主程的 ArrayAdapter（以支持 CuPy / torch）
"""

from __future__ import annotations
import os, re
from collections import OrderedDict
from typing import Tuple, Optional, Dict

import numpy as np
import cv2
import torch

# CuPy（可选）
try:
    import cupy as cp
    _ = cp.zeros((1,), dtype=cp.float32)
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# ================= 默认路径（已改为 images 而非 masked_rgb） =================
READ_ROOT = "./dataset"
DEFAULT_BASE = f"{READ_ROOT}/bunny_racer/video-05-pixel_7-PXL_20230728_005055979.TS"
DEFAULT_RGB_DIR   = f"{DEFAULT_BASE}/images"   # <— 改这里：不再用 masked_rgb
DEFAULT_EVENT_DIR = f"{DEFAULT_BASE}/masks"    # 实际会在 choose_event_dir 中与 events/ 自动择优
DEFAULT_INFER_DIR = f"{DEFAULT_BASE}/images"
DEFAULT_MASK_DIR  = f"{DEFAULT_BASE}/masks"

DEFAULT_TRAIN_OUT_DIR = "./results_ZEJING_TRAIN"
DEFAULT_INFER_OUT_DIR = "./Output_result"
DEFAULT_ONLINE_BEST   = "./weights/online_head_best.pth"

# ============== 基本工具 ==============
def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def default_matches_dir(infer_out_dir: str) -> str:
    return os.path.join(infer_out_dir, "matches")

# ============== 扫描帧 ==============
_FRAME_RE = re.compile(r'^(?:frame_)?(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)

def scan_frames(frames_dir: str, exts=(".png", ".jpg", ".jpeg")) -> "OrderedDict[int, str]":
    """返回有序映射 {idx: abs_path}，自动识别后缀与位数；若目录不存在或为空，返回空映射。"""
    paths: Dict[int, str] = {}
    if not os.path.isdir(frames_dir):
        return OrderedDict()
    for f in os.listdir(frames_dir):
        m = _FRAME_RE.match(f)
        if not m:
            continue
        idx = int(m.group(1)); ext = m.group(2).lower()
        if ('.' + ext) not in exts:
            continue
        p = os.path.join(frames_dir, f)
        if os.path.isfile(p):
            paths[idx] = p
    return OrderedDict(sorted(paths.items(), key=lambda kv: kv[0]))

# ============== 读取：RGB / EVT / MASK ==============
def load_rgb(path: str, size_hw: Tuple[int, int], adapter) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """
    读取 BGR 图并 resize 到 (H,W)，返回 (CHW float[0..1] 的 torch/cupy 对象, 原始 u8 BGR ndarray)
    - adapter: 主程的 ArrayAdapter 实例（决定是否使用 CuPy）
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = size_hw
    img = cv2.resize(img, (W, H))  # u8 BGR
    if getattr(adapter, "use_cupy", False):
        cp_img = cp.asarray(img)
        rgb_f = (cp_img.astype(cp.float32) / 255.0).transpose(2, 0, 1)  # CHW on GPU
        return rgb_f, img
    # torch
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)  # CHW
    return t, img

def load_evt(path: Optional[str], size_hw: Tuple[int, int], adapter) -> Optional[object]:
    """读取事件灰度图（单通道），返回 1×H×W 张量/数组；不存在返回 None。"""
    if path is None or not os.path.exists(path):
        return None
    evt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if evt is None:
        return None
    H, W = size_hw
    evt = cv2.resize(evt, (W, H))  # H,W
    if getattr(adapter, "use_cupy", False):
        cp_evt = cp.asarray(evt)
        return (cp_evt.astype(cp.float32) / 255.0)[None, ...]  # 1,H,W
    return torch.from_numpy(evt).unsqueeze(0).float().div_(255.0)  # 1,H,W

def load_mask_binary(path: Optional[str], size_hw: Tuple[int, int], device: torch.device) -> Optional[torch.Tensor]:
    """
    读取彩色 mask：非黑色像素 -> 1.0，黑色 -> 0.0；返回 torch.float32(H,W) on device
    """
    if path is None or not os.path.exists(path):
        return None
    mk = cv2.imread(path, cv2.IMREAD_COLOR)
    if mk is None:
        return None
    H, W = size_hw
    mk = cv2.resize(mk, (W, H))
    m = (mk[:, :, 0] | mk[:, :, 1] | mk[:, :, 2]) > 0
    m = torch.from_numpy(m.astype('float32')).to(device)  # H,W float 0/1
    return m

# ============== 事件目录自动选择 ==============
def choose_event_dir(arg_event_dir: Optional[str], base_train_dir: str):
    """
    在 [arg_event_dir, base_train_dir/event, base_train_dir/events] 中择优（以帧数最多为准）。
    返回 (evt_map, chosen_evt_dir)
    """
    candidates = []
    if arg_event_dir:
        candidates.append(arg_event_dir)
    candidates.extend([os.path.join(base_train_dir, "event"),
                       os.path.join(base_train_dir, "events")])
    best_map = OrderedDict()
    chosen = None
    for cand in candidates:
        m = scan_frames(cand)
        if len(m) > len(best_map):
            best_map = m
            chosen = cand if len(m) > 0 else chosen
    return best_map, chosen

# ============== 动态分辨率 ==============
def get_dynamic_hw(sample_path: Optional[str], max_size: int, fallback_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    根据 sample 图像计算 (H, W)，最长边限制为 max_size；缺省返回 fallback_hw=(H,W)
    """
    if sample_path is None or not os.path.isfile(sample_path):
        return fallback_hw
    im = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return fallback_hw
    h0, w0 = im.shape[:2]
    s = min(1.0, float(max_size) / float(max(h0, w0)))
    h = max(1, int(round(h0 * s)))
    w = max(1, int(round(w0 * s)))
    return h, w

__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_RGB_DIR", "DEFAULT_EVENT_DIR", "DEFAULT_INFER_DIR", "DEFAULT_MASK_DIR",
    "DEFAULT_TRAIN_OUT_DIR", "DEFAULT_INFER_OUT_DIR", "DEFAULT_ONLINE_BEST",
    "ensure_dir", "default_matches_dir",
    "scan_frames", "load_rgb", "load_evt", "load_mask_binary",
    "choose_event_dir", "get_dynamic_hw",
]

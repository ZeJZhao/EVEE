#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_parser.py — 独立的数据集注解解析模块

- 默认实现：NAVI 的 annotations.json
- 新增实现：BOP/TUDL 的 scene_gt.json + scene_camera.json
"""

import json
import os
import numpy as np


def _quat_wxyz_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:
        return np.eye(3, dtype=float)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)]
    ], dtype=float)


# ================= NAVI =================

def load_annotations(ann_path, item_parser=None):
    """
    加载 NAVI 的 annotations.json，返回 {filename: item}
    item 结构：
      {
        "image_size": [H, W],
        "camera": { "focal_length": float, "q": [qw,qx,qy,qz], "t": [tx,ty,tz] }
      }
    """
    if item_parser is None:
        def default_item_parser(raw_dict):
            # NAVI: image_size 是 [H, W]（acc_pix 中按此用法）
            item = {
                "image_size": raw_dict.get("image_size", [0, 0]),
                "camera": {
                    "focal_length": raw_dict.get("camera", {}).get("focal_length", 1.0),
                    "q": raw_dict.get("camera", {}).get("q", [1.0, 0.0, 0.0, 0.0]),
                    "t": raw_dict.get("camera", {}).get("t", [0.0, 0.0, 0.0])
                }
            }
            return item

    data = json.load(open(ann_path, "r"))
    by_name = {}
    for raw in data:
        filename = raw.get("filename", "")
        if filename:
            parsed_item = item_parser(raw) if item_parser else default_item_parser(raw)
            by_name[filename] = parsed_item
    return by_name


def find_annotation_item(by_name, stem, extensions=None):
    """
    根据 stem 查找对应的 annotation item。
    """
    if extensions is None:
        extensions = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")

    # 精确匹配：stem + ext
    for ext in extensions:
        key = f"{stem}.{ext}"
        if key in by_name:
            return by_name[key]

    # 模糊匹配：stem 在 key 中
    for key, item in by_name.items():
        if stem in key:
            return item

    return None


# ================= BOP / TUDL =================

def load_tudl_scene(scene_dir: str, img_ext: str = "png", obj_id: int | None = None):
    """
    加载 BOP/TUDL 一个场景目录（含 scene_gt.json / scene_camera.json），
    返回 {filename: item}，其中 filename 形如 "000123.png"，便于用 stem 匹配。

    item 结构（与 NAVI 尽量对齐）：
      {
        "image_size": [H, W],
        "camera": {
            "K": 3x3 ndarray,
            "R": 3x3 ndarray,    # 使用“相机到模型”(c2m) 姿态，便于与 NAVI 的 camera.q 一致逻辑
            "t": (3,) ndarray,   # 同上（c2m）
            "focal_length": float (兼容字段，取 K[0,0])
        }
      }
    """
    gt_path = os.path.join(scene_dir, "scene_gt.json")
    cam_path = os.path.join(scene_dir, "scene_camera.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Missing scene_gt.json under: {scene_dir}")
    # scene_camera.json 可选，但强烈建议存在（用于拿到 K 和图像尺寸）
    cam_data = {}
    if os.path.isfile(cam_path):
        cam_data = json.load(open(cam_path, "r"))

    gt_data = json.load(open(gt_path, "r"))

    by_name = {}

    # 遍历每个 image_id
    for img_id_str, gt_list in gt_data.items():
        # 可能一帧里有多个 obj；默认取 obj_id 匹配或第一个
        gt_item = None
        if obj_id is not None:
            for g in gt_list:
                if int(g.get("obj_id", -1)) == int(obj_id):
                    gt_item = g
                    break
        if gt_item is None and len(gt_list) > 0:
            gt_item = gt_list[0]
        if gt_item is None:
            continue

        # 读取 R_m2c（9 元素）和 t_m2c（3 元素）
        R_m2c_flat = gt_item.get("cam_R_m2c", None)
        t_m2c = gt_item.get("cam_t_m2c", None)
        if R_m2c_flat is None or t_m2c is None:
            continue
        R_m2c = np.array(R_m2c_flat, dtype=float).reshape(3, 3)
        t_m2c = np.array(t_m2c, dtype=float).reshape(3,)

        # 相机到模型（与 NAVI 中使用 camera 位姿的方向一致）
        R_c2m = R_m2c.T
        t_c2m = -R_c2m @ t_m2c

        # 读取相机内参/尺寸（如存在）
        K = None
        H = W = None
        if img_id_str in cam_data:
            c = cam_data[img_id_str]
            K_flat = c.get("cam_K", None)
            if K_flat is not None:
                K = np.array(K_flat, dtype=float).reshape(3, 3)
            H = c.get("height", None)
            W = c.get("width", None)

        # fallback：若 scene_camera.json 缺失，给出保底 K/H/W（建议尽快补全 scene_camera.json）
        if K is None:
            K = np.array([[1000.0, 0.0, 640.0],
                          [0.0, 1000.0, 360.0],
                          [0.0, 0.0, 1.0]], dtype=float)
        if H is None or W is None:
            H, W = 720, 1280

        # 组装 item
        item = {
            "image_size": [H, W],
            "camera": {
                "K": K,
                "R": R_c2m,
                "t": t_c2m,
                "focal_length": float(K[0, 0]),  # 兼容字段
            }
        }

        # 以 "000000.png" 作为 key，便于用 stem 匹配
        stem = f"{int(img_id_str):06d}"
        filename = f"{stem}.{img_ext}"
        by_name[filename] = item

    return by_name

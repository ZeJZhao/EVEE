# dataset_parser.py
"""
dataset_parser.py — 独立的数据集注解解析模块

此模块提供加载和解析 annotations.json 的功能，支持自定义解析器以适应不同数据集。
默认实现针对 NAVI 数据集的结构。
"""

import json
import os


def load_annotations(ann_path, item_parser=None):
    """
    加载 annotations.json 文件，并返回一个字典：{filename: annotation_item}

    Args:
        ann_path (str): JSON 文件路径。
        item_parser (callable, optional): 自定义解析函数，接收 raw_dict，返回处理后的 item。
                                         默认使用 NAVI 格式解析。

    Returns:
        dict: {filename: item}
    """
    if item_parser is None:
        def default_item_parser(raw_dict):
            # 默认 NAVI 格式解析：提取 image_size, camera (focal_length, q, t)
            # 注意：image_size 假设为 [H, W]，根据原始代码调整为 [W, H] 或反之
            # 在原始代码中：HA, WA = itemA["image_size"]，然后用 WA*0.5, HA*0.5，所以 image_size 是 [H, W]
            item = {
                "image_size": raw_dict.get("image_size", [0, 0]),  # [H, W]
                "camera": {
                    "focal_length": raw_dict.get("camera", {}).get("focal_length", 1.0),
                    "q": raw_dict.get("camera", {}).get("q", [1.0, 0.0, 0.0, 0.0]),  # [qw, qx, qy, qz]
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

    Args:
        by_name (dict): 从 load_annotations 返回的字典。
        stem (str): 文件 stem（如 "frame_00060"）。
        extensions (list, optional): 支持的文件扩展名，默认为常见图像格式。

    Returns:
        dict or None: 匹配的 item，或 None。
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
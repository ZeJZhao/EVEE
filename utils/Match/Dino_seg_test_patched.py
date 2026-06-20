import os
import shutil
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ================== 配置 ==================
# dinov3 仓库路径（相对当前脚本所在目录 /home/zejing/EVEE/utils）
REPO_DIR = "./dinov3"

# DINOv3 vith16plus 预训练权重路径（请根据你自己的实际路径改）
WEIGHT_PATH = "/home/zj/.cache/torch/hub/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

# 输入帧所在目录（你的视频已经按帧存成 frame_00000.jpg, frame_00005.jpg, ...）
IMAGE_DIR = "/home/zj/EVEE/dataset/BOP/tudl/tudl_test_all/000001/images"

# 每一帧的 mask（人工标注或由 event 转成的 mask）所在目录
MASK_DIR = "/home/zj/EVEE/dataset/BOP/tudl/tudl_test_all/000001/events"

# 每一帧对应的 event PNG 所在目录（可选，用于进一步约束）
EVENT_DIR = "/home/zj/EVEE/dataset/BOP/tudl/tudl_test_all/000001/events"

# 自动追踪生成的 mask 输出目录
OUTPUT_MASK_DIR = "/home/zj/EVEE/utils/dinov3/result"

INPUT_RES = 2048        # 输入尺寸（要是 PATCH_SIZE 的倍数）2048
PATCH_SIZE = 16         # DINOv3 ViT-H/16 的 patch_size=16

# 相似度阈值：越大越“收缩”，越小越“膨胀”
THRESH = 0.8

# 原型动量：越接近 1，更新越慢（更稳定）；越小，自适应越快（也更容易被噪声带跑）
PROTO_MOMENTUM = 0.7

# 如果当前帧预测目标 patch 数太少（可能 tracking 丢了），就不要更新原型
MIN_OBJECT_PATCHES = 10
# ================== 配置结束 ===============


def make_transform(size: int = 1024):
    """DINOv3 常用图像预处理：Resize + ToTensor + Normalize"""
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_dinov3_model(repo_dir: str, weight_path: str, device: torch.device):
    """只在程序开始时加载一次 DINOv3 backbone"""
    print("loading DINOv3 vith16plus backbone...")
    model = torch.hub.load(
        repo_dir,
        "dinov3_vith16plus",
        source="local",
        weights=weight_path,
    )
    model.eval().to(device)
    model.requires_grad_(False)
    return model


@torch.no_grad()
def extract_patch_features(
    image_path: str,
    model,
    device: torch.device,
    input_res: int = 1024,
    patch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """
    对单张图片提取 patch 级别特征

    返回:
        orig_img_t: 原始图像的 tensor (3, H_orig, W_orig)
        feat: patch 特征，shape=(N_patch, C)
        H_orig, W_orig: 原图高宽
        h, w: patch 网格高宽（H_in/patch_size, W_in/patch_size）
    """
    img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = img.size

    transform = make_transform(input_res)
    img_t = transform(img).unsqueeze(0).to(device)  # (1, 3, H_in, W_in)
    _, _, H_in, W_in = img_t.shape

    assert H_in % patch_size == 0 and W_in % patch_size == 0, \
        f"input_res 必须能被 patch_size={patch_size} 整除"

    outputs = model.forward_features(img_t)

    if isinstance(outputs, dict):
        if "x_norm_patchtokens" in outputs:
            patch_tokens = outputs["x_norm_patchtokens"]
        elif "x_norm_patchtoken" in outputs:
            patch_tokens = outputs["x_norm_patchtoken"]
        else:
            patch_tokens = next(iter(outputs.values()))
    else:
        patch_tokens = outputs

    if patch_tokens.ndim == 3:
        B, N_patch, C = patch_tokens.shape
    elif patch_tokens.ndim == 2:
        N_patch, C = patch_tokens.shape
        B = 1
        patch_tokens = patch_tokens.unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected patch_tokens ndim={patch_tokens.ndim}, shape={patch_tokens.shape}")

    h = H_in // patch_size
    w = W_in // patch_size
    assert h * w == N_patch, f"patch 数量不匹配: N_patch={N_patch}, h*w={h*w}"

    feat = patch_tokens[0]            # (N_patch, C)
    feat = F.normalize(feat, dim=-1)  # L2-normalize

    orig_img_t = transforms.ToTensor()(img)  # (3, H_orig, W_orig)
    return orig_img_t, feat, H_orig, W_orig, h, w


def load_and_downsample_mask(mask_path: str, h: int, w: int) -> torch.Tensor:
    """
    读取 mask，并下采样到 patch 网格 (h, w)
    返回: (h, w) 的 bool tensor，True 表示目标区域
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_small = mask_img.resize((w, h), resample=Image.NEAREST)
    mask_np = np.array(mask_small, dtype=np.uint8)
    mask_bool = mask_np > 0
    return torch.from_numpy(mask_bool)


def load_and_downsample_event(event_path: str, h: int, w: int) -> torch.Tensor:
    """
    读取当前帧的 event PNG，并下采样到 patch 网格 (h, w)
    返回: (h, w) 的 bool tensor，True 表示有 event（前景）
    """
    event_img = Image.open(event_path).convert("L")
    event_small = event_img.resize((w, h), resample=Image.NEAREST)
    event_np = np.array(event_small, dtype=np.uint8)
    event_bool = event_np > 0
    return torch.from_numpy(event_bool)


@torch.no_grad()
def build_initial_prototype(
    first_img_path: str,
    first_mask_path: str,
    model,
    device: torch.device,
    input_res: int = 1024,
    patch_size: int = 16,
):
    """
    用第一帧的图 + mask 构建目标原型
    """
    _, feat, H_orig, W_orig, h, w = extract_patch_features(
        first_img_path, model, device, input_res, patch_size
    )

    mask_patch = load_and_downsample_mask(first_mask_path, h, w)
    mask_flat = mask_patch.view(-1)

    if mask_flat.sum().item() == 0:
        raise RuntimeError(f"第一帧 mask 下采样后没有任何前景像素，请检查路径是否正确: {first_mask_path}")

    obj_feats = feat[mask_flat]
    proto = obj_feats.mean(dim=0, keepdim=True)
    proto = F.normalize(proto, dim=-1)

    print(f"Initial prototype built with {int(mask_flat.sum().item())} object patches.")
    return proto, H_orig, W_orig, h, w


@torch.no_grad()
def track_frame_with_prototype(
    image_path: str,
    model,
    device: torch.device,
    proto: torch.Tensor,
    H_ref: int,
    W_ref: int,
    h_ref: int,
    w_ref: int,
    input_res: int = 1024,
    patch_size: int = 16,
    thresh: float = 0.7,
    momentum: float = 0.9,
    event_path: str | None = None,
    mask_path: str | None = None,
):
    """
    用给定原型在当前帧上做 tracking，返回：
        mask_up_np: 上采样到原图大小的二值 mask (H_orig, W_orig)，0/255
        updated_proto: 更新后的原型 (1, C)
    """
    _, feat, H_orig, W_orig, h, w = extract_patch_features(
        image_path, model, device, input_res, patch_size
    )

    assert H_orig == H_ref and W_orig == W_ref, "后续帧的原图分辨率与第一帧不一致"
    assert h == h_ref and w == w_ref, "后续帧的 patch 网格与第一帧不一致"

    proto = F.normalize(proto, dim=-1)
    feat = F.normalize(feat, dim=-1)

    sim = (feat * proto).sum(dim=-1)
    sim_map = sim.view(h, w)

    sim_min = sim_map.min()
    sim_max = sim_map.max()
    sim_norm = (sim_map - sim_min) / (sim_max - sim_min + 1e-6)

    obj_mask_patch = sim_norm > thresh

    if event_path is not None and os.path.isfile(event_path):
        event_mask_patch = load_and_downsample_event(event_path, h, w).to(obj_mask_patch.device)
        obj_mask_patch = obj_mask_patch & event_mask_patch
    elif event_path is not None:
        print(f"  [WARN] event file not found: {event_path}, fall back to DINO only.")

    if mask_path is not None:
        if os.path.isfile(mask_path):
            gt_mask_patch = load_and_downsample_mask(mask_path, h, w).to(obj_mask_patch.device)
            obj_mask_patch = gt_mask_patch
        else:
            print(f"  [WARN] mask file not found: {mask_path}, fall back to DINO/event.")

    obj_mask_flat = obj_mask_patch.view(-1)
    num_obj_patches = int(obj_mask_flat.sum().item())
    print(f"  tracking: {image_path}, object patches = {num_obj_patches}")

    obj_mask_np = obj_mask_patch.cpu().numpy().astype(np.uint8) * 255
    obj_mask_img = Image.fromarray(obj_mask_np, mode="L")
    obj_mask_up_img = obj_mask_img.resize((W_orig, H_orig), resample=Image.NEAREST)
    mask_up_np = np.array(obj_mask_up_img, dtype=np.uint8)

    updated_proto = proto.clone()
    if num_obj_patches >= MIN_OBJECT_PATCHES:
        obj_feats_cur = feat[obj_mask_flat]
        new_proto = obj_feats_cur.mean(dim=0, keepdim=True)
        new_proto = F.normalize(new_proto, dim=-1)
        updated_proto = F.normalize(
            momentum * proto + (1.0 - momentum) * new_proto,
            dim=-1,
        )
    else:
        print("  [WARN] 当前帧目标 patch 太少，不更新原型，避免被噪声带跑。")

    return mask_up_np, updated_proto


def save_binary_mask(mask_np: np.ndarray, out_path: str):
    mask_img = Image.fromarray(mask_np, mode="L")
    mask_img.save(out_path)



# ================== API: 给 EVEE_BANK 调用的 DINOv3 dense 描述子 ==================
class DinoV3Descriptor:
    """最小封装：提供 dense(rgb_last, out_hw) -> (C, Hf, Wf) 的语义描述子。"""

    def __init__(
        self,
        repo_dir: str = REPO_DIR,
        weight_path: str = WEIGHT_PATH,
        device: str | torch.device = "cuda",
        input_res: int = 1024,
        patch_size: int = 16,
    ):
        self.repo_dir = repo_dir
        self.weight_path = weight_path
        self.device = torch.device(device)
        self.input_res = int(input_res)
        self.patch_size = int(patch_size)
        self.model = load_dinov3_model(self.repo_dir, self.weight_path, self.device)
        self.model.eval()

    def _to_pil(self, rgb: torch.Tensor) -> Image.Image:
        """rgb: (3,H,W) / (1,3,H,W) / uint8/float"""
        if rgb.dim() == 4:
            rgb = rgb[0]
        assert rgb.dim() == 3 and rgb.shape[0] == 3, f"Expect (3,H,W) or (1,3,H,W), got {tuple(rgb.shape)}"
        x = rgb.detach().to("cpu")
        if x.dtype != torch.uint8:
            # assume 0..1 or 0..255 float
            x = x.float()
            if x.max().item() <= 1.5:
                x = (x * 255.0).clamp(0, 255)
            x = x.to(torch.uint8)
        x = x.permute(1, 2, 0).numpy()  # HWC
        return Image.fromarray(x, mode="RGB")

    @torch.no_grad()
    def dense(self, rgb_last: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        """
        返回:
            sem: (C, Hf, Wf)  (L2 normalized on channel dim)
        """
        pil = self._to_pil(rgb_last)
        _, feat, _, _, h, w = extract_patch_features(
            pil, self.model, self.device, input_res=self.input_res, patch_size=self.patch_size
        )  # feat: (h*w, C) 已经 normalize
        C = feat.shape[-1]
        feat_map = feat.view(h, w, C).permute(2, 0, 1).unsqueeze(0)  # (1,C,h,w)

        # upsample 到 out_hw（与 EVEE 的 feat_last 对齐）
        sem = F.interpolate(feat_map, size=out_hw, mode="bilinear", align_corners=False)[0]  # (C,Hf,Wf)
        sem = F.normalize(sem, dim=0)
        return sem

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not os.path.isdir(EVENT_DIR):
        print(f"[WARN] EVENT_DIR not found: {EVENT_DIR} （会退回到只用 DINO）")
    if not os.path.isdir(MASK_DIR):
        print(f"[WARN] MASK_DIR not found: {MASK_DIR} （将只能用 DINO+event，不会使用每帧 GT mask）")

    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    model = load_dinov3_model(REPO_DIR, WEIGHT_PATH, device)

    img_files = sorted(
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Found {len(img_files)} frames in {IMAGE_DIR}")

    # 找到第一帧（你现在用的是 frame_00005）
    first_img_name = None
    for f in img_files:
        if os.path.splitext(f)[0] == "frame_00001":
            first_img_name = f
            break
    if first_img_name is None:
        raise RuntimeError("在 IMAGE_DIR 找不到 frame_00005.*，请检查文件名。")

    first_img_path = os.path.join(IMAGE_DIR, first_img_name)
    first_base = os.path.splitext(first_img_name)[0]

    # 第一帧的 mask 从 MASK_DIR/frame_00005.png 取
    first_mask_path = os.path.join(MASK_DIR, first_base + ".png")
    if not os.path.isfile(first_mask_path):
        raise FileNotFoundError(f"第一帧的 mask 没找到: {first_mask_path}")

    # === 第一步：用第一帧图像 + mask 建立初始原型 ===
    proto, H_ref, W_ref, h_ref, w_ref = build_initial_prototype(
        first_img_path,
        first_mask_path,
        model,
        device,
        INPUT_RES,
        PATCH_SIZE,
    )

    # 把第一帧的 mask 也拷贝到输出目录
    first_out_path = os.path.join(OUTPUT_MASK_DIR, first_base + ".png")
    shutil.copy(first_mask_path, first_out_path)
    print(f"Copied first frame mask to {first_out_path}")

    # === 第二步：对后续每一帧做 tracking + 原型更新 ===
    for i, fname in enumerate(img_files):
        image_path = os.path.join(IMAGE_DIR, fname)
        base, _ = os.path.splitext(fname)
        out_mask_path = os.path.join(OUTPUT_MASK_DIR, base + ".png")

        if base == first_base:
            print(f"[{i+1}/{len(img_files)}] Skip first frame (already has GT mask): {image_path}")
            continue

        event_path = None
        if os.path.isdir(EVENT_DIR):
            candidate_event = os.path.join(EVENT_DIR, base + ".png")
            if os.path.isfile(candidate_event):
                event_path = candidate_event

        mask_path = None
        if os.path.isdir(MASK_DIR):
            candidate_mask = os.path.join(MASK_DIR, base + ".png")
            if os.path.isfile(candidate_mask):
                mask_path = candidate_mask

        print(f"[{i+1}/{len(img_files)}] Tracking {image_path} -> {out_mask_path} (event: {event_path}, mask: {mask_path})")
        mask_up_np, proto = track_frame_with_prototype(
            image_path,
            model,
            device,
            proto,
            H_ref,
            W_ref,
            h_ref,
            w_ref,
            input_res=INPUT_RES,
            patch_size=PATCH_SIZE,
            thresh=THRESH,
            momentum=PROTO_MOMENTUM,
            event_path=event_path,
            mask_path=mask_path,
        )

        save_binary_mask(mask_up_np, out_mask_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("All frames tracked.")

from typing import List, Tuple, Optional
import math
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F

import core.CNN_FEAT
PointList = Tuple[List[int], List[int]]  # (xs, ys)

# =============================================================
# ORB 后备
# =============================================================
class ORBMatcher:
    def __init__(self, nfeatures: int = 2000):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
            WTA_K=4, scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=7
        )
    @staticmethod
    def _cv_kps_from_xy(pts_xy):
        return [cv2.KeyPoint(float(x), float(y), 16.0) for x, y in pts_xy]

    def compute_desc(self, img_gray, pts_xy):
        if len(pts_xy) == 0:
            return [], None
        kps = self._cv_kps_from_xy(pts_xy)
        kps, des = self.orb.compute(img_gray, kps)  # des: (N, 32) uint8
        return kps, des

    def match_by_points(self, img1_gray, img2_gray, pts1_xy, pts2_xy, max_matches=500, ratio=0.9):
        kps1, des1 = self.compute_desc(img1_gray, pts1_xy)
        kps2, des2 = self.compute_desc(img2_gray, pts2_xy)
        if des1 is None or des2 is None or len(kps1) == 0 or len(kps2) == 0:
            return [], [], []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        m12 = bf.knnMatch(des1, des2, k=2)
        cand12 = [(m[0].queryIdx, m[0].trainIdx) for m in m12 if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        m21 = bf.knnMatch(des2, des1, k=2)
        cand21 = {(m[0].queryIdx, m[0].trainIdx) for m in m21 if len(m) == 2 and m[0].distance < ratio * m[1].distance}
        mutual = [(qi, tj) for (qi, tj) in cand12 if (tj, qi) in cand21]

        mutual = mutual[:max_matches]
        p1 = [kps1[qi].pt for qi, _ in mutual]
        p2 = [kps2[tj].pt for _, tj in mutual]
        return p1, p2, mutual


def _to_gray(img_bgr_u8):
    if img_bgr_u8.ndim == 2:
        gray = img_bgr_u8
    else:
        gray = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# =============================================================
# 工具函数
# =============================================================
def _points_to_grid_norm(pts_xy: List[tuple], H: int, W: int, device) -> torch.Tensor:
    """把像素坐标映射到 grid_sample 用的 [-1,1]"""
    if len(pts_xy) == 0:
        return torch.empty((1, 0, 1, 2), device=device, dtype=torch.float32)
    xs = torch.tensor([p[0] for p in pts_xy], device=device, dtype=torch.float32)
    ys = torch.tensor([p[1] for p in pts_xy], device=device, dtype=torch.float32)
    gx = 2.0 * xs / max(W - 1, 1) - 1.0
    gy = 2.0 * ys / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    return grid


@torch.no_grad()
def _avg_pool_feat(feat: torch.Tensor, stride: int) -> torch.Tensor:
    """简单平均池化得到粗尺度特征 (C, Hc, Wc)"""
    if stride <= 1:
        return feat
    x = feat.unsqueeze(0)
    x = F.avg_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=True)
    return x.squeeze(0)


@torch.no_grad()
def _sample_learned_descriptor(
    feat_map: torch.Tensor,          # (C,Hf,Wf)
    img_hw: tuple,                   # (H,W)
    pts_xy: List[tuple],
    bank_sim: Optional[torch.Tensor] = None,  # (Hf,Wf) or (H,W) or None
    add_bank_channel: bool = True,
    bank_weight: float = 0.7,
) -> torch.Tensor:
    """在关键点处采样“学习特征 + Bank”，并 L2 归一化"""
    assert feat_map.dim() == 3, "feat_map must be (C,Hf,Wf)"
    C, Hf, Wf = feat_map.shape
    dev = feat_map.device
    H, W = img_hw

    grid = _points_to_grid_norm(pts_xy, H, W, dev)
    feats = F.grid_sample(
        feat_map.unsqueeze(0),  # (1,C,Hf,Wf)
        grid,
        mode="bilinear",
        align_corners=True
    )  # -> (1,C,N,1)
    desc = feats.squeeze(0).squeeze(-1).T  # (N,C)

    if add_bank_channel and bank_sim is not None:
        if bank_sim.dim() == 2 and bank_sim.shape == (Hf, Wf):
            sim_map = bank_sim.unsqueeze(0).unsqueeze(0)  # (1,1,Hf,Wf)
        else:
            if bank_sim.dim() == 2:
                sim_map = bank_sim.unsqueeze(0).unsqueeze(0).float()
            else:
                sim_map = bank_sim
                if sim_map.dim() == 3:
                    sim_map = sim_map.unsqueeze(0)
            sim_map = F.interpolate(sim_map, size=(Hf, Wf), mode="bilinear", align_corners=False)
        sim_vals = F.grid_sample(sim_map, grid, mode="bilinear", align_corners=True)  # (1,1,N,1)
        sim_vals = sim_vals.flatten(1).T  # (N,1)
        desc = torch.cat([desc, bank_weight * sim_vals], dim=1)  # (N,C+1)

    desc = F.normalize(desc, dim=1, eps=1e-8)
    return desc


# =============================================================
# 粗到细候选（Coarse→Fine TopK）
# =============================================================
@torch.no_grad()
def _build_topk_gate_from_coarse(
    featL: torch.Tensor, featR: torch.Tensor,
    ptsL_xy: List[tuple], ptsR_xy: List[tuple],
    img_hw: tuple, stride: int = 4, topk: int = 3
) -> Optional[torch.Tensor]:
    """粗尺度上为每个 i 选 Top-K 候选 j，返回 NA×NB 的布尔门控矩阵"""
    if stride <= 1 or len(ptsL_xy) == 0 or len(ptsR_xy) == 0:
        return None
    dev = featL.device
    cL = _avg_pool_feat(featL, stride)
    cR = _avg_pool_feat(featR, stride)
    H, W = img_hw
    descLc = _sample_learned_descriptor(cL, img_hw, ptsL_xy, None, add_bank_channel=False)
    descRc = _sample_learned_descriptor(cR, img_hw, ptsR_xy, None, add_bank_channel=False)
    S = torch.matmul(descLc, descRc.t())  # (NA,NB)
    NA, NB = S.shape
    k = min(topk, NB)
    _, idx = torch.topk(S, k=k, dim=1)  # (NA,k)
    mask = torch.zeros((NA, NB), dtype=torch.bool, device=dev)
    ar = torch.arange(NA, device=dev).unsqueeze(1).expand_as(idx)
    mask[ar, idx] = True
    return mask

@torch.no_grad()
def _epipolar_bias_from_F(
    xyL: torch.Tensor, xyR: torch.Tensor, Fmat: np.ndarray,
    sigma_px: float = 1.5, device=None
):
    """对称极线距离的负二次偏置：-d^2/(2σ^2)"""
    if Fmat is None:
        return None
    if device is None:
        device = xyL.device
    NA = xyL.shape[0]; NB = xyR.shape[0]
    xL = torch.cat([xyL, torch.ones((NA, 1), device=device)], dim=1)  # (NA,3)
    xR = torch.cat([xyR, torch.ones((NB, 1), device=device)], dim=1)  # (NB,3)
    F = torch.tensor(Fmat, dtype=torch.float32, device=device)
    lR = torch.matmul(xL, F.t())              # (NA,3)
    lL = torch.matmul(xR, F)                  # (NB,3)
    num = torch.matmul(torch.matmul(xL, F.t()), xR.t()).abs()  # (NA,NB)
    den1 = torch.sqrt(lR[:, 0:1] ** 2 + lR[:, 1:2] ** 2)       # (NA,1)
    den2 = torch.sqrt(lL[:, 0:1] ** 2 + lL[:, 1:2] ** 2).t()   # (1,NB)
    den = den1 + den2 + 1e-6
    d = num / den
    bias = -(d ** 2) / (2.0 * (sigma_px ** 2))
    return bias


# =============================================================
# 自注意 / 交叉注意
# =============================================================
def _knn_indices(xy: torch.Tensor, k: int) -> torch.Tensor:
    N = xy.shape[0]
    if N == 0:
        return torch.empty((0, 0), dtype=torch.long, device=xy.device)
    d2 = (xy[:, None, :] - xy[None, :, :]).pow(2).sum(-1)
    _, idx = torch.topk(-d2, k=min(k, N), dim=1)
    return idx


@torch.no_grad()
def _attn_agg(desc: torch.Tensor, xy: torch.Tensor, k_idx: torch.Tensor, pos_sigma: float = 24.0) -> torch.Tensor:
    if desc.shape[0] == 0:
        return desc
    N, D = desc.shape
    nbr = desc[k_idx]  # (N,k,D)
    dxy = xy[:, None, :] - xy[k_idx]
    pos_bias = -dxy.pow(2).sum(-1) / (2.0 * (pos_sigma ** 2) + 1e-8)
    attn_logits = (nbr * desc[:, None, :]).sum(-1) / math.sqrt(D)
    attn_logits = attn_logits + pos_bias
    w = F.softmax(attn_logits, dim=1)
    agg = (w[..., None] * nbr).sum(1)
    out = F.normalize(desc + agg, dim=1, eps=1e-8)
    return out


@torch.no_grad()
def _cross_attn_agg(descA: torch.Tensor, xyA: torch.Tensor,
                    descB: torch.Tensor, xyB: torch.Tensor,
                    gate_mask: Optional[torch.Tensor] = None,
                    geo_sigma: float = 32.0):
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        zA = torch.zeros((descA.shape[0],), device=descA.device)
        zB = torch.zeros((descB.shape[0],), device=descB.device)
        return descA, descB, (zA, zB)
    S = torch.matmul(descA, descB.t())
    dxy2 = (xyA[:, None, :] - xyB[None, :, :]).pow(2).sum(-1)
    geo_bias = -dxy2 / (2.0 * (geo_sigma ** 2) + 1e-8)
    S = S + geo_bias
    if gate_mask is not None:
        S = S.masked_fill(~gate_mask, -1e9)
    wAB = F.softmax(S, dim=1)
    aggA = torch.matmul(wAB, descB)
    outA = F.normalize(descA + aggA, dim=1, eps=1e-8)
    wBA = F.softmax(S.t(), dim=1)
    aggB = torch.matmul(wBA, descA)
    outB = F.normalize(descB + aggB, dim=1, eps=1e-8)
    respA = wAB.max(dim=1).values
    respB = wBA.max(dim=1).values
    return outA, outB, (respA, respB)


# =============================================================
# 极小 MLP 重加权头（可选）
# =============================================================

_REWEIGHT_HEAD = None
_REWEIGHT_CKPT = "/home/zj/EVEE/weights/PrEEVEE.pth"

def _extract_state_dict(ckpt_obj):
    # 兼容：纯 state_dict / Lightning {'state_dict': ...} / {'model': ...}
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "net", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj

def _load_matching_weights(dst_module, ckpt_sd: dict):
    """只把 shape 对得上的键加载进来，避免前缀/多余参数导致 strict load 失败。"""
    dst_sd = dst_module.state_dict()
    filtered = {}

    # 常见前缀：pair_reweighter., reweighter., module., model.
    prefixes = ["pair_reweighter.", "reweighter.", "module.", "model."]

    def strip_prefix(key: str):
        for p in prefixes:
            if key.startswith(p):
                return key[len(p):]
        return key

    for k, v in ckpt_sd.items():
        kk = strip_prefix(k)
        if kk in dst_sd and hasattr(v, "shape") and v.shape == dst_sd[kk].shape:
            filtered[kk] = v

    missing, unexpected = dst_module.load_state_dict(filtered, strict=False)
    return missing, unexpected, list(filtered.keys())

def _get_reweighter(device):
    global _REWEIGHT_HEAD
    if _REWEIGHT_HEAD is None:
        rw = core.CNN_FEAT.PairReweighter().to(device).eval()

        ckpt_path = os.environ.get("EVEE_REWEIGHT_CKPT", _REWEIGHT_CKPT)
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_sd = _extract_state_dict(ckpt)

        missing, unexpected, loaded_keys = _load_matching_weights(rw, ckpt_sd)

        # 你也可以临时打开下面这行做确认（确认后再注释掉）
        print(f"[ReWeight] loaded={len(loaded_keys)} missing={missing} unexpected={unexpected}")

        _REWEIGHT_HEAD = rw
    return _REWEIGHT_HEAD

# =============================================================
# dual-softmax with dustbin
# =============================================================
@torch.no_grad()
def _dual_softmax_with_dustbin(S: torch.Tensor, tau: float = 0.04, dustbin_bias: float = 0.1):
    """
    S: (NA, NB) 相似度（未除以 tau）
    返回:
      P         : (NA, NB) 概率矩阵
      row_db    : (NA,)    每个 i 选择“不匹配”的概率
      col_db    : (NB,)    每个 j 选择“不匹配”的概率
    """
    NA, NB = S.shape
    Sd = S / max(tau, 1e-6)

    # 行 softmax: NB+1 (含 dustbin)
    row_db_col = torch.full((NA, 1), dustbin_bias, device=S.device)
    P_row = F.softmax(torch.cat([Sd, row_db_col], dim=1), dim=1)  # (NA, NB+1)

    # 列 softmax: NA+1 (含 dustbin)
    col_db_row = torch.full((1, NB), dustbin_bias, device=S.device)
    P_col = F.softmax(torch.cat([Sd, col_db_row], dim=0), dim=0)  # (NA+1, NB)

    P = P_row[:, :NB] * P_col[:NA, :]  # (NA, NB)
    row_db = P_row[:, -1]              # (NA,)
    col_db = P_col[-1, :]              # (NB,)
    return P, row_db, col_db


# =============================================================
# 分布式匹配 + 先验/重加权
# =============================================================
@torch.no_grad()
def _match_with_priors(
    descA, descB, xyA, xyB,
    tau: float, conf_thr: float,
    gate_mask: Optional[torch.Tensor] = None,
    bank_lambda: float = 0.0,
    use_reweighter: bool = False,
    epi_bias: Optional[torch.Tensor] = None,
    geo_sigma_for_rew: float = 32.0,
    use_dustbin: bool = True,
    dustbin_bias: float = 0.1,
    has_bank_channel: bool = False,
):
    # 1) 余弦相似
    S = torch.matmul(descA, descB.t())  # (NA,NB)

    # 2) Bank 先验（仅当确实有 bank 通道时）
    if has_bank_channel and bank_lambda > 0:
        bankA = descA[:, -1]
        bankB = descB[:, -1]
        S = S + bank_lambda * (bankA[:, None] + bankB[None, :])

    # 3) 几何门控 + 极线偏置
    if gate_mask is not None:
        S = S.masked_fill(~gate_mask, -1e9)
    if epi_bias is not None:
        S = S + epi_bias

    # 4) 极小 MLP 重加权（可选）
    if use_reweighter:
        dxy2 = (xyA[:, None, :] - xyB[None, :, :]).pow(2).sum(-1)
        geo_bias = -dxy2 / (2.0 * (geo_sigma_for_rew ** 2) + 1e-8)
        # 若无 bank 通道，传零向量即可
        if has_bank_channel:
            bankA = descA[:, -1]
            bankB = descB[:, -1]
        else:
            bankA = torch.zeros(xyA.shape[0], device=xyA.device)
            bankB = torch.zeros(xyB.shape[0], device=xyB.device)
        w = _get_reweighter(descA.device).score(S, geo_bias, bankA, bankB)
        S = S + torch.log(w + 1e-6)  # 等价于概率乘 w

    # 5) 分布式匹配（支持 dustbin）
    if use_dustbin:
        P, row_db, col_db = _dual_softmax_with_dustbin(S, tau=tau, dustbin_bias=dustbin_bias)
    else:
        Sd = S / max(tau, 1e-6)
        P = F.softmax(Sd, dim=1) * F.softmax(Sd, dim=0)

    # 6) 互一致 + 置信/不匹配过滤
    i_max = P.argmax(dim=1)
    j_max = P.argmax(dim=0)
    matches = []
    for i in range(P.shape[0]):
        j = int(i_max[i])
        if j_max[j].item() == i and P[i, j].item() >= conf_thr:
            if use_dustbin:
                if P[i, j].item() <= max(float(row_db[i].item()), float(col_db[j].item())):
                    continue
            matches.append((i, j))
    return matches


# =============================================================
# 主流程：CGM（上下文图匹配）
# =============================================================
@torch.no_grad()
def _contextual_match(
    imgL_bgr, imgR_bgr,
    pts_L: PointList, pts_R: PointList,
    feat_L: torch.Tensor, feat_R: torch.Tensor,
    img_hw: Optional[tuple] = None,
    bank_sim_L: Optional[torch.Tensor] = None,
    bank_sim_R: Optional[torch.Tensor] = None,
    # 参数
    k_nn: int = 2,
    adaptive_k: bool = True,
    self_pos_sigma: float = 24.0,
    cross_geo_sigma: float = 32.0,
    tau: float = 0.04,
    conf_thr_stage: tuple = (0.7, 0.5, 0.3),
    max_matches: int = 1000,
    max_dist_px: Optional[float] = None,
    bank_prior_lambda: float = 0.08,
    use_reweighter: bool = True,
    # 粗到细
    use_coarse: bool = True,
    coarse_stride: int = 4,
    coarse_topk: int = 5,
    # 极线先验
    F_prior: Optional[np.ndarray] = None,
    epi_sigma_px: float = 1.5,
    epi_lambda: float = 0.6,
    # dustbin
    use_dustbin: bool = True,
    dustbin_bias: float = 0.1,
):
    H = img_hw[0] if img_hw is not None else imgR_bgr.shape[0]
    W = img_hw[1] if img_hw is not None else imgR_bgr.shape[1]

    ptsL_xy = list(zip(pts_L[0], pts_L[1]))
    ptsR_xy = list(zip(pts_R[0], pts_R[1]))
    if len(ptsL_xy) == 0 or len(ptsR_xy) == 0:
        return [], []

    # 采样“学习 + Bank”描述子
    descL = _sample_learned_descriptor(feat_L, (H, W), ptsL_xy, bank_sim_L)
    descR = _sample_learned_descriptor(feat_R, (H, W), ptsR_xy, bank_sim_R)
    has_bank_channel = (bank_sim_L is not None) and (bank_sim_R is not None)

    # 坐标张量
    dev = descL.device
    xyL = torch.tensor(ptsL_xy, dtype=torch.float32, device=dev)
    xyR = torch.tensor(ptsR_xy, dtype=torch.float32, device=dev)

    # 几何门控（max_dist_px）
    gate_mask = None
    if max_dist_px is not None and max_dist_px > 0:
        d2 = (xyL[:, None, :] - xyR[None, :, :]).pow(2).sum(-1)
        gate_mask = (d2 <= float(max_dist_px) ** 2)

    # 粗尺度 TopK 门控
    if use_coarse:
        coarse_mask = _build_topk_gate_from_coarse(feat_L, feat_R, ptsL_xy, ptsR_xy, (H, W),
                                                   stride=coarse_stride, topk=coarse_topk)
        if coarse_mask is not None:
            gate_mask = coarse_mask if gate_mask is None else (gate_mask & coarse_mask)

    # 自适应 k
    kL_val = k_nn
    kR_val = k_nn
    if adaptive_k:
        kL_val = max(8, min(k_nn, max(8, xyL.shape[0] // 200)))
        kR_val = max(8, min(k_nn, max(8, xyR.shape[0] // 200)))

    # 第 1 阶段：各自自注意
    kL = _knn_indices(xyL, kL_val)
    kR = _knn_indices(xyR, kR_val)
    descL = _attn_agg(descL, xyL, kL, pos_sigma=self_pos_sigma)
    descR = _attn_agg(descR, xyR, kR, pos_sigma=self_pos_sigma)

    # 极线偏置（先验）
    epi_bias = None
    if F_prior is not None and epi_lambda > 0:
        epi = _epipolar_bias_from_F(xyL, xyR, F_prior, sigma_px=epi_sigma_px, device=dev)
        if epi is not None:
            epi_bias = epi_lambda * epi

    # 尝试匹配 + 早停
    m12 = _match_with_priors(
        descL, descR, xyL, xyR, tau=tau, conf_thr=conf_thr_stage[0],
        gate_mask=gate_mask, bank_lambda=bank_prior_lambda,
        use_reweighter=use_reweighter, epi_bias=epi_bias,
        geo_sigma_for_rew=cross_geo_sigma, use_dustbin=use_dustbin,
        dustbin_bias=dustbin_bias, has_bank_channel=has_bank_channel
    )
    if len(m12) > 0:
        m12 = m12[:max_matches]
        good_L = [ptsL_xy[i] for i, _ in m12]
        good_R = [ptsR_xy[j] for _, j in m12]
        return good_L, good_R

    # 第 2 阶段：交叉注意 + matchability 蒙版
    descL, descR, (respL, respR) = _cross_attn_agg(descL, xyL, descR, xyR, gate_mask, geo_sigma=cross_geo_sigma)
    maskL = (respL >= 0.05)
    maskR = (respR >= 0.05)
    if maskL.any() and maskR.any():
        descL2 = descL[maskL]
        descR2 = descR[maskR]
        xyL2 = xyL[maskL]
        xyR2 = xyR[maskR]
        gate2 = None
        if gate_mask is not None:
            gate2 = gate_mask[maskL][:, maskR]
        epi2 = epi_bias[maskL][:, maskR] if epi_bias is not None else None
        m12 = _match_with_priors(
            descL2, descR2, xyL2, xyR2, tau=tau, conf_thr=conf_thr_stage[1],
            gate_mask=gate2, bank_lambda=bank_prior_lambda,
            use_reweighter=use_reweighter, epi_bias=epi2,
            geo_sigma_for_rew=cross_geo_sigma, use_dustbin=use_dustbin,
            dustbin_bias=dustbin_bias, has_bank_channel=has_bank_channel
        )
        if len(m12) > 0:
            idxL = torch.nonzero(maskL, as_tuple=False).squeeze(1)
            idxR = torch.nonzero(maskR, as_tuple=False).squeeze(1)
            m12 = [(int(idxL[i].item()), int(idxR[j].item())) for (i, j) in m12]
            m12 = m12[:max_matches]
            good_L = [ptsL_xy[i] for i, _ in m12]
            good_R = [ptsR_xy[j] for _, j in m12]
            return good_L, good_R

    # 第 3 阶段：再一轮自注意 + 交叉注意
    descL = _attn_agg(descL, xyL, kL, pos_sigma=self_pos_sigma)
    descR = _attn_agg(descR, xyR, kR, pos_sigma=self_pos_sigma)
    descL, descR, _ = _cross_attn_agg(descL, xyL, descR, xyR, gate_mask, geo_sigma=cross_geo_sigma)

    m12 = _match_with_priors(
        descL, descR, xyL, xyR, tau=tau, conf_thr=conf_thr_stage[2],
        gate_mask=gate_mask, bank_lambda=bank_prior_lambda,
        use_reweighter=use_reweighter, epi_bias=epi_bias,
        geo_sigma_for_rew=cross_geo_sigma, use_dustbin=use_dustbin,
        dustbin_bias=dustbin_bias, has_bank_channel=has_bank_channel
    )
    if len(m12) == 0:
        return [], []

    m12 = m12[:max_matches]
    good_L = [ptsL_xy[i] for i, _ in m12]
    good_R = [ptsR_xy[j] for _, j in m12]
    return good_L, good_R


# =============================================================
# 两段式(EM) 姿态引导再匹配
# =============================================================
@torch.no_grad()
def match_two_stage_with_pose(
    prev_img_bgr, curr_img_bgr,
    prev_pts, curr_pts,
    prev_feat_map, curr_feat_map,
    prev_bank_sim=None, curr_bank_sim=None,
    *,               # 第一阶段（无先验）
    k_nn=2, tau=0.04, conf_thr_stage=(0.7, 0.5, 0.3),
    max_dist=20.0, max_matches=1000,
    adaptive_k=True, bank_prior_lambda=0.08,
    use_reweighter=True, use_coarse=True,
    coarse_stride=4, coarse_topk=5,
    use_dustbin=True, dustbin_bias=0.1,
    # 第二阶段（带极线先验 + 更严阈值）
    second_tau=0.03, second_conf=(0.8, 0.6, 0.3),
    epi_sigma_px=1.5, epi_lambda=0.6,
):
    H, W = curr_img_bgr.shape[:2]

    # —— 第一轮：无先验 ——
    L1, R1 = _contextual_match(
        prev_img_bgr, curr_img_bgr, prev_pts, curr_pts,
        prev_feat_map, curr_feat_map,
        img_hw=(H, W),
        bank_sim_L=prev_bank_sim, bank_sim_R=curr_bank_sim,
        k_nn=k_nn, adaptive_k=adaptive_k, tau=tau, conf_thr_stage=conf_thr_stage,
        max_matches=max_matches, max_dist_px=max_dist,
        bank_prior_lambda=bank_prior_lambda, use_reweighter=use_reweighter,
        use_coarse=use_coarse, coarse_stride=coarse_stride, coarse_topk=coarse_topk,
        F_prior=None, epi_sigma_px=epi_sigma_px, epi_lambda=0.0,
        use_dustbin=use_dustbin, dustbin_bias=dustbin_bias
    )
    if len(L1) < 8:
        return L1, R1, None

    # —— 用第一轮结果估 F ——
    F_hat, _ = cv2.findFundamentalMat(np.float32(L1), np.float32(R1), cv2.FM_RANSAC, 1.0, 0.999)

    # —— 第二轮：带 F 先验（更严格阈值/温度 + dustbin）——
    L2, R2 = _contextual_match(
        prev_img_bgr, curr_img_bgr, prev_pts, curr_pts,
        prev_feat_map, curr_feat_map,
        img_hw=(H, W),
        bank_sim_L=prev_bank_sim, bank_sim_R=curr_bank_sim,
        k_nn=k_nn, adaptive_k=adaptive_k, tau=second_tau, conf_thr_stage=second_conf,
        max_matches=max_matches, max_dist_px=max_dist,
        bank_prior_lambda=bank_prior_lambda, use_reweighter=use_reweighter,
        use_coarse=use_coarse, coarse_stride=coarse_stride, coarse_topk=coarse_topk,
        F_prior=F_hat if (F_hat is not None and F_hat.shape == (3, 3)) else None,
        epi_sigma_px=epi_sigma_px, epi_lambda=epi_lambda,
        use_dustbin=use_dustbin, dustbin_bias=dustbin_bias
    )
    return L2, R2, F_hat


# =============================================================
# 姿态用的空间多样性采样（可在外部RANSAC前调用）
# =============================================================
def select_diverse_matches(good_L, good_R, img_hw, grid: int = 8, per_cell: int = 5):
    H, W = img_hw
    cell_h = max(1, H // grid)
    cell_w = max(1, W // grid)
    buckets = {}
    for idx, ((xl, yl), (xr, yr)) in enumerate(zip(good_L, good_R)):
        gx = min(grid - 1, int(xl // cell_w))
        gy = min(grid - 1, int(yl // cell_h))
        buckets.setdefault((gx, gy), []).append(idx)
    keep = []
    for _, idxs in buckets.items():
        keep.extend(idxs[:per_cell])
    keep = sorted(keep)
    L2 = [good_L[i] for i in keep]
    R2 = [good_R[i] for i in keep]
    return L2, R2


# =============================================================
# 统一入口：优先 CGM（两段式+增强），其次 ORB
# =============================================================
@torch.no_grad()
def match_two_sets(prev_img_bgr,
                   curr_img_bgr,
                   prev_pts: PointList,
                   curr_pts: PointList,
                   *,
                   max_dist: float = 20.0,
                   orb_nfeatures: int = 2000,
                   max_matches: int = 500,
                   prev_feat_map: Optional[torch.Tensor] = None,   # (C,Hf,Wf)
                   curr_feat_map: Optional[torch.Tensor] = None,   # (C,Hf,Wf)
                   prev_bank_sim: Optional[torch.Tensor] = None,   # (Hf,Wf) or (H,W)
                   curr_bank_sim: Optional[torch.Tensor] = None,   # (Hf,Wf) or (H,W)
                   ratio: float = 0.8,
                   # CGM/增强默认参数（已针对姿态指标调优）
                   k_nn: int = 2,
                   tau: float = 0.04,
                   conf_thr_stage: tuple = (0.7, 0.5, 0.3),
                   adaptive_k: bool = True,
                   bank_prior_lambda: float = 0.08,
                   use_reweighter: bool = True,
                   use_coarse: bool = True,
                   coarse_stride: int = 4,
                   coarse_topk: int = 5,
                   use_dustbin: bool = True,
                   dustbin_bias: float = 0.1,
                   second_tau: float = 0.03,
                   second_conf: tuple = (0.8, 0.6, 0.3),
                   epi_sigma_px: float = 1.5,
                   epi_lambda: float = 0.6):
    use_learned = (prev_feat_map is not None) and (curr_feat_map is not None)
    if use_learned:
        good_L, good_R, _F = match_two_stage_with_pose(
            prev_img_bgr, curr_img_bgr, prev_pts, curr_pts,
            prev_feat_map, curr_feat_map,
            prev_bank_sim=prev_bank_sim, curr_bank_sim=curr_bank_sim,
            k_nn=k_nn, tau=tau, conf_thr_stage=conf_thr_stage,
            max_dist=max_dist, max_matches=max_matches,
            adaptive_k=adaptive_k, bank_prior_lambda=bank_prior_lambda,
            use_reweighter=use_reweighter, use_coarse=use_coarse,
            coarse_stride=coarse_stride, coarse_topk=coarse_topk,
            use_dustbin=use_dustbin, dustbin_bias=dustbin_bias,
            second_tau=second_tau, second_conf=second_conf,
            epi_sigma_px=epi_sigma_px, epi_lambda=epi_lambda
        )
        return good_L[:max_matches], good_R[:max_matches]

    # —— 后备 ORB ——（保持原样）
    left_gray = _to_gray(prev_img_bgr)
    right_gray = _to_gray(curr_img_bgr)
    matcher = ORBMatcher(nfeatures=orb_nfeatures)
    pts_prev_xy = list(zip(prev_pts[0], prev_pts[1]))
    pts_curr_xy = list(zip(curr_pts[0], curr_pts[1]))
    pL, pR, _ = matcher.match_by_points(left_gray, right_gray, pts_prev_xy, pts_curr_xy,
                                        max_matches=max_matches, ratio=0.9)

    dthr2 = float(max_dist) * float(max_dist)
    good_L, good_R = [], []
    for (xl, yl), (xr, yr) in zip(pL, pR):
        dx = xr - xl; dy = yr - yl
        if dx * dx + dy * dy <= dthr2:
            good_L.append((xl, yl)); good_R.append((xr, yr))
    return good_L, good_R


@torch.no_grad()
def match_and_draw_2panel(prev_img_bgr,
                          curr_img_bgr,
                          prev_pts: PointList,
                          curr_pts: PointList,
                          *,
                          max_dist: float = 20.0,
                          orb_nfeatures: int = 2000,
                          max_matches: int = 500,
                          draw_right_points: bool = True,
                          right_point_color=(255, 255, 255),
                          thickness: int = 1,
                          draw_match_lines: bool = True,
                          prev_feat_map: Optional[torch.Tensor] = None,
                          curr_feat_map: Optional[torch.Tensor] = None,
                          prev_bank_sim: Optional[torch.Tensor] = None,
                          curr_bank_sim: Optional[torch.Tensor] = None,
                          ratio: float = 0.8,
                          # 传递增强参数（用默认即可）
                          k_nn: int = 2,
                          tau: float = 0.04,
                          conf_thr_stage: tuple = (0.7, 0.5, 0.3),
                          adaptive_k: bool = True,
                          bank_prior_lambda: float = 0.08,
                          use_reweighter: bool = True,
                          use_coarse: bool = True,
                          coarse_stride: int = 4,
                          coarse_topk: int = 5,
                          use_dustbin: bool = True,
                          dustbin_bias: float = 0.1,
                          second_tau: float = 0.03,
                          second_conf: tuple = (0.8, 0.6, 0.3),
                          epi_sigma_px: float = 1.5,
                          epi_lambda: float = 0.6):
    good_L, good_R = match_two_sets(
        prev_img_bgr, curr_img_bgr, prev_pts, curr_pts,
        max_dist=max_dist, orb_nfeatures=orb_nfeatures, max_matches=max_matches,
        prev_feat_map=prev_feat_map, curr_feat_map=curr_feat_map,
        prev_bank_sim=prev_bank_sim, curr_bank_sim=curr_bank_sim,
        ratio=ratio, k_nn=k_nn, tau=tau, conf_thr_stage=conf_thr_stage,
        adaptive_k=adaptive_k, bank_prior_lambda=bank_prior_lambda,
        use_reweighter=use_reweighter, use_coarse=use_coarse,
        coarse_stride=coarse_stride, coarse_topk=coarse_topk,
        use_dustbin=use_dustbin, dustbin_bias=dustbin_bias,
        second_tau=second_tau, second_conf=second_conf,
        epi_sigma_px=epi_sigma_px, epi_lambda=epi_lambda
    )

    h = max(prev_img_bgr.shape[0], curr_img_bgr.shape[0])
    w = prev_img_bgr.shape[1] + curr_img_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:prev_img_bgr.shape[0], :prev_img_bgr.shape[1]] = prev_img_bgr
    canvas[:curr_img_bgr.shape[0], prev_img_bgr.shape[1]:prev_img_bgr.shape[1] + curr_img_bgr.shape[1]] = curr_img_bgr

    xoff = prev_img_bgr.shape[1]
    color = (0, 255, 0)
    for (xl, yl), (xr, yr) in zip(good_L, good_R):
        pt1 = (int(round(xl)), int(round(yl)))
        pt2 = (int(round(xr + xoff)), int(round(yr)))
        if draw_match_lines:
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 2, color, -1, cv2.LINE_AA)

    if draw_right_points and curr_pts[0]:
        for x, y in zip(curr_pts[0], curr_pts[1]):
            cv2.circle(canvas, (int(x) + xoff, int(y)), 2, right_point_color, -1, cv2.LINE_AA)

    return canvas


@torch.no_grad()
def match_and_draw_overlay(prev_img_bgr,
                           curr_img_bgr,
                           prev_pts: PointList,
                           curr_pts: PointList,
                           *,
                           max_dist: float = 20.0,
                           orb_nfeatures: int = 2000,
                           max_matches: int = 500,
                           line_color=(0, 255, 0),
                           thickness: int = 1,
                           draw_match_lines: bool = True,
                           prev_feat_map: Optional[torch.Tensor] = None,
                           curr_feat_map: Optional[torch.Tensor] = None,
                           prev_bank_sim: Optional[torch.Tensor] = None,
                           curr_bank_sim: Optional[torch.Tensor] = None,
                           ratio: float = 0.8,
                           k_nn: int = 2,
                           tau: float = 0.04,
                           conf_thr_stage: tuple = (0.7, 0.5, 0.3),
                           adaptive_k: bool = True,
                           bank_prior_lambda: float = 0.08,
                           use_reweighter: bool = True,
                           use_coarse: bool = True,
                           coarse_stride: int = 4,
                           coarse_topk: int = 5,
                           use_dustbin: bool = True,
                           dustbin_bias: float = 0.1,
                           second_tau: float = 0.03,
                           second_conf: tuple = (0.8, 0.6, 0.3),
                           epi_sigma_px: float = 1.5,
                           epi_lambda: float = 0.6):
    good_L, good_R = match_two_sets(
        prev_img_bgr, curr_img_bgr, prev_pts, curr_pts,
        max_dist=max_dist, orb_nfeatures=orb_nfeatures, max_matches=max_matches,
        prev_feat_map=prev_feat_map, curr_feat_map=curr_feat_map,
        prev_bank_sim=prev_bank_sim, curr_bank_sim=curr_bank_sim,
        ratio=ratio, k_nn=k_nn, tau=tau, conf_thr_stage=conf_thr_stage,
        adaptive_k=adaptive_k, bank_prior_lambda=bank_prior_lambda,
        use_reweighter=use_reweighter, use_coarse=use_coarse,
        coarse_stride=coarse_stride, coarse_topk=coarse_topk,
        use_dustbin=use_dustbin, dustbin_bias=dustbin_bias,
        second_tau=second_tau, second_conf=second_conf,
        epi_sigma_px=epi_sigma_px, epi_lambda=epi_lambda
    )

    canvas = curr_img_bgr.copy()
    for (xl, yl), (xr, yr) in zip(good_L, good_R):
        if draw_match_lines:
            cv2.line(canvas, (int(xl), int(yl)), (int(xr), int(yr)), line_color, thickness, cv2.LINE_AA)
    return canvas


# =============================================================
# 有状态封装（接口兼容原版本）
# =============================================================
class StatefulMatcher:
    def __init__(self,
                 max_dist: float = 20.0,
                 orb_nfeatures: int = 2000,
                 max_matches: int = 500,
                 ratio: float = 0.8,
                 k_nn: int = 2,
                 tau: float = 0.04,
                 conf_thr_stage: tuple = (0.7, 0.5, 0.3)):
        self.max_dist = float(max_dist)
        self.orb_nfeatures = int(orb_nfeatures)
        self.max_matches = int(max_matches)
        self.ratio = float(ratio)
        self.k_nn = int(k_nn)
        self.tau = float(tau)
        self.conf_thr_stage = tuple(conf_thr_stage)
        self._last_img = {}
        self._last_pts = {}
        self._last_feat = {}   # key -> torch.Tensor (C,Hf,Wf)
        self._last_bank = {}   # key -> torch.Tensor (Hf,Wf) or (H,W)

    def reset(self, key: Optional[str] = None):
        if key is None:
            self._last_img.clear(); self._last_pts.clear()
            self._last_feat.clear(); self._last_bank.clear()
        else:
            self._last_img.pop(key, None); self._last_pts.pop(key, None)
            self._last_feat.pop(key, None); self._last_bank.pop(key, None)

    @torch.no_grad()
    def step(self,
             key: str,
             curr_img_bgr,
             curr_pts: PointList,
             *,
             right_point_color=(255, 255, 255),
             draw_right_points: bool = True,
             draw_match_lines: bool = True,
             thickness: int = 1,
             two_panel: bool = True,
             curr_feat_map: Optional[torch.Tensor] = None,  # (C,Hf,Wf)
             curr_bank_sim: Optional[torch.Tensor] = None,  # (Hf,Wf) or (H,W)
             use_reweighter: bool = True,
             ratio: Optional[float] = None):
        prev_img = self._last_img.get(key, None)
        prev_pts = self._last_pts.get(key, ([], []))
        prev_feat = self._last_feat.get(key, None)
        prev_bank = self._last_bank.get(key, None)

        # 更新缓存
        self._last_img[key] = curr_img_bgr.copy()
        self._last_pts[key] = curr_pts
        if curr_feat_map is not None:
            self._last_feat[key] = curr_feat_map
        if curr_bank_sim is not None:
            self._last_bank[key] = curr_bank_sim

        if prev_img is None or not prev_pts[0] or not curr_pts[0]:
            return None

        use_learned = (prev_feat is not None) and (curr_feat_map is not None)

        if two_panel:
            return match_and_draw_2panel(
                prev_img, curr_img_bgr, prev_pts, curr_pts,
                max_dist=self.max_dist, orb_nfeatures=self.orb_nfeatures, max_matches=self.max_matches,
                draw_right_points=draw_right_points, right_point_color=right_point_color, thickness=thickness,
                draw_match_lines=draw_match_lines,
                prev_feat_map=prev_feat if use_learned else None,
                curr_feat_map=curr_feat_map if use_learned else None,
                prev_bank_sim=prev_bank if use_learned else None,
                curr_bank_sim=curr_bank_sim if use_learned else None,
                ratio=(self.ratio if ratio is None else ratio),
                use_reweighter=use_reweighter  # 新增：传递给下游
            )
        else:
            return match_and_draw_overlay(
                prev_img, curr_img_bgr, prev_pts, curr_pts,
                max_dist=self.max_dist, orb_nfeatures=self.orb_nfeatures, max_matches=self.max_matches,
                line_color=(0, 255, 0), thickness=thickness,
                draw_match_lines=draw_match_lines,
                prev_feat_map=prev_feat if use_learned else None,
                curr_feat_map=curr_feat_map if use_learned else None,
                prev_bank_sim=prev_bank if use_learned else None,
                curr_bank_sim=curr_bank_sim if use_learned else None,
                ratio=(self.ratio if ratio is None else ratio),
                use_reweighter=use_reweighter  # 新增：传递给下游
            )
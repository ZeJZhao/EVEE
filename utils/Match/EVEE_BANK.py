from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EVEE_Bank(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        max_len: int = 32,
        max_protos: int | None = None,
        merge_thr: float = 0.9,
        vote_decay: float = 0.6,
        vote_alpha: float = 2,
        spatial_sigma: float = 800.0,  # 位置敏感度，越小越依赖空间位置

        # ===== 新增：鲁棒池化/对比式Bank/时序平滑 =====
        pool: str = "topk",            # "max" | "topk" | "lse"
        topk: int = 5,                 # pool="topk" 时使用（你的默认值）
        temp: float = 0.07,            # pool="lse" 或 contrast 时的温度
        enable_neg_bank: bool = True,  # 负样本Bank
        neg_lambda: float = 0.5,       # 对比项系数：pos - λ*neg
        smooth_alpha: float = 0.0,     # 时序平滑系数[0,1)，0为关闭

        # ===== 新增：小波门控/去噪（默认关闭，保持向后兼容）=====
        enable_wavelet: bool = True,  # 开关：小波门控/去噪
        wavelet_mode: str = "gate",    # "gate" | "denoise"
        wavelet_levels: int = 1,       # 1 足够；想更稳可用 2
        wavelet_thr: float = 0.15,     # 门槛：越大越挑剔
        wavelet_gamma: float = 6.0,    # Sigmoid 陡峭度
        wavelet_beta: float = 0.3,     # 对 Δ 的放大量（0.2~0.4）
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_protos = max_protos or max_len
        self.merge_thr = float(merge_thr)
        self.vote_decay = float(vote_decay)
        self.vote_alpha = float(vote_alpha)
        self.spatial_sigma = float(spatial_sigma)

        # 旧有buffer
        self.register_buffer("protos", None)        # (P,C)
        self.register_buffer("votes", None)         # (P,)
        self.register_buffer("proto_coords", None)  # (P,2)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

        # 新增：负样本Bank & 时序平滑
        self.register_buffer("neg_protos", None)        # (Pn,C)
        self.register_buffer("neg_votes", None)         # (Pn,)
        self.register_buffer("neg_coords", None)        # (Pn,2)
        self.register_buffer("prev_s01", None)          # 上一帧相似度

        # 新增：控制项
        self.pool = pool
        self.topk = int(topk)
        self.temp = float(temp)
        self.enable_neg_bank = bool(enable_neg_bank)
        self.neg_lambda = float(neg_lambda)
        self.smooth_alpha = float(smooth_alpha)

        # 小波参数
        self.enable_wavelet = bool(enable_wavelet)
        self.wavelet_mode = str(wavelet_mode)
        self.wavelet_levels = int(wavelet_levels)
        self.wavelet_thr = float(wavelet_thr)
        self.wavelet_gamma = float(wavelet_gamma)
        self.wavelet_beta = float(wavelet_beta)

    # ---------- 内部工具 ----------
    @torch.no_grad()
    def _weighted_mean_feature(
        self, feat_last: torch.Tensor, weight_last: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], float, Tuple[float, float]]:
        """
        feat_last:   (C,Hf,Wf)
        weight_last: (1,Hf,Wf) ∈ [0,1]
        返回: (mu[C], mass, (cx, cy))
        """
        C, Hf, Wf = feat_last.shape
        w = weight_last.view(-1)
        mass = float(w.sum().item())
        if mass <= 1e-8:
            return None, 0.0, (0.0, 0.0)

        f = feat_last.view(C, -1)
        mu = (f * w.unsqueeze(0)).sum(dim=1) / (w.sum() + 1e-8)
        mu = F.normalize(mu, dim=0)

        ys, xs = torch.nonzero(weight_last[0] > 0, as_tuple=True)
        if len(xs) > 0:
            cx = float(xs.float().mean().item())
            cy = float(ys.float().mean().item())
        else:
            cx, cy = 0.0, 0.0
        return mu, mass, (cx, cy)

    @torch.no_grad()
    def _update_bank_once(self, mu, mass, cx, cy, is_neg=False):
        """合并/新增到正/负Bank（代码复用）"""
        if mu is None or mass <= 1e-8:
            return
        dtype = mu.dtype
        device = mu.device

        # 选Bank指针
        if not is_neg:
            Bp, Vp, Cp = "protos", "votes", "proto_coords"
        else:
            Bp, Vp, Cp = "neg_protos", "neg_votes", "neg_coords"

        bank = getattr(self, Bp)
        votes = getattr(self, Vp)
        coords = getattr(self, Cp)

        if bank is None:
            setattr(self, Bp, mu.unsqueeze(0).to(device=device, dtype=dtype))
            setattr(self, Vp, torch.tensor([mass], device=device, dtype=dtype))
            setattr(self, Cp, torch.tensor([[cx, cy]], device=device, dtype=dtype))
            self.num_updates += 1
            return
        else:
            # 票数衰减
            votes.mul_((self.vote_decay if not is_neg else min(self.vote_decay, 0.9)))

        # 查找最相似的 proto
        bank_n = F.normalize(bank, dim=1)
        sim = torch.mv(bank_n, mu)
        smax, idx = torch.max(sim, dim=0)

        thr = self.merge_thr if not is_neg else min(self.merge_thr + 0.02, 0.99)
        if float(smax.item()) >= thr:
            old_vote = float(votes[idx].item())
            new_vote = old_vote + mass
            mix = float(mass / max(new_vote, 1e-8))
            updated = F.normalize((1.0 - mix) * bank[idx] + mix * mu, dim=0)
            bank[idx] = updated.to(dtype=dtype)
            votes[idx] = torch.tensor(new_vote, device=device, dtype=dtype)
            old_coord = coords[idx]
            new_coord = (1.0 - mix) * old_coord + mix * torch.tensor([cx, cy], device=device, dtype=dtype)
            coords[idx] = new_coord
        else:
            if bank.shape[0] < self.max_protos:
                setattr(self, Bp, torch.cat([bank, mu.unsqueeze(0).to(device=device, dtype=dtype)], dim=0))
                setattr(self, Vp, torch.cat([votes, torch.tensor([mass], device=device, dtype=dtype)], dim=0))
                setattr(self, Cp, torch.cat([coords, torch.tensor([[cx, cy]], device=device, dtype=dtype)], dim=0))
            else:
                j = torch.argmin(votes)
                bank[j] = mu.to(device=device, dtype=dtype)
                votes[j] = torch.tensor(mass, device=device, dtype=dtype)
                coords[j] = torch.tensor([cx, cy], device=device, dtype=dtype)

        # 回写
        setattr(self, Bp, bank)
        setattr(self, Vp, votes)
        setattr(self, Cp, coords)
        self.num_updates += 1

    def _pool_sim(self, sim_feat: torch.Tensor, pool: str) -> torch.Tensor:
        """
        sim_feat: (N,P) / (N,Pn)  逐像素到各(proto)的相似度
        返回：逐像素聚合后的标量 (N,)
        """
        if pool == "max":
            return sim_feat.max(dim=1).values
        elif pool == "topk":
            k = min(self.topk, sim_feat.shape[1])
            topk_vals, _ = torch.topk(sim_feat, k=k, dim=1)
            return topk_vals.mean(dim=1)  # 较 max 更稳，召回更高
        elif pool == "lse":
            # LogSumExp / temperature
            x = sim_feat / max(self.temp, 1e-6)
            m = x.max(dim=1, keepdim=True).values
            return (m.squeeze(1) + torch.log(torch.exp(x - m).sum(dim=1) + 1e-9) * self.temp)
        else:
            # 保底
            return sim_feat.max(dim=1).values

    # ---------- 小波工具 ----------
    @staticmethod
    def _haar_filters(dtype, device):
        # Haar 分解滤波器（正交，缩放 1/sqrt(2)）
        h = torch.tensor([1.0, 1.0], dtype=dtype, device=device) / math.sqrt(2.0)
        g = torch.tensor([1.0, -1.0], dtype=dtype, device=device) / math.sqrt(2.0)
        LL = torch.outer(h, h); LH = torch.outer(h, g)
        HL = torch.outer(g, h); HH = torch.outer(g, g)
        w = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # (4,1,2,2)
        return w

    @torch.no_grad()
    def _dwt2_once(self, x: torch.Tensor):
        # x: (H,W) -> 返回 (LL,LH,HL,HH)，尺寸为 (H/2,W/2)
        if x.dim() != 2:
            raise ValueError("expect (H,W)")
        H, W = x.shape
        pad_h = (H % 2); pad_w = (W % 2)
        if pad_h or pad_w:
            x = F.pad(x.unsqueeze(0).unsqueeze(0),
                      (0, pad_w, 0, pad_h), mode="replicate").squeeze(0).squeeze(0)
        w = self._haar_filters(x.dtype, x.device)
        y = F.conv2d(x.unsqueeze(0).unsqueeze(0), w, stride=2)  # (1,4,H/2,W/2)
        LL, LH, HL, HH = y[0,0], y[0,1], y[0,2], y[0,3]
        return LL, LH, HL, HH

    @torch.no_grad()
    def _wavelet_gate(self, s_map: torch.Tensor) -> torch.Tensor:
        """
        输入: s_map=(H,W)，一般用 s_pos 的聚合图或 Δ 的粗图
        输出: g=(H,W)∈[0,1]，越靠近结构/纹理区域越大
        """
        H, W = s_map.shape
        g_acc = torch.zeros_like(s_map)
        cur = s_map
        for _ in range(max(1, self.wavelet_levels)):
            LL, LH, HL, HH = self._dwt2_once(cur)
            # 细节能量（高频）：纹理/角点更大
            D = (LH.abs() + HL.abs() + HH.abs()) / 3.0
            # 归一化到原分辨率
            Du = F.interpolate(D.unsqueeze(0).unsqueeze(0),
                               size=(H, W), mode="bilinear", align_corners=False)[0,0]
            g_acc += Du
            cur = LL  # 多层递归：继续往低频走
        g_acc /= float(max(1, self.wavelet_levels))
        # Sigmoid 门控（以 wavelet_thr 为阈，gamma 控制陡峭）
        g = torch.sigmoid(self.wavelet_gamma * (g_acc - self.wavelet_thr))
        return g.clamp(0, 1)

    @torch.no_grad()
    def _wavelet_denoise(self, s01: torch.Tensor) -> torch.Tensor:
        """
        对 s01 做一层小波软阈值去噪并重构；仅用于 mode="denoise"
        """
        LL, LH, HL, HH = self._dwt2_once(s01)
        tau = self.wavelet_thr * (LH.abs().mean() + HL.abs().mean() + HH.abs().mean()).clamp(min=1e-6)
        def soft(x): return torch.sign(x) * torch.relu(x.abs() - tau)
        LHs, HLs, HHs = soft(LH), soft(HL), soft(HH)
        # 用转置卷积做简单的 Haar 逆变换
        w = self._haar_filters(s01.dtype, s01.device)
        y = torch.stack([LL, LHs, HLs, HHs], dim=0).unsqueeze(0)  # (1,4,h,w)
        rec = F.conv_transpose2d(y, w, stride=2)[0,0]
        # 裁回原尺寸
        rec = rec[:s01.shape[0], :s01.shape[1]]
        return rec.clamp(0.0, 1.0)

    # ---------- 对外接口 ----------
    @torch.no_grad()
    def update_from_teacher(self, feat_last: torch.Tensor, weight_last: torch.Tensor):
        # 正样本：来自 teacher 的置信图
        mu, mass, (cx, cy) = self._weighted_mean_feature(feat_last, weight_last)
        self._update_bank_once(mu, mass, cx, cy, is_neg=False)

        # 负样本Bank（可选）：用 (1 - weight) 近似背景统计，缩放其权重防止压制过强
        if self.enable_neg_bank:
            bg_w = (1.0 - weight_last).clamp(0.0, 1.0)
            mu_n, mass_n, (cxn, cyn) = self._weighted_mean_feature(feat_last, bg_w)
            if mass > 0 and mass_n > 0:
                # 让负样本投票尺度与正样本同量级，避免一上来被“海量背景”淹没
                scale = (mass / (mass_n + 1e-6)) * 0.5
                self._update_bank_once(mu_n, mass_n * scale, cxn, cyn, is_neg=True)

    @torch.no_grad()
    def similarity(
        self,
        feat_last: torch.Tensor,
        mode: str = "contrast",  # "geom" | "spatial" | "contrast"
    ) -> torch.Tensor:
        """
        返回相似度图 (Hf,Wf)
        - geom: 仅Bank特征（默认 top-k 聚合）
        - spatial: 特征 + 空间权重融合（与原版一致，但用 top-k）
        - contrast: 正Bank 与 负Bank 的对比式相似度
        """
        C, Hf, Wf = feat_last.shape
        device = feat_last.device
        dtype = feat_last.dtype

        # 空Bank直接返回零图
        if self.protos is None or self.protos.shape[0] == 0:
            return torch.zeros((Hf, Wf), device=device, dtype=dtype)

        v = feat_last.view(C, -1).transpose(0, 1)      # (N,C)
        v = F.normalize(v, dim=1)
        protos_n = F.normalize(self.protos, dim=1)     # (P,C)

        # 票数加权（凸显“常见/可靠”的Proto）
        votes = self.votes.clamp(min=0)
        w_pos = (votes / (votes.max().clamp(min=1e-6))).pow(self.vote_alpha)
        sim_pos = torch.matmul(v, protos_n.t()) * w_pos.unsqueeze(0)  # (N,P)

        # ---- 聚合（替代原来的 max）----
        spooled = self._pool_sim(sim_pos, self.pool)  # (N,)

        if mode == "contrast":
            # 负Bank（可选）
            if self.enable_neg_bank and self.neg_protos is not None and self.neg_protos.shape[0] > 0:
                neg_n = F.normalize(self.neg_protos, dim=1)  # (Pn,C)
                w_neg = self.neg_votes.clamp(min=0)
                w_neg = (w_neg / (w_neg.max().clamp(min=1e-6))).pow(self.vote_alpha)
                sim_neg = torch.matmul(v, neg_n.t()) * w_neg.unsqueeze(0)  # (N,Pn)
                sneg = self._pool_sim(sim_neg, self.pool)
            else:
                sneg = torch.zeros_like(spooled)

            # 对比式打分 Δ + 小波门控（可选）
            delta = spooled - self.neg_lambda * sneg                     # (N,)

            delta_map = delta.view(Hf, Wf)
            if self.enable_wavelet and self.wavelet_mode == "gate":
                # 用前景聚合强度的结构纹理作为门控
                g = self._wavelet_gate(spooled.view(Hf, Wf))
                delta_map = delta_map * (1.0 + self.wavelet_beta * g)    # 结构区域放大
            s01 = torch.sigmoid(delta_map / max(self.temp, 1e-6)).clamp(0, 1)
            if self.enable_wavelet and self.wavelet_mode == "denoise":
                s01 = self._wavelet_denoise(s01)

        else:
            # 位置权重融合或纯几何
            if mode == "spatial":
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(Hf, device=device),
                    torch.arange(Wf, device=device),
                    indexing="ij",
                )
                coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()
                dist2 = torch.cdist(coords, self.proto_coords)  # (N,P)
                sigma = max(self.spatial_sigma, 1e-6)
                spatial_w = torch.exp(-dist2 / sigma)           # (N,P)
                sim_mix = 0.8 * sim_pos + 0.2 * spatial_w
                s = self._pool_sim(sim_mix, self.pool)
            else:
                s = spooled

            # 小波门控/去噪（可选）
            s_map = s.view(Hf, Wf)
            if self.enable_wavelet and self.wavelet_mode == "gate":
                g = self._wavelet_gate(s_map)
                s_map = s_map * (1.0 + self.wavelet_beta * g)
            s01 = ((s_map.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
            if self.enable_wavelet and self.wavelet_mode == "denoise":
                s01 = self._wavelet_denoise(s01)

        # 可选：时序平滑（放在最后）
        if self.smooth_alpha > 1e-6:
            if self.prev_s01 is None or self.prev_s01.shape != s01.shape:
                self.prev_s01 = s01.clone()
            s01 = (1.0 - self.smooth_alpha) * s01 + self.smooth_alpha * self.prev_s01
            self.prev_s01 = s01.detach()

        return s01

    # ---------- 便捷函数 ----------
    @torch.no_grad()
    def extract_peaks(self, sim01: torch.Tensor, k: int = 500, nms: int = 4, thr: float = 0.5):
        """
        在相似度图上做简易 NMS，返回 top-k 峰值坐标 (xy, score)
        不改外部接口；需要时在外部直接调用即可。
        """
        H, W = sim01.shape
        s = sim01.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        pooled = F.max_pool2d(s, kernel_size=2*nms+1, stride=1, padding=nms)
        keep = (s == pooled) & (s >= thr)
        ys, xs = torch.nonzero(keep[0, 0], as_tuple=True)
        scores = sim01[ys, xs]
        if len(scores) == 0:
            return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=sim01.dtype)
        topk = min(k, scores.shape[0])
        vals, idx = torch.topk(scores, k=topk)
        sel = torch.stack([xs[idx], ys[idx]], dim=1)  # (K,2) -> (x,y)
        return sel, vals

    @torch.no_grad()
    def reset(self):
        self.protos = None
        self.votes = None
        self.proto_coords = None
        self.neg_protos = None
        self.neg_votes = None
        self.neg_coords = None
        self.prev_s01 = None
        self.num_updates.zero_()

    def export_state(self) -> dict:
        return {
            "protos": None if self.protos is None else self.protos.detach().cpu(),
            "votes": None if self.votes is None else self.votes.detach().cpu(),
            "proto_coords": None if self.proto_coords is None else self.proto_coords.detach().cpu(),
            "neg_protos": None if self.neg_protos is None else self.neg_protos.detach().cpu(),
            "neg_votes": None if self.neg_votes is None else self.neg_votes.detach().cpu(),
            "neg_coords": None if self.neg_coords is None else self.neg_coords.detach().cpu(),
            "num_updates": int(self.num_updates.item()),
            "cfg": {
                "max_protos": self.max_protos,
                "merge_thr": self.merge_thr,
                "vote_decay": self.vote_decay,
                "vote_alpha": self.vote_alpha,
                "spatial_sigma": self.spatial_sigma,
                "pool": self.pool,
                "topk": self.topk,
                "temp": self.temp,
                "enable_neg_bank": self.enable_neg_bank,
                "neg_lambda": self.neg_lambda,
                "smooth_alpha": self.smooth_alpha,
                # 新增导出
                "enable_wavelet": self.enable_wavelet,
                "wavelet_mode": self.wavelet_mode,
                "wavelet_levels": self.wavelet_levels,
                "wavelet_thr": self.wavelet_thr,
                "wavelet_gamma": self.wavelet_gamma,
                "wavelet_beta": self.wavelet_beta,
            },
        }

    def load_state(self, state: dict):
        if state is None:
            return
        device = self.device
        for k in ["protos", "votes", "proto_coords", "neg_protos", "neg_votes", "neg_coords"]:
            v = state.get(k, None)
            if v is not None:
                setattr(self, k, v.to(device=device))
        self.num_updates = torch.tensor(state.get("num_updates", 0), dtype=torch.long, device=device)
        cfg = state.get("cfg", {})
        for k in [
            "max_protos", "merge_thr", "vote_decay", "vote_alpha", "spatial_sigma",
            "pool", "topk", "temp", "enable_neg_bank", "neg_lambda", "smooth_alpha",
            "enable_wavelet", "wavelet_mode", "wavelet_levels", "wavelet_thr", "wavelet_gamma", "wavelet_beta"
        ]:
            if k in cfg:
                setattr(self, k, cfg[k])

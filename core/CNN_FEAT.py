import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from utils.heatmap import event_to_heatmap
import time

time_start = time.time()

# --------------------- Haar Wavelet 细节提取器（替换原 Sobel） ---------------------
class HaarWaveletEdge(nn.Module):
    """
    使用 2x2 Haar 小波核计算 LH/HL/HH 三个细节子带，返回它们的能量图（sqrt(LH^2+HL^2+HH^2)）并做[0,1]归一化。
    不做下采样（stride=1、padding=1），尺寸与输入一致，方便后续插值到特征尺度。
    """
    def __init__(self):
        super().__init__()
        # Haar 一维低/高通：1/sqrt(2) * [1, 1] 与 [1, -1]
        s = 2**0.5
        l = torch.tensor([1.0/s, 1.0/s], dtype=torch.float32)   # low
        h = torch.tensor([1.0/s, -1.0/s], dtype=torch.float32)  # high

        # 构造 2D 核（外积）
        LL = torch.ger(l, l)  # 2x2
        LH = torch.ger(l, h)  # 垂直细节
        HL = torch.ger(h, l)  # 水平细节
        HH = torch.ger(h, h)  # 对角细节

        # 注册为 buffer，形状是 (out_ch=1, in_ch=1, kH=2, kW=2)
        self.register_buffer("w_LH", LH.view(1, 1, 2, 2))
        self.register_buffer("w_HL", HL.view(1, 1, 2, 2))
        self.register_buffer("w_HH", HH.view(1, 1, 2, 2))

    def forward(self, x_gray: torch.Tensor) -> torch.Tensor:
        # x_gray: [B,1,H,W]
        # 与卷积核做同尺寸卷积（不下采样），保持与原 Sobel 接口一致
        LH = F.conv2d(x_gray, self.w_LH, stride=1, padding=1)
        HL = F.conv2d(x_gray, self.w_HL, stride=1, padding=1)
        HH = F.conv2d(x_gray, self.w_HH, stride=1, padding=1)
        mag = torch.sqrt(LH * LH + HL * HL + HH * HH + 1e-6)
        return mag / (mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))


def rgb_to_gray(x):
    # x: [B,3,H,W]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


# --------------------- （原名保留）Contour-aware Mamba -> 内部改为 Wavelet-aware ---------------------
class ContourAwareMamba(nn.Module):
    """
    名称保留以减少外部调用改动，但内部“contour”全部由小波细节图替代。
    """
    def __init__(self, d_model, gamma_token_bias=0.3):
        super().__init__()
        self.d_model = d_model
        # 用小波细节替换 Sobel
        self.edge = HaarWaveletEdge()
        self.gamma_token_bias = gamma_token_bias

        self.motion_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )
        # 输入仍为单通道（小波细节能量图），接口不变
        self.contour_conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.Sigmoid()
        )
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )

    @torch.no_grad()
    def _build_contour_map(self, rgb_t, evt_t, Hf, Wf):
        """
        现在的“contour”是小波细节能量图：
        - 若有事件帧：先归一化事件帧，再取小波细节；
        - 否则：从 RGB 转灰度后取小波细节。
        之后插值到 (Hf, Wf)，并限定到 [0,1]。
        """
        if evt_t is not None:
            # evt_t: [B,1,H,W]
            evt_norm = evt_t / evt_t.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            contour = self.edge(evt_norm)
        else:
            # rgb_t: [B,3,H,W]
            gray = rgb_to_gray(rgb_t)
            # [0,1] 归一化
            gmin = gray.amin(dim=(2, 3), keepdim=True)
            gmax = gray.amax(dim=(2, 3), keepdim=True)
            gray = (gray - gmin) / (gmax - gmin + 1e-6)
            contour = self.edge(gray)

        return F.interpolate(contour, size=(Hf, Wf), mode='bilinear', align_corners=False).clamp(0, 1)

    def forward(self, feat_seq, rgb_seq=None, evt_seq=None):
        B, T, C, Hf, Wf = feat_seq.shape
        enhanced, token_bias = [], []

        for t in range(T):
            feat_t = feat_seq[:, t]
            diff = torch.zeros_like(feat_t) if t == 0 else torch.abs(feat_t - feat_seq[:, t - 1])
            motion_feat = self.motion_conv(diff)

            contour = self._build_contour_map(
                rgb_seq[:, t] if rgb_seq is not None else None,
                evt_seq[:, t] if evt_seq is not None else None,
                Hf, Wf
            )

            # 运动掩膜逻辑不变
            if t == 0:
                motion_mask = torch.zeros_like(contour)
            else:
                diff_map = torch.abs(feat_seq[:, t] - feat_seq[:, t - 1]).mean(1, keepdim=True)
                diff_map = diff_map / diff_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                motion_mask = (diff_map > 0.1).float()

            contour = contour * motion_mask  # 小波细节 × 运动区域

            contour_feat = self.contour_conv(contour)
            w = self.fuse_conv(torch.cat([motion_feat, contour_feat], dim=1))
            enhanced.append(feat_t * (1.0 + w))
            token_bias.append(contour)

        enhanced = torch.stack(enhanced, dim=1)             # [B,T,C,Hf,Wf]
        contour_tokens = torch.stack(token_bias, dim=1)     # [B,T,1,Hf,Wf]

        BT, L = B * T, Hf * Wf
        x_bt = enhanced.permute(0, 1, 3, 4, 2).contiguous().view(BT, L, C)
        m_bt = contour_tokens.view(BT, 1, Hf, Wf).permute(0, 2, 3, 1).contiguous().view(BT, L, 1)
        x_bt = x_bt * (1.0 + self.gamma_token_bias * m_bt)

        x_bt = self.mamba(x_bt).view(B, T, Hf, Wf, C).permute(0, 1, 4, 2, 3).contiguous()

        out = []
        for t in range(T):
            r = self.refine(x_bt[:, t])
            out.append(x_bt[:, t] + 0.5 * r)
        return torch.stack(out, dim=1)


# --------------------- 主干网络 ---------------------
class MambaCNN_Fast(nn.Module):
    def __init__(self, in_channels=3, seq_len=5, d_model=128, train_all_frames=True):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.train_all_frames = train_all_frames

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 内部已改为小波细节感知
        self.contour_mamba = ContourAwareMamba(d_model=d_model, gamma_token_bias=0.7)
        self.temporal_fusion = nn.Sequential(
            nn.Conv3d(d_model, d_model, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(d_model),
            nn.ReLU()
        )
        self.detector = nn.Sequential(
            nn.Conv2d(d_model, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, evt_seq=None, apply_rigid=False):
        B, T, C, H, W = x.shape
        frame_features = [self.init_conv(x[:, t]) for t in range(T)]
        temporal_features = torch.stack(frame_features, dim=1)  # [B,T,Cf,Hf,Wf]
        B, T, C_feat, Hf, Wf = temporal_features.shape

        features = self.contour_mamba(temporal_features, rgb_seq=x, evt_seq=evt_seq)
        fused = self.temporal_fusion(features.permute(0, 2, 1, 3, 4))

        fused_bt = fused.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C_feat, Hf, Wf)
        det_up = F.interpolate(self.detector(fused_bt), size=(H, W), mode='bilinear', align_corners=False)
        pred_heatmaps = det_up.view(B, T, 1, H, W)

        if apply_rigid and evt_seq is not None:
            gts = [event_to_heatmap(evt_seq[:, t], sigma=3, thresh=0.2) for t in range(T)]
            evt_heatmaps = torch.stack(gts, dim=1).to(pred_heatmaps.device)
            return pred_heatmaps, evt_heatmaps

        return pred_heatmaps

class PairReweighter(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1), torch.nn.Sigmoid()
        )
        with torch.no_grad():
            for m in self.mlp:
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)

    @torch.no_grad()
    def score(self, cos_sim, geo_bias, bankL, bankR):
        NA, NB = cos_sim.shape
        feat = torch.stack([
            cos_sim.reshape(-1),
            geo_bias.reshape(-1),
            bankL.reshape(-1, 1).repeat(1, NB).reshape(-1),
            bankR.reshape(1, -1).repeat(NA, 1).reshape(-1),
        ], dim=1)
        w = self.mlp(feat).view(NA, NB)
        return w.clamp(1e-3, 1.0)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mamba_ssm import Mamba
# from utils.heatmap import event_to_heatmap
# import time
#
# time_start = time.time()
# # --------------------- Sobel Edge 提取器 ---------------------
# class SobelEdge(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
#         ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
#         self.register_buffer("kx", kx.view(1,1,3,3))
#         self.register_buffer("ky", ky.view(1,1,3,3))
#
#     def forward(self, x_gray):
#         gx = F.conv2d(x_gray, self.kx, padding=1)
#         gy = F.conv2d(x_gray, self.ky, padding=1)
#         mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
#         return mag / (mag.amax(dim=(2,3), keepdim=True).clamp_min(1e-6))
#
#
# def rgb_to_gray(x):
#     r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
#     return 0.2989*r + 0.5870*g + 0.1140*b
#
#
# # --------------------- Contour-aware Mamba ---------------------
# class ContourAwareMamba(nn.Module):
#     def __init__(self, d_model, gamma_token_bias=0.3):
#         super().__init__()
#         self.d_model = d_model
#         self.edge = SobelEdge()
#         self.gamma_token_bias = gamma_token_bias
#
#         self.motion_conv = nn.Sequential(
#             nn.Conv2d(d_model, d_model, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(d_model, d_model, 3, padding=1)
#         )
#         self.contour_conv = nn.Sequential(
#             nn.Conv2d(1, d_model, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(d_model, d_model, 3, padding=1)
#         )
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(d_model*2, d_model, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(d_model, d_model, 1),
#             nn.Sigmoid()
#         )
#         self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
#         self.refine = nn.Sequential(
#             nn.Conv2d(d_model, d_model, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(d_model, d_model, 3, padding=1)
#         )
#
#     @torch.no_grad()
#     def _build_contour_map(self, rgb_t, evt_t, Hf, Wf):
#         if evt_t is not None:
#             evt_norm = evt_t / evt_t.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
#             # evt_norm = (evt_norm > 0.3).float() * evt_norm  # ✅ 去掉小噪声
#             contour = self.edge(evt_norm)
#
#         else:
#             gray = rgb_to_gray(rgb_t)
#             gray = (gray - gray.amin(dim=(2,3), keepdim=True)) / \
#                    (gray.amax(dim=(2,3), keepdim=True) - gray.amin(dim=(2,3), keepdim=True) + 1e-6)
#             contour = self.edge(gray)
#
#         return F.interpolate(contour, size=(Hf, Wf), mode='bilinear', align_corners=False).clamp(0, 1)
#
#     def forward(self, feat_seq, rgb_seq=None, evt_seq=None):
#         B, T, C, Hf, Wf = feat_seq.shape
#         enhanced, token_bias = [], []
#
#         for t in range(T):
#             feat_t = feat_seq[:, t]
#             diff = torch.zeros_like(feat_t) if t == 0 else torch.abs(feat_t - feat_seq[:, t-1])
#             motion_feat = self.motion_conv(diff)
#
#             contour = self._build_contour_map(
#                 rgb_seq[:, t] if rgb_seq is not None else None,
#                 evt_seq[:, t] if evt_seq is not None else None,
#                 Hf, Wf
#             )
#
#             if t == 0:
#                 motion_mask = torch.zeros_like(contour)
#             else:
#                 diff_map = torch.abs(feat_seq[:, t] - feat_seq[:, t-1]).mean(1, keepdim=True)
#                 diff_map = diff_map / diff_map.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
#                 motion_mask = (diff_map > 0.1).float()
#             contour = contour * motion_mask
#
#             contour_feat = self.contour_conv(contour)
#             w = self.fuse_conv(torch.cat([motion_feat, contour_feat], dim=1))
#             enhanced.append(feat_t * (1.0 + w))
#             token_bias.append(contour)
#
#         enhanced = torch.stack(enhanced, dim=1)
#         contour_tokens = torch.stack(token_bias, 1)
#
#         BT, L = B * T, Hf * Wf
#         x_bt = enhanced.permute(0,1,3,4,2).contiguous().view(BT, L, C)
#         m_bt = contour_tokens.view(BT, 1, Hf, Wf).permute(0,2,3,1).contiguous().view(BT, L, 1)
#         x_bt = x_bt * (1.0 + self.gamma_token_bias * m_bt)
#
#         x_bt = self.mamba(x_bt).view(B, T, Hf, Wf, C).permute(0,1,4,2,3).contiguous()
#
#         out = []
#         for t in range(T):
#             r = self.refine(x_bt[:, t])
#             out.append(x_bt[:, t] + 0.5 * r)
#         return torch.stack(out, dim=1)
#
#
# # --------------------- 主干网络 ---------------------
# class MambaCNN_Fast(nn.Module):
#     def __init__(self, in_channels=3, seq_len=5, d_model=128, train_all_frames=True):
#         super().__init__()
#         self.seq_len = seq_len
#         self.d_model = d_model
#         self.train_all_frames = train_all_frames
#
#         self.init_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, d_model, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.contour_mamba = ContourAwareMamba(d_model=d_model, gamma_token_bias=0.7)
#         self.temporal_fusion = nn.Sequential(
#             nn.Conv3d(d_model, d_model, (3, 3, 3), padding=(1, 1, 1)),
#             nn.BatchNorm3d(d_model),
#             nn.ReLU()
#         )
#         self.detector = nn.Sequential(
#             nn.Conv2d(d_model, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, evt_seq=None, apply_rigid=False):
#         B, T, C, H, W = x.shape
#         frame_features = [self.init_conv(x[:, t]) for t in range(T)]
#         temporal_features = torch.stack(frame_features, dim=1)
#         B, T, C_feat, Hf, Wf = temporal_features.shape
#
#         features = self.contour_mamba(temporal_features, rgb_seq=x, evt_seq=evt_seq)
#         fused = self.temporal_fusion(features.permute(0, 2, 1, 3, 4))
#
#         fused_bt = fused.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C_feat, Hf, Wf)
#         det_up = F.interpolate(self.detector(fused_bt), size=(H, W), mode='bilinear', align_corners=False)
#         pred_heatmaps = det_up.view(B, T, 1, H, W)
#
#         if apply_rigid and evt_seq is not None:
#             gts = [event_to_heatmap(evt_seq[:, t], sigma=3, thresh=0.2) for t in range(T)]
#             evt_heatmaps = torch.stack(gts, dim=1).to(pred_heatmaps.device)
#             return pred_heatmaps, evt_heatmaps
#
#         return  pred_heatmaps
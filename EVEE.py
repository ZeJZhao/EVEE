from __future__ import annotations
import os, time, queue, threading, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any, Dict
from collections import deque  # [MOD] 用于高效滑动窗口

from configs import configs as cfg
from core.CNN_FEAT import MambaCNN_Fast as STM
from utils.Match.EVEE_BANK import EVEE_Bank
from utils.Match.EVEE_Matcher import StatefulMatcher, ORBMatcher

# ===== CuPy 支持（保持原逻辑） =====
CUPY_AVAILABLE = True
try:
    import cupy as cp
    _ = cp.zeros((1,), dtype=cp.float32)
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# ====== 读取/路径 相关：统一从 reader 模块导入 ======
from utils.reader.reader import (
    DEFAULT_BASE,
    DEFAULT_RGB_DIR, DEFAULT_EVENT_DIR, DEFAULT_INFER_DIR, DEFAULT_MASK_DIR,
    DEFAULT_TRAIN_OUT_DIR, DEFAULT_INFER_OUT_DIR, DEFAULT_ONLINE_BEST,
    scan_frames, load_rgb, load_evt, load_mask_binary,
    choose_event_dir, get_dynamic_hw,
    default_matches_dir, ensure_dir
)

# ------------------- 工具：cupy/torch 适配 -------------------
class ArrayAdapter:
    """统一处理 cupy/torch 转换。优先使用 CuPy；无 CuPy 则走 torch。"""
    def __init__(self, use_cupy: bool = True):
        self.use_cupy = bool(use_cupy and CUPY_AVAILABLE)

    def to_gpu(self, arr: Any):
        if self.use_cupy and not isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr

    def stack(self, arrs: List[Any]):
        if self.use_cupy:
            return cp.stack(arrs, 0)
        else:
            if isinstance(arrs[0], torch.Tensor):
                return torch.stack(arrs, dim=0)
            # [MOD] 优化：直接 np.stack，避免多余 asarray（假设 arrs 已为 np/兼容）
            return np.stack(arrs, axis=0)

    def to_torch(self, arr: Any, device: torch.device):
        # 仅在确实是 CuPy 时才同步默认流，避免不必要的同步
        if isinstance(arr, torch.Tensor):
            return arr.to(device, non_blocking=True)
        if self.use_cupy and isinstance(arr, cp.ndarray):
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            t = torch.utils.dlpack.from_dlpack(arr.toDlpack())
            return t.to(device, non_blocking=True)
        return torch.as_tensor(arr).to(device, non_blocking=True)

# ------------------- 遮罩辅助 -------------------
def _mask_u8_bgr(img_u8: np.ndarray, mask_t: Optional[torch.Tensor]) -> np.ndarray:
    if mask_t is None:
        return img_u8
    m = (mask_t > 0.5).detach().cpu().numpy().astype(np.uint8)  # H,W in {0,1}
    if m.ndim == 2:
        m3 = np.repeat(m[:, :, None], 3, axis=2)
    else:
        m3 = m
    return img_u8 * m3


def _mask_chw_float(x_chw: Any, mask_t: Optional[torch.Tensor], adapter: ArrayAdapter):
    if mask_t is None:
        return x_chw
    if isinstance(x_chw, torch.Tensor):
        m = mask_t
        if m.dim() == 2:
            m = m.unsqueeze(0)  # 1,H,W
        return x_chw * m
    if CUPY_AVAILABLE and isinstance(x_chw, cp.ndarray):
        m_np = (mask_t > 0.5).detach().cpu().numpy().astype(np.float32)
        m_cp = cp.asarray(m_np)[None, ...]
        return x_chw * m_cp
    t = torch.as_tensor(x_chw)
    m = mask_t
    if m.dim() == 2:
        m = m.unsqueeze(0)
    y = t * m
    return y

# ------------------- 数学/绘制工具 -------------------
@torch.no_grad()
def upsample_logits(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)[0, 0]


@torch.no_grad()
def local_max_mask(x01: torch.Tensor, nms: int = 5) -> torch.Tensor:
    x = x01.unsqueeze(0).unsqueeze(0)
    pad = nms // 2
    maxw = F.max_pool2d(x, kernel_size=nms, stride=1, padding=pad)
    return (x == maxw)[0, 0]


@torch.no_grad()
def extract_kpts(x01: torch.Tensor, thr: float, nms: int, topk: int):
    if x01.numel() == 0: return [], []
    mask = (x01 >= thr) & local_max_mask(x01, nms)
    if mask.any():
        scores = x01[mask]
        k = min(int(topk), int(scores.numel()))
        _, idxs = torch.topk(scores, k)
        coords = torch.nonzero(mask, as_tuple=False)[idxs]
        return coords[:, 1].int().cpu().tolist(), coords[:, 0].int().cpu().tolist()
    flat = x01.view(-1)
    k = min(int(topk), int(flat.numel()))
    _, idxs = torch.topk(flat, k)
    H, W = x01.shape
    ys = (idxs // W).int().cpu().tolist()
    xs = (idxs % W).int().cpu().tolist()
    return xs, ys

@torch.no_grad()
def qth(x01: torch.Tensor, abs_thr: float, q: float) -> float:
    return float(max(abs_thr, float(torch.quantile(x01.flatten(), q).item())))

# ------------------- 形态学开闭（布尔） -------------------
def _morph_open_close(img_bgr: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    if (k_open <= 1 and k_close <= 1) or img_bgr.size == 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    if k_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel)
    if k_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel)
    if img_bgr.ndim == 3:
        mask3 = np.repeat((binm > 0)[:, :, None].astype(np.uint8), 3, axis=2)
        return img_bgr * mask3
    else:
        return (img_bgr * (binm > 0).astype(np.uint8))

# ------------------- 进度条/ETA -------------------
def _fmt_time(sec: float) -> str:
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _print_progress(tag: str, i: int, n: int, t0: float, final_fps: Optional[float] = None):  # [MOD] 添加 final_fps 参数
    done = i
    total = max(1, n)
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    pct = 100.0 * done / total
    barw, filled = 24, int(24 * done / total)
    bar = "█" * filled + "·" * (barw - filled)

    # 核心修改：添加 \r 返回行首，并用 end='' 保持单行
    msg = (f"\r[{tag}] [{bar}] {done}/{total} ({pct:5.1f}%)  "  # \r 在这里！
           f"FPS={rate:6.2f}  ETA={_fmt_time(eta)}")
    print(msg, end="", flush=True)

    if done == total:
        print()  # 结束时换行
        # [MOD] 只在结束时写 FPS 文件，避免高频 I/O
        if final_fps is not None:
            try:
                with open("Output_result/FPS_result", "a", encoding="utf-8") as f:
                    f.write(f"{final_fps:.6f}\n")  # 使用最终 FPS
            except Exception as e:
                # 不影响主流程，仅提示
                print(f"[WARN] 追加写入 FPS_result.txt 失败: {e}")

# ------------------- 模型与权重 -------------------
@torch.no_grad()
def teacher_forward(teacher: STM, rgb_seq: torch.Tensor, evt_seq: Optional[torch.Tensor]):
    B, T, C, H, W = rgb_seq.shape
    x_bt = rgb_seq.view(B * T, C, H, W).contiguous(memory_format=torch.channels_last)
    feat_bt = teacher.init_conv(x_bt)  # (B*T,Cf,Hf,Wf)
    Cf, Hf, Wf = feat_bt.shape[1:]
    temporal_features = feat_bt.view(B, T, Cf, Hf, Wf)
    features = teacher.contour_mamba(temporal_features, rgb_seq=rgb_seq, evt_seq=evt_seq)
    fused_5d = teacher.temporal_fusion(features.permute(0, 2, 1, 3, 4))
    fused_last = fused_5d[:, :, -1].contiguous().view(Cf, Hf, Wf)
    det_fm = teacher.detector(fused_last.unsqueeze(0))  # (1,1,Hf,Wf)
    det_up = upsample_logits(det_fm, (H, W))
    return fused_last, det_up, det_fm[0]

# ------------------- Online Head -------------------
class OnlineHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        mid = max(16, in_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, 1, 1)
        )

    def forward(self, feat_last: torch.Tensor) -> torch.Tensor:
        if feat_last.dim() == 3: feat_last = feat_last.unsqueeze(0)
        x = self.net(feat_last.contiguous(memory_format=torch.channels_last))
        return x[:, 0]

# ------------------- Best State (共享) -------------------
class SharedBest:
    def __init__(self, save_path: Optional[str]):
        self.lock = threading.Lock()
        self.best_loss = float('inf')
        self.state_dict = None
        self.version = 0
        self.save_path = save_path

    def maybe_update(self, new_loss: float, state_dict: dict, min_improve: float) -> bool:
        with self.lock:
            if (new_loss + min_improve) < self.best_loss:
                self.best_loss = float(new_loss)
                self.state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
                self.version += 1
                if self.save_path:
                    ensure_dir(os.path.dirname(self.save_path))
                    torch.save({"version": self.version,
                                "best_loss": self.best_loss,
                                "state_dict": self.state_dict}, self.save_path)
                print(f"[Best] v{self.version} loss={self.best_loss:.6f}")
                return True
        return False

    def get(self):
        with self.lock:
            return self.version, self.state_dict

# ------------------- 匹配 + RANSAC 导出 -------------------
def _to_gray(img_bgr_u8: np.ndarray) -> np.ndarray:
    return img_bgr_u8 if img_bgr_u8.ndim == 2 else cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2GRAY)


def _to_xy_array(pts) -> np.ndarray:
    if isinstance(pts, tuple) and len(pts) == 2:
        xs, ys = pts
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.column_stack([np.asarray(xs, dtype=np.float32),
                                np.asarray(ys, dtype=np.float32)])
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1 and pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.ndim == 1 and pts.size == 2:
        pts = pts[None, :]
    return pts.astype(np.float32)


def _pre_filter_by_motion(pL: np.ndarray, pR: np.ndarray, max_px: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(pL) == 0 or len(pR) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    d2 = np.sum((pR - pL) ** 2, axis=1)
    keep = d2 <= float(max_px) ** 2
    return pL[keep], pR[keep]


def _desc_ransac_matches(prev_img_bgr: np.ndarray,
                         curr_img_bgr: np.ndarray,
                         prev_xy: np.ndarray,
                         curr_xy: np.ndarray,
                         *,
                         orb_nfeatures: int,
                         desc_max_matches: int,
                         ransac_model: str,
                         ransac_thr_px: float,
                         ransac_conf: float,
                         ransac_max_iters: int,
                         pre_px_thr: Optional[float] = None) -> np.ndarray:
    """返回 (N,4): x1,y1,x2,y2（RANSAC 内点）。不足 8/4 个则返回空。
    - ransac_model: 'F' or 'H'
    """
    matcher = ORBMatcher(nfeatures=int(orb_nfeatures))
    g1, g2 = _to_gray(prev_img_bgr), _to_gray(curr_img_bgr)
    pts1_list = [tuple(map(float, xy)) for xy in prev_xy.tolist()]
    pts2_list = [tuple(map(float, xy)) for xy in curr_xy.tolist()]
    p1, p2, _ = matcher.match_by_points(g1, g2, pts1_list, pts2_list, max_matches=int(desc_max_matches))
    if len(p1) == 0 or len(p2) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    P1 = np.asarray(p1, dtype=np.float32)
    P2 = np.asarray(p2, dtype=np.float32)

    if pre_px_thr is not None and pre_px_thr > 0:
        P1, P2 = _pre_filter_by_motion(P1, P2, pre_px_thr)
        if len(P1) == 0:
            return np.zeros((0, 4), dtype=np.float32)

    if ransac_model.upper() == 'H':
        if len(P1) < 4:
            return np.zeros((0, 4), dtype=np.float32)
        Hm, mask = cv2.findHomography(P1, P2, method=cv2.RANSAC,
                                      ransacReprojThreshold=float(ransac_thr_px),
                                      confidence=float(ransac_conf),
                                      maxIters=int(ransac_max_iters))
    else:
        if len(P1) < 8:
            return np.zeros((0, 4), dtype=np.float32)
        method = cv2.FM_RANSAC
        if hasattr(cv2, "USAC_MAGSAC"):
            method = cv2.USAC_MAGSAC
        elif hasattr(cv2, "USAC_DEFAULT"):
            method = cv2.USAC_DEFAULT
        try:
            Fm, mask = cv2.findFundamentalMat(
                P1, P2, method=method,
                ransacReprojThreshold=float(ransac_thr_px),
                confidence=float(ransac_conf),
                maxIters=int(ransac_max_iters)
            )
        except Exception:
            Fm, mask = cv2.findFundamentalMat(
                P1, P2, method=cv2.FM_RANSAC,
                ransacReprojThreshold=float(ransac_thr_px),
                confidence=float(ransac_conf),
                maxIters=int(ransac_max_iters)
            )

    if mask is None:
        return np.zeros((0, 4), dtype=np.float32)
    inl = mask.ravel().astype(bool)
    P1i, P2i = P1[inl], P2[inl]
    if len(P1i) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    M = np.concatenate([P1i, P2i], axis=1).astype(np.float32)  # (N,4)
    return M

# ------------------- 导出匹配（替换点）：描述子 + RANSAC -------------------
def _dump_matches_pair_desc_ransac(prev_idx: int,
                                   prev_img: np.ndarray,
                                   curr_idx: int,
                                   curr_img: np.ndarray,
                                   prev_xy: np.ndarray,
                                   curr_xy: np.ndarray,
                                   out_dir: str,
                                   *,
                                   orb_nfeatures: int,
                                   desc_max_matches: int,
                                   ransac_model: str,
                                   ransac_thr_px: float,
                                   ransac_conf: float,
                                   ransac_max_iters: int,
                                   pre_px_thr: Optional[float] = None) -> tuple[str, int]:
    ensure_dir(out_dir)
    mat = _desc_ransac_matches(prev_img, curr_img, prev_xy, curr_xy,
                               orb_nfeatures=orb_nfeatures,
                               desc_max_matches=desc_max_matches,
                               ransac_model=ransac_model,
                               ransac_thr_px=ransac_thr_px,
                               ransac_conf=ransac_conf,
                               ransac_max_iters=ransac_max_iters,
                               pre_px_thr=pre_px_thr)
    outp = os.path.join(out_dir, f"matches_{prev_idx:05d}_{curr_idx:05d}.txt")
    with open(outp, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in mat:
            f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
    return outp, int(len(mat))

# ------------------- Bank 异步线程（similarity + update） -------------------
class _Future:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.exc: Optional[BaseException] = None

    def set_result(self, value):
        self.result = value
        self.event.set()

    def set_exception(self, exc: BaseException):
        self.exc = exc
        self.event.set()

    def wait(self, timeout: Optional[float] = None):
        self.event.wait(timeout)
        if self.exc is not None:
            raise self.exc
        return self.result

class BankAsync:
    """将 Bank 的 heavy 操作放入独立线程；支持跳帧 update，优先执行 similarity。 [MOD] 添加缓存，避免阻塞 """
    def __init__(self, bank: EVEE_Bank, *, mode: str = "contrast", workers: int = 1, skip: int = 0, sim_stride: int = 1):  # [MOD] 添加 sim_stride
        self.bank = bank
        self.mode = mode
        self.skip = max(0, int(skip))
        self.sim_stride = max(1, int(sim_stride))  # [MOD] 支持 sim 降频
        self._sim_q: "queue.Queue[tuple]" = queue.Queue(maxsize=64)
        self._upd_q: "queue.Queue[tuple]" = queue.Queue(maxsize=64)
        self._alive = True
        self._workers: List[threading.Thread] = []
        self._lock = threading.Lock()  # 防止同时读写银行内部状态
        # [MOD] 缓存：类似 EVEE.py 的 _last_sim
        self._last_sim = None  # (idx, sim_map: torch.Tensor)
        self._last_lock = threading.Lock()
        for wid in range(max(1, int(workers))):
            t = threading.Thread(target=self._worker, name=f"BankWorker-{wid}", daemon=True)
            t.start()
            self._workers.append(t)

    def close(self):
        self._alive = False
        # 放入 None 以唤醒
        try:
            self._sim_q.put_nowait((None,))
        except Exception:
            pass
        try:
            self._upd_q.put_nowait((None,))
        except Exception:
            pass
        for t in self._workers:
            t.join(timeout=0.1)

    def _worker(self):
        while self._alive:
            item = None
            try:
                item = self._sim_q.get(timeout=0.002)
            except queue.Empty:
                pass
            if item is None:
                try:
                    item = self._upd_q.get(timeout=0.01)
                except queue.Empty:
                    continue
            if item is None:
                continue
            if item and item[0] is None:
                continue
            kind = item[0]
            try:
                if kind == "sim":
                    _, key, feat_last, fut = item
                    with torch.no_grad():
                        with self._lock:
                            sim_map = self.bank.similarity(feat_last, mode=self.mode).clamp(0, 1)
                    # [MOD] 缓存结果
                    with self._last_lock:
                        self._last_sim = (key, sim_map.detach())
                    fut.set_result(sim_map)
                elif kind == "upd":
                    _, feat_last, w_small = item
                    with torch.no_grad():
                        with self._lock:
                            self.bank.update_from_teacher(feat_last, w_small)
            except BaseException as e:
                if kind == "sim":
                    fut.set_exception(e)
                else:
                    print(f"[BankAsync] update failed: {e}")

    def submit_similarity(self, key: int, feat_last: torch.Tensor) -> _Future:
        fut = _Future()
        try:
            self._sim_q.put_nowait(("sim", key, feat_last.detach(), fut))
        except queue.Full:
            # 拥塞时直接在当前线程同步计算，避免等待
            with torch.no_grad():
                with self._lock:
                    sim_map = self.bank.similarity(feat_last, mode=self.mode).clamp(0, 1)
            # [MOD] 仍缓存
            with self._last_lock:
                self._last_sim = (key, sim_map.detach())
            fut.set_result(sim_map)
        return fut

    def maybe_submit_update(self, idx: int, feat_last: torch.Tensor, w_small: torch.Tensor):
        if self.skip > 0 and (idx % (self.skip + 1)) != 0:
            return
        try:
            self._upd_q.put_nowait(("upd", feat_last.detach(), w_small.detach()))
        except queue.Full:
            # 拥塞时丢弃本次 update（可接受）
            pass

    # [MOD] 新增：非阻塞获取最新 sim（用缓存，避免 wait）
    def get_latest_similarity(self, current_idx: int) -> Optional[torch.Tensor]:
        with self._last_lock:
            if self._last_sim is not None and self._last_sim[0] >= current_idx - self.sim_stride:  # 用最近的
                return self._last_sim[1]
            return None

# ------------------- 预取器（DataLoader 风格） -------------------
class Prefetcher:
    """简易预取器：后台线程读取/预处理到队列，主线程消费。 [MOD] 移除后台 to_torch，避免同步 """
    def __init__(self, indices: List[int], loader_fn, *, maxsize: int = 16, name: str = "Prefetch"):
        self.indices = indices
        self.loader_fn = loader_fn
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=maxsize)
        self.th = threading.Thread(target=self._run, name=name, daemon=True)
        self.th.start()

    def _run(self):
        for idx in self.indices:
            try:
                item = self.loader_fn(idx)
            except BaseException as e:
                print(f"[Prefetch] idx={idx} failed: {e}")
                item = None
            self.q.put((idx, item))
        self.q.put(None)

    def get(self):
        return self.q.get()

# ------------------- 性能设置 -------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass
torch.set_num_threads(1)

# ------------------- 主流程 -------------------
def main():
    parser = argparse.ArgumentParser()
    # ===== 输出目录 =====
    parser.add_argument("--train_out_dir", default=None, type=str, help="训练阶段可视化/导出目录（不指定则默认）")
    parser.add_argument("--infer_out_dir", default=None, type=str, help="推理阶段可视化/导出目录（不指定则默认）")
    parser.add_argument("--matches_dump_dir", default=None, type=str,
                        help="匹配结果导出目录（默认 <infer_out_dir>/matches)")
    # ===== 数据/路径 =====
    parser.add_argument("--rgb_dir", default=DEFAULT_RGB_DIR, type=str)
    parser.add_argument("--event_dir", default=DEFAULT_EVENT_DIR, type=str)
    parser.add_argument("--infer_dir", default=DEFAULT_INFER_DIR, type=str)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR, type=str)
    parser.add_argument("--px_match_thr", default=25.0, type=float, help="像素级预过滤阈值（RANSAC 前，可选）")

    # ===== 导出匹配：新参数 =====
    parser.add_argument("--dump_matcher", choices=["nn_px", "desc_ransac"], default="desc_ransac",
                        help="导出匹配方式：'nn_px'=互为最近邻(旧)；'desc_ransac'=描述子+RANSAC(新)")
    parser.add_argument("--orb_nfeatures", default=2000, type=int)
    parser.add_argument("--desc_max_matches", default=7000, type=int)
    parser.add_argument("--ransac_model", choices=["F", "H"], default="F",
                        help="RANSAC 模型：F=Fundamental（默认，无需内参）；H=Homography")
    parser.add_argument("--ransac_thr_px", default=20, type=float, help="RANSAC 重投影阈值（像素）")
    parser.add_argument("--ransac_conf", default=0.9995, type=float, help="RANSAC 置信度")
    parser.add_argument("--ransac_max_iters", default=8000, type=int,
                        help="RANSAC 最大迭代次数（仅 H 模型支持；F 的迭代由 OpenCV 内部控制）")

    # ===== Bank 超参/模式 =====
    parser.add_argument("--bank_max_protos", default=1024, type=int)
    parser.add_argument("--bank_merge_thr", default=0.9, type=float)
    parser.add_argument("--bank_vote_decay", default=0.6, type=float)
    parser.add_argument("--bank_vote_alpha", default=3, type=float)
    parser.add_argument("--bank_mode", default="contrast", choices=["geom", "spatial", "contrast"])  # 保持原可选 + contrast
    parser.add_argument("--bank_state", default="", type=str, help="可选：加载/保存 Bank 状态的路径（.pt）")
    parser.add_argument("--bank_update_infer", default=False, action="store_true",
                        help="推理阶段也把高置信区域写入记忆")

    # ===== 形态学开闭 =====
    parser.add_argument("--morph_open", default=1, type=int, help="开运算核大小（像素，0=不启用）")
    parser.add_argument("--morph_close", default=1, type=int, help="闭运算核大小（像素，0=不启用）")

    # ===== 其它 =====
    parser.add_argument("--thr_re", default=getattr(cfg, "real_THRESHOLD_RE", 0.02), type=float)
    parser.add_argument("--thr_rb", default=getattr(cfg, "real_THRESHOLD_RB", 0.02), type=float)
    parser.add_argument("--ronly_q", default=0.9, type=float)
    parser.add_argument("--bank_q", default=0.9, type=float)
    parser.add_argument("--topk_ronly", default=480, type=int)
    parser.add_argument("--topk_bank", default=480, type=int)
    parser.add_argument("--topk_evt", default=480, type=int, help="事件伪特征点 top-k（推理阶段）")
    parser.add_argument("--max_train_frames", default=64, type=int)
    parser.add_argument("--batch_size", default=2, type=int)  # 保持=2 以确保精度
    parser.add_argument("--w_evt", default=0.6, type=float)
    parser.add_argument("--w_re", default=1, type=float) #原本是0.4
    parser.add_argument("--ckpt", default=getattr(cfg, "val_ckpt", ""), type=str)
    parser.add_argument("--seq_len", default=getattr(cfg, "seq_len", 5), type=int)
    parser.add_argument("--max_size", default=360, type=int, help="最长边限制（避免 OOM）")
    parser.add_argument("--train_threads", default=23, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-2, type=float) #原本是1e-4
    parser.add_argument("--amp", default=False, action='store_true')
    parser.add_argument("--min_improve", default=0.0001, type=float)
    parser.add_argument("--save_vis", default=True, action='store_true')
    parser.add_argument("--save_best", default="./weights/online_head_best.pth", type=str)

    # ===== [MOD] 新增/调整参数：预取 & Bank 异步 =====
    parser.add_argument("--prefetch", default=12, type=int, help="预取队列大小")
    parser.add_argument("--bank_async", default=True, action="store_true", help="启用 Bank 异步")
    parser.add_argument("--bank_workers", default=1, type=int, help="Bank 线程数（>1 提升并行）")  # [MOD] 默认2，提升并行
    parser.add_argument("--bank_skip", default=2, type=int, help="Bank update 跳帧（0=每帧）")
    parser.add_argument("--bank_sim_stride", default=1, type=int, help="Bank sim 降频（1=每帧）")  # [MOD] 新增，支持降频

    args = parser.parse_args()

    # 输出目录默认值
    train_out_dir = args.train_out_dir or DEFAULT_TRAIN_OUT_DIR
    infer_out_dir = args.infer_out_dir or DEFAULT_INFER_OUT_DIR
    # ensure_dir(train_out_dir)  # 当前没有训练阶段可视化写入，避免生成空的 results_ZEJING_TRAIN
    ensure_dir(infer_out_dir)
    if not args.matches_dump_dir:
        args.matches_dump_dir = default_matches_dir(infer_out_dir)
    ensure_dir(args.matches_dump_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} | CuPy={'ON' if CUPY_AVAILABLE else 'OFF'}")
    adapter = ArrayAdapter()

    # Teacher
    teacher = STM(seq_len=args.seq_len).to(device)
    if os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        teacher.load_state_dict(state, strict=False)
    teacher.eval()

    # Bank
    bank = EVEE_Bank(
        device=str(device),
        max_len=args.bank_max_protos,
        max_protos=args.bank_max_protos,
        merge_thr=args.bank_merge_thr,
        vote_decay=args.bank_vote_decay,
        vote_alpha=args.bank_vote_alpha,
    ).to(device)
    if args.bank_state and os.path.isfile(args.bank_state):
        try:
            st = torch.load(args.bank_state, map_location=device)
            bank.load_state(st)
            print(f"[Bank] Loaded state from {args.bank_state}")
        except Exception as e:
            print(f"[Bank] Load failed: {e}")

    bank_async: Optional[BankAsync] = None
    if args.bank_async:
        bank_async = BankAsync(bank, mode=args.bank_mode, workers=args.bank_workers, skip=args.bank_skip, sim_stride=args.bank_sim_stride)

    # ====== 扫描训练/推理/掩膜帧 ======
    rgb_map = scan_frames(args.rgb_dir)
    base_train = os.path.dirname(args.rgb_dir.rstrip("/"))
    evt_map, chosen_evt_dir = choose_event_dir(args.event_dir, base_train)
    infer_map = scan_frames(args.infer_dir)
    mask_map = scan_frames(args.mask_dir)

    has_train = (len(rgb_map) > 0 and len(evt_map) > 0)
    has_infer = (len(infer_map) > 0)

    print(f"[Check] Train RGB frames: {len(rgb_map)} | EVT frames: {len(evt_map)} | event_dir={chosen_evt_dir}")
    print(f"[Check] Infer frames: {len(infer_map)} @ {args.infer_dir}")


    # 动态分辨率
    if len(rgb_map) > 0:
        sample_path = next(iter(rgb_map.values()))
    elif len(infer_map) > 0:
        sample_path = next(iter(infer_map.values()))
    else:
        sample_path = None
    fallback_hw = (
        getattr(cfg, "val_img_size", (640, 480))[1],
        getattr(cfg, "val_img_size", (640, 480))[0]
    )
    H, W = get_dynamic_hw(sample_path, args.max_size, fallback_hw)

    # ====== 线程与共享区 ======
    train_queues = [queue.Queue(maxsize=16) for _ in range(args.train_threads)]
    # infer_q, vis_q = queue.Queue(maxsize=6), queue.Queue(maxsize=6)
    infer_q, vis_q = queue.Queue(0), queue.Queue(0)


    sample_buffer: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    sample_lock = threading.Lock()
    shared_best = SharedBest(args.save_best)
    if os.path.isfile(args.save_best):
        best_state = torch.load(args.save_best, map_location=device)
        shared_best.state_dict = best_state["state_dict"]
        shared_best.best_loss = best_state["best_loss"]
        shared_best.version = best_state["version"]
        # print(f"[Init] Loaded existing online head best v{shared_best.version} loss={shared_best.best_loss:.6f}")
    else:
        print("[Init] No online head best found, will start from scratch using val_ckpt features")

    def put_latest(q: "queue.Queue", item) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        q.put(item)

    # --- 训练线程（与原版一致） ---
    def train_worker(tid: int, my_q: "queue.Queue[int]"):
        local_head, local_version = None, -1
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
        optimizer: Optional[torch.optim.Optimizer] = None
        Cf: Optional[int] = None
        while True:
            end_n = my_q.get()
            if end_n is None: break
            if local_head is None:
                while True:
                    with sample_lock:
                        if len(sample_buffer) > 0:
                            Cf = sample_buffer[0][0].shape[0]
                            break
                    time.sleep(0.001)
                local_head = OnlineHead(Cf).to(device, memory_format=torch.channels_last)
                optimizer = torch.optim.Adam(local_head.parameters(), lr=args.lr, weight_decay=args.wd)
            # 同步 best
            v, sd = shared_best.get()
            if sd is not None and v != local_version:
                local_head.load_state_dict(sd, strict=False)
                local_version = v

            with sample_lock:
                L = min(end_n, tid + 1, args.max_train_frames)
                snapshot = sample_buffer[:end_n][-L:]

            local_head.train()
            feats_batch, tgts_batch = [], []
            for feat_dev, tgt_dev in snapshot:
                if tgt_dev is None: continue
                feats_batch.append(feat_dev)
                tgts_batch.append(tgt_dev)
                if len(feats_batch) == args.batch_size:
                    feats = torch.stack(feats_batch, dim=0).contiguous(memory_format=torch.channels_last)
                    tgts = torch.stack(tgts_batch, dim=0)
                    if tgts.dim() == 4 and tgts.shape[1] == 1: tgts = tgts[:, 0]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                        pred_fm = local_head(feats)
                        loss_evt = F.binary_cross_entropy_with_logits(pred_fm, tgts)
                        re_logits = teacher.detector(feats)[:, 0]
                        loss_re = F.binary_cross_entropy_with_logits(
                            pred_fm, (re_logits.sigmoid() >= args.thr_re).float()
                        )
                        loss = args.w_evt * loss_evt + args.w_re * loss_re
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    shared_best.maybe_update(float(loss.item()), local_head.state_dict(), args.min_improve)
                    feats_batch, tgts_batch = [], []
            if len(feats_batch) > 0:
                feats = torch.stack(feats_batch, dim=0).contiguous(memory_format=torch.channels_last)
                tgts = torch.stack(tgts_batch, dim=0)
                if tgts.dim() == 4 and tgts.shape[1] == 1: tgts = tgts[:, 0]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    pred_fm = local_head(feats)
                    loss_evt = F.binary_cross_entropy_with_logits(pred_fm, tgts)
                    re_logits = teacher.detector(feats)[:, 0]
                    loss_re = F.binary_cross_entropy_with_logits(
                        pred_fm, (re_logits.sigmoid() >= args.thr_re).float()
                    )
                    loss = args.w_evt * loss_evt + args.w_re * loss_re
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                shared_best.maybe_update(float(loss.item()), local_head.state_dict(), args.min_improve)

    for tid in range(args.train_threads):
        threading.Thread(target=train_worker, args=(tid, train_queues[tid]), daemon=True).start()

    # --- 推理线程（并行 Bank） ---
    def infer_worker():
        infer_head, infer_version = None, -1
        if not hasattr(infer_worker, "_matcher_RBD"):
            infer_worker._matcher_RBD = StatefulMatcher(max_dist=10.0, orb_nfeatures=1000)
        fcount = 0  # [MOD] 添加计数器，支持 stride
        while True:
            item = infer_q.get()
            if item is None: break
            idx, feat_last, det_up_T, rgb_u8, evt_ready, target_fm_dev, infer_mask = item
            Cf, Hf, Wf = feat_last.shape
            fcount += 1

            if infer_head is None:
                infer_head = OnlineHead(Cf).to(device, memory_format=torch.channels_last)
            v, sd = shared_best.get()
            if sd is not None and v != infer_version:
                infer_head.load_state_dict(sd, strict=False)
                infer_version = v

            online01 = None
            if infer_head is not None:
                infer_head.eval()
                with torch.no_grad():
                    pred_fm = infer_head(feat_last)
                    up = upsample_logits(pred_fm, (H, W))
                    online01 = up.sigmoid().clamp(0, 1)

            teach01 = det_up_T.sigmoid().clamp(0, 1)
            if infer_mask is not None:
                m = infer_mask.clamp(0, 1)
                teach01 = teach01 * m
                if online01 is not None:
                    online01 = online01 * m

            # Keep the RGB canvas unmasked so saved visualizations retain context.
            # The actual ROI gating is still applied to rgb_f/model maps/keypoints.
            rgb_u8_vis = rgb_u8

            # [MOD] Bank.similarity：非阻塞 submit + 缓存（避免 wait）
            sim_map = None
            if bank_async is not None:
                if (fcount % args.bank_sim_stride) == 0:  # [MOD] 支持降频
                    bank_async.submit_similarity(idx, feat_last)
                sim_map = bank_async.get_latest_similarity(idx)
                if sim_map is None:  # 无缓存时同步计算（fallback）
                    with torch.no_grad():
                        sim_map = bank.similarity(feat_last, mode=args.bank_mode).clamp(0, 1)
            else:
                with torch.no_grad():
                    sim_map = bank.similarity(feat_last, mode=args.bank_mode).clamp(0, 1)

            sim_up = upsample_logits(sim_map, (H, W)).sigmoid().clamp(0, 1)
            if infer_mask is not None:
                sim_up = sim_up * infer_mask
            thr_b = qth(sim_up, args.thr_rb, args.bank_q)
            pts_b = extract_kpts(sim_up, thr_b, 5, args.topk_bank)

            # Ronly / Online / Event 提前计算（顺序调整，无依赖）
            thr_r = qth(teach01, args.thr_rb, args.ronly_q)
            pts_r = extract_kpts(teach01, thr_r, 5, args.topk_ronly)

            pts_g = ([], [])
            if online01 is not None:
                thr_d = qth(online01, args.thr_rb, args.ronly_q)
                pts_g = extract_kpts(online01, thr_d, 5, args.topk_ronly)

            pts_e = ([], [])
            if isinstance(target_fm_dev, torch.Tensor):
                try:
                    evt_curr = target_fm_dev
                    if evt_curr.dim() == 3 and evt_curr.shape[0] == 1:
                        evt_curr = evt_curr[0]
                    evt01 = evt_curr.float().clamp(0, 1)
                    if infer_mask is not None:
                        evt01 = evt01 * infer_mask
                    thr_e = qth(evt01, args.thr_rb, args.bank_q)
                    pts_e = extract_kpts(evt01, thr_e, 5, args.topk_evt)
                except Exception:
                    pts_e = ([], [])

            # 合并并去重
            # xs = pts_r[0] + pts_b[0] + pts_g[0]
            # ys = pts_r[1] + pts_b[1] + pts_g[1]
            xs = pts_r[0] + pts_b[0] + pts_g[0] + pts_e[0]
            ys = pts_r[1] + pts_b[1] + pts_g[1] + pts_e[1]
            if xs:
                uniq = sorted(set(zip(xs, ys)))
                xs = [a for a, _ in uniq]
                ys = [b for _, b in uniq]
            pts_big = (xs, ys)

            canvas = infer_worker._matcher_RBD.step(
                key="RBD",
                curr_img_bgr=rgb_u8_vis,
                curr_pts=pts_big,
                draw_right_points=False,
                two_panel=True,
                thickness=2,
                curr_feat_map=feat_last,
                use_reweighter=True,
                curr_bank_sim=sim_map
            )
            if canvas is None: canvas = rgb_u8_vis.copy()

            # （可选）推理阶段将高置信区域写入记忆（跳帧）
            if args.bank_update_infer:
                with torch.no_grad():
                    conf_for_bank = online01 if online01 is not None else teach01
                    if isinstance(target_fm_dev, torch.Tensor):
                        evt_tmp = target_fm_dev
                        if evt_tmp.dim() == 3 and evt_tmp.shape[0] == 1:
                            evt_tmp = evt_tmp[0]
                        evt_tmp = evt_tmp.float().clamp(0, 1)
                        if infer_mask is not None:
                            evt_tmp = evt_tmp * infer_mask
                        conf_for_bank = torch.maximum(conf_for_bank, evt_tmp)
                    thr_up = qth(conf_for_bank, args.thr_rb, args.bank_q)
                    bin_up = (conf_for_bank >= thr_up).float()  # H,W
                    w_small = F.interpolate(bin_up[None, None, ...], size=(feat_last.shape[1], feat_last.shape[2]),
                                            mode='nearest')[0]
                    if bank_async is not None:
                        bank_async.maybe_submit_update(idx, feat_last, w_small)
                    else:
                        bank.update_from_teacher(feat_last, w_small)

            curr_xy = _to_xy_array(pts_big)
            prev_idx = getattr(infer_worker, '_prev_idx', None)
            prev_xy = getattr(infer_worker, '_prev_xy', None)
            prev_img = getattr(infer_worker, '_prev_img', None)
            if prev_idx is not None and prev_img is not None and args.dump_matcher == 'desc_ransac':
                _dump_matches_pair_desc_ransac(
                    prev_idx, prev_img, idx, rgb_u8_vis,
                    prev_xy, curr_xy, args.matches_dump_dir,
                    orb_nfeatures=args.orb_nfeatures,
                    desc_max_matches=args.desc_max_matches,
                    ransac_model=args.ransac_model,
                    ransac_thr_px=args.ransac_thr_px,
                    ransac_conf=args.ransac_conf,
                    ransac_max_iters=args.ransac_max_iters,
                    pre_px_thr=args.px_match_thr
                )
            elif prev_idx is not None and prev_img is not None and args.dump_matcher == 'nn_px':
                _dump_matches_pair_desc_ransac(
                    prev_idx, prev_img, idx, rgb_u8_vis,
                    prev_xy, curr_xy, args.matches_dump_dir,
                    orb_nfeatures=args.orb_nfeatures,
                    desc_max_matches=args.desc_max_matches,
                    ransac_model='H',
                    ransac_thr_px=1e9, ransac_conf=0.0, ransac_max_iters=0,
                    pre_px_thr=args.px_match_thr
                )
            infer_worker._prev_idx = idx
            infer_worker._prev_xy = curr_xy
            infer_worker._prev_img = rgb_u8_vis.copy()

            vis_q.put((idx, canvas,
                       f"RBD(+evt pseudo)  head_v={infer_version if infer_version >= 0 else 0}"))

    thr_infer = threading.Thread(target=infer_worker); thr_infer.start()

    # --- 可视化线程（与原版一致） ---
    def vis_worker():
        win = "ZeJing"
        if args.save_vis:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            item = vis_q.get()
            if item is None:
                break
            idx, show_img, tag = item
            show_img = _morph_open_close(show_img, args.morph_open, args.morph_close)
            if args.save_vis:
                cv2.imshow(win, show_img)
                if cv2.waitKey(1) == ord('q'):
                    args.save_vis = False
            out_dir = infer_out_dir if ("RE(" in tag or "RBD" in tag or "Bank+Detect" in tag) else train_out_dir
            if args.save_vis:
                cv2.imwrite(f"{out_dir}/{idx:06d}.png", show_img)
        if args.save_vis:
            cv2.destroyAllWindows()

    thr_vis = threading.Thread(target=vis_worker); thr_vis.start()

    final_fps_train = None
    final_fps_infer = None

    # =========================
    # 一、训练：images + event(s)  —— 引入预取 & 批量转换
    # =========================
    if has_train:
        print("[MODE] Train on images + event(s) with optional masks")
        indices = list(rgb_map.keys())

        def load_train_item(idx: int):  # [MOD] 移除后台 mask/rgb mask 应用 & to_torch（延迟到主线程）
            rgb_path = rgb_map[idx]
            evt_path = evt_map.get(idx, None)
            mask_path = mask_map.get(idx, None)
            # [MOD] mask 保持 raw（binary np），不 to_torch 到 device
            mask_raw = load_mask_binary(mask_path, (H, W), 'cpu') if mask_path else None  # 'cpu' 避免 GPU 上传
            rgb_loaded = load_rgb(rgb_path, (H, W), adapter)
            if rgb_loaded is None:
                return None
            rgb_f, rgb_u8 = rgb_loaded
            evt = load_evt(evt_path, (H, W), adapter)  # 保持 adapter 格式，无 to_torch
            return (rgb_f, rgb_u8, evt, mask_raw)  # mask_raw 为 np/None

        prefetch = Prefetcher(indices, load_train_item, maxsize=args.prefetch, name="Prefetch-Train")
        # [MOD] 用 deque 高效 popleft
        rgb_cache = deque(maxlen=args.seq_len)
        evt_cache = deque(maxlen=args.seq_len)
        u8_cache = deque(maxlen=args.seq_len)
        mask_cache = deque(maxlen=args.seq_len)
        t0 = time.time()
        n_total = len(indices)
        it = 0
        while True:
            got = prefetch.get()
            if got is None:  # 结束标记
                break
            idx, payload = got
            it += 1
            if payload is None:
                _print_progress("TRAIN", it, n_total, t0, final_fps_train)
                continue
            rgb_f, rgb_u8, evt, mask_raw = payload
            # [MOD] 主线程应用 mask（避免后台同步）
            mask_t = adapter.to_torch(mask_raw, device) if mask_raw is not None else None
            rgb_f = _mask_chw_float(rgb_f, mask_t, adapter)

            rgb_cache.append(rgb_f)
            u8_cache.append(rgb_u8)
            evt_cache.append(evt)
            mask_cache.append(mask_t)

            if len(rgb_cache) < args.seq_len:
                _print_progress("TRAIN", it, n_total, t0, final_fps_train)
                continue

            # —— 在“批处理级别”做 stack + to_torch（减少多余转换与同步）
            rgb_seq_arr = adapter.stack(list(rgb_cache))            # (T,3,H,W) [cupy/np]
            rgb_seq = adapter.to_torch(rgb_seq_arr, device).unsqueeze(0)  # (1,T,3,H,W)

            evt_ready = all([e is not None for e in evt_cache])
            evt_seq = None
            if evt_ready:
                evt_seq_arr = adapter.stack(list(evt_cache))      # (T,1,H,W)
                evt_seq = adapter.to_torch(evt_seq_arr, device).unsqueeze(0)

            with torch.no_grad():
                feat_last, det_up_T, det_fm_T = teacher_forward(teacher, rgb_seq, evt_seq)

            # 用 event 监督更新 Bank —— 交给异步线程（跳帧）
            if evt_ready:
                C, Hf, Wf = feat_last.shape
                evt_last = evt_seq[0, -1]  # (1,H,W)
                evt_fm = F.interpolate(evt_last.unsqueeze(0), size=(Hf, Wf), mode='area')[0].clamp(0, 1)
                mlast = mask_cache[-1]
                if mlast is not None:
                    mask_small = F.interpolate(mlast[None, None, ...], size=(Hf, Wf), mode='nearest')[0]
                    evt_fm = evt_fm * mask_small
                if bank_async is not None:
                    bank_async.maybe_submit_update(idx, feat_last, evt_fm)
                else:
                    bank.update_from_teacher(feat_last, evt_fm)

            target_fm_dev = None
            if evt_ready:
                C, Hf, Wf = feat_last.shape
                evt_last = evt_seq[0, -1]
                evt_fm = F.interpolate(evt_last.unsqueeze(0), size=(Hf, Wf), mode='area')[0]
                target_fm_dev = evt_fm.clamp(0, 1)

            infer_q.put((idx, feat_last.detach(), det_up_T.detach(), u8_cache[-1], evt_ready, target_fm_dev, mask_cache[-1]))

            with sample_lock:
                sample_buffer.append((feat_last.detach(), target_fm_dev))
                max_keep = max(4 * args.max_train_frames, args.max_train_frames)
                if len(sample_buffer) > max_keep:
                    del sample_buffer[:len(sample_buffer) - max_keep]
                end_n = len(sample_buffer)

            if evt_ready:
                for _tid in range(args.train_threads):
                    put_latest(train_queues[_tid], end_n)

            _print_progress("TRAIN", it, n_total, t0, final_fps_train)
        final_fps_train = (it / max(time.time() - t0, 1e-6))  # [MOD] 计算最终 FPS
        _print_progress("TRAIN", it, n_total, t0, final_fps_train)  # 触发写文件
    else:
        print("[MODE] No train pairs found (images+events). Skip training.")

    # =========================
    # 二、推理：infer_dir 帧 + 事件伪特征点 + ROI gating  —— 引入预取 & 批量转换
    # =========================
    if has_infer:
        print("[MODE] Infer on frames (infer_dir) with RBD + event pseudo-kpts + mask gating")
        all_items = list(infer_map.items())
        indices = [idx for idx, _ in all_items]

        def load_infer_item(idx: int):  # [MOD] 移除后台 to_torch & mask 应用
            rgb_path = infer_map[idx]
            mask_path = mask_map.get(idx, None)
            # [MOD] mask 保持 raw
            mask_raw = load_mask_binary(mask_path, (H, W), 'cpu') if mask_path else None
            rgb_loaded = load_rgb(rgb_path, (H, W), adapter)
            if rgb_loaded is None:
                return None
            rgb_f, rgb_u8 = rgb_loaded
            # 事件帧（若存在）按 HxW 对齐，保持 adapter 格式
            evt_path = evt_map.get(idx, None)
            evt_curr = None
            if evt_path is not None:
                try:
                    evt_arr = load_evt(evt_path, (H, W), adapter)
                    if evt_arr is not None:
                        evt_curr = evt_arr  # [MOD] 无 to_torch
                except Exception:
                    evt_curr = None
            return (rgb_f, rgb_u8, mask_raw, evt_curr)

        prefetch = Prefetcher(indices, load_infer_item, maxsize=args.prefetch, name="Prefetch-Infer")
        # [MOD] 用 deque
        rgb_cache = deque(maxlen=args.seq_len)
        u8_cache = deque(maxlen=args.seq_len)
        t0 = time.time()
        it = 0
        n_total = len(indices)
        while True:
            got = prefetch.get()
            if got is None:
                break
            idx, payload = got
            it += 1
            if payload is None:
                _print_progress("INFER", it, n_total, t0, final_fps_infer)
                continue
            rgb_f, rgb_u8, mask_raw, evt_curr = payload
            # [MOD] 主线程应用 mask
            mask_t = adapter.to_torch(mask_raw, device) if mask_raw is not None else None
            rgb_f = _mask_chw_float(rgb_f, mask_t, adapter)

            rgb_cache.append(rgb_f)
            u8_cache.append(rgb_u8)
            if len(rgb_cache) < args.seq_len:
                _print_progress("INFER", it, n_total, t0, final_fps_infer)
                continue

            rgb_seq_arr = adapter.stack(list(rgb_cache))
            rgb_seq = adapter.to_torch(rgb_seq_arr, device).unsqueeze(0)
            with torch.no_grad():
                feat_last, det_up_T, det_fm_T = teacher_forward(teacher, rgb_seq, None)

            # [MOD] 延迟 to_torch evt_curr
            evt_pack = None
            if evt_curr is not None:
                try:
                    evt_pack = adapter.to_torch(evt_curr, device)
                    if evt_pack.dim() == 2:
                        evt_pack = evt_pack.unsqueeze(0)  # (1,H,W)
                except Exception:
                    evt_pack = None

            infer_q.put((idx, feat_last.detach(), det_up_T.detach(), u8_cache[-1], False, evt_pack, mask_t))

            _print_progress("INFER", it, n_total, t0, final_fps_infer)
        final_fps_infer = (it / max(time.time() - t0, 1e-6))
        _print_progress("INFER", it, n_total, t0, final_fps_infer)  # 触发写文件
    else:
        print("[MODE] No infer frames found. Skip inference.")

    # ===== 收尾 =====
    for q in train_queues:
        q.put(None)
    infer_q.put(None)
    thr_infer.join()
    vis_q.put(None)
    thr_vis.join()

    # === 新增：将最终 online_head_best.pth 复制到与 matches_*.txt 相同目录 ===
    try:
        import shutil
        src_best = args.save_best if os.path.isfile(args.save_best) else DEFAULT_ONLINE_BEST
        if os.path.isfile(src_best):
            ensure_dir(args.matches_dump_dir)
            dst_best = os.path.join(args.matches_dump_dir, "online_head_best.pth")
            shutil.copy2(src_best, dst_best)
            print(f"[Persist] Copied best online head to: {dst_best}")
        else:
            print(f"[Persist] No best weight found at {args.save_best} or {DEFAULT_ONLINE_BEST}, skip copy.")
    except Exception as e:
        print(f"[Persist] Failed to copy best weight: {e}")

    if args.bank_state:
        try:
            torch.save(bank.export_state(), args.bank_state)
            print(f"[Bank] Saved state to {args.bank_state}")
        except Exception as e:
            print(f"[Bank] Save failed: {e}")

    if bank_async is not None:
        bank_async.close()

if __name__ == "__main__":
    online_train = DEFAULT_ONLINE_BEST
    if os.path.exists(online_train):
        os.remove(online_train)
        print(f"已删除旧文件: {online_train}")
    else:
        print(f"文件不存在，无需删除: {online_train}")

    main()

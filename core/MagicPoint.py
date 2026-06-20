#-*-coding:utf8-*-
import torch
from core.solver.nms import box_nms
from core.modules.cnn.vgg_backbone import VGGBackboneBN,VGGBackbone
from core.modules.cnn.cnn_heads import DetectorHead

# ====== 模型定义 ======
class MagicPoint(torch.nn.Module):
    def __init__(self, config, input_channel=1, grid_size=8, using_bn=True, device='cuda'):
        super().__init__()
        self.nms        = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk       = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
        self.detector_head = DetectorHead(input_channel=128, grid_size=grid_size, using_bn=using_bn)

    def forward(self, x):
        feat_map = self.backbone(x['img'] if isinstance(x, dict) else x)
        outputs  = self.detector_head(feat_map)
        prob = outputs.get('prob_nms', outputs.get('prob'))
        if prob.ndim == 4 and prob.size(1) == 1:
            prob_iter = prob[:, 0]
        elif prob.ndim == 3:
            prob_iter = prob
        else:
            raise RuntimeError(f"Unexpected prob shape: {prob.shape}")
        if self.nms is not None and "prob_nms" not in outputs:
            prob_list = []
            for p in prob_iter:
                p1 = box_nms(p.unsqueeze(0), self.nms,
                             min_prob=self.det_thresh,
                             keep_top_k=self.topk).squeeze(0)
                prob_list.append(p1)
            outputs['prob_nms'] = torch.stack(prob_list, dim=0)
        outputs.setdefault('pred', (outputs['prob_nms'] if 'prob_nms' in outputs else prob_iter) >= self.det_thresh)
        return outputs

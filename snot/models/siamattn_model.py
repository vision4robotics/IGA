from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from snot.core.config_attn import cfg
from snot.models.backbone import get_backbone
from snot.models.head import get_rpn_head
from snot.models.neck import get_neck
from snot.models.attn.feature_fusion import FeatureFusionNeck
from snot.models.attn.enhance import FeatureEnhance
from snot.models.attn.mask import FusedSemanticHead
from snot.models.attn.detection import FCx2DetHead


class ModelBuilderAttn(nn.Module):
    def __init__(self):
        super(ModelBuilderAttn, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)
            self.feature_fusion = FeatureFusionNeck(num_ins=5, fusion_level=1,
                                                    in_channels=[64, 256, 256, 256, 256], conv_out_channels=256)
            self.mask_head = FusedSemanticHead(pooling_func=None,
                                               num_convs=4, in_channels=256,
                                               upsample_ratio=(cfg.MASK.MASK_OUTSIZE // cfg.TRAIN.ROIPOOL_OUTSIZE))
            self.bbox_head = FCx2DetHead(pooling_func=None,
                                         in_channels=256 * (cfg.TRAIN.ROIPOOL_OUTSIZE // 4)**2)

    def template(self, z):
        with torch.no_grad():
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf[2:] = self.neck(zf[2:])
            self.zf = zf

    def track(self, x):
        with torch.no_grad():
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf[2:] = self.neck(xf[2:])

            zf, xf[2:] = self.feature_enhance(self.zf[2:], xf[2:])
            cls, loc = self.rpn_head(zf, xf[2:])
            enhanced_zf = self.zf[:2] + zf
            if cfg.MASK.MASK:
                self.b_fused_features, self.m_fused_features = self.feature_fusion(enhanced_zf, xf)
            return {
                'cls': cls,
                'loc': loc
            }

    def mask_refine(self, roi):
        with torch.no_grad():
            mask_pred = self.mask_head(self.m_fused_features, roi)

        return mask_pred

    def bbox_refine(self, roi):
        with torch.no_grad():
            bbox_pred = self.bbox_head(self.b_fused_features, roi)

        return bbox_pred

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

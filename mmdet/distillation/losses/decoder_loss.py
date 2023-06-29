# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from mmcv.cnn import constant_init, kaiming_init
# from ..builder import DISTILL_LOSSES
#
#
# @DISTILL_LOSSES.register_module()
# class DecoderFeatureLoss(nn.Module):
#     def __init__(self,ratio=0.5,name=None):
#         self.ratio = ratio
#         self.name = name
#
#         return
#
#     def forward(self,pred_S,pred_T,img_metas):
#         feature_loss = torch.cosine_similarity(pred_S,pred_T, dim=0).mean()
#         return feature_loss
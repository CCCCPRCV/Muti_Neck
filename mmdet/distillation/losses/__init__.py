from .fgd import  FeatureLoss
from .merge_fgd_loss import MergeLoss
from .queue_feature import EncoderFeatureLoss,DecoderFeatureLoss,FeatureLoss_small
from .merge_fgd_loss_add_mask import MergeEDLoss
__all__ = [
    'FeatureLoss','EncoderFeatureLoss','DecoderFeatureLoss','FeatureLoss_small',
    'MergeLoss','MergeEDLoss'
]

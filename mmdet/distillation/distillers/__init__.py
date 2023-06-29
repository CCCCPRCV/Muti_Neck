from .detection_distiller import DetectionDistiller
from .merge_distiller import MergeDistiller
from .center_collecter import CenterCollerter
from .merge_distiller_add_proposal import MergeEDDistiller

__all__ = [
    'DetectionDistiller','MergeDistiller','CenterCollerter','MergeEDDistiller'
]
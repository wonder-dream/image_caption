"""
工具模块
"""

from .deepfashion_dataset import DeepFashionDataset, create_deepfashion_dataset
from .eval_metrics import COCOScoreEvaluator
from .scst_loss import SCSTLoss, CiderRewardCalculator, get_reference_captions
from .optimizations import (
    LabelSmoothingLoss,
    FocalLoss,
    WarmupCosineScheduler,
    TransformerScheduler,
    CaptionAugmentation,
    ExponentialMovingAverage,
    EarlyStopping,
    GradientClipping,
    R_Drop,
    DropPath,
    get_optimized_config,
)

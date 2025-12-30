"""
图像描述生成模型模块
"""

from .vit_transformer_model import ViTTransformerCaptioning, build_model as build_vit_model
from .grid_transformer_model import GridTransformerCaptioning, build_model as build_grid_model

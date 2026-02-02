"""
olive-auto: Batch optimization pipeline generator for Olive.

Generate complete optimization pipelines for HuggingFace models
across all supported targets and precisions.
"""

__version__ = "0.1.0"
__author__ = "Microsoft"

from olive_auto.task_detection import detect_model_info, infer_task_from_model
from olive_auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY
from olive_auto.command_generator import CommandGenerator
from olive_auto.environment import EnvironmentScriptGenerator

__all__ = [
    "detect_model_info",
    "infer_task_from_model",
    "TARGET_PRECISION_MATRIX",
    "CATEGORY_TARGET_COMPATIBILITY",
    "CommandGenerator",
    "EnvironmentScriptGenerator",
]

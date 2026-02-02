"""Auto task detection for HuggingFace models."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default task when detection fails
DEFAULT_HF_TASK = "text-generation-with-past"

# Task to category mapping
TASK_CATEGORY_MAP = {
    # Text/NLP
    "text-generation": "nlp",
    "text-generation-with-past": "nlp",
    "text2text-generation": "nlp",
    "text2text-generation-with-past": "nlp",
    "text-classification": "nlp",
    "token-classification": "nlp",
    "question-answering": "nlp",
    "fill-mask": "nlp",
    "feature-extraction": "nlp",
    "zero-shot-classification": "nlp",
    "multiple-choice": "nlp",
    "summarization": "nlp",
    "translation": "nlp",
    # Vision
    "image-classification": "vision",
    "image-segmentation": "vision",
    "semantic-segmentation": "vision",
    "object-detection": "vision",
    "image-to-text": "vision",
    "zero-shot-image-classification": "vision",
    "zero-shot-object-detection": "vision",
    "depth-estimation": "vision",
    "image-feature-extraction": "vision",
    # Audio
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
    "text-to-audio": "audio",
    "audio-to-audio": "audio",
    "audio-frame-classification": "audio",
    "audio-xvector": "audio",
    # Multimodal
    "image-text-to-text": "multimodal",
    "visual-question-answering": "multimodal",
    "document-question-answering": "multimodal",
    # Generative
    "stable-diffusion": "diffusers",
    "stable-diffusion-xl": "diffusers",
    "text-to-image": "diffusers",
}

# Architecture suffix to task mapping
ARCHITECTURE_TASK_MAP = {
    "ForCausalLM": "text-generation-with-past",
    "ForSeq2SeqLM": "text2text-generation-with-past",
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForQuestionAnswering": "question-answering",
    "ForMaskedLM": "fill-mask",
    "ForImageClassification": "image-classification",
    "ForObjectDetection": "object-detection",
    "ForSemanticSegmentation": "image-segmentation",
    "ForAudioClassification": "audio-classification",
    "ForCTC": "automatic-speech-recognition",
    "ForSpeechSeq2Seq": "automatic-speech-recognition",
    "ForVision2Seq": "image-to-text",
    "ForConditionalGeneration": "text2text-generation-with-past",
    "ForMultipleChoice": "multiple-choice",
    "ForNextSentencePrediction": "text-classification",
    "ForPreTraining": "feature-extraction",
    "ForFeatureExtraction": "feature-extraction",
    "ForImageSegmentation": "image-segmentation",
    "ForDepthEstimation": "depth-estimation",
}


def infer_task_from_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> Optional[str]:
    """
    Infer the task from a HuggingFace model.

    Uses multiple strategies:
    1. Optimum's TasksManager (most reliable)
    2. Model config's architectures field
    3. Model card/README metadata

    Args:
        model_name_or_path: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code

    Returns:
        Inferred task string or None if detection fails
    """
    task = None

    # Strategy 1: Use Optimum's TasksManager
    task = _infer_via_tasks_manager(model_name_or_path)
    if task:
        logger.debug(f"Task inferred via TasksManager: {task}")
        return task

    # Strategy 2: Infer from model architecture name
    task = _infer_from_architecture(model_name_or_path, trust_remote_code)
    if task:
        logger.debug(f"Task inferred from architecture: {task}")
        return task

    # Strategy 3: Check model card metadata
    task = _infer_from_model_card(model_name_or_path)
    if task:
        logger.debug(f"Task inferred from model card: {task}")
        return task

    return None


def _infer_via_tasks_manager(model_name_or_path: str) -> Optional[str]:
    """Use Optimum's TasksManager for task inference."""
    try:
        from optimum.exporters.tasks import TasksManager

        task = TasksManager.infer_task_from_model(model_name_or_path)
        if task in ("text-generation", "text2text-generation"):
            task = f"{task}-with-past"
        return task
    except ImportError:
        logger.debug("Optimum not installed, skipping TasksManager inference")
        return None
    except Exception as e:
        logger.debug(f"TasksManager inference failed: {e}")
        return None


def _infer_from_architecture(
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> Optional[str]:
    """Infer task from model architecture name pattern."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        architectures = getattr(config, "architectures", []) or []

        for arch in architectures:
            for suffix, task in ARCHITECTURE_TASK_MAP.items():
                if arch.endswith(suffix):
                    return task
        return None
    except ImportError:
        logger.debug("transformers not installed, skipping architecture inference")
        return None
    except Exception as e:
        logger.debug(f"Architecture inference failed: {e}")
        return None


def _infer_from_model_card(model_name_or_path: str) -> Optional[str]:
    """Infer task from HuggingFace model card metadata."""
    try:
        from huggingface_hub import model_info

        info = model_info(model_name_or_path)
        if info.pipeline_tag:
            task = info.pipeline_tag
            if task in ("text-generation", "text2text-generation"):
                task = f"{task}-with-past"
            return task
        return None
    except ImportError:
        logger.debug("huggingface_hub not installed, skipping model card inference")
        return None
    except Exception as e:
        logger.debug(f"Model card inference failed: {e}")
        return None


def get_model_category(task: str) -> str:
    """
    Get the model category from task.

    Args:
        task: The task string

    Returns:
        Category string (nlp, vision, audio, multimodal, diffusers, or unknown)
    """
    return TASK_CATEGORY_MAP.get(task, "unknown")


def detect_model_info(
    model_name_or_path: str,
    user_task: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[str, str]:
    """
    Detect model task and category.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        user_task: User-specified task (overrides detection)
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (task, category)
    """
    if user_task:
        task = user_task
        logger.info(f"Using user-specified task: {task}")
    else:
        task = infer_task_from_model(model_name_or_path, trust_remote_code)
        if task is None:
            task = DEFAULT_HF_TASK
            logger.warning(
                f"Could not detect task for {model_name_or_path}, "
                f"defaulting to {task}"
            )
        else:
            logger.info(f"Detected task: {task}")

    category = get_model_category(task)
    logger.info(f"Model category: {category}")

    return task, category

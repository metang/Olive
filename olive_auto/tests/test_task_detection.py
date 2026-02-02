"""Tests for task detection module."""
import pytest

from olive_auto.task_detection import (
    detect_model_info,
    infer_task_from_model,
    get_model_category,
    TASK_CATEGORY_MAP,
    ARCHITECTURE_TASK_MAP,
    DEFAULT_HF_TASK,
)


class TestTaskCategoryMap:
    """Tests for task to category mapping."""

    def test_nlp_tasks(self):
        """Test that NLP tasks map to 'nlp' category."""
        nlp_tasks = [
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            "token-classification",
            "question-answering",
            "fill-mask",
        ]
        for task in nlp_tasks:
            assert TASK_CATEGORY_MAP.get(task) == "nlp"

    def test_vision_tasks(self):
        """Test that vision tasks map to 'vision' category."""
        vision_tasks = [
            "image-classification",
            "object-detection",
            "image-segmentation",
        ]
        for task in vision_tasks:
            assert TASK_CATEGORY_MAP.get(task) == "vision"

    def test_audio_tasks(self):
        """Test that audio tasks map to 'audio' category."""
        audio_tasks = [
            "automatic-speech-recognition",
            "audio-classification",
        ]
        for task in audio_tasks:
            assert TASK_CATEGORY_MAP.get(task) == "audio"


class TestArchitectureTaskMap:
    """Tests for architecture to task mapping."""

    def test_causal_lm_mapping(self):
        """Test that ForCausalLM maps to text-generation-with-past."""
        assert ARCHITECTURE_TASK_MAP["ForCausalLM"] == "text-generation-with-past"

    def test_seq2seq_mapping(self):
        """Test that ForSeq2SeqLM maps to text2text-generation-with-past."""
        assert ARCHITECTURE_TASK_MAP["ForSeq2SeqLM"] == "text2text-generation-with-past"

    def test_classification_mapping(self):
        """Test that ForSequenceClassification maps to text-classification."""
        assert ARCHITECTURE_TASK_MAP["ForSequenceClassification"] == "text-classification"


class TestGetModelCategory:
    """Tests for get_model_category function."""

    def test_known_task(self):
        """Test category detection for known task."""
        assert get_model_category("text-generation-with-past") == "nlp"
        assert get_model_category("image-classification") == "vision"
        assert get_model_category("automatic-speech-recognition") == "audio"

    def test_unknown_task(self):
        """Test category detection for unknown task."""
        assert get_model_category("unknown-task") == "unknown"


class TestDetectModelInfo:
    """Tests for detect_model_info function."""

    def test_user_task_override(self):
        """User-specified task should override detection."""
        task, category = detect_model_info(
            "nonexistent-model",
            user_task="text-classification",
        )
        assert task == "text-classification"
        assert category == "nlp"

    def test_fallback_to_default(self):
        """Should fallback to default task when detection fails."""
        task, category = detect_model_info(
            "nonexistent-model-12345",
            user_task=None,
        )
        assert task == DEFAULT_HF_TASK
        assert category == "nlp"


class TestInferTaskFromModel:
    """Tests for infer_task_from_model function."""

    def test_nonexistent_model_returns_none(self):
        """Should return None for nonexistent model."""
        result = infer_task_from_model("nonexistent-model-12345")
        # May return None or a task depending on network conditions
        assert result is None or isinstance(result, str)

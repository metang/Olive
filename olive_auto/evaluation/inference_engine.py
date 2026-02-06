"""
Inference engine for evaluating ONNX models.

Handles ONNX Runtime session creation, input preprocessing, inference execution,
and metric computation (latency, memory, accuracy).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


class OnnxInferenceEngine:
    """Wrapper around ONNX Runtime for inference and metrics."""

    def __init__(
        self,
        model_path: Path,
        providers: Optional[List[str]] = None,
        enable_profiling: bool = False,
    ):
        """
        Initialize ONNX Runtime inference engine.

        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            enable_profiling: Whether to enable ONNX Runtime profiling
        """
        self.model_path = Path(model_path)
        self.providers = providers or ort.get_available_providers()
        self.enable_profiling = enable_profiling
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize ONNX Runtime inference session."""
        try:
            # Create session options
            sess_opts = ort.SessionOptions()
            sess_opts.enable_profiling = self.enable_profiling
            sess_opts.log_severity_level = 3  # Suppress verbose logging

            # Create session with specified providers
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=self.providers,
            )

            # Extract input/output names and shapes
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            logger.debug(
                f"Initialized ONNX session for {self.model_path.name} "
                f"with providers: {self.session.get_providers()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            raise

    def run_inference(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on the model.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
            output_names: List of output names to return (default: all outputs)

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            outputs = self.session.run(output_names, inputs)
            if output_names:
                return dict(zip(output_names, outputs))
            else:
                return dict(zip(self.output_names, outputs))
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def measure_latency(
        self,
        inputs: Dict[str, np.ndarray],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Measure latency percentiles (p50, p99).

        Args:
            inputs: Input data dictionary
            num_runs: Number of inference runs to measure
            warmup_runs: Number of warmup runs before measurement

        Returns:
            Dictionary with 'p50_ms' and 'p99_ms' keys
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        # Warmup runs to stabilize performance
        for _ in range(warmup_runs):
            try:
                self.session.run(None, inputs)
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
                return {}

        # Measure latencies
        latencies_ms = []
        for _ in range(num_runs):
            try:
                start = time.perf_counter()
                self.session.run(None, inputs)
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                latencies_ms.append(elapsed)
            except Exception as e:
                logger.warning(f"Latency measurement failed: {e}")
                continue

        if not latencies_ms:
            logger.warning("No successful latency measurements")
            return {}

        # Calculate percentiles
        latencies_ms.sort()
        p50 = np.percentile(latencies_ms, 50)
        p99 = np.percentile(latencies_ms, 99)

        return {
            "latency_p50_ms": float(p50),
            "latency_p99_ms": float(p99),
        }

    def get_input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get input tensor shapes from model.

        Returns:
            Dictionary mapping input names to shape tuples
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        shapes = {}
        for inp in self.session.get_inputs():
            # Convert shape that may contain strings like 'batch_size' to actual ints
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    # Use 1 for dynamic dimensions
                    shape.append(1)
            shapes[inp.name] = tuple(shape)

        return shapes

    def get_input_dtypes(self) -> Dict[str, str]:
        """
        Get input tensor data types from model.

        Returns:
            Dictionary mapping input names to dtype strings
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        dtypes = {}
        for inp in self.session.get_inputs():
            dtypes[inp.name] = inp.type

        return dtypes

    def create_dummy_inputs(self) -> Dict[str, np.ndarray]:
        """
        Create dummy input tensors matching model input shapes.

        Returns:
            Dictionary of dummy input arrays
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        dummy_inputs = {}
        for inp in self.session.get_inputs():
            # Convert shape that may contain strings to actual ints
            shape = []
            for dim in inp.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    # Use 1 for dynamic dimensions
                    shape.append(1)

            # Create dummy data based on type
            if "float" in inp.type.lower():
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
            elif "int64" in inp.type.lower():
                dummy_inputs[inp.name] = np.random.randint(0, 100, shape).astype(
                    np.int64
                )
            elif "int32" in inp.type.lower():
                dummy_inputs[inp.name] = np.random.randint(0, 100, shape).astype(
                    np.int32
                )
            else:
                # Default to float32
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        return dummy_inputs

    def close(self) -> None:
        """Close the ONNX Runtime session."""
        self.session = None


class AccuracyComputor:
    """Compute accuracy metrics for language models using various methods."""

    @staticmethod
    def compute_perplexity(
        model_engine: OnnxInferenceEngine,
        texts: List[str],
        tokenizer_fn: Optional[callable] = None,
    ) -> Optional[float]:
        """
        Compute perplexity on text samples.

        Args:
            model_engine: ONNX inference engine
            texts: List of text samples to evaluate
            tokenizer_fn: Function to tokenize text (must return input dict for model)

        Returns:
            Perplexity score or None if computation failed
        """
        if not texts or not tokenizer_fn:
            logger.warning("Cannot compute perplexity: missing texts or tokenizer")
            return None

        try:
            log_probs = []

            for text in texts:
                try:
                    # Tokenize
                    inputs = tokenizer_fn(text)

                    # Run inference
                    outputs = model_engine.run_inference(inputs)

                    # Extract log probabilities
                    # Assuming output is logits, compute log softmax
                    if outputs:
                        # Get first output
                        logits = list(outputs.values())[0]
                        # Compute softmax (simplified approach)
                        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                        # Use average log prob
                        log_prob = np.mean(np.log(np.maximum(probs, 1e-10)))
                        log_probs.append(log_prob)
                except Exception as e:
                    logger.debug(f"Failed to compute log prob for sample: {e}")
                    continue

            if not log_probs:
                return None

            # Perplexity = exp(-mean(log_prob))
            perplexity = float(np.exp(-np.mean(log_probs)))
            return perplexity

        except Exception as e:
            logger.error(f"Failed to compute perplexity: {e}")
            return None

    @staticmethod
    def compute_sklearn_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_type: str = "accuracy",
    ) -> Optional[float]:
        """
        Compute sklearn metrics (accuracy, F1, etc.) for classification.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            metric_type: Type of metric (accuracy, f1, precision, recall)

        Returns:
            Metric value or None if computation failed
        """
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            if metric_type == "accuracy":
                return float(accuracy_score(y_true, y_pred))
            elif metric_type == "f1":
                return float(f1_score(y_true, y_pred, average="weighted"))
            elif metric_type == "precision":
                return float(precision_score(y_true, y_pred, average="weighted"))
            elif metric_type == "recall":
                return float(recall_score(y_true, y_pred, average="weighted"))
            else:
                logger.warning(f"Unknown metric type: {metric_type}")
                return None

        except ImportError:
            logger.warning("sklearn not available for metric computation")
            return None
        except Exception as e:
            logger.error(f"Failed to compute {metric_type}: {e}")
            return None

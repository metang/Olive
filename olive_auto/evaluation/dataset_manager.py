"""
Dataset management for evaluation.

Handles automatic downloading and sampling of datasets for model evaluation,
with support for 10% (normal) and 1% (test mode) sampling fractions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manage dataset download and preparation for evaluation."""
    
    # Mapping of task to dataset configuration
    TASK_TO_DATASET = {
        "text-generation": {
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-103-v1",
            "split": "validation",
            "text_field": "text",
        },
        "text-classification": {
            "dataset_name": "glue",
            "dataset_config": "mnli",
            "split": "validation_matched",
            "text_field": "premise",
        },
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, test_mode: bool = False):
        """
        Initialize dataset manager.
        
        Args:
            cache_dir: Directory to cache datasets (default: ~/.cache/olive-auto)
            test_mode: If True, use 1% of data. If False, use 10%
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "olive-auto"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        self.sample_fraction = 0.01 if test_mode else 0.10
    
    def get_evaluation_data(
        self,
        task: str,
        model_name: str,
        num_samples: Optional[int] = None,
    ) -> Optional[List[Dict]]:
        """
        Get evaluation data for a given task and model.
        
        Automatically downloads dataset if not cached. Returns sampled data
        according to test_mode (1% or 10%).
        
        Args:
            task: Model task type (text-generation, text-classification, etc.)
            model_name: Name of model (for logging)
            num_samples: Override sample count (if None, use sample_fraction)
            
        Returns:
            List of data samples, or None if dataset unavailable
        """
        if task not in self.TASK_TO_DATASET:
            logger.warning(
                f"Skipping evaluation for {model_name}: no dataset for task '{task}'"
            )
            return None
        
        config = self.TASK_TO_DATASET[task]
        
        try:
            dataset = self._load_dataset(
                dataset_name=config["dataset_name"],
                dataset_config=config["dataset_config"],
                split=config["split"],
            )
            
            if dataset is None:
                return None
            
            # Determine sample count
            if num_samples is None:
                num_samples = max(1, int(len(dataset) * self.sample_fraction))
            
            # Sample data
            indices = list(range(min(num_samples, len(dataset))))
            sampled_data = [dataset[i] for i in indices]
            
            logger.info(
                f"Loaded {len(sampled_data)} samples from {config['dataset_name']} "
                f"for {model_name} (fraction: {self.sample_fraction})"
            )
            
            return sampled_data
        
        except Exception as e:
            logger.warning(
                f"Failed to load dataset for {model_name} ({task}): {e}"
            )
            return None
    
    def _load_dataset(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
    ):
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: Name of dataset (e.g., 'wikitext')
            dataset_config: Config/subset name (e.g., 'wikitext-103-v1')
            split: Split to load (e.g., 'validation')
            
        Returns:
            Loaded dataset, or None if load failed
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error(
                "Failed to import 'datasets' library. "
                "Install with: pip install datasets"
            )
            return None
        
        try:
            logger.debug(f"Loading {dataset_name}/{dataset_config}:{split}")
            
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )
            
            return dataset
        
        except Exception as e:
            logger.warning(
                f"Failed to load {dataset_name}/{dataset_config}:{split}: {e}"
            )
            return None
    
    def verify_dataset_available(self, task: str) -> bool:
        """
        Check if dataset is available for a given task.
        
        Args:
            task: Model task type
            
        Returns:
            True if dataset is available, False otherwise
        """
        return task in self.TASK_TO_DATASET

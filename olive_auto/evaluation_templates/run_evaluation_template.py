"""
Template for auto-generated run_evaluation.py script.

This file is rendered with pipeline-specific values and saved to the
pipeline output directory for user execution.
"""

# This file will be rendered with Jinja2 or string formatting
# Variables: pipeline_dir, manifest_path, model_name

EVALUATION_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
Auto-generated evaluation script for olive-auto pipeline

Generated: {timestamp}
Pipeline: {pipeline_name}

This script evaluates all converted models in the olive-auto pipeline,
measuring accuracy, latency, memory usage, and model size.

Usage:
    python run_evaluation.py              # Evaluate all models (10% dataset)
    python run_evaluation.py --test       # Evaluate all models (1% dataset)  
    python run_evaluation.py --help       # Show all options

Output:
    evaluation_results/
    ├── results_summary.json              # Detailed results in JSON
    ├── results_summary.csv               # Results in CSV (for Excel)
    └── results_dashboard.html            # Interactive HTML dashboard

Requirements:
    - Python 3.8+
    - onnx
    - datasets
    - plotly (for HTML dashboard)
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate olive-auto generated models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py              # Full evaluation (10%% dataset)
  python run_evaluation.py --test       # Quick test (1%% dataset)
  python run_evaluation.py --output-dir ./my_results

        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: use only 1%% of dataset for faster evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)"
    )
    
    args = parser.parse_args()
    
    # Import evaluation components
    try:
        from olive_auto.evaluation.evaluator import PipelineEvaluator
        from olive_auto.evaluation.config import EvaluationConfig
    except ImportError as e:
        logger.error(
            f"Failed to import olive_auto evaluation modules: {e}\\n"
            "Install with: pip install -e ."
        )
        sys.exit(1)
    
    # Get pipeline directory (parent of this script)
    pipeline_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 70)
    logger.info("Olive-Auto Model Evaluation")
    logger.info("=" * 70)
    logger.info(f"Pipeline directory: {pipeline_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Test mode: {args.test} (dataset fraction: {1 if args.test else 10}%%)")
    logger.info("=" * 70)
    
    try:
        # Create evaluator
        evaluator = PipelineEvaluator(
            pipeline_dir=pipeline_dir,
            test_mode=args.test,
            output_dir=output_dir
        )
        
        # Run evaluation
        logger.info("Starting model evaluation...")
        summary = evaluator.evaluate_all_models()
        
        # Print summary
        logger.info("=" * 70)
        logger.info("Evaluation Complete!")
        logger.info("=" * 70)
        logger.info(f"Total models: {summary.total_models}")
        logger.info(f"Evaluated: {summary.evaluated_models}")
        logger.info(f"Skipped: {summary.skipped_models}")
        logger.info(f"Failed: {summary.failed_models}")
        logger.info(f"Total time: {summary.total_time_sec:.1f}s")
        logger.info("=" * 70)
        
        logger.info(f"Results saved to: {output_dir}/")
        logger.info(f"  - JSON: results_summary.json")
        logger.info(f"  - CSV: results_summary.csv (for Excel)")
        logger.info(f"  - HTML: results_dashboard.html (interactive)")
        logger.info("")
        
        # Print recommendations
        if summary.recommendations.best_overall:
            logger.info("Recommendations:")
            if summary.recommendations.best_overall:
                logger.info(f"  Best overall: {summary.recommendations.best_overall}")
            if summary.recommendations.best_accuracy:
                logger.info(f"  Best accuracy: {summary.recommendations.best_accuracy}")
            if summary.recommendations.best_latency:
                logger.info(f"  Best latency: {summary.recommendations.best_latency}")
            if summary.recommendations.best_efficiency:
                logger.info(f"  Most efficient: {summary.recommendations.best_efficiency}")
        
        logger.info("=" * 70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''


def get_evaluation_script_template(
    timestamp: str = None,
    pipeline_name: str = "olive-auto pipeline"
) -> str:
    """
    Get the evaluation script template with values filled in.
    
    Args:
        timestamp: Timestamp for when script was generated
        pipeline_name: Name/path of the pipeline
        
    Returns:
        Rendered script content
    """
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    return EVALUATION_SCRIPT_TEMPLATE.format(
        timestamp=timestamp,
        pipeline_name=pipeline_name,
    )

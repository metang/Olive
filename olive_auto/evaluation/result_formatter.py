"""
Result formatting for evaluation output.

Formats evaluation results into CSV, JSON, and HTML dashboards with
smart recommendations based on metric performance.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import EvaluationSummary, ModelEvaluationResult, RecommendationSet

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Format evaluation results for multiple output formats."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize result formatter.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, summary: EvaluationSummary) -> Dict[str, Path]:
        """
        Save evaluation results in all formats (JSON, CSV, HTML).
        
        Args:
            summary: Evaluation summary with results
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        results_paths = {}
        
        # Save JSON
        json_path = self.output_dir / "results_summary.json"
        self.to_json(summary, json_path)
        results_paths["json"] = json_path
        
        # Save CSV
        csv_path = self.output_dir / "results_summary.csv"
        self.to_csv(summary, csv_path)
        results_paths["csv"] = csv_path
        
        # Save HTML
        html_path = self.output_dir / "results_dashboard.html"
        self.to_html(summary, html_path)
        results_paths["html"] = html_path
        
        logger.info(
            f"Results saved: JSON, CSV, HTML in {self.output_dir}"
        )
        
        return results_paths
    
    def to_json(self, summary: EvaluationSummary, output_path: Path) -> None:
        """
        Save results as JSON.
        
        Args:
            summary: Evaluation summary
            output_path: Path to save JSON file
        """
        data = {
            "metadata": {
                "timestamp": summary.timestamp,
                "pipeline_dir": str(summary.pipeline_dir) if summary.pipeline_dir else None,
                "total_models": summary.total_models,
                "evaluated_models": summary.evaluated_models,
                "skipped_models": summary.skipped_models,
                "failed_models": summary.failed_models,
                "total_time_sec": summary.total_time_sec,
                "sample_fraction": summary.sample_fraction_used,
            },
            "models": [],
            "best_by_metric": {},
            "recommendations": {
                "best_overall": summary.recommendations.best_overall,
                "best_accuracy": summary.recommendations.best_accuracy,
                "best_latency": summary.recommendations.best_latency,
                "best_efficiency": summary.recommendations.best_efficiency,
            },
        }
        
        # Add model results
        for model in summary.models:
            model_data = {
                "name": model.model_name,
                "target": model.target,
                "precision": model.precision,
                "status": model.status.value,
                "elapsed_time_sec": model.elapsed_time_sec,
                "num_samples": model.num_samples_evaluated,
            }
            
            if model.is_success():
                model_data["metrics"] = model.metrics
                model_data["accuracy_method"] = model.accuracy_method.value
                model_data["accuracy_confidence"] = model.accuracy_confidence
            else:
                model_data["error_message"] = model.error_message
            
            data["models"].append(model_data)
        
        # Save JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved JSON results to {output_path}")
    
    def to_csv(self, summary: EvaluationSummary, output_path: Path) -> None:
        """
        Save results as CSV for Excel import.
        
        Args:
            summary: Evaluation summary
            output_path: Path to save CSV file
        """
        successful_models = summary.get_successful_models()
        
        if not successful_models:
            logger.warning("No successful models to export to CSV")
            return
        
        # Collect all metric names
        metric_names = set()
        for model in successful_models:
            if model.metrics:
                metric_names.update(model.metrics.keys())
        metric_names = sorted(list(metric_names))
        
        # Prepare CSV headers
        headers = [
            "model_name",
            "target",
            "precision",
            "status",
            "elapsed_time_sec",
            "num_samples",
            "accuracy_method",
            "accuracy_confidence",
        ]
        headers.extend(metric_names)
        
        # Write CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for model in successful_models:
                row = {
                    "model_name": model.model_name,
                    "target": model.target,
                    "precision": model.precision,
                    "status": model.status.value,
                    "elapsed_time_sec": f"{model.elapsed_time_sec:.2f}",
                    "num_samples": model.num_samples_evaluated,
                    "accuracy_method": model.accuracy_method.value if model.accuracy_method else "",
                    "accuracy_confidence": model.accuracy_confidence or "",
                }
                
                # Add metrics
                if model.metrics:
                    for metric_name in metric_names:
                        value = model.metrics.get(metric_name)
                        if value is not None:
                            row[metric_name] = f"{value:.4f}"
                
                writer.writerow(row)
        
        logger.info(f"Saved CSV results to {output_path}")
    
    def to_html(self, summary: EvaluationSummary, output_path: Path) -> None:
        """
        Save results as interactive HTML dashboard with charts.
        
        Args:
            summary: Evaluation summary
            output_path: Path to save HTML file
        """
        successful_models = summary.get_successful_models()
        
        # Generate recommendation boxes HTML
        recommendations_html = self._generate_recommendations_html(
            summary.recommendations,
            successful_models
        )
        
        # Generate table HTML
        table_html = self._generate_table_html(successful_models)
        
        # Generate charts HTML
        charts_html = self._generate_charts_html(successful_models)
        
        # Combine into final HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Olive-Auto Evaluation Results</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
            border-left: 4px solid #4caf50;
        }}
        
        .recommendations h2 {{
            color: #2e7d32;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .recommendation-item {{
            margin-bottom: 12px;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #4caf50;
        }}
        
        .recommendation-item strong {{
            color: #1b5e20;
        }}
        
        .recommendation-item .metric {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .chart-container {{
            height: 500px;
            margin-bottom: 30px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}
        
        th {{
            background: #f5f5f5;
            color: #333;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f9f9f9;
        }}
        
        .status-success {{
            color: #4caf50;
            font-weight: 600;
        }}
        
        .status-skipped {{
            color: #ff9800;
        }}
        
        .status-failed {{
            color: #f44336;
        }}
        
        .metric-value {{
            font-family: "Courier New", monospace;
            font-size: 0.9em;
            color: #555;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .summary-card .label {{
            color: #666;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        
        .summary-card .value {{
            color: #333;
            font-size: 2em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✓ Model Evaluation Results</h1>
        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            | Total Time: {summary.total_time_sec:.1f}s
        </div>
        
        {recommendations_html}
        
        <div class="section">
            <h2>Evaluation Summary</h2>
            <div class="summary">
                <div class="summary-card">
                    <div class="label">Total Models</div>
                    <div class="value">{summary.total_models}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Evaluated</div>
                    <div class="value">{summary.evaluated_models}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Skipped</div>
                    <div class="value">{summary.skipped_models}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Failed</div>
                    <div class="value">{summary.failed_models}</div>
                </div>
            </div>
        </div>
        
        {charts_html}
        
        <div class="section">
            <h2>Detailed Results</h2>
            {table_html}
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML dashboard to {output_path}")
    
    @staticmethod
    def _generate_recommendations_html(
        recommendations: RecommendationSet,
        models: List[ModelEvaluationResult]
    ) -> str:
        """Generate HTML for recommendations section."""
        html = '<div class="section recommendations">\n'
        html += '<h2>✓ Smart Recommendations</h2>\n'
        html += '<div>\n'
        
        if recommendations.best_overall:
            best_model = next((m for m in models if m.model_name == recommendations.best_overall), None)
            if best_model and best_model.metrics:
                html += '<div class="recommendation-item">\n'
                html += f'<strong>Best Overall:</strong> {recommendations.best_overall}\n'
                acc = best_model.metrics.get("accuracy", 0)
                lat = best_model.metrics.get("latency_p50_ms", 0)
                html += f'<div class="metric">Accuracy: {acc:.4f} | Latency: {lat:.2f}ms</div>\n'
                html += '</div>\n'
        
        if recommendations.best_accuracy:
            best_model = next((m for m in models if m.model_name == recommendations.best_accuracy), None)
            if best_model and best_model.metrics:
                html += '<div class="recommendation-item">\n'
                html += f'<strong>Best Accuracy:</strong> {recommendations.best_accuracy}\n'
                acc = best_model.metrics.get("accuracy", 0)
                html += f'<div class="metric">Accuracy: {acc:.4f}</div>\n'
                html += '</div>\n'
        
        if recommendations.best_latency:
            best_model = next((m for m in models if m.model_name == recommendations.best_latency), None)
            if best_model and best_model.metrics:
                html += '<div class="recommendation-item">\n'
                html += f'<strong>Best Latency:</strong> {recommendations.best_latency}\n'
                lat = best_model.metrics.get("latency_p50_ms", 0)
                html += f'<div class="metric">Latency (p50): {lat:.2f}ms</div>\n'
                html += '</div>\n'
        
        if recommendations.best_efficiency:
            best_model = next((m for m in models if m.model_name == recommendations.best_efficiency), None)
            if best_model and best_model.metrics:
                html += '<div class="recommendation-item">\n'
                html += f'<strong>Most Efficient:</strong> {recommendations.best_efficiency}\n'
                acc = best_model.metrics.get("accuracy", 0)
                size = best_model.metrics.get("model_size_mb", 1)
                ratio = acc / (size / 1024) if size > 0 else 0
                html += f'<div class="metric">Accuracy: {acc:.4f} | Size: {size:.0f}MB | Ratio: {ratio:.4f}</div>\n'
                html += '</div>\n'
        
        html += '</div>\n'
        html += '</div>\n'
        
        return html
    
    @staticmethod
    def _generate_table_html(models: List[ModelEvaluationResult]) -> str:
        """Generate HTML table of results."""
        if not models:
            return '<p>No successful evaluations to display.</p>'
        
        html = '<table>\n'
        html += '<thead><tr>\n'
        html += '<th>Model</th>\n'
        html += '<th>Target</th>\n'
        html += '<th>Precision</th>\n'
        html += '<th>Accuracy</th>\n'
        html += '<th>Latency (p50)</th>\n'
        html += '<th>Latency (p99)</th>\n'
        html += '<th>Memory (MB)</th>\n'
        html += '<th>Size (MB)</th>\n'
        html += '<th>Status</th>\n'
        html += '</tr></thead>\n'
        html += '<tbody>\n'
        
        for model in models:
            if not model.metrics:
                continue
            
            acc = model.metrics.get("accuracy", 0)
            lat_p50 = model.metrics.get("latency_p50_ms", 0)
            lat_p99 = model.metrics.get("latency_p99_ms", 0)
            mem = model.metrics.get("memory_system_peak_mb", 0)
            size = model.metrics.get("model_size_mb", 0)
            
            html += '<tr>\n'
            html += f'<td><strong>{model.model_name}</strong></td>\n'
            html += f'<td>{model.target}</td>\n'
            html += f'<td>{model.precision}</td>\n'
            html += f'<td class="metric-value">{acc:.4f}</td>\n'
            html += f'<td class="metric-value">{lat_p50:.2f}ms</td>\n'
            html += f'<td class="metric-value">{lat_p99:.2f}ms</td>\n'
            html += f'<td class="metric-value">{mem:.1f}</td>\n'
            html += f'<td class="metric-value">{size:.0f}</td>\n'
            html += f'<td><span class="status-success">✓ Success</span></td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n'
        html += '</table>\n'
        
        return html
    
    @staticmethod
    def _generate_charts_html(models: List[ModelEvaluationResult]) -> str:
        """Generate HTML with Plotly charts."""
        if not models:
            return ''
        
        # Prepare data for charts
        model_names = [m.model_name for m in models]
        accuracies = [m.metrics.get("accuracy", 0) if m.metrics else 0 for m in models]
        latencies = [m.metrics.get("latency_p50_ms", 0) if m.metrics else 0 for m in models]
        
        html = '<div class="section">\n'
        html += '<h2>Performance Analysis</h2>\n'
        html += '<div class="chart-container" id="accuracy-chart"></div>\n'
        html += '<div class="chart-container" id="latency-chart"></div>\n'
        html += '<script>\n'
        
        # Accuracy chart
        html += 'Plotly.newPlot("accuracy-chart", [{\n'
        html += f'x: {model_names},\n'
        html += f'y: {accuracies},\n'
        html += 'type: "bar",\n'
        html += 'marker: {{color: "#4caf50"}}\n'
        html += '}], {title: "Accuracy by Model", xaxis: {title: "Model"}, yaxis: {title: "Accuracy"}});\n'
        
        # Latency chart
        html += 'Plotly.newPlot("latency-chart", [{\n'
        html += f'x: {model_names},\n'
        html += f'y: {latencies},\n'
        html += 'type: "bar",\n'
        html += 'marker: {{color: "#2196f3"}}\n'
        html += '}], {title: "Latency (p50) by Model", xaxis: {title: "Model"}, yaxis: {title: "Latency (ms)"}});\n'
        
        html += '</script>\n'
        html += '</div>\n'
        
        return html

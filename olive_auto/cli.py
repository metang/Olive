"""
olive-auto: Standalone CLI for batch model optimization pipeline generation.

Usage:
    olive-auto -m microsoft/phi-2 -o ./pipeline
    olive-auto -m openai/whisper-tiny --targets cpu cuda --platform both
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from olive_auto import __version__
from olive_auto.task_detection import detect_model_info
from olive_auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY
from olive_auto.environment import EnvironmentScriptGenerator
from olive_auto.command_generator import CommandGenerator
from olive_auto.executor import PipelineExecutor

# Initialize CLI app
app = typer.Typer(
    name="olive-auto",
    help="Generate optimization pipelines for HuggingFace models across all targets and precisions.",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"olive-auto version {__version__}")
        raise typer.Exit()


@app.command()
def main(
    # Model specification
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        help="HuggingFace model ID or local path",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Model task (auto-detected if not specified)",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Trust remote code from HuggingFace",
    ),
    # Target/precision selection
    targets: Optional[List[str]] = typer.Option(
        None,
        "--targets",
        "-t",
        help="Specific targets (default: all compatible). Options: cpu, cuda, qnn, openvino, tensorrt, directml, webgpu, vitisai, rocm",
    ),
    precisions: Optional[List[str]] = typer.Option(
        None,
        "--precisions",
        "-p",
        help="Specific precisions (default: all per target). Options: fp32, fp16, bf16, int8, int4, bnb4",
    ),
    exclude_targets: Optional[List[str]] = typer.Option(
        None,
        "--exclude-targets",
        help="Targets to exclude from generation",
    ),
    # Output configuration
    output_dir: Path = typer.Option(
        Path("./olive_auto_pipeline"),
        "-o",
        "--output-dir",
        help="Output directory for generated scripts",
    ),
    platform: str = typer.Option(
        "auto",
        "--platform",
        help="Target platform: linux, windows, or auto (detect current)",
    ),
    # Execution options
    parallel: int = typer.Option(
        1,
        "--parallel",
        help="Number of parallel optimizations (for --run)",
    ),
    run: bool = typer.Option(
        False,
        "--run",
        help="Execute pipeline after generation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without creating files",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Test mode: generate only CPU target with fp32/fp16 precisions",
    ),
    # Misc options
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    Generate optimization pipeline for a HuggingFace model.

    This command analyzes a model, determines compatible targets and precisions,
    and generates a complete batch pipeline with environment setup and cleanup scripts.

    Examples:

        # Generate pipeline for all targets
        olive-auto -m microsoft/phi-2 -o ./phi2_pipeline

        # Generate for specific targets only
        olive-auto -m openai/whisper-tiny --targets cpu cuda

        # Generate and run immediately
        olive-auto -m bert-base-uncased --targets cpu --run

        # Generate for Windows
        olive-auto -m microsoft/phi-2 --platform windows
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Detect platform if auto
    if platform == "auto":
        platform = "windows" if os.name == "nt" else "linux"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Analyze model
        task_id = progress.add_task("Analyzing model...", total=None)
        detected_task, category = detect_model_info(
            model,
            user_task=task,
            trust_remote_code=trust_remote_code,
        )
        progress.update(task_id, completed=True)

        # Step 2: Determine targets
        progress.add_task("Selecting targets...", total=None)

        # Test mode: override to CPU only with fp32/fp16
        if test:
            selected_targets = ["cpu"]
            target_precisions = {"cpu": ["fp32", "fp16"]}
            console.print("[yellow]Test mode: CPU target with fp32/fp16 only[/yellow]")
        else:
            selected_targets = _get_targets(category, targets, exclude_targets)

            if not selected_targets:
                console.print(
                    "[red]Error: No compatible targets found for this model category.[/red]"
                )
                raise typer.Exit(1)

            # Step 3: Determine precisions
            target_precisions = _get_precisions(selected_targets, precisions)

        # Count total commands
        total_commands = sum(len(p) for p in target_precisions.values())

    # Display analysis results
    _display_analysis(model, detected_task, category, selected_targets, target_precisions)

    if dry_run:
        console.print("\n[yellow]Dry run mode - no files created.[/yellow]")
        raise typer.Exit(0)

    # Step 4: Generate pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create output directory
        task_id = progress.add_task("Creating output directory...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "outputs").mkdir(exist_ok=True)
        (output_dir / "environments").mkdir(exist_ok=True)
        (output_dir / "commands").mkdir(exist_ok=True)
        progress.update(task_id, completed=True)

        # Generate environment scripts
        task_id = progress.add_task("Generating environment scripts...", total=None)
        env_generator = EnvironmentScriptGenerator(output_dir, platform)
        for target in selected_targets:
            env_generator.generate_setup_script(target)
        env_generator.generate_master_setup_script(selected_targets)
        progress.update(task_id, completed=True)

        # Generate command scripts
        task_id = progress.add_task("Generating optimization commands...", total=None)
        cmd_generator = CommandGenerator(
            model_id=model,
            task=detected_task,
            category=category,
            output_dir=output_dir,
            targets=selected_targets,
            precisions=target_precisions,
            platform=platform,
        )
        cmd_generator.generate_command_scripts()
        cmd_generator.generate_master_script()
        progress.update(task_id, completed=True)

        # Generate cleanup script
        task_id = progress.add_task("Generating cleanup script...", total=None)
        env_generator.generate_cleanup_script(selected_targets)
        progress.update(task_id, completed=True)

        # Generate manifest
        task_id = progress.add_task("Generating manifest...", total=None)
        _generate_manifest(
            output_dir,
            model,
            detected_task,
            category,
            selected_targets,
            target_precisions,
        )
        progress.update(task_id, completed=True)

    # Display summary
    _display_summary(output_dir, selected_targets, total_commands, platform)

    # Execute if requested
    if run:
        console.print("\n[bold]Executing pipeline...[/bold]")
        executor = PipelineExecutor(output_dir, parallel=parallel)
        success = executor.run()
        if not success:
            raise typer.Exit(1)


def _get_targets(
    category: str,
    user_targets: Optional[List[str]],
    exclude_targets: Optional[List[str]],
) -> List[str]:
    """Determine which targets to use based on category and user preferences."""
    compatible = CATEGORY_TARGET_COMPATIBILITY.get(category, ["cpu"])

    if user_targets:
        # Validate user targets
        invalid = [t for t in user_targets if t not in TARGET_PRECISION_MATRIX]
        if invalid:
            console.print(f"[yellow]Warning: Unknown targets ignored: {invalid}[/yellow]")
        targets = [t for t in user_targets if t in compatible]
    else:
        targets = list(compatible)

    if exclude_targets:
        targets = [t for t in targets if t not in exclude_targets]

    return targets


def _get_precisions(
    targets: List[str],
    user_precisions: Optional[List[str]],
) -> dict:
    """Determine precisions for each target."""
    result = {}
    for target in targets:
        available = TARGET_PRECISION_MATRIX[target]["precisions"]
        if user_precisions:
            result[target] = [p for p in user_precisions if p in available]
            if not result[target]:
                result[target] = available  # Fallback to all if none match
        else:
            result[target] = available
    return result


def _display_analysis(
    model: str,
    task: str,
    category: str,
    targets: List[str],
    precisions: dict,
):
    """Display model analysis results."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Task:[/bold] {task}\n"
            f"[bold]Category:[/bold] {category}",
            title="Model Analysis",
            border_style="blue",
        )
    )

    # Create targets table
    table = Table(title="Target/Precision Matrix", show_header=True, header_style="bold")
    table.add_column("Target", style="cyan")
    table.add_column("Precisions", style="green")
    table.add_column("Commands", justify="right")

    for target in targets:
        precs = precisions[target]
        table.add_row(target, ", ".join(precs), str(len(precs)))

    console.print(table)


def _display_summary(
    output_dir: Path,
    targets: List[str],
    total_commands: int,
    platform: str,
):
    """Display generation summary."""
    ext = ".ps1" if platform == "windows" else ".sh"

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Pipeline generated successfully![/bold green]\n\n"
            f"[bold]Output:[/bold] {output_dir.absolute()}\n"
            f"[bold]Targets:[/bold] {len(targets)}\n"
            f"[bold]Total commands:[/bold] {total_commands}\n\n"
            f"[bold]Generated files:[/bold]\n"
            f"  - setup_environments{ext}\n"
            f"  - run_all_optimizations{ext}\n"
            f"  - cleanup{ext}\n"
            f"  - environments/ ({len(targets)} scripts)\n"
            f"  - commands/ ({total_commands} scripts)\n"
            f"  - manifest.json",
            title="Generation Complete",
            border_style="green",
        )
    )

    # Usage instructions
    if platform == "windows":
        console.print("\n[bold]To run the pipeline:[/bold]")
        console.print(f"  cd {output_dir}")
        console.print("  .\\setup_environments.ps1   # First time only")
        console.print("  .\\run_all_optimizations.ps1")
    else:
        console.print("\n[bold]To run the pipeline:[/bold]")
        console.print(f"  cd {output_dir}")
        console.print("  ./setup_environments.sh   # First time only")
        console.print("  ./run_all_optimizations.sh")


def _generate_manifest(
    output_dir: Path,
    model: str,
    task: str,
    category: str,
    targets: List[str],
    precisions: dict,
):
    """Generate pipeline manifest JSON."""
    manifest = {
        "model_id": model,
        "task": task,
        "category": category,
        "targets": targets,
        "precisions": precisions,
        "generated_by": "olive-auto",
        "version": __version__,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


# Entry point
if __name__ == "__main__":
    app()

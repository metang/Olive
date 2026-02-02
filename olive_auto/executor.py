"""Pipeline execution engine for olive-auto."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

console = Console()


@dataclass
class ExecutionResult:
    """Result of a single optimization command."""

    target: str
    precision: str
    success: bool
    return_code: int
    log_file: Path
    error_message: Optional[str] = None


class PipelineExecutor:
    """Execute generated optimization pipeline."""

    def __init__(self, pipeline_dir: Path, parallel: int = 1):
        self.pipeline_dir = Path(pipeline_dir)
        self.parallel = parallel
        self.manifest = self._load_manifest()
        self.results: List[ExecutionResult] = []

    def _load_manifest(self) -> dict:
        """Load pipeline manifest."""
        manifest_path = self.pipeline_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text())

    def run(self) -> bool:
        """Execute the full pipeline."""
        console.print("\n[bold]Starting pipeline execution...[/bold]")

        # Get all commands to run
        commands = self._get_commands()
        total = len(commands)

        console.print(f"Total commands: {total}")
        console.print(f"Parallel workers: {self.parallel}\n")

        if self.parallel == 1:
            # Sequential execution
            self._run_sequential(commands)
        else:
            # Parallel execution
            self._run_parallel(commands)

        # Print summary
        return self._print_summary()

    def _get_commands(self) -> List[tuple]:
        """Get list of (target, precision, script_path) tuples."""
        commands = []
        commands_dir = self.pipeline_dir / "commands"

        for target, precisions in self.manifest["precisions"].items():
            for precision in precisions:
                # Determine script extension
                ext = ".ps1" if sys.platform == "win32" else ".sh"
                script_path = commands_dir / f"{target}_{precision}{ext}"
                if script_path.exists():
                    commands.append((target, precision, script_path))

        return commands

    def _run_sequential(self, commands: List[tuple]):
        """Run commands sequentially."""
        with Progress(console=console) as progress:
            task = progress.add_task("Running optimizations...", total=len(commands))

            for target, precision, script_path in commands:
                progress.update(task, description=f"Running {target}/{precision}...")
                result = self._execute_command(target, precision, script_path)
                self.results.append(result)
                progress.advance(task)

    def _run_parallel(self, commands: List[tuple]):
        """Run commands in parallel."""
        with Progress(console=console) as progress:
            task = progress.add_task("Running optimizations...", total=len(commands))

            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(
                        self._execute_command, target, precision, script_path
                    ): (target, precision)
                    for target, precision, script_path in commands
                }

                for future in as_completed(futures):
                    target, precision = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.results.append(
                            ExecutionResult(
                                target=target,
                                precision=precision,
                                success=False,
                                return_code=-1,
                                log_file=Path(),
                                error_message=str(e),
                            )
                        )
                    progress.advance(task)

    def _execute_command(
        self,
        target: str,
        precision: str,
        script_path: Path,
    ) -> ExecutionResult:
        """Execute a single optimization command."""
        log_file = self.pipeline_dir / "logs" / f"{target}_{precision}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(log_file, "w") as log:
                if sys.platform == "win32":
                    result = subprocess.run(
                        [
                            "powershell",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(script_path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=3600,  # 1 hour timeout
                    )
                else:
                    result = subprocess.run(
                        ["bash", str(script_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                    )

            return ExecutionResult(
                target=target,
                precision=precision,
                success=result.returncode == 0,
                return_code=result.returncode,
                log_file=log_file,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                target=target,
                precision=precision,
                success=False,
                return_code=-1,
                log_file=log_file,
                error_message="Timeout after 1 hour",
            )
        except Exception as e:
            return ExecutionResult(
                target=target,
                precision=precision,
                success=False,
                return_code=-1,
                log_file=log_file,
                error_message=str(e),
            )

    def _print_summary(self) -> bool:
        """Print execution summary and return success status."""
        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)

        console.print("\n")

        # Create results table
        table = Table(title="Execution Results", show_header=True)
        table.add_column("Target", style="cyan")
        table.add_column("Precision", style="magenta")
        table.add_column("Status")
        table.add_column("Log File")

        for result in self.results:
            status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
            if result.error_message:
                status += f" ({result.error_message})"
            table.add_row(
                result.target,
                result.precision,
                status,
                str(result.log_file.name) if result.log_file.exists() else "-",
            )

        console.print(table)

        # Summary
        console.print(f"\n[bold]Summary:[/bold] {passed} passed, {failed} failed")

        if failed > 0:
            console.print("\n[yellow]Check log files for failure details:[/yellow]")
            for result in self.results:
                if not result.success:
                    console.print(f"  - {result.log_file}")

        return failed == 0

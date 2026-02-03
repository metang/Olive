"""Generate olive optimize commands for all target/precision combinations."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from olive_auto.targets import (
    TARGET_PRECISION_MATRIX,
    CATEGORY_TARGET_COMPATIBILITY,
    get_provider_for_target,
    get_device_for_target,
)


@dataclass
class OptimizationCommand:
    """Represents a single optimization command."""

    target: str
    precision: str
    model_id: str
    task: str
    output_dir: Path
    extra_args: Dict[str, str] = field(default_factory=dict)

    def to_olive_command(self) -> str:
        """Generate the olive optimize command string."""
        cmd_parts = [
            "olive optimize",
            f'-m "{self.model_id}"',
            f"--task {self.task}",
            f"--provider {self._get_provider()}",
            f"--precision {self.precision}",
            f'-o "{self.output_dir}"',
        ]

        # Add device flag based on target
        device = get_device_for_target(self.target)
        if device != "cpu":
            cmd_parts.append(f"--device {device}")

        # Add extra arguments
        if self.extra_args:
            for key, value in self.extra_args.items():
                cmd_parts.append(f"--{key} {value}")

        return " \\\n    ".join(cmd_parts)

    def to_olive_command_windows(self) -> str:
        """Generate the olive optimize command string for Windows (PowerShell)."""
        cmd_parts = [
            "olive optimize",
            f'-m "{self.model_id}"',
            f"--task {self.task}",
            f"--provider {self._get_provider()}",
            f"--precision {self.precision}",
            f'-o "{self.output_dir}"',
        ]

        # Add device flag based on target
        device = get_device_for_target(self.target)
        if device != "cpu":
            cmd_parts.append(f"--device {device}")

        # Add extra arguments
        if self.extra_args:
            for key, value in self.extra_args.items():
                cmd_parts.append(f"--{key} {value}")

        return " `\n    ".join(cmd_parts)

    def _get_provider(self) -> str:
        """Map target to execution provider name."""
        return get_provider_for_target(self.target)


class CommandGenerator:
    """Generate optimization commands and scripts."""

    def __init__(
        self,
        model_id: str,
        task: str,
        category: str,
        output_dir: Path,
        targets: Optional[List[str]] = None,
        precisions: Optional[Dict[str, List[str]]] = None,
        platform: str = "linux",
    ):
        self.model_id = model_id
        self.task = task
        self.category = category
        self.output_dir = Path(output_dir)
        self.platform = platform
        self.commands_dir = self.output_dir / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=True)

        # Determine compatible targets
        self.targets = targets or self._get_compatible_targets()

        # Determine precisions per target
        self.precisions = precisions or self._get_default_precisions()

    def _get_compatible_targets(self) -> List[str]:
        """Get targets compatible with the model category."""
        return CATEGORY_TARGET_COMPATIBILITY.get(self.category, ["cpu"])

    def _get_default_precisions(self) -> Dict[str, List[str]]:
        """Get default precisions for each target."""
        precisions = {}
        for target in self.targets:
            if target in TARGET_PRECISION_MATRIX:
                precisions[target] = TARGET_PRECISION_MATRIX[target]["precisions"]
        return precisions

    def generate_all_commands(self) -> List[OptimizationCommand]:
        """Generate all optimization commands."""
        commands = []

        for target in self.targets:
            for precision in self.precisions.get(target, ["fp32"]):
                output_path = self.output_dir / "outputs" / f"{target}_{precision}"

                cmd = OptimizationCommand(
                    target=target,
                    precision=precision,
                    model_id=self.model_id,
                    task=self.task,
                    output_dir=output_path,
                )
                commands.append(cmd)

        return commands

    def generate_command_scripts(self) -> Dict[str, Path]:
        """Generate individual command scripts."""
        commands = self.generate_all_commands()
        scripts = {}

        for cmd in commands:
            if self.platform == "windows":
                script_path = self._generate_powershell_script(cmd)
            else:
                script_path = self._generate_bash_script(cmd)

            scripts[f"{cmd.target}_{cmd.precision}"] = script_path

        return scripts

    def _generate_bash_script(self, cmd: OptimizationCommand) -> Path:
        """Generate bash script for a single command."""
        script_name = f"{cmd.target}_{cmd.precision}.sh"
        script_path = self.commands_dir / script_name

        # Convert to absolute paths
        output_dir_abs = str(cmd.output_dir.resolve())
        log_dir_abs = str((self.output_dir / "logs").resolve())
        env_dir_abs = str((self.output_dir / "environments" / f"olive_{cmd.target}").resolve())

        lines = [
            "#!/bin/bash",
            f"# Optimization: {cmd.target} with {cmd.precision} precision",
            f"# Model: {cmd.model_id}",
            "set -e",
            "",
            f'LOG_FILE="{log_dir_abs}/{cmd.target}_{cmd.precision}.log"',
            f'OUTPUT_DIR="{output_dir_abs}"',
            f'ENV_DIR="{env_dir_abs}"',
            "",
            'echo "========================================"',
            f'echo "Target: {cmd.target}"',
            f'echo "Precision: {cmd.precision}"',
            f'echo "Model: {cmd.model_id}"',
            'echo "========================================"',
            "",
            "# Ensure log directory exists",
            'mkdir -p "$(dirname "$LOG_FILE")"',
            "",
            "# Activate environment",
            'source "$ENV_DIR/bin/activate"',
            "",
            "# Create output directory",
            'mkdir -p "$OUTPUT_DIR"',
            "",
            'echo "Starting optimization..."',
            'echo "Log file: $LOG_FILE"',
            'echo "Output dir: $OUTPUT_DIR"',
            "",
            "# Run optimization with better error capture",
            "{",
            f'{cmd.to_olive_command()} 2>&1 | tee "$LOG_FILE"',
            'OUTPUT_STATUS="${PIPESTATUS[0]}"',
            "",
            "    # Check if output was produced",
            '    if [ "$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)" -eq 0 ]; then',
            '        echo "WARNING: No output files produced by optimization. Check log for details."',
            "    else",
            '        echo "Successfully produced $(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l) output file(s)"',
            "    fi",
            "",
            "    if [ $OUTPUT_STATUS -ne 0 ]; then",
            '        echo "ERROR: Optimization failed with status code $OUTPUT_STATUS"',
            "        exit $OUTPUT_STATUS",
            "    fi",
            "}",
            "",
            'echo ""',
            f'echo "Optimization complete: {cmd.target} {cmd.precision}"',
        ]

        script_path.write_text("\n".join(lines))
        if os.name != "nt":
            script_path.chmod(0o755)

        return script_path


    def _generate_powershell_script(self, cmd: OptimizationCommand) -> Path:
        """Generate PowerShell script for a single command."""
        script_name = f"{cmd.target}_{cmd.precision}.ps1"
        script_path = self.commands_dir / script_name

        # Convert to absolute paths
        output_dir_abs = str(cmd.output_dir.resolve())
        log_dir_abs = str((self.output_dir / "logs").resolve())
        env_dir_abs = str((self.output_dir / "environments" / f"olive_{cmd.target}").resolve())

        lines = [
            f"# Optimization: {cmd.target} with {cmd.precision} precision",
            f"# Model: {cmd.model_id}",
            "$ErrorActionPreference = 'Stop'",
            "",
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            f'$LogFile = "{log_dir_abs}\\{cmd.target}_{cmd.precision}.log"',
            f'$OutputDir = "{output_dir_abs}"',
            f'$EnvDir = "{env_dir_abs}"',
            "",
            "Write-Host '========================================'",
            f"Write-Host 'Target: {cmd.target}'",
            f"Write-Host 'Precision: {cmd.precision}'",
            f"Write-Host 'Model: {cmd.model_id}'",
            "Write-Host '========================================'",
            "",
            "# Ensure log directory exists",
            'New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null',
            "",
            "# Activate environment",
            f'& "$EnvDir\\Scripts\\Activate.ps1"',
            "",
            "# Create output directory",
            'New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null',
            "",
            "Write-Host 'Starting optimization...'",
            "Write-Host \"Log file: $LogFile\"",
            "Write-Host \"Output dir: $OutputDir\"",
            "",
            "# Run optimization with better error capture",
            "try {",
            f'    olive optimize `',
            f'        -m "{cmd.model_id}" `',
            f'        --task {cmd.task} `',
            f'        --provider {cmd._get_provider()} `',
            f'        --precision {cmd.precision} `',
            f'        -o $OutputDir 2>&1 | Tee-Object -FilePath $LogFile',
            "    ",
            "    # Check if output was produced",
            "    $outputFiles = @(Get-ChildItem -Path $OutputDir -File -Recurse -ErrorAction SilentlyContinue)",
            "    if ($outputFiles.Count -eq 0) {",
            "        Write-Host 'WARNING: No output files produced by optimization. Check log for details.'",
            "    } else {",
            "        Write-Host \"Successfully produced $($outputFiles.Count) output file(s)\"",
            "    }",
            "} catch {",
            "    Write-Host \"ERROR: Optimization failed with exception: $_\"",
            "    throw",
            "}",
            "",
            "Write-Host ''",
            f"Write-Host 'Optimization complete: {cmd.target} {cmd.precision}'",
        ]

        script_path.write_text("\n".join(lines))
        return script_path

    def generate_master_script(self) -> Path:
        """Generate master orchestration script."""
        if self.platform == "windows":
            return self._generate_master_powershell()
        else:
            return self._generate_master_bash()

    def _generate_master_bash(self) -> Path:
        """Generate master bash orchestration script."""
        script_path = self.output_dir / "run_all_optimizations.sh"
        commands = self.generate_all_commands()

        lines = [
            "#!/bin/bash",
            "# Master orchestration script for olive-auto",
            f"# Model: {self.model_id}",
            "set -e",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            "",
            "# Create logs directory",
            'mkdir -p "$SCRIPT_DIR/logs"',
            "",
            'echo "========================================"',
            'echo "olive-auto Pipeline"',
            'echo "========================================"',
            f'echo "Model: {self.model_id}"',
            f'echo "Task: {self.task}"',
            f'echo "Category: {self.category}"',
            f'echo "Total combinations: {len(commands)}"',
            'echo "========================================"',
            "",
            "# Track results",
            "PASSED=0",
            "FAILED=0",
            "",
        ]

        for cmd in commands:
            lines.extend([
                f"# {cmd.target} with {cmd.precision}",
                'echo ""',
                f'echo "[$(date)] Running: {cmd.target} {cmd.precision}"',
                f'if "$SCRIPT_DIR/commands/{cmd.target}_{cmd.precision}.sh"; then',
                f'    echo "[PASS] {cmd.target} {cmd.precision}"',
                "    ((PASSED++))",
                "else",
                f'    echo "[FAIL] {cmd.target} {cmd.precision}"',
                "    ((FAILED++))",
                "fi",
                "",
            ])

        lines.extend([
            "# Print summary",
            'echo ""',
            'echo "========================================"',
            'echo "Optimization Summary"',
            'echo "========================================"',
            'echo "Passed: $PASSED"',
            'echo "Failed: $FAILED"',
            'echo ""',
            "",
            "if [ $FAILED -gt 0 ]; then",
            "    exit 1",
            "fi",
        ])

        script_path.write_text("\n".join(lines))
        if os.name != "nt":
            script_path.chmod(0o755)

        return script_path

    def _generate_master_powershell(self) -> Path:
        """Generate master PowerShell orchestration script."""
        script_path = self.output_dir / "run_all_optimizations.ps1"
        commands = self.generate_all_commands()

        lines = [
            "# Master orchestration script for olive-auto",
            f"# Model: {self.model_id}",
            "$ErrorActionPreference = 'Stop'",
            "",
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            "",
            "# Create logs directory",
            'New-Item -ItemType Directory -Force -Path "$ScriptDir\\logs" | Out-Null',
            "",
            "Write-Host '========================================'",
            "Write-Host 'olive-auto Pipeline'",
            "Write-Host '========================================'",
            f"Write-Host 'Model: {self.model_id}'",
            f"Write-Host 'Task: {self.task}'",
            f"Write-Host 'Category: {self.category}'",
            f"Write-Host 'Total combinations: {len(commands)}'",
            "Write-Host '========================================'",
            "",
            "# Track results",
            "$Passed = 0",
            "$Failed = 0",
            "",
        ]

        for cmd in commands:
            lines.extend([
                f"# {cmd.target} with {cmd.precision}",
                "Write-Host ''",
                f"Write-Host \"[$(Get-Date)] Running: {cmd.target} {cmd.precision}\"",
                "try {",
                f'    & "$ScriptDir\\commands\\{cmd.target}_{cmd.precision}.ps1"',
                f"    Write-Host '[PASS] {cmd.target} {cmd.precision}'",
                "    $Passed++",
                "} catch {",
                f"    Write-Host '[FAIL] {cmd.target} {cmd.precision}'",
                "    $Failed++",
                "}",
                "",
            ])

        lines.extend([
            "# Print summary",
            "Write-Host ''",
            "Write-Host '========================================'",
            "Write-Host 'Optimization Summary'",
            "Write-Host '========================================'",
            "Write-Host \"Passed: $Passed\"",
            "Write-Host \"Failed: $Failed\"",
            "Write-Host ''",
            "",
            "if ($Failed -gt 0) {",
            "    exit 1",
            "}",
        ])

        script_path.write_text("\n".join(lines))
        return script_path

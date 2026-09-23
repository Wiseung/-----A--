from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvaluationError(RuntimeError):
    def __init__(self, message: str, elapsed_sec: float = 0.0) -> None:
        super().__init__(message)
        self.elapsed_sec = elapsed_sec


@dataclass(slots=True)
class EvaluationResult:
    result: dict[str, Any]
    elapsed_sec: float
    stdout: str


class Evaluator:
    def __init__(
        self,
        official_root: Path,
        config_path: Path,
        results_dir: Path,
        timeout_sec: float,
        retain_traces: bool = True,
    ) -> None:
        self.official_root = official_root.resolve()
        self.config_path = config_path.resolve()
        self.results_dir = results_dir.resolve()
        self.timeout_sec = timeout_sec
        self.retain_traces = retain_traces
        self.calls = 0
        self.temp_root = self.results_dir / "temp"
        for name in ("raw", "schedules", "traces", "logs", "temp"):
            (self.results_dir / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_tag(tag: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._")
        return value or "evaluation"

    def _run(
        self,
        command: list[str],
        result_path: Path,
        trace_path: Path,
        log_path: Path,
        tag: str,
    ) -> EvaluationResult:
        self.calls += 1
        started = time.monotonic()

        def save_failure(message: str) -> None:
            failed_log = self.results_dir / "logs" / f"{self._safe_tag(tag)}.failed.txt"
            failed_log.write_text(message.rstrip() + "\n", encoding="utf-8")

        try:
            completed = subprocess.run(
                command,
                cwd=self.official_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            elapsed = time.monotonic() - started
            message = (
                f"evaluator timeout after {self.timeout_sec:g}s\n"
                f"stdout:\n{error.stdout or ''}\nstderr:\n{error.stderr or ''}\n"
            )
            save_failure(message)
            raise EvaluationError(message, elapsed) from error
        except OSError as error:
            elapsed = time.monotonic() - started
            message = f"could not launch evaluator: {error}"
            save_failure(message)
            raise EvaluationError(message, elapsed) from error

        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            message = (
                f"evaluator exited with code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
            save_failure(message)
            raise EvaluationError(message, elapsed)
        if not result_path.is_file():
            message = f"evaluator succeeded without result file: {result_path}"
            save_failure(message)
            raise EvaluationError(message, elapsed)

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            message = f"could not read evaluator result: {error}"
            save_failure(message)
            raise EvaluationError(message, elapsed) from error

        suffix = self._safe_tag(tag)
        shutil.copy2(result_path, self.results_dir / "raw" / f"{suffix}.json")
        if trace_path.is_file() and self.retain_traces:
            shutil.copy2(trace_path, self.results_dir / "traces" / f"{suffix}.json")
        if log_path.is_file():
            shutil.copy2(log_path, self.results_dir / "logs" / f"{suffix}.txt")
        return EvaluationResult(result=result, elapsed_sec=elapsed, stdout=completed.stdout)

    def evaluate_multicore(
        self,
        graph_path: Path,
        plan: dict[str, Any],
        problem: int,
        tag: str,
    ) -> EvaluationResult:
        script = self.official_root / "code" / f"multicore_cut_evaluate_problem_{problem}.py"
        with tempfile.TemporaryDirectory(prefix="candidate_", dir=self.temp_root) as temp:
            temp_dir = Path(temp)
            plan_path = temp_dir / "plan.json"
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result_path = temp_dir / "result.json"
            trace_path = temp_dir / "trace.json"
            log_path = temp_dir / "log.txt"
            command = [
                sys.executable, str(script), str(graph_path.resolve()), str(plan_path),
                "--config", str(self.config_path),
                "-o", str(result_path),
                "--trace-output", str(trace_path),
                "--log-output", str(log_path),
            ]
            return self._run(command, result_path, trace_path, log_path, tag)

    def evaluate_singlecore(self, graph_path: Path, tag: str) -> EvaluationResult:
        script = self.official_root / "code" / "singlecore_evaluate.py"
        with tempfile.TemporaryDirectory(prefix="singlecore_", dir=self.temp_root) as temp:
            temp_dir = Path(temp)
            result_path = temp_dir / "result.json"
            trace_path = temp_dir / "trace.json"
            log_path = temp_dir / "log.txt"
            command = [
                sys.executable, str(script), str(graph_path.resolve()),
                "--config", str(self.config_path),
                "-o", str(result_path),
                "--trace-output", str(trace_path),
                "--log-output", str(log_path),
            ]
            return self._run(command, result_path, trace_path, log_path, tag)
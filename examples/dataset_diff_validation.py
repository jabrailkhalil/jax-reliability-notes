"""Run independent dataset oracles, CLI fixtures and isolated broken variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CountingResult(unittest.TextTestResult):
    subtests = 0

    def addSubTest(self, test, subtest, error):
        self.subtests += 1
        super().addSubTest(test, subtest, error)


def run_python(
    arguments: list[str], *, directory: Path = ROOT
) -> subprocess.CompletedProcess:
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )


def check_cli() -> dict:
    fixtures = Path("examples/dataset-diff-fixtures")
    results = {}
    expected = {
        "base": ("equal", True, False, False),
        "relabel": ("different", True, True, False),
        "config-only": ("different", True, False, True),
        "repacked": ("equal", False, False, False),
        "row-edits": ("different", False, False, False),
    }
    for name, answer in expected.items():
        process = run_python(
            ["examples/dataset_diff.py", str(fixtures / "base"), str(fixtures / name)]
        )
        if process.returncode != 0:
            raise AssertionError(f"CLI {name}: {process.returncode}: {process.stderr}")
        report = json.loads(process.stdout)
        actual = tuple(
            report[field]
            for field in (
                "status",
                "data_files_equal",
                "features_changed",
                "config_changed",
            )
        )
        if actual != answer or report["complete"] is not True:
            raise AssertionError(f"CLI {name}: {report}")
        results[name] = report
    for name, bound in (
        ("row-overflow", "--max-rows"),
        ("byte-overflow", "--max-bytes"),
    ):
        process = run_python(
            [
                "examples/dataset_diff.py",
                str(fixtures / "base"),
                str(fixtures / "base"),
                bound,
                "1",
            ]
        )
        report = json.loads(process.stdout)
        if (
            process.returncode != 2
            or report["complete"] is not False
            or report["status"] != "invalid"
        ):
            raise AssertionError(f"CLI {name}: {process.returncode}: {report}")
        results[name] = report
    return results


def check_negative_controls() -> list[dict]:
    source = (ROOT / "examples/dataset_diff.py").read_text(encoding="utf-8")
    tests = ROOT / "tests/test_dataset_diff.py"
    controls = [
        (
            "equal-data-hashes-ignore-metadata",
            "    old = {row.key: row for row in before.rows}",
            "    if dict(before.file_hashes)[before.data_files[0]] == dict(after.file_hashes)[after.data_files[0]]:\n        after = before\n    old = {row.key: row for row in before.rows}",
            "test_identical_data_bytes_can_change_label_meanings",
        ),
        (
            "positional-instead-of-keyed-matching",
            "    old = {row.key: row for row in before.rows}\n    new = {row.key: row for row in after.rows}",
            "    old = {str(i): row for i, row in enumerate(before.rows)}\n    new = {str(i): row for i, row in enumerate(after.rows)}",
            "test_reordered_repacked_shards_have_no_keyed_row_edits",
        ),
        (
            "ignore-decoded-label-meaning",
            "        if old[key].label_name != new[key].label_name:",
            "        if False:",
            "test_identical_data_bytes_can_change_label_meanings",
        ),
        (
            "overwrite-duplicate-keys",
            '            _require(key not in seen, "duplicate row key id")',
            "            pass  # Deliberately omit global key uniqueness validation.",
            "test_invalid_rows_are_rejected_on_both_identical_digest_sides",
        ),
        (
            "certify-only-a-capped-prefix",
            '            _require(len(raw_rows) < max_rows, "row limit exceeded")',
            "            if len(raw_rows) >= max_rows:\n                break",
            "test_invalid_manifests_paths_and_resource_limits",
        ),
    ]
    results = []
    for name, original, broken, method in controls:
        if source.count(original) != 1:
            raise AssertionError(
                f"Control {name} no longer matches exactly one source site"
            )
        with tempfile.TemporaryDirectory(prefix="dataset-diff-control-") as temporary:
            directory = Path(temporary)
            (directory / "examples").mkdir()
            (directory / "tests").mkdir()
            (directory / "examples/dataset_diff.py").write_bytes(
                source.replace(original, broken).encode("utf-8")
            )
            shutil.copyfile(tests, directory / "tests/test_dataset_diff.py")
            process = run_python(
                [
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_dataset_diff.py",
                    "-k",
                    method,
                ],
                directory=directory,
            )
            output = process.stdout + process.stderr
            if (
                process.returncode != 1
                or f"FAIL: {method}" not in output
                or "ERROR:" in output
                or "Ran 1 test" not in output
            ):
                raise AssertionError(
                    f"Control {name} did not produce the expected assertion failure: {output}"
                )
            results.append(
                {
                    "control": name,
                    "exit_code": process.returncode,
                    "test": method,
                    "log": output,
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "pyarrow", "huggingface-hub", "ruff", "mypy")
        },
        "scope": "Offline synthetic local snapshots; no Hub revisions, models or remote datasets executed.",
        "success": False,
    }
    try:
        log = io.StringIO()
        suite = unittest.defaultTestLoader.discover(
            str(ROOT / "tests"), pattern="test_dataset_diff.py"
        )
        result = unittest.TextTestRunner(
            stream=log, verbosity=2, resultclass=CountingResult
        ).run(suite)
        report["unittest"] = {
            "tests_run": result.testsRun,
            "subtests": result.subtests,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "log": log.getvalue(),
        }
        if not result.wasSuccessful() or result.testsRun == 0:
            raise AssertionError("Independent dataset oracles failed")
        report["cli"] = check_cli()
        report["negative_controls"] = check_negative_controls()
        report["success"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run actual-model OCR oracles, CLI fixtures and isolated broken controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples/ocr-review-fixtures"
MODEL_SHA256 = "faa49ece375f521a1164f772553828c0d514edce11bbdf5578b130b168bf5831"


class CountingResult(unittest.TextTestResult):
    subtests = 0

    def addSubTest(self, test, subtest, error):
        self.subtests += 1
        super().addSubTest(test, subtest, error)


def make_counting_result(
    stream: Any, descriptions: bool, verbosity: int
) -> CountingResult:
    return CountingResult(stream, descriptions, verbosity)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_upstream() -> dict[str, Any]:
    """Verify the normal installed package before importing any lab tests."""
    version = importlib.metadata.version("ocrmypdf")
    require(version == "17.13.0", f"Unexpected OCRmyPDF version: {version}")
    package = importlib.import_module("ocrmypdf")
    element = package.OcrElement
    require(
        element.__module__ == "ocrmypdf.models.ocr_element",
        "OcrElement must come from the published model module",
    )
    source = Path(inspect.getfile(element)).resolve()
    distribution = importlib.metadata.distribution("ocrmypdf")
    installed_source = Path(
        str(distribution.locate_file("ocrmypdf/models/ocr_element.py"))
    ).resolve()
    require(source == installed_source, "The imported model is not the installed file")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    require(
        source_sha256 == MODEL_SHA256, "Installed model source differs from the pin"
    )

    word = element(
        ocr_class=package.OcrClass.WORD,
        bbox=package.BoundingBox(1, 1, 2, 2),
        text="synthetic probe",
    )
    line = element(ocr_class=package.OcrClass.LINE, children=[word])
    page = element(
        ocr_class=package.OcrClass.PAGE,
        bbox=package.BoundingBox(0, 0, 10, 10),
        children=[line],
    )
    require(page.words == [word], "Actual upstream words traversal did not match")
    return {
        "package": "ocrmypdf",
        "version": version,
        "public_import": "from ocrmypdf import OcrElement, BoundingBox, OcrClass",
        "model_module": element.__module__,
        "model_source_sha256": source_sha256,
        "model_from_installed_distribution": True,
        "actual_dataclass_construction_and_words_checked": True,
        "ocr_pipeline_executed": False,
    }


def run_python(
    arguments: list[str], *, directory: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(directory),
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


def fixture_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FIXTURES.glob("*.json"))
    }


def signature(group: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    return (
        tuple(word["path"] for word in group["a"]),
        tuple(word["path"] for word in group["b"]),
        group["kind"],
    )


def check_cli() -> dict[str, Any]:
    before = fixture_hashes()
    provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    require(
        {name: value for name, value in before.items() if name != "provenance.json"}
        == provenance["files"],
        "Committed fixture bytes do not match provenance; retain LF checkout bytes",
    )
    results: dict[str, Any] = {}

    def invoke(name: str, a: str, b: str, *options: str) -> dict[str, Any]:
        process = run_python(
            [
                "examples/ocr_review.py",
                str(FIXTURES / a),
                str(FIXTURES / b),
                *options,
            ]
        )
        try:
            report = json.loads(process.stdout)
        except (ValueError, TypeError) as error:
            raise AssertionError(
                f"CLI {name} did not emit JSON: {process.returncode}: {process.stderr}"
            ) from error
        require(
            process.returncode == 0
            and report["status"] == "compared"
            and report["complete"] is True,
            f"CLI {name}: {process.returncode}: {report}: {process.stderr}",
        )
        results[name] = {"exit_code": process.returncode, "report": report}
        return report

    primary = invoke(
        "independent-four-region-gold-and-decisions",
        "regions-a.json",
        "regions-b.json",
        "--gold",
        str(FIXTURES / "regions-gold.json"),
        "--decisions",
        str(FIXTURES / "regions-decisions.json"),
    )
    expected_groups = {
        (("/0/0",), ("/0/0",), "text"),
        (("/0/1",), ("/0/1",), "agreement"),
        (("/0/2",), ("/0/2",), "agreement"),
        (("/0/3",), ("/0/3",), "text"),
    }
    require(
        {signature(group) for group in primary["groups"]} == expected_groups,
        "Four-region grouping differs from the literal oracle",
    )
    require(
        {
            signature(group)
            for group in primary["groups"]
            if group["id"] in primary["queue"]
        }
        == {(("/0/0",), ("/0/0",), "text"), (("/0/3",), ("/0/3",), "text")},
        "The literal queue must contain only g1 and g4",
    )
    expected_metrics = {
        "regions": 4,
        "a_error_regions": ["g2", "g4"],
        "b_error_regions": ["g1", "g2"],
        "error_regions": ["g1", "g2", "g4"],
        "caught_error_regions": ["g1", "g4"],
        "missed_error_regions": ["g2"],
        "denominator": 3,
        "caught": 2,
        "missed": 1,
    }
    require(
        all(
            primary["evaluation"][key] == value
            for key, value in expected_metrics.items()
        )
        and math.isclose(primary["evaluation"]["recall"], 2 / 3, rel_tol=1e-12),
        "Gold must count shared wrong g2 even after its independent correction",
    )
    require(
        primary["a_source"] == before["regions-a.json"]
        and primary["b_source"] == before["regions-b.json"],
        "Report lost the actual raw source hashes",
    )
    decisions = {item["outcome"]: item for item in primary["decisions"]}
    require(
        len(primary["decisions"]) == 3
        and {outcome: item["text"] for outcome, item in decisions.items()}
        == {"accept_a": "100", "accept_b": "7", "correct": "80"},
        "Decisions must retain the two acceptance choices and correction80",
    )
    corrected = decisions["correct"]
    require(
        corrected["a"][0]["path"] == corrected["b"][0]["path"] == "/0/1"
        and corrected["a"][0]["text"] == corrected["b"][0]["text"] == "B0"
        and corrected["a"][0]["bbox"] == corrected["b"][0]["bbox"] == [50, 10, 70, 20]
        and corrected["comparison_id"] == primary["comparison_id"]
        and corrected["a_source"] == primary["a_source"]
        and corrected["b_source"] == primary["b_source"]
        and corrected["frame"] == primary["frame"],
        "The agreement correction must preserve both raw readings and provenance",
    )

    cases = [
        (
            "same-sidecar-order-shifted-positions",
            "shifted-a.json",
            "shifted-b.json",
            {
                (("/0/0",), ("/0/1",), "text"),
                (("/0/1",), ("/0/0",), "text"),
            },
        ),
        (
            "equal-text-disjoint-positions",
            "disjoint-a.json",
            "disjoint-b.json",
            {(("/0/0",), (), "unmatched"), ((), ("/0/0",), "unmatched")},
        ),
        (
            "split-word-retains-segmentation",
            "split-a.json",
            "split-b.json",
            {(("/0/0",), ("/0/0", "/0/1"), "segmentation")},
        ),
        (
            "many-to-many-ambiguity",
            "ambiguous-a.json",
            "ambiguous-b.json",
            {(("/0/0", "/0/1"), ("/0/0", "/0/1"), "ambiguous")},
        ),
    ]
    for name, a, b, expected in cases:
        report = invoke(name, a, b)
        require(
            {signature(group) for group in report["groups"]} == expected
            and len(report["queue"]) == len(expected),
            f"CLI {name} differs from the independent literal membership oracle",
        )

    empty = invoke(
        "empty-universe-has-undefined-error-recall",
        "empty-a.json",
        "empty-b.json",
        "--gold",
        str(FIXTURES / "empty-gold.json"),
    )
    require(
        empty["groups"] == []
        and empty["queue"] == []
        and empty["evaluation"]["regions"] == 0
        and empty["evaluation"]["denominator"] == 0
        and empty["evaluation"]["recall"] is None,
        "Empty error denominator must be explicit and recall must be null",
    )
    invalid = [
        ("different-declared-frame", "different-frame-b.json", []),
        (
            "stale-gold-source-binding",
            "regions-b.json",
            ["--gold", str(FIXTURES / "stale-gold.json")],
        ),
        (
            "word-overflow-does-not-certify-prefix",
            "regions-b.json",
            ["--max-words", "1"],
        ),
    ]
    for name, b, options in invalid:
        process = run_python(
            [
                "examples/ocr_review.py",
                str(FIXTURES / "regions-a.json"),
                str(FIXTURES / b),
                *options,
            ]
        )
        report = json.loads(process.stdout)
        require(
            process.returncode == 2
            and report["status"] == "invalid"
            and report["complete"] is False,
            f"Invalid CLI {name} was certified or crashed: {process.returncode}: {report}",
        )
        results[name] = {"exit_code": process.returncode, "report": report}
    require(fixture_hashes() == before, "CLI changed its original fixture inputs")
    return results


def check_negative_controls() -> list[dict[str, Any]]:
    source = (ROOT / "examples/ocr_review.py").read_text(encoding="utf-8")
    tests = ROOT / "tests/test_ocr_review.py"
    controls = [
        (
            "text-only-pairing-ignores-spatial-alignment",
            [
                (
                    "if _iou(old_word, new_word) >= threshold:",
                    "if old_word.text == new_word.text:",
                )
            ],
            "test_same_text_disjoint_positions_remain_unmatched",
        ),
        (
            "queue-only-gold-denominator",
            [
                (
                    'error_regions = errors["a"] | errors["b"]',
                    "error_regions = set(caught)",
                )
            ],
            "test_gold_counts_agreed_wrong_and_both_missing_regions",
        ),
        (
            "ignore-explicit-correction-outcome",
            [
                (
                    'text = decision["correction"]',
                    'text = " ".join(word["text"] for word in groups[identifier]["a"])',
                )
            ],
            "test_decisions_preserve_sources_and_correct_agreement_errors",
        ),
        (
            "certify-only-a-capped-prefix",
            [
                (
                    '_require(words <= max_words, "word limit exceeded")',
                    "pass  # Deliberately remove complete-input word bound.",
                ),
                ("for native in root.words:", "for native in root.words[:max_words]:"),
            ],
            "test_word_limit_rejects_overflow_instead_of_comparing_prefix",
        ),
    ]
    results = []
    for name, replacements, method in controls:
        mutated = source
        for original, broken in replacements:
            require(
                mutated.count(original) == 1,
                f"Control {name} must match exactly one source site: {original}",
            )
            mutated = mutated.replace(original, broken, 1)
        with tempfile.TemporaryDirectory(prefix="ocr-review-control-") as temporary:
            directory = Path(temporary)
            (directory / "examples").mkdir()
            (directory / "tests").mkdir()
            (directory / "examples/ocr_review.py").write_bytes(mutated.encode("utf-8"))
            shutil.copyfile(tests, directory / "tests/test_ocr_review.py")
            process = run_python(
                [
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_ocr_review.py",
                    "-k",
                    method,
                ],
                directory=directory,
            )
            output = process.stdout + process.stderr
            require(
                process.returncode == 1
                and re.search(rf"^FAIL: {re.escape(method)}\b", output, re.MULTILINE)
                is not None
                and re.search(r"^ERROR:", output, re.MULTILINE) is None
                and re.search(r"Ran 1 test\b", output) is not None,
                f"Control {name} did not yield its intended single-test assertion failure: {output}",
            )
            results.append(
                {
                    "control": name,
                    "exit_code": process.returncode,
                    "test": method,
                    "intended_assertion_failure": True,
                    "log": output,
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "scope": "Synthetic positional observations and literal gold; actual installed upstream models, no OCR/PDF/image/browser or human evaluation.",
        "success": False,
    }
    try:
        report["packages"] = dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        )
        report["upstream"] = verify_upstream()
        log = io.StringIO()
        suite = unittest.defaultTestLoader.discover(
            str(ROOT / "tests"), pattern="test_ocr_review.py"
        )
        result = unittest.TextTestRunner(
            stream=log, verbosity=2, resultclass=make_counting_result
        ).run(suite)
        if not isinstance(result, CountingResult):
            raise AssertionError("Expected a counting unittest result")
        report["unittest"] = {
            "tests_run": result.testsRun,
            "subtests": result.subtests,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "log": log.getvalue(),
        }
        require(
            result.wasSuccessful() and result.testsRun > 0 and len(result.skipped) == 0,
            "Independent OCR oracles failed, did not run, or skipped",
        )
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

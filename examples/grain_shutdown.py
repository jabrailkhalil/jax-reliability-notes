"""Check pinned Grain close methods without loading native dependencies.

This executes trusted source, like importing a package. Only the two documented
revisions and their exact file hashes are accepted. The fixtures isolate close;
they do not emulate worker processes, data iteration, or checkpoint restoration.
"""

import argparse
import ast
import atexit
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import unittest
from concurrent import futures
from pathlib import Path
from unittest import mock

SOURCE_FILE = "grain/_src/python/dataset/transformations/interleave.py"
REVISIONS = {
    "baseline": (
        "7e36b7231574d0f9ab106f903a8d832f58157594",
        "af48b7e02246b2eef3fbfb18cc39c957b235d48b8a21ef0fc3b924a305d0e0d2",
    ),
    "fixed": (
        "b24133c928899e86b649d624bea32c3e494483e4",
        "7fbb97a37ebb0c63d79c63c5093b5eef523c25cd34fd566a2abd14543b03285c",
    ),
}
EXPECTED_BASELINE_FAILURES = {
    "overlap_2",
    "overlap_4",
    "overlap_8",
    "siblings_after_ValueError",
    "siblings_after_RuntimeError",
}


def load_close(source, mode):
    """Compile the actual method, not a reimplementation of its algorithm."""
    revision, expected_hash = REVISIONS[mode]
    content = subprocess.check_output(
        ["git", "-C", str(source), "show", f"{revision}:{SOURCE_FILE}"]
    )
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Unexpected source hash for {revision}: {digest}")
    tree = ast.parse(content)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InterleaveDatasetIterator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "close"
    )
    namespace = {"futures": futures, "threading": threading, "sys": sys}
    exec(  # noqa: S102 -- Only SHA-256-verified, trusted pinned source.
        compile(ast.Module(body=[method], type_ignores=[]), SOURCE_FILE, "exec"),
        namespace,
    )
    return namespace["close"], digest


class Child:
    def __init__(self, action=None):
        self.calls = 0
        self.action = action

    def close(self):
        self.calls += 1
        if self.action is not None:
            self.action()


def fixture(close, children):
    class Fixture:
        pass

    Fixture.close = close
    obj = Fixture()
    obj._closed = False
    obj._prefetch_ds_iter = Child()
    obj._iterators_in_use = children
    return obj


def shutdown_child(close, phase):
    """Keep output handles alive through real interpreter final collection."""

    class FinalChild:
        def __init__(self):
            self.write = os.write

        def close(self):
            self.write(1, b"CLOSED\n")

    class FinalFixture:
        def __init__(self):
            self._closed = False
            self._prefetch_ds_iter = FinalChild()
            self._iterators_in_use = [FinalChild(), FinalChild()]
            self.write = os.write
            self.is_finalizing = sys.is_finalizing
            self.cycle = self

        def __del__(self):
            self.write(1, b"FINAL_GC=" + str(self.is_finalizing()).encode() + b"\n")
            self.close()

    FinalFixture.close = close
    gc.disable()
    obj = FinalFixture()
    if phase == "atexit":
        atexit.register(obj.__del__)
        # atexit already invokes cleanup; avoid a second destructor print.
        FinalFixture.__del__ = lambda self: None
    else:
        globals()["KEEP_ALIVE"] = obj


def run_checks(close, source, mode):
    cases = {}
    check = unittest.TestCase()

    def record(name, function):
        try:
            function()
        except (AssertionError, threading.BrokenBarrierError) as error:
            cases[name] = {"passed": False, "error": repr(error)}
        else:
            cases[name] = {"passed": True}

    def overlap(count):
        # This timeout detects a blocked protocol, not a latency regression.
        barrier = threading.Barrier(count, timeout=5)
        children = [Child(lambda: barrier.wait()) for _ in range(count)]
        obj = fixture(close, children)
        try:
            obj.close()
        finally:
            barrier.abort()
        check.assertEqual([child.calls for child in children], [1] * count)
        obj.close()
        check.assertEqual([child.calls for child in children], [1] * count)

    def siblings_after_error(error_type):
        problem = error_type("deliberate child close failure")
        entered = futures.Future()
        returned = futures.Future()
        release = threading.Event()
        finished = threading.Event()

        def fail():
            raise problem

        def slow_cleanup():
            entered.set_result(None)
            if not release.wait(timeout=5):
                raise TimeoutError("The test did not release sibling cleanup")
            finished.set()

        children = [Child(fail), Child(slow_cleanup)]
        obj = fixture(close, children)

        def call_close():
            try:
                obj.close()
            except (ValueError, RuntimeError, TimeoutError) as error:
                returned.set_exception(error)
            else:
                returned.set_result(None)

        caller = threading.Thread(target=call_close, daemon=False)
        caller.start()
        try:
            # No timing threshold: observe whichever protocol event occurs first.
            futures.wait(
                [entered, returned], timeout=5, return_when=futures.FIRST_COMPLETED
            )
            check.assertTrue(entered.done(), "Sibling cleanup never started")
            check.assertFalse(returned.done(), "close returned before sibling release")
        finally:
            release.set()
            caller.join(timeout=10)
        check.assertFalse(caller.is_alive())
        check.assertTrue(finished.is_set())
        with check.assertRaises(error_type) as raised:
            returned.result(timeout=5)
        check.assertIs(raised.exception, problem)
        check.assertEqual([child.calls for child in children], [1, 1])
        obj.close()
        check.assertEqual([child.calls for child in children], [1, 1])

    def slots(children):
        obj = fixture(close, children)
        for child in children:
            if child is not None:
                child.action = lambda: check.assertEqual(obj._prefetch_ds_iter.calls, 1)
        obj.close()
        obj.close()
        check.assertTrue(obj._closed)
        check.assertEqual(obj._prefetch_ds_iter.calls, 1)
        check.assertEqual(
            [c.calls for c in children if c is not None],
            [1] * sum(c is not None for c in children),
        )

    def unavailable_threads():
        children = [Child(), Child()]
        obj = fixture(close, children)
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError):
            obj.close()
        check.assertEqual([child.calls for child in children], [1, 1])

    def finalizing():
        children = [Child(), Child()]
        obj = fixture(close, children)
        with (
            mock.patch.object(sys, "is_finalizing", return_value=True),
            mock.patch.object(threading.Thread, "start") as start,
        ):
            obj.close()
        start.assert_not_called()
        check.assertEqual([child.calls for child in children], [1, 1])

    def real_shutdown(phase):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(source),
                "--expect",
                mode,
                "--shutdown-child",
                phase,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        check.assertEqual(result.returncode, 0, result.stderr)
        check.assertEqual(result.stderr, "")
        expected_phase = "True" if phase == "final-gc" else "False"
        check.assertEqual(
            result.stdout, f"FINAL_GC={expected_phase}\n" + "CLOSED\n" * 3
        )

    for count in (2, 4, 8):
        record(f"overlap_{count}", lambda count=count: overlap(count))
    for error_type in (ValueError, RuntimeError):
        record(
            f"siblings_after_{error_type.__name__}",
            lambda error_type=error_type: siblings_after_error(error_type),
        )
    record("empty", lambda: slots([]))
    record("uninitialized", lambda: slots([None, None]))
    record("partial_single_child", lambda: slots([None, Child(), None]))
    record("unavailable_threads", unavailable_threads)
    record("finalizing_no_threads", finalizing)
    for phase in ("atexit", "final-gc"):
        record(phase, lambda phase=phase: real_shutdown(phase))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Trusted Grain Git clone containing both revisions",
    )
    parser.add_argument("--expect", required=True, choices=REVISIONS)
    parser.add_argument(
        "--shutdown-child", choices=("atexit", "final-gc"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    source = args.source.resolve()
    close, digest = load_close(source, args.expect)
    if args.shutdown_child:
        shutdown_child(close, args.shutdown_child)
        return
    cases = run_checks(close, source, args.expect)
    failed = {name for name, result in cases.items() if not result["passed"]}
    expected = EXPECTED_BASELINE_FAILURES if args.expect == "baseline" else set()
    report = {
        "python": sys.version,
        "platform": sys.platform,
        "revision": REVISIONS[args.expect][0],
        "source_sha256": digest,
        "expect": args.expect,
        "cases": cases,
        "failed": sorted(failed),
        "expected_failures": sorted(expected),
        "matched_expectation": failed == expected,
    }
    print(json.dumps(report, indent=2))
    if failed != expected:
        raise SystemExit("Unexpected result: see the failed/expected sets above")


if __name__ == "__main__":
    main()

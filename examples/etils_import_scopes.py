"""Reproduce import-scope leakage against exact installed etils source.

Run in a disposable environment with one of the two documented revisions.
The named-semaphore probe runs only on Linux and cleans up only its own names.
"""

import argparse
import builtins
import gc
import hashlib
import json
import sys
import threading
import weakref
from pathlib import Path

from etils import epy
from etils.epy import lazy_imports_utils

SOURCES = {
    "baseline": (
        "d18f006162a8da2b7eb79de2e1e1cac0d07774c5",
        "e7e27c3312b2cbb35acaf392b832b7834824c811f0ee12779c2f77c1ba7f8d09",
    ),
    "fixed": (
        "7875132c8a7567ad71035704081d1c9a210353fc",
        "82f4ae62f9ebeffb9b50d0242eb07621251db322f267f7cb2e57c8eef3566fd5",
    ),
}
BASE_FAILURES = {
    "other_thread_plain",
    "other_thread_relative",
    "finalizer_plain",
    "finalizer_relative",
    "non_lifo_restoration",
    "non_lifo_remaining_scope",
}


def plain_import():
    from json import dumps

    return type(dumps).__name__


def relative_import():
    module = __import__(
        "env_utils", {"__package__": "etils.epy"}, None, ("__name__",), 1
    )
    return module.__name__


def observe(import_fn):
    try:
        return import_fn()
    except ValueError as error:
        # This exact error is the reported relative-import interception, not
        # permission to accept unrelated exceptions as a baseline success.
        if "Relative import statements not supported" not in str(error):
            raise
        return "relative import intercepted"


def on_other_thread(import_fn):
    ready = threading.Event()
    done = threading.Event()
    result, errors = [], []

    def worker():
        try:
            if not ready.wait(10):
                raise TimeoutError("Lazy scope did not open")
            result.append(observe(import_fn))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        with epy.lazy_imports():
            ready.set()
            if not done.wait(10):
                raise TimeoutError("Worker import did not finish")
    finally:
        ready.set()
        thread.join(10)
    if thread.is_alive() or errors or len(result) != 1:
        raise AssertionError((thread.is_alive(), errors, result))
    return result[0]


def during_finalization(import_fn):
    class Node:
        pass

    result = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        node = Node()
        node.cycle = node
        finalizer = weakref.finalize(node, lambda: result.append(observe(import_fn)))
        del node
        with epy.lazy_imports():
            gc.collect()
        if finalizer.alive or len(result) != 1:
            raise AssertionError((finalizer.alive, result))
        return result[0]
    finally:
        if gc_enabled:
            gc.enable()


def overlapping_scopes(first_exit):
    original = builtins.__import__
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    exited = [threading.Event(), threading.Event()]
    results = [[], []]
    errors = []

    def worker(index):
        try:
            with epy.lazy_imports():
                import json as before

                results[index].append(isinstance(before, lazy_imports_utils.LazyModule))
                entered[index].set()
                if not release[index].wait(10):
                    raise TimeoutError("Lazy scope was not released")
                import json as after

                results[index].append(isinstance(after, lazy_imports_utils.LazyModule))
        except BaseException as error:
            errors.append(error)
        finally:
            exited[index].set()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    try:
        threads[0].start()
        if not entered[0].wait(10):
            raise TimeoutError("First scope did not open")
        threads[1].start()
        if not entered[1].wait(10):
            raise TimeoutError("Second scope did not open")
        release[first_exit].set()
        if not exited[first_exit].wait(10):
            raise TimeoutError("First exit did not finish")
        release[1 - first_exit].set()
    finally:
        for event in release:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
        restored = builtins.__import__ is original
        # Restore before evaluating the broken baseline's observations.
        builtins.__import__ = original
    if any(thread.is_alive() for thread in threads) or errors:
        raise AssertionError(errors)
    if any(len(result) != 2 for result in results):
        raise AssertionError(results)
    return restored, all(all(result) for result in results)


def direct_import():
    with epy.lazy_imports():
        import missing_scope_lab_module as module

    return isinstance(module, lazy_imports_utils.LazyModule)


def nested_callbacks():
    outer, inner = [], []
    with epy.lazy_imports(success_callback=outer.append):
        import json as first

        with epy.lazy_imports(success_callback=inner.append):
            import json as second

        import json as third

    assert first.dumps([]) == second.dumps([]) == third.dumps([]) == "[]"
    return [outer, inner]


def linux_semaphores():
    import _multiprocessing
    import multiprocessing
    from multiprocessing import resource_tracker

    class Node:
        pass

    original = builtins.__import__
    gc_enabled = gc.isenabled()
    gc.disable()
    names = []
    try:
        for _ in range(4):
            node = Node()
            node.lock = multiprocessing.get_context("spawn").Lock()
            node.cycle = node
            names.append(node.lock._semlock.name)
            del node
        with epy.lazy_imports():
            gc.collect()
        linked = sum(
            Path("/dev/shm/sem." + name.lstrip("/")).exists() for name in names
        )
    finally:
        builtins.__import__ = original
        for name in names:
            if Path("/dev/shm/sem." + name.lstrip("/")).exists():
                _multiprocessing.sem_unlink(name)
                resource_tracker.unregister(name, "semaphore")
        if gc_enabled:
            gc.enable()
    if any(Path("/dev/shm/sem." + name.lstrip("/")).exists() for name in names):
        raise AssertionError("Probe resource cleanup failed")
    return linked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=SOURCES, required=True)
    args = parser.parse_args()
    revision, expected_sha = SOURCES[args.expect]
    source_sha = hashlib.sha256(
        Path(lazy_imports_utils.__file__).read_bytes()
    ).hexdigest()
    if source_sha != expected_sha:
        raise ValueError(f"Wrong installed source for {revision}: {source_sha}")
    records = []

    def check(name, actual, expected):
        records.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "ok": actual == expected,
            }
        )

    for name, import_fn, expected in (
        ("plain", plain_import, "function"),
        ("relative", relative_import, "etils.epy.env_utils"),
    ):
        check(f"other_thread_{name}", on_other_thread(import_fn), expected)
        check(f"finalizer_{name}", during_finalization(import_fn), expected)
    restored, remaining_lazy = overlapping_scopes(first_exit=0)
    check("non_lifo_restoration", restored, True)
    check("non_lifo_remaining_scope", remaining_lazy, True)
    check("direct_import_remains_lazy", direct_import(), True)
    check("nested_callbacks", nested_callbacks(), [["json", "json"], ["json"]])
    check("lifo_control", overlapping_scopes(first_exit=1), (True, True))
    expected_failures = set(BASE_FAILURES)
    if sys.platform == "linux":
        check("native_linux_finalizer", linux_semaphores(), 0)
        expected_failures.add("native_linux_finalizer")
    if args.expect == "fixed":
        expected_failures.clear()
    failures = {record["name"] for record in records if not record["ok"]}
    # Failures must also have the exact reported observation, not just any
    # inequality that happens to share a case name.
    baseline_observations = {
        "other_thread_plain": "LazyModule",
        "other_thread_relative": "relative import intercepted",
        "finalizer_plain": "LazyModule",
        "finalizer_relative": "relative import intercepted",
        "non_lifo_restoration": False,
        "non_lifo_remaining_scope": False,
        "native_linux_finalizer": 4,
    }
    if args.expect == "baseline":
        for record in records:
            if record["name"] in expected_failures:
                assert record["actual"] == baseline_observations[record["name"]], record
    assert failures == expected_failures, (records, failures, expected_failures)
    # One log record prevents expected baseline finalizer stderr from being
    # interleaved between the lines of a pretty-printed JSON report in CI.
    print(
        "ETILS_LAB_RESULT="
        + json.dumps(
            {
                "mode": args.expect,
                "revision": revision,
                "source_sha256": source_sha,
                "python": sys.version,
                "platform": sys.platform,
                "passed_controls": len(records) - len(failures),
                "expected_failures": sorted(failures),
                "native_semaphore_probe": "executed"
                if sys.platform == "linux"
                else "Linux only",
                "records": records,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

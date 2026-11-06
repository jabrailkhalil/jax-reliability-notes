"""Independent oracles for the bounded, local dataset snapshot lab."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import dataset_diff as dataset_diff_module  # noqa: E402
from dataset_diff import compare_snapshots, load_snapshot  # noqa: E402


FEATURES = {
    "id": {"dtype": "string", "_type": "Value"},
    "text": {"dtype": "string", "_type": "Value"},
    "label": {"names": ["negative", "positive"], "_type": "ClassLabel"},
}
CONFIG = {"name": "toy", "split": "train"}
ROWS = [
    {"id": "r1", "text": "alpha", "label": 0},
    {"id": "r2", "text": "beta", "label": 1},
]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def row_bytes(rows):
    return b"".join(json_bytes(row) for row in rows)


def copy_features():
    return json.loads(json.dumps(FEATURES))


class DatasetDiffOracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dataset-diff-oracle-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.counter = 0

    def make_snapshot(
        self, *, shards=None, features=None, config=None, column_order=None
    ):
        self.counter += 1
        directory = self.directory / f"snapshot-{self.counter}"
        directory.mkdir()
        shards = [("rows.jsonl", row_bytes(ROWS))] if shards is None else shards
        payloads = {
            "features.json": json_bytes(FEATURES if features is None else features),
            "config.json": json_bytes(CONFIG if config is None else config),
        }
        payloads.update(shards)
        for name, payload in payloads.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        manifest = {
            "version": 1,
            "key": "id",
            "column_order": ["id", "text", "label"]
            if column_order is None
            else column_order,
            "data_files": [name for name, _ in shards],
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            },
        }
        self.write_manifest(directory, manifest)
        return directory

    def read_manifest(self, directory):
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, directory, manifest):
        (directory / "manifest.json").write_bytes(json_bytes(manifest))

    def replace_file(self, directory, name, payload):
        (directory / name).write_bytes(payload)
        manifest = self.read_manifest(directory)
        manifest["files"][name] = hashlib.sha256(payload).hexdigest()
        self.write_manifest(directory, manifest)

    def assert_invalid(self, directory, **bounds):
        with self.assertRaises((ValueError, OSError)):
            load_snapshot(directory, **bounds)

    def run_cli(self, before, after):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["dataset_diff.py", str(before), str(after)]),
            redirect_stdout(output),
        ):
            exit_code = dataset_diff_module.main()
        return exit_code, json.loads(output.getvalue())

    def assert_report(self, report, **changes):
        expected = {
            "status": "equal",
            "complete": True,
            "data_files_equal": True,
            "features_changed": False,
            "column_order_changed": False,
            "config_changed": False,
            "added": [],
            "removed": [],
            "changed": [],
            "label_meaning_changes": [],
        }
        expected.update(changes)
        for field, answer in expected.items():
            with self.subTest(report_field=field):
                self.assertEqual(report[field], answer)
        for field in ("before", "after"):
            self.assertIsInstance(report[field], str)
            self.assertTrue(report[field])

    def test_identical_data_bytes_can_change_label_meanings(self):
        features = copy_features()
        features["label"]["names"] = ["positive", "negative"]
        before = self.make_snapshot()
        after = self.make_snapshot(features=features)
        self.assertEqual(
            (before / "rows.jsonl").read_bytes(), (after / "rows.jsonl").read_bytes()
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(
            report,
            status="different",
            features_changed=True,
            label_meaning_changes=[
                {
                    "key": "r1",
                    "before": {"kind": "class", "name": "negative"},
                    "after": {"kind": "class", "name": "positive"},
                },
                {
                    "key": "r2",
                    "before": {"kind": "class", "name": "positive"},
                    "after": {"kind": "class", "name": "negative"},
                },
            ],
        )
        self.assertNotEqual(report["before"], report["after"])

    def test_configuration_change_is_separate_from_features_and_rows(self):
        before = self.make_snapshot()
        after = self.make_snapshot(config={"name": "reviewed", "split": "train"})
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(report, status="different", config_changed=True)
        self.assertNotEqual(report["before"], report["after"])

    def test_reordered_repacked_shards_have_no_keyed_row_edits(self):
        before = self.make_snapshot()
        after = self.make_snapshot(
            shards=[
                ("parts/b.jsonl", row_bytes([ROWS[1]])),
                ("parts/a.jsonl", row_bytes([ROWS[0]])),
            ],
            config={"split": "train", "name": "toy"},
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(report, data_files_equal=False)

    def test_added_removed_and_changed_keys_have_literal_oracles(self):
        before = self.make_snapshot()
        after = self.make_snapshot(
            shards=[
                (
                    "rows.jsonl",
                    row_bytes(
                        [
                            {"id": "r3", "text": "gamma", "label": 0},
                            {"id": "r1", "text": "edited", "label": 1},
                        ]
                    ),
                )
            ]
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(
            report,
            status="different",
            data_files_equal=False,
            added=["r3"],
            removed=["r2"],
            changed=[{"key": "r1", "fields": ["label", "text"]}],
            label_meaning_changes=[
                {
                    "key": "r1",
                    "before": {"kind": "class", "name": "negative"},
                    "after": {"kind": "class", "name": "positive"},
                },
            ],
        )

    def test_keys_preserve_whitespace_unicode_and_numeric_looking_strings(self):
        keys = ["1", "01", " 1", "1 ", " ", "\u00e9", "e\u0301"]
        rows = [{"id": key, "text": "unchanged", "label": 0} for key in keys]
        edited = [
            {"id": key, "text": "edited" if key == "01" else "unchanged", "label": 0}
            for key in reversed(keys)
        ]
        before = self.make_snapshot(shards=[("rows.jsonl", row_bytes(rows))])
        after = self.make_snapshot(shards=[("rows.jsonl", row_bytes(edited))])
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(
            report,
            status="different",
            data_files_equal=False,
            changed=[{"key": "01", "fields": ["text"]}],
        )
        self.assertEqual(report["row_counts"], {"before": 7, "after": 7})

    def test_unknown_label_is_distinct_from_a_class_named_unknown(self):
        features = copy_features()
        features["label"]["names"] = ["unknown", "known"]
        before = self.make_snapshot(
            features=features,
            shards=[
                (
                    "rows.jsonl",
                    row_bytes(
                        [
                            {"id": "minus", "text": "a", "label": -1},
                            {"id": "named", "text": "b", "label": 0},
                        ]
                    ),
                )
            ],
        )
        after = self.make_snapshot(
            features=features,
            shards=[
                (
                    "rows.jsonl",
                    row_bytes(
                        [
                            {"id": "minus", "text": "a", "label": 0},
                            {"id": "named", "text": "b", "label": -1},
                        ]
                    ),
                )
            ],
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(
            report,
            status="different",
            data_files_equal=False,
            changed=[
                {"key": "minus", "fields": ["label"]},
                {"key": "named", "fields": ["label"]},
            ],
            label_meaning_changes=[
                {
                    "key": "minus",
                    "before": {"kind": "unknown"},
                    "after": {"kind": "class", "name": "unknown"},
                },
                {
                    "key": "named",
                    "before": {"kind": "class", "name": "unknown"},
                    "after": {"kind": "unknown"},
                },
            ],
        )

    def test_empty_data_files_and_no_data_files_are_both_empty(self):
        before = self.make_snapshot(shards=[])
        after = self.make_snapshot(shards=[("empty.jsonl", b"")])
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(report, data_files_equal=False)

    def test_declared_column_order_is_separate_from_feature_mapping(self):
        features = {
            "label": FEATURES["label"],
            "id": FEATURES["id"],
            "text": FEATURES["text"],
        }
        before = self.make_snapshot()
        after = self.make_snapshot(
            features=features, column_order=["label", "id", "text"]
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(report, status="different", column_order_changed=True)
        self.assertNotEqual(report["before"], report["after"])

    def test_feature_mapping_reorder_changes_identity_without_semantic_change(self):
        features = {
            "label": FEATURES["label"],
            "id": FEATURES["id"],
            "text": FEATURES["text"],
        }
        before = self.make_snapshot()
        after = self.make_snapshot(features=features)
        self.assertNotEqual(
            (before / "features.json").read_bytes(),
            (after / "features.json").read_bytes(),
        )
        report = compare_snapshots(load_snapshot(before), load_snapshot(after))
        self.assert_report(report)
        self.assertNotEqual(report["before"], report["after"])

    def test_exact_row_and_total_byte_limits_are_inclusive(self):
        directory = self.make_snapshot(
            shards=[
                ("a.jsonl", row_bytes([ROWS[0]])),
                ("b.jsonl", row_bytes([{"id": "r2", "text": "\u03b2", "label": 1}])),
            ]
        )
        manifest = self.read_manifest(directory)
        total_bytes = len((directory / "manifest.json").read_bytes()) + sum(
            len((directory / name).read_bytes()) for name in manifest["files"]
        )
        reference = load_snapshot(directory)
        exact = load_snapshot(directory, max_rows=2, max_bytes=total_bytes)
        self.assert_report(compare_snapshots(reference, exact))
        with self.assertRaisesRegex(ValueError, "row limit"):
            load_snapshot(directory, max_rows=1, max_bytes=total_bytes)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            load_snapshot(directory, max_rows=2, max_bytes=total_bytes - 1)

    def test_loaded_snapshot_does_not_reread_mutated_input_files(self):
        directory = self.make_snapshot()
        identical = self.make_snapshot()
        loaded = load_snapshot(directory)
        reference = load_snapshot(identical)
        original = compare_snapshots(loaded, reference)
        self.assert_report(original)
        self.assertEqual(original["before"], original["after"])
        (directory / "rows.jsonl").write_text("invalid now", encoding="utf-8")
        (directory / "features.json").write_text("{}", encoding="utf-8")
        (directory / "config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(compare_snapshots(loaded, reference), original)
        self.assert_invalid(directory)

    def test_cli_success_emits_complete_equal_and_different_reports(self):
        before = self.make_snapshot()
        cases = [
            ("equal", CONFIG, False),
            ("different", {"name": "reviewed", "split": "train"}, True),
        ]
        for status, config, config_changed in cases:
            with self.subTest(status=status):
                after = self.make_snapshot(config=config)
                exit_code, report = self.run_cli(before, after)
                self.assertEqual(exit_code, 0)
                self.assert_report(report, status=status, config_changed=config_changed)

    def test_cli_invalid_json_on_either_side_is_incomplete_with_exit_two(self):
        for side in ("before", "after"):
            with self.subTest(side=side):
                before = self.make_snapshot()
                after = self.make_snapshot()
                invalid = before if side == "before" else after
                self.replace_file(invalid, "rows.jsonl", b'{"id":"r1"\n')
                exit_code, report = self.run_cli(before, after)
                self.assertEqual(exit_code, 2)
                self.assertEqual(set(report), {"status", "complete", "error"})
                self.assertEqual(report["status"], "invalid")
                self.assertIs(report["complete"], False)
                self.assertIsInstance(report["error"], str)
                self.assertTrue(report["error"])

    def test_cli_unexpected_construction_failure_is_incomplete_with_exit_three(self):
        before = self.make_snapshot()
        after = self.make_snapshot()
        with patch.object(
            dataset_diff_module.Dataset,
            "from_dict",
            side_effect=RuntimeError("injected construction failure"),
        ) as construct:
            exit_code, report = self.run_cli(before, after)
        construct.assert_called_once()
        self.assertEqual(exit_code, 3)
        self.assertEqual(
            report, {"status": "error", "complete": False, "error": "RuntimeError"}
        )

    def test_invalid_rows_are_rejected_on_both_identical_digest_sides(self):
        cases = [
            ("duplicate within shard", [("rows.jsonl", row_bytes([ROWS[0], ROWS[0]]))]),
            (
                "duplicate across shards",
                [("a.jsonl", row_bytes([ROWS[0]])), ("b.jsonl", row_bytes([ROWS[0]]))],
            ),
            (
                "missing key after valid shard",
                [
                    ("a.jsonl", row_bytes([ROWS[0]])),
                    ("b.jsonl", row_bytes([{"text": "beta", "label": 1}])),
                ],
            ),
        ]
        invalid_rows = [
            {"text": "alpha", "label": 0},
            {"id": None, "text": "alpha", "label": 0},
            {"id": "", "text": "alpha", "label": 0},
            {"id": 1, "text": "alpha", "label": 0},
            {"id": True, "text": "alpha", "label": 0},
            {"id": "r1", "text": None, "label": 0},
            {"id": "r1", "text": 1, "label": 0},
            {"id": "r1", "label": 0},
            {"id": "r1", "text": "alpha", "label": 0, "extra": 1},
        ]
        invalid_rows.extend(
            {"id": "r1", "text": "alpha", "label": value}
            for value in (None, True, 0.0, "negative", -2, 2)
        )
        cases.extend(
            (f"invalid raw row {index}", [("rows.jsonl", row_bytes([row]))])
            for index, row in enumerate(invalid_rows)
        )
        for name, shards in cases:
            with self.subTest(case=name):
                before = self.make_snapshot(shards=shards)
                after = self.make_snapshot(shards=shards)
                self.assertEqual(
                    self.read_manifest(before)["files"],
                    self.read_manifest(after)["files"],
                )
                self.assert_invalid(before)
                self.assert_invalid(after)

    def test_malformed_json_and_metadata_are_invalid_with_correct_digests(self):
        cases = [
            ("rows.jsonl", b'{"id":"r1"\n'),
            ("rows.jsonl", b"[]\n"),
            ("rows.jsonl", b'{"id":"r1","id":"r2","text":"a","label":0}\n'),
            ("rows.jsonl", b'{"id":"r1","text":"a","label":NaN}\n'),
            ("config.json", b'{"name":"toy","name":"other","split":"train"}\n'),
            ("features.json", b'{"id":{},"id":{},"text":{},"label":{}}\n'),
        ]
        for config in (
            {"name": "", "split": "train"},
            {"name": 1, "split": "train"},
            {"name": "toy"},
            {"name": "toy", "split": "train", "extra": True},
        ):
            cases.append(("config.json", json_bytes(config)))
        invalid_features = []
        for names in ([], ["same", "same"], ["", "positive"], [0, "positive"]):
            features = copy_features()
            features["label"]["names"] = names
            invalid_features.append(features)
        features = copy_features()
        features["id"]["dtype"] = "int64"
        invalid_features.append(features)
        features = copy_features()
        del features["text"]
        invalid_features.append(features)
        features = copy_features()
        features["extra"] = {"dtype": "string", "_type": "Value"}
        invalid_features.append(features)
        features = copy_features()
        features["label"]["names_file"] = "not-part-of-this-contract.txt"
        invalid_features.append(features)
        cases.extend(("features.json", json_bytes(value)) for value in invalid_features)
        for index, (name, payload) in enumerate(cases):
            with self.subTest(case=index, file=name):
                directory = self.make_snapshot()
                self.replace_file(directory, name, payload)
                self.assert_invalid(directory)

    def test_invalid_manifests_paths_and_resource_limits(self):
        mutations = [
            ("version", 2),
            ("version", True),
            ("key", "guessed"),
            ("column_order", ["id", "id", "label"]),
            ("column_order", ["id", "text"]),
            ("column_order", ["id", "text", "label", "extra"]),
            ("data_files", ["rows.jsonl", "rows.jsonl"]),
            ("data_files", "rows.jsonl"),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                directory = self.make_snapshot()
                manifest = self.read_manifest(directory)
                manifest[field] = value
                self.write_manifest(directory, manifest)
                self.assert_invalid(directory)
        for kind in (
            "missing metadata",
            "extra file",
            "uppercase digest",
            "wrong digest",
            "extra manifest field",
            "duplicate manifest field",
        ):
            with self.subTest(case=kind):
                directory = self.make_snapshot()
                manifest = self.read_manifest(directory)
                if kind == "missing metadata":
                    del manifest["files"]["features.json"]
                elif kind == "extra file":
                    manifest["files"]["unlisted.jsonl"] = hashlib.sha256(
                        b""
                    ).hexdigest()
                elif kind == "uppercase digest":
                    manifest["files"]["rows.jsonl"] = manifest["files"][
                        "rows.jsonl"
                    ].upper()
                elif kind == "wrong digest":
                    manifest["files"]["rows.jsonl"] = "0" * 64
                elif kind == "extra manifest field":
                    manifest["extra"] = True
                if kind == "duplicate manifest field":
                    payload = b'{"version":1,' + json_bytes(manifest)[1:]
                    (directory / "manifest.json").write_bytes(payload)
                else:
                    self.write_manifest(directory, manifest)
                self.assert_invalid(directory)
        outside = self.directory / "outside.jsonl"
        outside.write_bytes(row_bytes(ROWS))
        for escaped in ("../outside.jsonl", str(outside.resolve())):
            with self.subTest(path=escaped):
                directory = self.make_snapshot(shards=[])
                manifest = self.read_manifest(directory)
                manifest["data_files"] = [escaped]
                manifest["files"][escaped] = hashlib.sha256(
                    outside.read_bytes()
                ).hexdigest()
                self.write_manifest(directory, manifest)
                self.assert_invalid(directory)
        directory = self.make_snapshot()
        self.assert_invalid(directory, max_rows=1)
        self.assert_invalid(directory, max_bytes=1)
        for field in ("max_rows", "max_bytes"):
            for value in (0, -1, True, 1.5, "1000"):
                with self.subTest(bound=field, value=value):
                    self.assert_invalid(directory, **{field: value})


if __name__ == "__main__":
    unittest.main()

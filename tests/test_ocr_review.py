"""Independent synthetic oracles for positional OCR review, not OCR accuracy."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
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

import ocr_review as ocr_review_module  # noqa: E402
from ocr_review import (  # noqa: E402
    compare_observations,
    evaluate_comparison,
    load_observation,
    record_decisions,
)


FRAME = {"id": "synthetic-page-1", "page_number": 0, "width": 300, "height": 200}


def clone(value):
    return json.loads(json.dumps(value))


def word(text, bbox, **fields):
    return {"ocr_class": "ocrx_word", "text": text, "bbox": list(bbox), **fields}


def document(words, *, engine="synthetic-A", frame=None):
    frame = clone(FRAME if frame is None else frame)
    return {
        "version": 1,
        "engine": engine,
        "frame": frame,
        "tree": {
            "ocr_class": "ocr_page",
            "bbox": [0, 0, frame["width"], frame["height"]],
            "children": [{"ocr_class": "ocr_line", "children": words}],
        },
    }


def raw_json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


class OcrReviewOracleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ocr-review-oracle-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.counter = 0

    def save(self, value=None, *, raw=None):
        self.counter += 1
        path = self.directory / f"input-{self.counter}.json"
        path.write_bytes(raw_json(value) if raw is None else raw)
        return path

    def source_hash(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def pair(self, a_words, b_words):
        a_path = self.save(document(a_words))
        b_path = self.save(document(b_words, engine="synthetic-B"))
        a, b = load_observation(a_path), load_observation(b_path)
        return a_path, b_path, a, b, compare_observations(a, b)

    def group(self, report, *, a, b):
        matches = [
            group
            for group in report["groups"]
            if {item["path"] for item in group["a"]} == set(a)
            and {item["path"] for item in group["b"]} == set(b)
        ]
        self.assertEqual(
            len(matches), 1, f"expected one spatial group for A={a}, B={b}"
        )
        return matches[0]

    def four_regions(self):
        boxes = [
            [10, 10, 40, 30],
            [60, 10, 90, 30],
            [110, 10, 150, 30],
            [170, 10, 200, 30],
        ]
        a_words = [
            word(text, box) for text, box in zip(["100", "B0", "OK", "?"], boxes)
        ]
        b_words = [
            word(text, box) for text, box in zip(["101", "B0", "OK", "7"], boxes)
        ]
        a_path, b_path, a, b, report = self.pair(a_words, b_words)
        gold = {
            "version": 1,
            "a_source": self.source_hash(a_path),
            "b_source": self.source_hash(b_path),
            "regions": [
                {"id": "g1", "a": ["/0/0"], "b": ["/0/0"], "text": "100"},
                {"id": "g2", "a": ["/0/1"], "b": ["/0/1"], "text": "80"},
                {"id": "g3", "a": ["/0/2"], "b": ["/0/2"], "text": "OK"},
                {"id": "g4", "a": ["/0/3"], "b": ["/0/3"], "text": "7"},
            ],
        }
        return a_path, b_path, a, b, report, gold

    def run_cli(self, a_path, b_path, *arguments):
        output = io.StringIO()
        with (
            patch.object(
                sys, "argv", ["ocr_review.py", str(a_path), str(b_path), *arguments]
            ),
            redirect_stdout(output),
        ):
            exit_code = ocr_review_module.main()
        return exit_code, json.loads(output.getvalue())

    def test_native_projection_ignores_aggregate_text_and_keeps_raw_paths(self):
        value = document(
            [
                word("e\u0301  ", [10, 10, 30, 20], confidence=0.7),
                word("\u00e9", [40, 10, 60, 20]),
            ]
        )
        value["tree"]["text"] = "THIS PAGE AGGREGATE IS NOT A WORD"
        value["tree"]["children"][0]["text"] = "THIS LINE AGGREGATE IS NOT A WORD"
        value["tree"]["children"][0]["bbox"] = [5, 5, 70, 25]
        value["tree"]["children"].insert(
            0, {"ocr_class": "ocr_par", "text": "not a leaf word"}
        )
        path = self.save(value)
        observed = load_observation(path)
        self.assertEqual(observed.source_id, self.source_hash(path))
        self.assertEqual(observed.engine, "synthetic-A")
        self.assertEqual(
            [
                (item.path, item.line, item.text, item.bbox, item.confidence)
                for item in observed.words
            ],
            [
                ("/1/0", "/1", "e\u0301  ", (10, 10, 30, 20), 0.7),
                ("/1/1", "/1", "\u00e9", (40, 10, 60, 20), None),
            ],
        )
        self.assertIsInstance(observed.words, tuple)
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            observed.words[0].text = "overwritten"
        original = compare_observations(observed, observed)
        path.write_text("{}", encoding="utf-8")
        self.assertEqual(compare_observations(observed, observed), original)

    def test_text_comparison_does_not_normalize_case_spaces_or_unicode(self):
        for first, second in [("Word", "word"), ("a b", "a  b"), ("\u00e9", "e\u0301")]:
            with self.subTest(a=first, b=second):
                _, _, _, _, report = self.pair(
                    [word(first, [10, 10, 30, 20])], [word(second, [10, 10, 30, 20])]
                )
                group = self.group(report, a=["/0/0"], b=["/0/0"])
                self.assertEqual(group["kind"], "text")
                self.assertEqual(group["a"][0]["text"], first)
                self.assertEqual(group["b"][0]["text"], second)
                self.assertEqual(report["queue"], [group["id"]])

    def test_same_text_disjoint_positions_remain_unmatched(self):
        _, _, _, _, report = self.pair(
            [word("Total", [10, 10, 30, 20])], [word("Total", [100, 10, 120, 20])]
        )
        left = self.group(report, a=["/0/0"], b=[])
        right = self.group(report, a=[], b=["/0/0"])
        self.assertEqual([left["kind"], right["kind"]], ["unmatched", "unmatched"])
        self.assertEqual(set(report["queue"]), {left["id"], right["id"]})
        self.assertTrue(left["needs_review"])
        self.assertTrue(right["needs_review"])

    def test_reordered_duplicate_words_match_positions_not_tree_order(self):
        _, _, _, _, report = self.pair(
            [word("same", [10, 10, 30, 20]), word("same", [100, 10, 120, 20])],
            [word("same", [100, 10, 120, 20]), word("same", [10, 10, 30, 20])],
        )
        left = self.group(report, a=["/0/0"], b=["/0/1"])
        right = self.group(report, a=["/0/1"], b=["/0/0"])
        self.assertEqual([left["kind"], right["kind"]], ["agreement", "agreement"])
        self.assertEqual(report["queue"], [])
        self.assertEqual(left["b"][0]["bbox"], [10, 10, 30, 20])
        self.assertEqual(right["b"][0]["bbox"], [100, 10, 120, 20])

    def test_split_and_merge_are_reviewable_even_when_joined_text_matches(self):
        whole = [word("note book", [0, 0, 100, 10])]
        # Reverse supplied tree order: geometric reading order must remain note, book.
        split = [word("book", [20, 0, 100, 10]), word("note", [0, 0, 20, 10])]
        for a_words, b_words, a_paths, b_paths in [
            (whole, split, ["/0/0"], ["/0/0", "/0/1"]),
            (split, whole, ["/0/0", "/0/1"], ["/0/0"]),
        ]:
            with self.subTest(a_count=len(a_words), b_count=len(b_words)):
                _, _, _, _, report = self.pair(a_words, b_words)
                group = self.group(report, a=a_paths, b=b_paths)
                self.assertEqual(group["kind"], "segmentation")
                self.assertEqual(report["queue"], [group["id"]])
                split_side = group["a"] if len(a_words) == 2 else group["b"]
                self.assertEqual(
                    [item["text"] for item in split_side], ["note", "book"]
                )
                self.assertEqual(
                    [item["path"] for item in split_side], ["/0/1", "/0/0"]
                )

    def test_many_to_many_overlap_is_ambiguous_without_arbitrary_pairing(self):
        _, _, _, _, report = self.pair(
            [word("a", [0, 0, 20, 10]), word("b", [20, 0, 40, 10])],
            [word("a", [0, 0, 30, 10]), word("b", [10, 0, 40, 10])],
        )
        group = self.group(report, a=["/0/0", "/0/1"], b=["/0/0", "/0/1"])
        self.assertEqual(group["kind"], "ambiguous")
        self.assertEqual(len(report["groups"]), 1)
        self.assertEqual(report["queue"], [group["id"]])

    def test_iou_threshold_is_inclusive_and_requires_real_intersection(self):
        # Intersection/union = 20/200 = 0.1, exactly the declared threshold.
        a_path = self.save(document([word("A", [0, 0, 11, 10])]))
        for box, expected_kind, expected_groups in [
            ([9, 0, 20, 10], "text", 1),
            ([10, 0, 21, 10], "unmatched", 2),
            ([11, 0, 22, 10], "unmatched", 2),
            ([0, 30, 11, 40], "unmatched", 2),
        ]:
            with self.subTest(bbox=box):
                b_path = self.save(document([word("B", box)], engine="synthetic-B"))
                report = compare_observations(
                    load_observation(a_path), load_observation(b_path)
                )
                self.assertEqual(len(report["groups"]), expected_groups)
                self.assertEqual(
                    {group["kind"] for group in report["groups"]}, {expected_kind}
                )
                self.assertEqual(len(report["queue"]), expected_groups)
        # A contained small word has IoU 0.05, despite containment coverage 1.
        b_path = self.save(document([word("A", [0, 0, 20, 10])], engine="synthetic-B"))
        a_path = self.save(document([word("A", [0, 0, 1, 10])]))
        report = compare_observations(
            load_observation(a_path), load_observation(b_path)
        )
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual({group["kind"] for group in report["groups"]}, {"unmatched"})

    def test_gold_counts_agreed_wrong_and_both_missing_regions(self):
        _, _, a, b, report, gold = self.four_regions()
        g1 = self.group(report, a=["/0/0"], b=["/0/0"])
        g2 = self.group(report, a=["/0/1"], b=["/0/1"])
        g4 = self.group(report, a=["/0/3"], b=["/0/3"])
        self.assertEqual(set(report["queue"]), {g1["id"], g4["id"]})
        self.assertEqual(g2["kind"], "agreement")
        metrics = evaluate_comparison(report, a, b, gold)
        self.assertEqual(metrics["regions"], 4)
        self.assertEqual(set(metrics["a_error_regions"]), {"g2", "g4"})
        self.assertEqual(set(metrics["b_error_regions"]), {"g1", "g2"})
        self.assertEqual(set(metrics["error_regions"]), {"g1", "g2", "g4"})
        self.assertEqual(set(metrics["caught_error_regions"]), {"g1", "g4"})
        self.assertEqual(metrics["missed_error_regions"], ["g2"])
        self.assertEqual(
            (metrics["denominator"], metrics["caught"], metrics["missed"]), (3, 2, 1)
        )
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        gold["regions"].append(
            {"id": "both-missing", "a": [], "b": [], "text": "omitted"}
        )
        extended = evaluate_comparison(report, a, b, gold)
        self.assertEqual(
            (extended["denominator"], extended["caught"], extended["missed"]), (4, 2, 2)
        )
        self.assertEqual(set(extended["missed_error_regions"]), {"g2", "both-missing"})
        self.assertEqual(extended["recall"], 0.5)

    def test_gold_annotation_order_controls_joined_text_and_counts_region_once(self):
        a_path, b_path, a, b, report = self.pair(
            [word("left", [10, 10, 30, 20]), word("right", [100, 10, 130, 20])],
            [word("left", [10, 10, 30, 20]), word("right", [100, 10, 130, 20])],
        )
        gold = {
            "version": 1,
            "a_source": self.source_hash(a_path),
            "b_source": self.source_hash(b_path),
            "regions": [
                {
                    "id": "reverse",
                    "a": ["/0/1", "/0/0"],
                    "b": ["/0/1", "/0/0"],
                    "text": "right left",
                }
            ],
        }
        metrics = evaluate_comparison(report, a, b, gold)
        self.assertEqual(metrics["denominator"], 0)
        self.assertIsNone(metrics["recall"])
        _, _, a, b, report, gold = self.four_regions()
        gold["regions"] = [
            {
                "id": "one-error-region",
                "a": ["/0/0", "/0/1", "/0/2", "/0/3"],
                "b": ["/0/0", "/0/1", "/0/2", "/0/3"],
                "text": "100 80 OK 7",
            }
        ]
        metrics = evaluate_comparison(report, a, b, gold)
        self.assertEqual(
            (metrics["denominator"], metrics["caught"], metrics["missed"]), (1, 1, 0)
        )
        self.assertEqual(metrics["recall"], 1)

    def test_zero_error_and_empty_observations_have_null_recall(self):
        for words, regions in [
            (
                [word("OK", [10, 10, 30, 20])],
                [{"id": "ok", "a": ["/0/0"], "b": ["/0/0"], "text": "OK"}],
            ),
            ([], []),
        ]:
            with self.subTest(words=len(words)):
                a_path, b_path, a, b, report = self.pair(words, clone(words))
                self.assertEqual(report["status"], "compared")
                self.assertIs(report["complete"], True)
                self.assertEqual(report["queue"], [])
                gold = {
                    "version": 1,
                    "a_source": self.source_hash(a_path),
                    "b_source": self.source_hash(b_path),
                    "regions": regions,
                }
                metrics = evaluate_comparison(report, a, b, gold)
                self.assertEqual(
                    (metrics["denominator"], metrics["caught"], metrics["missed"]),
                    (0, 0, 0),
                )
                self.assertIsNone(metrics["recall"])

    def test_decisions_preserve_sources_and_correct_agreement_errors(self):
        a_path, b_path, a, b, report, gold = self.four_regions()
        groups = [
            self.group(report, a=[f"/0/{index}"], b=[f"/0/{index}"])
            for index in (0, 3, 1)
        ]
        original = clone(report)
        before_metrics = evaluate_comparison(report, a, b, gold)
        decisions = {
            "version": 1,
            "comparison_id": report["comparison_id"],
            "decisions": [
                {"group": groups[0]["id"], "outcome": "accept_a"},
                {"group": groups[1]["id"], "outcome": "accept_b"},
                {"group": groups[2]["id"], "outcome": "correct", "correction": "80"},
            ],
        }
        receipts = record_decisions(report, decisions)
        self.assertEqual([item["text"] for item in receipts], ["100", "7", "80"])
        self.assertEqual(
            [item["outcome"] for item in receipts], ["accept_a", "accept_b", "correct"]
        )
        self.assertEqual([item["text"] for item in receipts[2]["a"]], ["B0"])
        self.assertEqual([item["text"] for item in receipts[2]["b"]], ["B0"])
        for receipt, group in zip(receipts, groups):
            self.assertEqual(receipt["a"], group["a"])
            self.assertEqual(receipt["b"], group["b"])
            self.assertEqual(receipt["comparison_id"], report["comparison_id"])
            self.assertEqual(receipt["a_source"], self.source_hash(a_path))
            self.assertEqual(receipt["b_source"], self.source_hash(b_path))
            self.assertEqual(receipt["frame"], FRAME)
        self.assertEqual(report, original)
        self.assertEqual(evaluate_comparison(report, a, b, gold), before_metrics)
        receipts[2]["a"][0]["text"] = "changed receipt only"
        self.assertEqual(report, original)
        decisions["decisions"] = [
            {"group": groups[2]["id"], "outcome": "correct", "correction": ""}
        ]
        self.assertEqual(record_decisions(report, decisions)[0]["text"], "")

    def test_accepting_an_empty_unmatched_side_is_explicit_deletion(self):
        _, _, _, _, report = self.pair([], [word("extra", [10, 10, 30, 20])])
        group = self.group(report, a=[], b=["/0/0"])
        receipt = record_decisions(
            report,
            {
                "version": 1,
                "comparison_id": report["comparison_id"],
                "decisions": [{"group": group["id"], "outcome": "accept_a"}],
            },
        )[0]
        self.assertEqual(receipt["text"], "")
        self.assertEqual(receipt["a"], [])
        self.assertEqual(receipt["b"][0]["text"], "extra")

    def test_raw_source_bytes_and_threshold_bind_comparison_and_group_ids(self):
        value = document([word("same", [10, 10, 30, 20])])
        a_path = self.save(value)
        b_path = self.save(
            document([word("same", [10, 10, 30, 20])], engine="synthetic-B")
        )
        equivalent_path = self.save(
            raw=(json.dumps(value, indent=2) + "\n").encode("utf-8")
        )
        a, equivalent, b = (
            load_observation(a_path),
            load_observation(equivalent_path),
            load_observation(b_path),
        )
        first = compare_observations(a, b)
        reformatted = compare_observations(equivalent, b)
        threshold = compare_observations(a, b, iou_threshold=0.5)
        self.assertEqual(first["a_source"], self.source_hash(a_path))
        self.assertEqual(first["b_source"], self.source_hash(b_path))
        self.assertNotEqual(first["a_source"], reformatted["a_source"])
        self.assertNotEqual(first["comparison_id"], reformatted["comparison_id"])
        self.assertNotEqual(first["comparison_id"], threshold["comparison_id"])
        self.assertNotEqual(first["groups"][0]["id"], threshold["groups"][0]["id"])
        self.assertEqual(reformatted["queue"], [])
        self.assertEqual(threshold["queue"], [])

    def test_invalid_gold_rejects_partial_duplicate_or_stale_annotations(self):
        _, _, a, b, report, base = self.four_regions()
        cases = []
        for key, value in [
            ("version", True),
            ("version", 2),
            ("a_source", "0" * 64),
            ("b_source", "0" * 64),
            ("regions", "not a list"),
            ("extra", True),
        ]:
            gold = clone(base)
            gold[key] = value
            cases.append((key, gold))
        gold = clone(base)
        gold["regions"].pop()
        cases.append(("uncovered observed words", gold))
        gold = clone(base)
        gold["regions"][1]["id"] = "g1"
        cases.append(("duplicate region ID", gold))
        gold = clone(base)
        gold["regions"][0]["a"].append("/0/1")
        cases.append(("word reused across regions", gold))
        for key, value in [
            ("a", ["/0/0", "/0/0"]),
            ("a", ["/99/0"]),
            ("b", "/0/0"),
            ("id", ""),
            ("id", 1),
            ("text", None),
            ("text", 80),
            ("extra", True),
        ]:
            gold = clone(base)
            gold["regions"][0][key] = value
            cases.append((f"invalid region {key}={value!r}", gold))
        for label, gold in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    evaluate_comparison(report, a, b, gold)
        a_other_path = self.save(document([word("new source", [10, 10, 30, 20])]))
        with self.assertRaises(ValueError):
            evaluate_comparison(report, load_observation(a_other_path), b, base)

    def test_invalid_decisions_reject_stale_unknown_and_duplicate_targets(self):
        _, _, a, b, report, _ = self.four_regions()
        group = self.group(report, a=["/0/1"], b=["/0/1"])
        base = {
            "version": 1,
            "comparison_id": report["comparison_id"],
            "decisions": [
                {"group": group["id"], "outcome": "correct", "correction": "80"}
            ],
        }
        cases = []
        for key, value in [
            ("version", True),
            ("version", 2),
            ("comparison_id", "stale"),
            ("decisions", "not a list"),
            ("extra", True),
        ]:
            decisions = clone(base)
            decisions[key] = value
            cases.append((key, decisions))
        decisions = clone(base)
        decisions["decisions"].append(clone(decisions["decisions"][0]))
        cases.append(("duplicate target", decisions))
        for key, value in [
            ("group", "unknown"),
            ("outcome", "guess"),
            ("correction", None),
            ("correction", 80),
            ("extra", True),
            ("outcome", "accept_a"),
        ]:
            decisions = clone(base)
            decisions["decisions"][0][key] = value
            cases.append((f"invalid decision {key}={value!r}", decisions))
        decisions = clone(base)
        del decisions["decisions"][0]["correction"]
        cases.append(("correct without correction", decisions))
        other = compare_observations(a, b, iou_threshold=0.5)
        decisions = clone(base)
        decisions["decisions"][0]["group"] = self.group(other, a=["/0/1"], b=["/0/1"])[
            "id"
        ]
        cases.append(("group from another threshold", decisions))
        for label, decisions in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    record_decisions(report, decisions)

    def test_invalid_observation_schema_geometry_and_layout_are_rejected(self):
        base = document([word("one", [10, 10, 30, 20])])
        cases = []
        for key, value in [
            ("version", True),
            ("version", 2),
            ("engine", ""),
            ("engine", 1),
            ("tree", []),
            ("extra", True),
        ]:
            value_document = clone(base)
            value_document[key] = value
            cases.append((f"document {key}={value!r}", value_document))
        for key, value in [
            ("id", ""),
            ("page_number", -1),
            ("page_number", True),
            ("page_number", 0.5),
            ("width", 0),
            ("height", -1),
            ("width", True),
            ("extra", True),
        ]:
            value_document = clone(base)
            value_document["frame"][key] = value
            cases.append((f"frame {key}={value!r}", value_document))
        for key, value in [
            ("bbox", None),
            ("bbox", [10, 10, 10, 20]),
            ("bbox", [30, 10, 10, 20]),
            ("bbox", [-1, 10, 30, 20]),
            ("bbox", [10, 10, 301, 20]),
            ("bbox", [10, 10, 30]),
            ("bbox", [10, 10, 30, 20, 40]),
            ("bbox", [True, 10, 30, 20]),
            ("bbox", ["10", 10, 30, 20]),
            ("bbox", [10, 10, float("nan"), 20]),
            ("bbox", [10, 10, float("inf"), 20]),
            ("text", ""),
            ("text", None),
            ("text", 100),
            ("confidence", True),
            ("confidence", -0.1),
            ("confidence", 1.1),
            ("confidence", float("nan")),
            ("confidence", "0.9"),
            ("direction", "rtl"),
            ("textangle", 1),
            ("poly", [[10, 10], [30, 20]]),
            ("children", [word("nested", [12, 12, 18, 18])]),
            ("children", {}),
            ("ocr_class", 1),
            ("extra", True),
        ]:
            value_document = clone(base)
            value_document["tree"]["children"][0]["children"][0][key] = value
            cases.append((f"word {key}={value!r}", value_document))
        for missing in ("bbox", "text", "ocr_class"):
            value_document = clone(base)
            del value_document["tree"]["children"][0]["children"][0][missing]
            cases.append((f"missing word {missing}", value_document))
        value_document = clone(base)
        value_document["tree"]["children"][0]["bbox"] = [0, 0, 5, 5]
        cases.append(("word outside declared parent bbox", value_document))
        value_document = clone(base)
        del value_document["tree"]["bbox"]
        cases.append(("missing page bbox", value_document))
        for label, value_document in cases:
            with self.subTest(case=label):
                with self.assertRaises((ValueError, OSError)):
                    load_observation(self.save(value_document))

    def test_frame_mismatch_and_invalid_thresholds_fail_closed(self):
        a_path = self.save(document([word("same", [10, 10, 30, 20])]))
        a = load_observation(a_path)
        for key, value in [
            ("id", "another-page"),
            ("page_number", 1),
            ("width", 400),
            ("height", 300),
        ]:
            with self.subTest(frame_field=key):
                frame = clone(FRAME)
                frame[key] = value
                b_path = self.save(
                    document(
                        [word("same", [10, 10, 30, 20])],
                        engine="synthetic-B",
                        frame=frame,
                    )
                )
                with self.assertRaises(ValueError):
                    compare_observations(a, load_observation(b_path))
        for value in (0, -1, 1.1, True, None, "0.1", float("nan"), float("inf")):
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    compare_observations(a, a, iou_threshold=value)

    def test_word_limit_rejects_overflow_instead_of_comparing_prefix(self):
        path = self.save(
            document(
                [
                    word("one", [10, 10, 30, 20]),
                    word("two", [40, 10, 60, 20]),
                    word("three", [70, 10, 100, 20]),
                ]
            )
        )
        with self.assertRaises(ValueError):
            load_observation(path, max_words=2)

    def test_tiny_finite_boxes_keep_their_geometric_overlap(self):
        a_path, b_path, a, b, _ = self.pair(
            [word("tiny", [0, 0, 1e-200, 1e-200])],
            [word("tiny", [0, 0, 1e-200, 1e-200])],
        )
        identical = compare_observations(a, b, iou_threshold=1)
        self.assertEqual(len(identical["groups"]), 1)
        self.assertEqual(identical["groups"][0]["kind"], "agreement")
        self.assertEqual(identical["queue"], [])
        exit_code, result = self.run_cli(a_path, b_path, "--iou-threshold", "1")
        self.assertEqual(exit_code, 0)
        self.assertIs(result["complete"], True)
        # Both areas are the smallest positive subnormal float, while their
        # intersection would round to zero if products were computed in floats.
        # The mathematical overlap ratio of these exact dyadic boxes is 1/3.
        extent = 2.0**-537
        _, _, _, _, overlapping = self.pair(
            [word("tiny", [0, 0, extent, extent])],
            [word("tiny", [extent / 2, 0, 1.5 * extent, extent])],
        )
        self.assertEqual(len(overlapping["groups"]), 1)
        self.assertEqual(overlapping["groups"][0]["kind"], "agreement")
        self.assertEqual(overlapping["queue"], [])

    def test_deep_json_is_invalid_not_an_unexpected_cli_failure(self):
        deeply_nested = self.save(raw=b"[" * 2000 + b"0" + b"]" * 2000)
        with self.assertRaises(ValueError):
            load_observation(deeply_nested)
        valid = self.save(document([word("one", [10, 10, 30, 20])]))
        exit_code, result = self.run_cli(valid, deeply_nested)
        self.assertEqual(exit_code, 2)
        self.assertEqual(result["status"], "invalid")
        self.assertIs(result["complete"], False)

    def test_exact_limits_and_malformed_json_never_silently_truncate(self):
        path = self.save(
            document([word("\u00e9", [10, 10, 30, 20]), word("two", [40, 10, 60, 20])])
        )
        byte_count = len(path.read_bytes())
        observed = load_observation(path, max_words=2, max_bytes=byte_count)
        self.assertEqual([item.text for item in observed.words], ["\u00e9", "two"])
        with self.assertRaises(ValueError):
            load_observation(path, max_words=2, max_bytes=byte_count - 1)
        with self.assertRaises(ValueError):
            load_observation(path, max_words=1, max_bytes=byte_count)
        for limit in ("max_words", "max_bytes"):
            for value in (0, -1, True, 1.5, "100"):
                with self.subTest(limit=limit, value=value):
                    with self.assertRaises(ValueError):
                        load_observation(path, **{limit: value})
        for raw in (b'{"version":1,', b"[]", b"\xff", b'{"version":1,"version":1}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    load_observation(self.save(raw=raw))
        too_many_nodes = document([])
        too_many_nodes["tree"]["children"] = [
            {"ocr_class": "ocr_par"} for _ in range(500)
        ]
        with self.assertRaises(ValueError):
            load_observation(self.save(too_many_nodes))
        too_deep = document([])
        node = too_deep["tree"]
        for _ in range(10):
            child = {"ocr_class": "ocr_par", "children": []}
            node["children"] = [child]
            node = child
        with self.assertRaises(ValueError):
            load_observation(self.save(too_deep))

    def test_cli_complete_success_and_invalid_input_on_either_side(self):
        a_path, b_path, _, _, _, _ = self.four_regions()
        exit_code, report = self.run_cli(a_path, b_path)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "compared")
        self.assertIs(report["complete"], True)
        self.assertEqual(len(report["queue"]), 2)
        invalid = self.save(raw=b'{"version":1,')
        for before, after in [(invalid, b_path), (a_path, invalid)]:
            with self.subTest(invalid_side="a" if before == invalid else "b"):
                exit_code, report = self.run_cli(before, after)
                self.assertEqual(exit_code, 2)
                self.assertEqual(report["status"], "invalid")
                self.assertIs(report["complete"], False)
                self.assertTrue(report["error"])

    def test_cli_rejects_stale_gold_and_unknown_decisions(self):
        a_path, b_path, _, _, report, gold = self.four_regions()
        gold["a_source"] = "0" * 64
        stale_gold = self.save(gold)
        decisions = self.save(
            {
                "version": 1,
                "comparison_id": report["comparison_id"],
                "decisions": [
                    {"group": "unknown", "outcome": "correct", "correction": "80"}
                ],
            }
        )
        for arguments in [
            ("--gold", str(stale_gold)),
            ("--decisions", str(decisions)),
            ("--max-words", "1"),
        ]:
            with self.subTest(arguments=arguments):
                exit_code, result = self.run_cli(a_path, b_path, *arguments)
                self.assertEqual(exit_code, 2)
                self.assertEqual(result["status"], "invalid")
                self.assertIs(result["complete"], False)

    def test_cli_unexpected_native_construction_error_is_incomplete_exit_three(self):
        a_path = self.save(document([word("one", [10, 10, 30, 20])]))
        b_path = self.save(
            document([word("one", [10, 10, 30, 20])], engine="synthetic-B")
        )
        with patch.object(
            ocr_review_module,
            "OcrElement",
            side_effect=RuntimeError("injected native failure"),
        ) as construct:
            exit_code, result = self.run_cli(a_path, b_path)
        construct.assert_called_once()
        self.assertEqual(exit_code, 3)
        self.assertEqual(result["status"], "error")
        self.assertIs(result["complete"], False)
        self.assertEqual(result["error"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()

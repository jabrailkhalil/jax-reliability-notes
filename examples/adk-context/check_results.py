"""Validate the lab's measured observations, including wrong-revision controls."""

import json
import sys
from pathlib import Path


def check(folder: Path) -> None:
    failures = {
        "empty_foreign_part_does_not_cut_out_user",
        "foreign_signature_does_not_cut_out_user",
        "foreign_metadata_does_not_cut_out_user",
        "inline_payload_keeps_attached_signature",
    }
    user_retention = failures - {"inline_payload_keeps_attached_signature"}
    for mode in ("baseline", "fixed"):
        report = json.loads((folder / f"adk-{mode}.json").read_text())
        wrong = json.loads((folder / f"adk-{mode}-wrong-expectation.json").read_text())
        assert report["expected"] == mode
        assert report["expected_behavior_matches"] == 6
        assert report["fixed_contract_matches"] == (2 if mode == "baseline" else 6)
        assert len(report["cases"]) == len(wrong["cases"]) == 6
        assert wrong["expected_behavior_matches"] == 2
        assert wrong["expected"] == ("fixed" if mode == "baseline" else "baseline")
        assert {
            c["name"] for c in wrong["cases"] if not c["expected_behavior_matched"]
        } == failures
        assert len({c["name"] for c in report["cases"]}) == 6
        for case, wrong_case in zip(report["cases"], wrong["cases"], strict=True):
            name = case["name"]
            assert case["expected_behavior_matched"]
            assert case["fixed_contract_matched"] == (mode == "fixed" or name not in failures)
            assert case.get("error") == (
                "outbound contents mismatch" if mode == "baseline" and name in failures else None
            )
            observation = case["observation"]
            assert observation == wrong_case["observation"]
            assert observation["producer_calls"] == observation["reviewer_calls"] == 1
            assert observation["stored_events_unchanged"]
            assert observation["stored_events"] == (4 if name == "own_agent_thought_unchanged" else 3)
            if name in user_retention:
                assert observation["current_user_present"] == (mode == "fixed")
            else:
                assert observation["current_user_present"]
    print("Both exact-revision observations and both wrong-expectation controls match.")


if __name__ == "__main__":
    check(Path(sys.argv[1]))

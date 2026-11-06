"""Compare bounded positional OCR observations and retain explicit review choices."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from ocrmypdf import BoundingBox, OcrClass, OcrElement

_NODE_FIELDS = {
    "ocr_class",
    "bbox",
    "text",
    "confidence",
    "children",
    "direction",
    "textangle",
    "poly",
}
_CONTAINERS = {
    "ocr_carea",
    "ocr_par",
    "ocr_line",
    "ocr_header",
    "ocr_footer",
    "ocr_caption",
    "ocr_textfloat",
}
_LINES = {"ocr_line", "ocr_header", "ocr_footer", "ocr_caption", "ocr_textfloat"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: Any, required: set[str], allowed: set[str]) -> None:
    _require(
        isinstance(value, dict) and required <= value.keys() <= allowed,
        "invalid object fields",
    )


def _number(value: Any) -> float:
    _require(type(value) in (int, float), "coordinate/value must be numeric")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("non-finite coordinate/value") from error
    _require(math.isfinite(number), "non-finite coordinate/value")
    return number


def _digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON member")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the supported depth") from error


def read_json(path: Path, *, max_bytes: int = 1024 * 1024) -> Any:
    _require(type(max_bytes) is int and max_bytes > 0, "byte limit must be positive")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    _require(len(raw) <= max_bytes, "byte limit exceeded")
    return _parse_json(raw)


@dataclass(frozen=True, slots=True)
class Frame:
    id: str
    page_number: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Word:
    path: str
    line: str | None
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


@dataclass(frozen=True, slots=True)
class Observation:
    source_id: str
    engine: str
    frame: Frame
    words: tuple[Word, ...]


def _box(value: Any, parent: tuple[float, float, float, float]) -> BoundingBox:
    _require(
        isinstance(value, list) and len(value) == 4, "bbox must have four coordinates"
    )
    left, top, right, bottom = [_number(v) for v in value]
    _require(left < right and top < bottom, "bbox must have positive area")
    _require(
        parent[0] <= left < right <= parent[2]
        and parent[1] <= top < bottom <= parent[3],
        "bbox outside parent/frame",
    )
    return BoundingBox(left, top, right, bottom)


def load_observation(
    path: Path, *, max_words: int = 100, max_bytes: int = 1024 * 1024
) -> Observation:
    """Construct actual upstream models after validating the supported raw tree."""
    _require(type(max_words) is int and max_words > 0, "word limit must be positive")
    _require(type(max_bytes) is int and max_bytes > 0, "byte limit must be positive")
    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    _require(len(raw) <= max_bytes, "byte limit exceeded")
    document = _parse_json(raw)
    _fields(
        document,
        {"version", "engine", "frame", "tree"},
        {"version", "engine", "frame", "tree"},
    )
    _require(
        type(document["version"]) is int and document["version"] == 1,
        "unsupported observation version",
    )
    _require(
        isinstance(document["engine"], str) and bool(document["engine"]),
        "engine must be nonempty",
    )
    envelope = document["frame"]
    _fields(
        envelope,
        {"id", "page_number", "width", "height"},
        {"id", "page_number", "width", "height"},
    )
    _require(
        isinstance(envelope["id"], str) and bool(envelope["id"]),
        "frame id must be nonempty",
    )
    _require(
        type(envelope["page_number"]) is int and envelope["page_number"] >= 0,
        "invalid physical page number",
    )
    for dimension in ("width", "height"):
        _require(
            type(envelope[dimension]) is int and 0 < envelope[dimension] <= 1_000_000,
            "invalid canvas dimensions",
        )
    frame = Frame(**envelope)
    canvas = (0.0, 0.0, float(frame.width), float(frame.height))
    paths: dict[int, tuple[str, str | None]] = {}
    nodes = 0
    words = 0

    def construct(
        value: Any,
        location: str,
        line: str | None,
        parent: tuple[float, float, float, float],
        depth: int,
    ) -> OcrElement:
        nonlocal nodes, words
        nodes += 1
        _require(nodes <= 500 and depth <= 8, "tree node/depth limit exceeded")
        _fields(value, {"ocr_class"}, _NODE_FIELDS)
        kind = value["ocr_class"]
        _require(isinstance(kind, str), "invalid OCR class")
        if depth == 0:
            _require(kind == OcrClass.PAGE, "tree root must be ocr_page")
        else:
            _require(
                kind in _CONTAINERS or kind == OcrClass.WORD, "unsupported OCR class"
            )
        _require(
            value.get("direction") in (None, "ltr"),
            "only LTR observations are supported",
        )
        angle = value.get("textangle")
        _require(angle is None or _number(angle) == 0, "rotated text is unsupported")
        _require(value.get("poly") is None, "polygon geometry is unsupported")
        text = value.get("text", "")
        _require(isinstance(text, str), "text must be a string")
        confidence = value.get("confidence")
        if confidence is not None:
            confidence = _number(confidence)
            _require(0 <= confidence <= 1, "confidence must be within 0..1")
        children = value.get("children", [])
        _require(isinstance(children, list), "children must be an array")
        bbox = _box(value["bbox"], parent) if value.get("bbox") is not None else None
        if depth == 0:
            _require(
                bbox is not None
                and (bbox.left, bbox.top, bbox.right, bbox.bottom) == canvas,
                "page bbox must match the canvas",
            )
        if kind == OcrClass.WORD:
            words += 1
            _require(words <= max_words, "word limit exceeded")
            _require(
                bool(text) and bbox is not None and not children,
                "word requires text/bbox and no children",
            )
        if kind in _LINES:
            line = location
        bounds = (bbox.left, bbox.top, bbox.right, bbox.bottom) if bbox else parent
        native = OcrElement(
            ocr_class=kind,
            bbox=bbox,
            text=text,
            confidence=confidence,
            direction=value.get("direction"),
            textangle=angle,
            page_number=frame.page_number if depth == 0 else None,
            children=[
                construct(child, f"{location}/{i}", line, bounds, depth + 1)
                for i, child in enumerate(children)
            ],
        )
        paths[id(native)] = (location, line)
        return native

    root = construct(document["tree"], "", None, canvas, 0)
    projected = []
    for native in root.words:
        location, line = paths[id(native)]
        bbox = native.bbox
        _require(bbox is not None, "native word lost its bbox")
        projected.append(
            Word(
                location,
                line,
                native.text,
                (bbox.left, bbox.top, bbox.right, bbox.bottom),
                native.confidence,
            )
        )
    return Observation(
        hashlib.sha256(raw).hexdigest(), document["engine"], frame, tuple(projected)
    )


def _iou(a: Word, b: Word) -> float:
    # Bounded inputs permit exact products even for subnormal box dimensions.
    first = tuple(Fraction(coordinate) for coordinate in a.bbox)
    second = tuple(Fraction(coordinate) for coordinate in b.bbox)
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(Fraction(0), right - left) * max(Fraction(0), bottom - top)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return float(intersection / (area_a + area_b - intersection))


def _word_dict(word: Word) -> dict[str, Any]:
    return {**asdict(word), "bbox": list(word.bbox)}


def compare_observations(
    a: Observation, b: Observation, *, iou_threshold: float = 0.1
) -> dict[str, Any]:
    """Use spatial components; segmentation/ambiguous groups remain reviewable."""
    threshold = _number(iou_threshold)
    _require(0 < threshold <= 1, "IoU threshold must be within 0..1")
    _require(
        a.frame == b.frame, "observations must share the declared frame/page/canvas"
    )
    comparison_id = _digest(
        {
            "a": a.source_id,
            "b": b.source_id,
            "iou": threshold,
            "algorithm": "spatial-components-v1",
        }
    )
    vertices = [("a", i) for i in range(len(a.words))] + [
        ("b", i) for i in range(len(b.words))
    ]
    neighbors: dict[tuple[str, int], set[tuple[str, int]]] = {
        vertex: set() for vertex in vertices
    }
    for i, old_word in enumerate(a.words):
        for j, new_word in enumerate(b.words):
            if _iou(old_word, new_word) >= threshold:
                neighbors[("a", i)].add(("b", j))
                neighbors[("b", j)].add(("a", i))
    seen: set[tuple[str, int]] = set()
    groups: list[dict[str, Any]] = []
    for vertex in vertices:
        if vertex in seen:
            continue
        pending = [vertex]
        component: set[tuple[str, int]] = set()
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component.add(item)
            pending.extend(neighbors[item] - component)
        seen.update(component)
        selected = {
            side: sorted(
                [observation.words[index] for s, index in component if s == side],
                key=lambda w: (w.bbox[1], w.bbox[0], w.path),
            )
            for side, observation in (("a", a), ("b", b))
        }
        old, new = selected["a"], selected["b"]
        if not old or not new:
            kind = "unmatched"
        elif len(old) == len(new) == 1:
            kind = "agreement" if old[0].text == new[0].text else "text"
        elif len(old) > 1 and len(new) > 1:
            kind = "ambiguous"
        else:
            kind = "segmentation"
        group_id = _digest(
            {
                "comparison": comparison_id,
                "a": sorted(w.path for w in old),
                "b": sorted(w.path for w in new),
            }
        )
        groups.append(
            {
                "id": group_id,
                "kind": kind,
                "needs_review": kind != "agreement",
                "a": [_word_dict(w) for w in old],
                "b": [_word_dict(w) for w in new],
            }
        )
    groups.sort(key=lambda group: group["id"])
    return {
        "status": "compared",
        "complete": True,
        "comparison_id": comparison_id,
        "a_source": a.source_id,
        "b_source": b.source_id,
        "frame": asdict(a.frame),
        "groups": groups,
        "queue": [group["id"] for group in groups if group["needs_review"]],
    }


def evaluate_comparison(
    report: dict[str, Any], a: Observation, b: Observation, gold: dict[str, Any]
) -> dict[str, Any]:
    """Measure queue recall on ALL independently annotated exact-text regions."""
    _fields(
        gold,
        {"version", "a_source", "b_source", "regions"},
        {"version", "a_source", "b_source", "regions"},
    )
    _require(
        type(gold["version"]) is int and gold["version"] == 1,
        "unsupported gold version",
    )
    _require(
        gold["a_source"] == a.source_id == report["a_source"]
        and gold["b_source"] == b.source_id == report["b_source"],
        "gold/comparison source mismatch",
    )
    _require(isinstance(gold["regions"], list), "gold regions must be an array")
    by_path = {"a": {w.path: w for w in a.words}, "b": {w.path: w for w in b.words}}
    covered: dict[str, set[str]] = {"a": set(), "b": set()}
    identifiers: set[str] = set()
    errors: dict[str, set[str]] = {"a": set(), "b": set()}
    queued_words = {
        (side, word["path"])
        for group in report["groups"]
        if group["id"] in report["queue"]
        for side in ("a", "b")
        for word in group[side]
    }
    caught: set[str] = set()
    for region in gold["regions"]:
        _fields(region, {"id", "a", "b", "text"}, {"id", "a", "b", "text"})
        name = region["id"]
        _require(
            isinstance(name, str) and bool(name) and name not in identifiers,
            "invalid/duplicate gold region id",
        )
        _require(isinstance(region["text"], str), "gold text must be a string")
        identifiers.add(name)
        has_queued_word = False
        for side in ("a", "b"):
            refs = region[side]
            _require(
                isinstance(refs, list) and all(isinstance(ref, str) for ref in refs),
                "gold refs must be arrays of paths",
            )
            texts = []
            for ref in refs:
                _require(
                    ref in by_path[side] and ref not in covered[side],
                    "unknown/duplicate gold word path",
                )
                covered[side].add(ref)
                texts.append(by_path[side][ref].text)
                has_queued_word |= (side, ref) in queued_words
            if " ".join(texts) != region["text"]:
                errors[side].add(name)
        if has_queued_word and (name in errors["a"] or name in errors["b"]):
            caught.add(name)
    _require(
        all(covered[side] == set(by_path[side]) for side in ("a", "b")),
        "gold must cover every observed word exactly once",
    )
    error_regions = errors["a"] | errors["b"]
    missed = error_regions - caught
    return {
        "unit": "annotated exact-text region",
        "regions": len(identifiers),
        "a_error_regions": sorted(errors["a"]),
        "b_error_regions": sorted(errors["b"]),
        "error_regions": sorted(error_regions),
        "caught_error_regions": sorted(caught),
        "missed_error_regions": sorted(missed),
        "denominator": len(error_regions),
        "caught": len(caught),
        "missed": len(missed),
        "recall": len(caught) / len(error_regions) if error_regions else None,
    }


def record_decisions(
    report: dict[str, Any], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    """Keep both raw readings even for agreement-region corrections."""
    _fields(
        decisions,
        {"version", "comparison_id", "decisions"},
        {"version", "comparison_id", "decisions"},
    )
    _require(
        type(decisions["version"]) is int and decisions["version"] == 1,
        "unsupported decision version",
    )
    _require(
        decisions["comparison_id"] == report["comparison_id"],
        "stale comparison decision",
    )
    _require(isinstance(decisions["decisions"], list), "decisions must be an array")
    groups = {group["id"]: group for group in report["groups"]}
    seen: set[str] = set()
    result = []
    for decision in decisions["decisions"]:
        _fields(decision, {"group", "outcome"}, {"group", "outcome", "correction"})
        identifier = decision["group"]
        _require(
            isinstance(identifier, str)
            and identifier in groups
            and identifier not in seen,
            "unknown/duplicate decision group",
        )
        outcome = decision["outcome"]
        _require(
            outcome in ("accept_a", "accept_b", "correct"), "invalid review outcome"
        )
        if outcome == "correct":
            _require(
                isinstance(decision.get("correction"), str),
                "correction must be an explicit string",
            )
            text = decision["correction"]
        else:
            _require(
                "correction" not in decision, "acceptance cannot contain a correction"
            )
            side = "a" if outcome == "accept_a" else "b"
            text = " ".join(word["text"] for word in groups[identifier][side])
        seen.add(identifier)
        group = groups[identifier]
        result.append(
            {
                "group": identifier,
                "outcome": outcome,
                "text": text,
                "a": deepcopy(group["a"]),
                "b": deepcopy(group["b"]),
                "frame": deepcopy(report["frame"]),
                "comparison_id": report["comparison_id"],
                "a_source": report["a_source"],
                "b_source": report["b_source"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-words", type=int, default=100)
    args = parser.parse_args()
    try:
        a = load_observation(args.a, max_words=args.max_words)
        b = load_observation(args.b, max_words=args.max_words)
        report = compare_observations(a, b, iou_threshold=args.iou_threshold)
        report["evaluation"] = (
            evaluate_comparison(report, a, b, read_json(args.gold))
            if args.gold
            else None
        )
        report["decisions"] = (
            record_decisions(report, read_json(args.decisions))
            if args.decisions
            else []
        )
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "invalid", "complete": False, "error": str(error)}))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "complete": False, "error": type(error).__name__}
            )
        )
        return 3
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

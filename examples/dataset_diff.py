"""Compare bounded local dataset snapshots by explicit keys and declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import ClassLabel, Dataset, Features

_COLUMNS = {"id", "text", "label"}
_METADATA = {"features.json", "config.json"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(raw: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=invalid_constant,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and bool(name), "invalid file path")
    relative = PurePosixPath(name)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "\\" not in name
        and ":" not in name,
        "file path must stay within the snapshot",
    )
    path = (root / name).resolve()
    _require(path.is_relative_to(root), "file path escapes the snapshot")
    _require(path.is_file(), "snapshot file is missing")
    return path


@dataclass(frozen=True, slots=True)
class Row:
    key: str
    text: str
    label: int
    label_name: str | None


@dataclass(frozen=True, slots=True)
class Snapshot:
    snapshot_id: str
    column_order: tuple[str, ...]
    features_json: str
    config_json: str
    file_hashes: tuple[tuple[str, str], ...]
    data_files: tuple[str, ...]
    rows: tuple[Row, ...]


def load_snapshot(
    directory: Path, *, max_rows: int = 1000, max_bytes: int = 1024 * 1024
) -> Snapshot:
    """Validate all input before constructing a native Hugging Face Dataset."""
    _require(type(max_rows) is int and max_rows > 0, "row limit must be positive")
    _require(type(max_bytes) is int and max_bytes > 0, "byte limit must be positive")
    root = Path(directory).resolve()
    with _file(root, "manifest.json").open("rb") as stream:
        manifest_bytes = stream.read(max_bytes + 1)
    _require(len(manifest_bytes) <= max_bytes, "byte limit exceeded")
    manifest = _json(manifest_bytes)
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(
        set(manifest) == {"version", "key", "column_order", "data_files", "files"},
        "invalid manifest fields",
    )
    _require(
        type(manifest["version"]) is int and manifest["version"] == 1,
        "unsupported manifest version",
    )
    _require(manifest["key"] == "id", "this lab requires the explicit key id")
    columns = manifest["column_order"]
    _require(
        isinstance(columns, list)
        and all(isinstance(c, str) for c in columns)
        and len(columns) == 3
        and set(columns) == _COLUMNS,
        "invalid declared column order",
    )
    data_files = manifest["data_files"]
    _require(
        isinstance(data_files, list)
        and all(isinstance(f, str) for f in data_files)
        and len(data_files) == len(set(data_files))
        and not (set(data_files) & _METADATA),
        "invalid data file list",
    )
    hashes = manifest["files"]
    _require(
        isinstance(hashes, dict) and set(hashes) == _METADATA | set(data_files),
        "manifest must seal all data and metadata files",
    )
    contents: dict[str, bytes] = {}
    total_bytes = len(manifest_bytes)
    for name, digest in hashes.items():
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "invalid SHA256 digest",
        )
        with _file(root, name).open("rb") as stream:
            raw = stream.read(max_bytes - total_bytes + 1)
        total_bytes += len(raw)
        _require(total_bytes <= max_bytes, "byte limit exceeded")
        _require(hashlib.sha256(raw).hexdigest() == digest, "file digest mismatch")
        contents[name] = raw

    feature_document = _json(contents["features.json"])
    _require(
        isinstance(feature_document, dict) and set(feature_document) == _COLUMNS,
        "unsupported feature columns",
    )
    for column in ("id", "text"):
        _require(
            feature_document[column] == {"_type": "Value", "dtype": "string"},
            "id and text must use Value(string)",
        )
    label = feature_document["label"]
    _require(
        isinstance(label, dict)
        and set(label) == {"_type", "names"}
        and label["_type"] == "ClassLabel",
        "label must use explicit ClassLabel names",
    )
    names = label["names"]
    _require(
        isinstance(names, list)
        and bool(names)
        and all(isinstance(n, str) and bool(n) for n in names)
        and len(names) == len(set(names)),
        "invalid ClassLabel names",
    )
    config = _json(contents["config.json"])
    _require(
        isinstance(config, dict)
        and set(config) == {"name", "split"}
        and all(isinstance(v, str) and bool(v) for v in config.values()),
        "config must declare nonempty name and split strings",
    )

    raw_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in data_files:
        for line in contents[name].splitlines():
            if not line.strip():
                continue
            row = _json(line)
            _require(isinstance(row, dict), "row must be an object")
            _require("id" in row, "missing row key id")
            key = row["id"]
            _require(isinstance(key, str) and bool(key), "id must be a nonempty string")
            _require(key not in seen, "duplicate row key id")
            _require(set(row) == _COLUMNS, "row fields do not match the schema")
            _require(isinstance(row["text"], str), "text must be a string")
            _require(
                type(row["label"]) is int and -1 <= row["label"] < len(names),
                "invalid ClassLabel code",
            )
            _require(len(raw_rows) < max_rows, "row limit exceeded")
            seen.add(key)
            raw_rows.append(row)

    features = Features.from_dict({c: feature_document[c] for c in columns})
    features_json = _canonical(features.to_dict())
    dataset = Dataset.from_dict(
        {c: [row[c] for row in raw_rows] for c in columns}, features=features
    )
    _require(dataset.column_names == columns, "native column order changed")
    class_label = features["label"]
    _require(isinstance(class_label, ClassLabel), "native label feature changed")
    rows = tuple(
        Row(
            key=row["id"],
            text=row["text"],
            label=row["label"],
            label_name=None
            if row["label"] == -1
            else class_label.int2str(row["label"]),
        )
        for row in dataset
    )
    return Snapshot(
        snapshot_id=hashlib.sha256(_canonical(manifest).encode()).hexdigest(),
        column_order=tuple(columns),
        features_json=features_json,
        config_json=_canonical(config),
        file_hashes=tuple(sorted(hashes.items())),
        data_files=tuple(data_files),
        rows=rows,
    )


def _meaning(row: Row) -> dict[str, str]:
    if row.label_name is None:
        return {"kind": "unknown"}
    return {"kind": "class", "name": row.label_name}


def compare_snapshots(before: Snapshot, after: Snapshot) -> dict[str, Any]:
    """Compare complete validated snapshots; file hashes never bypass row checks."""
    old = {row.key: row for row in before.rows}
    new = {row.key: row for row in after.rows}
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    changed = []
    meanings = []
    for key in sorted(old.keys() & new.keys()):
        fields = [
            f for f in ("label", "text") if getattr(old[key], f) != getattr(new[key], f)
        ]
        if fields:
            changed.append({"key": key, "fields": fields})
        if old[key].label_name != new[key].label_name:
            meanings.append(
                {"key": key, "before": _meaning(old[key]), "after": _meaning(new[key])}
            )
    features_changed = before.features_json != after.features_json
    order_changed = before.column_order != after.column_order
    config_changed = before.config_json != after.config_json
    different = bool(
        added
        or removed
        or changed
        or meanings
        or features_changed
        or order_changed
        or config_changed
    )
    old_files = dict(before.file_hashes)
    new_files = dict(after.file_hashes)
    return {
        "status": "different" if different else "equal",
        "complete": True,
        "source_kind": "local-content-sha256",
        "before": before.snapshot_id,
        "after": after.snapshot_id,
        "file_hashes": {"before": old_files, "after": new_files},
        "row_counts": {"before": len(old), "after": len(new)},
        "data_files_equal": {f: old_files[f] for f in before.data_files}
        == {f: new_files[f] for f in after.data_files},
        "features_changed": features_changed,
        "column_order_changed": order_changed,
        "config_changed": config_changed,
        "added": added,
        "removed": removed,
        "changed": changed,
        "label_meaning_changes": meanings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    try:
        limits = {"max_rows": args.max_rows, "max_bytes": args.max_bytes}
        before = load_snapshot(args.before, **limits)
        after = load_snapshot(args.after, **limits)
        report = compare_snapshots(before, after)
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

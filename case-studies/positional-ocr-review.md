# A disagreement queue still misses shared OCR errors

I built a small positional comparison lab after [OCRmyPDF's suggestion to use `OcrElement` trees in a separate application](https://github.com/ocrmypdf/OCRmyPDF/discussions/1777#discussioncomment-19380096). I consume the public models from the published OCRmyPDF 17.13.0 package. The inputs are inspectable synthetic trees; I do not run OCR or a PDF pipeline in this lab.

The distinction I want to make visible is between **finding disagreements** and **finding errors**. Two readings can disagree because one is wrong, both are wrong, or their segmentation differs. They can also agree on the same wrong answer. An agreement should remain available for independent audit.

## Run the example

From the repository root, use a fresh Python 3.11+ environment:

```sh
python -m venv .venv-ocr-review
```

Activate it with `source .venv-ocr-review/bin/activate` on Linux/macOS or `.\.venv-ocr-review\Scripts\Activate.ps1` in PowerShell, then:

```sh
python -m pip install -r requirements-ocr-review.txt
python examples/ocr_review.py examples/ocr-review-fixtures/regions-a.json examples/ocr-review-fixtures/regions-b.json --gold examples/ocr-review-fixtures/regions-gold.json --decisions examples/ocr-review-fixtures/regions-decisions.json
```

The four annotated regions have literal synthetic references, independent of the comparator:

| Region | Reading A | Reading B | Reference | Automatic disagreement queue |
| --- | --- | --- | --- | --- |
| g1 | `100` | `101` | `100` | Included |
| g2 | `B0` | `B0` | `80` | Omitted |
| g3 | `OK` | `OK` | `OK` | Omitted |
| g4 | `?` | `7` | `7` | Included |

There are three regions where at least one reading is wrong. The queue catches two and misses g2, so its exact-text error-region recall is **2/3**. Counting only queued regions would incorrectly report 2/2. This is a deliberately small oracle, not an OCR accuracy benchmark or an estimate for real documents.

The decision fixture accepts A for g1, accepts B for g4, and explicitly enters `80` for the agreed-but-wrong g2. Each receipt retains both raw readings, bounding boxes, tree paths, the selected text and source/comparison IDs. Recording the correction does not change the original queue or retroactively count g2 as detected. I write no corrected PDF and do not overwrite the source observations.

## Position and segmentation

I validate a common declared frame: physical page number, frame ID, canvas dimensions and top-left pixel coordinates. OCRmyPDF's [published `BoundingBox`/`OcrElement` model](https://github.com/ocrmypdf/OCRmyPDF/blob/1a1cfb595b9a504a4689b23193a76bae7c3a156d/src/ocrmypdf/models/ocr_element.py) supplies the native objects. I project actual word descendants and retain their nearest line and tree paths; aggregate parent text does not become another word.

I connect words from opposite readings when their bounding-box IoU is at least 0.1, then inspect each connected component. The threshold is an explicit fixture parameter, not a value tuned on real OCR. A one-to-one exact-text match is an agreement. Different text, unmatched words, split/merged segmentation and ambiguous many-to-many groups enter the queue. I preserve the raw per-side word lists without guessing a winner or joining away segmentation differences.

The other fixtures exercise these boundaries:

- `shifted-*` retain the same sidecar text order while moving words to other positions.
- `disjoint-*` contain identical text in boxes with no overlap.
- `split-*` align one word against two pieces and retain a segmentation review item.
- `ambiguous-*` retain an overlapping many-to-many group for review.
- `empty-*` have no errors; recall is `null` with a zero denominator.
- Wrong-frame and stale-reference inputs fail instead of producing a successful empty report.

A tree path identifies an element within one supplied observation. It is not a stable ID across OCR runs. Source IDs hash the complete raw input bytes, and comparison IDs also bind the algorithm and IoU threshold. Gold annotations explicitly associate source word paths with independently supplied reference regions, including regions missed by both readings. Each observed word must be covered exactly once. This keeps the metric's denominator separate from the comparator's alignment decisions.

## Validation and scope

Run the complete oracle file, CLI fixtures and deliberately broken variants with:

```sh
python examples/ocr_review_validation.py --output .lab/ocr-review/results.json
```

The runner records the installed package version and verifies the exact upstream model-file SHA256 before running the lab. The [workflow](../.github/workflows/ocr-review.yml) uses Linux Python 3.11/3.14 and Windows Python 3.12.10. It installs the actual package through pip rather than substituting model classes.

This first slice accepts one declared common frame, LTR axis-aligned word geometry and a bounded tree. The defaults are 100 words, 500 nodes, depth 8 and 1 MiB per observation; canvas dimensions are integral pixels from 1 to 1,000,000. Non-finite/zero-area/out-of-bounds boxes, unsupported rotations/polygons/directions, stale decisions, incomplete gold coverage and budget overflow fail validation. I calculate IoU area products with exact rational arithmetic before converting the ratio to a float, so tiny valid boxes cannot silently lose an overlap through area underflow. After dependencies load, the CLI returns 2 for invalid input and 3 for an unexpected runtime error, with `complete: false`. A successful full comparison returns 0; `complete` describes the comparison inputs, not the correctness of the OCR or completion of human review.

The frame ID is a caller declaration. I do not validate that two real image rasters share that identity or perform registration, deskewing, DPI conversion, PDF rendering, live OCR, browser interaction or real human review. Text scoring uses exact strings joined in the annotation's explicit path order with single spaces; it does not measure layout accuracy, CER or WER. I make no general claim that this queue improves recognition quality or is ready for every document layout. This is self-study material and an initial comparison contract for review.

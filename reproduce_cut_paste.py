#!/usr/bin/env python3
"""Standalone reuse-aware cut-paste reproduction from manifest U2Net masks."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from appendix_data import (
    ConcatItem,
    concat_source_manifest,
    load_concat_items,
    output_dir,
    public_source_ref,
    sha256_file,
    work_dir,
)


WORK = work_dir("cut_paste")
OUT = output_dir("cut_paste")

SEED = 20260527
EXPORT_H = 680
EXPORT_W = 340
CLASS_COUNT = 136
NOMINAL_ORDERED_NON_SELF_COUNT = CLASS_COUNT * (CLASS_COUNT - 1)
THESIS_DETECTOR_COUNT = 18360
SOURCE_POOL_FRACTION = 0.10
SOURCE_POOL_MIN_ITEMS = 1
MAX_PASTE_COVERAGE = 0.70
MIN_PASTE_COVERAGE = 0.01
MIN_MASK_AREA = 1e-6
MASK_THRESHOLD = 127
SOFT_EDGE_PX = 3
SOFT_EDGE_GAUSS = 3.0
MAX_USAGE_DELTA = 8
CANDIDATE_BEAM = 24
SOFT_VALIDATION_CANDIDATES = 64

CUTPASTE_TRAINING_HISTORY = [
    {
        "stage": "shared_u2net_masks",
        "role": "Consume the manifest-provided U2Net foreground masks used by the direct-composition sources.",
    },
    {
        "stage": "reusecap10_source_pool",
        "role": "Select a deterministic 10% per-class source pool, with at least one source retained per class.",
        "source_pool_fraction": SOURCE_POOL_FRACTION,
        "source_pool_min_items": SOURCE_POOL_MIN_ITEMS,
    },
    {
        "stage": "cut_paste_export",
        "role": "Attempt the full ordered non-self class-pair grid with reuse-aware source selection and direction search.",
        "nominal_ordered_nonself_count": NOMINAL_ORDERED_NON_SELF_COUNT,
        "thesis_detector_count": THESIS_DETECTOR_COUNT,
    },
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def select_pool(items: list[ConcatItem], class_id: int) -> list[ConcatItem]:
    limit = max(SOURCE_POOL_MIN_ITEMS, int(math.floor(SOURCE_POOL_FRACTION * len(items))))
    limit = min(len(items), max(1, limit))
    rng = random.Random(f"{SEED}:{int(class_id)}:cutpaste_source_pool")
    selected = list(items)
    rng.shuffle(selected)
    return selected[:limit]


def load_mask(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.shape[:2] != (EXPORT_H, EXPORT_W):
        raise ValueError(f"Mask must be {EXPORT_H}x{EXPORT_W}: {path} shape={arr.shape}")
    return arr >= MASK_THRESHOLD


def soften_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() == 0 or m.min() == 1:
        return m.astype(np.float32)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    soft = m.astype(np.float32)
    if SOFT_EDGE_PX > 0:
        edge_band = (dist_in < SOFT_EDGE_PX) & (m == 1)
        soft[edge_band] = dist_in[edge_band] / float(SOFT_EDGE_PX)
        outside_band = (dist_out < SOFT_EDGE_PX) & (m == 0)
        soft[outside_band] = 1.0 - (dist_out[outside_band] / float(SOFT_EDGE_PX))
    soft = np.clip(soft, 0.0, 1.0)
    if SOFT_EDGE_GAUSS > 0:
        k = int(2 * round(SOFT_EDGE_GAUSS * 2) + 1)
        soft = cv2.GaussianBlur(soft, (k, k), SOFT_EDGE_GAUSS)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def soft_mask(path: str) -> np.ndarray:
    return soften_mask(load_mask(path))


def overlap_bbox(target: ConcatItem, donor: ConcatItem) -> tuple[int, int, int, int] | None:
    y0 = max(int(target.y0), int(donor.y0))
    y1 = min(int(target.y1), int(donor.y1))
    x0 = max(int(target.x0), int(donor.x0))
    x1 = min(int(target.x1), int(donor.x1))
    if y1 <= y0 or x1 <= x0:
        return None
    if ((y1 - y0) * (x1 - x0)) / float(target.area) < MIN_PASTE_COVERAGE:
        return None
    return y0, y1, x0, x1


def binary_coverage(target: ConcatItem, donor: ConcatItem) -> float | None:
    if target.sha256 == donor.sha256:
        return None
    total_pixels = EXPORT_H * EXPORT_W
    if float(donor.area) / float(total_pixels) <= MIN_MASK_AREA:
        return None
    overlap = overlap_bbox(target, donor)
    if overlap is None:
        return None
    y0, y1, x0, x1 = overlap
    target_mask = load_mask(str(target.mask_path))
    donor_mask = load_mask(str(donor.mask_path))
    coverage = float(np.logical_and(target_mask[y0:y1, x0:x1], donor_mask[y0:y1, x0:x1]).sum()) / float(target.area)
    if MIN_PASTE_COVERAGE <= coverage <= MAX_PASTE_COVERAGE:
        return coverage
    return None


def rendered_coverage(target: ConcatItem, donor: ConcatItem) -> tuple[float, float] | None:
    binary_cov = binary_coverage(target, donor)
    if binary_cov is None:
        return None
    target_mask = load_mask(str(target.mask_path))
    soft = soft_mask(str(donor.mask_path))
    target_crop = target_mask[target.y0 : target.y1, target.x0 : target.x1].astype(np.float32)
    soft_crop = soft[target.y0 : target.y1, target.x0 : target.x1]
    soft_coverage = float((soft_crop * target_crop).sum()) / float(target.area)
    if MIN_PASTE_COVERAGE <= soft_coverage <= MAX_PASTE_COVERAGE:
        return float(binary_cov), float(soft_coverage)
    return None


def render_pair(target: ConcatItem, donor: ConcatItem) -> tuple[np.ndarray, float]:
    target_img = np.asarray(Image.open(target.image_path).convert("RGB"), dtype=np.float32)
    donor_img = np.asarray(Image.open(donor.image_path).convert("RGB"), dtype=np.float32)
    target_mask = load_mask(str(target.mask_path)).astype(np.float32)
    soft = soft_mask(str(donor.mask_path))
    intersection = float((soft * target_mask).sum())
    target_area = float(target_mask.sum())
    coverage = intersection / target_area if target_area > 0 else float(soft.mean())
    out = target_img * (1.0 - soft[..., None]) + donor_img * soft[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), float(coverage)


def ordered_items_by_usage(items: list[ConcatItem], usage: Counter[str], rng: random.Random) -> list[ConcatItem]:
    keyed = [(int(usage[item.sha256]), rng.random(), item) for item in items]
    keyed.sort(key=lambda row: (row[0], row[1]))
    return [item for _, _, item in keyed]


def choose_pair(
    items_a: list[ConcatItem],
    items_b: list[ConcatItem],
    usage: Counter[str],
    rng: random.Random,
) -> tuple[ConcatItem, ConcatItem, ConcatItem, ConcatItem, float, float] | None:
    ordered_a = ordered_items_by_usage(items_a, usage, rng)
    ordered_b = ordered_items_by_usage(items_b, usage, rng)
    min_a = int(usage[ordered_a[0].sha256])
    min_b = int(usage[ordered_b[0].sha256])
    for delta in range(MAX_USAGE_DELTA + 1):
        a_all = [item for item in ordered_a if int(usage[item.sha256]) <= min_a + delta]
        b_all = [item for item in ordered_b if int(usage[item.sha256]) <= min_b + delta]
        limits = [CANDIDATE_BEAM, CANDIDATE_BEAM * 4, max(len(a_all), len(b_all))]
        for limit in sorted(set(min(max(len(a_all), len(b_all)), value) for value in limits)):
            candidates: list[tuple[tuple[int, int, float, float], ConcatItem, ConcatItem, ConcatItem, ConcatItem, float]] = []
            for source_a in a_all[: min(len(a_all), limit)]:
                for source_b in b_all[: min(len(b_all), limit)]:
                    for render_target, render_donor in ((source_a, source_b), (source_b, source_a)):
                        binary_cov = binary_coverage(render_target, render_donor)
                        if binary_cov is None:
                            continue
                        score = (
                            max(int(usage[source_a.sha256]), int(usage[source_b.sha256])) + 1,
                            int(usage[source_a.sha256]) + int(usage[source_b.sha256]),
                            abs(float(binary_cov) - 0.30),
                            rng.random(),
                        )
                        candidates.append((score, source_a, source_b, render_target, render_donor, float(binary_cov)))
            if not candidates:
                continue
            candidates.sort(key=lambda row: row[0])
            validated = 0
            best_valid: tuple[tuple[int, int, float, float], ConcatItem, ConcatItem, ConcatItem, ConcatItem, float, float] | None = None
            validate_limits = [SOFT_VALIDATION_CANDIDATES, SOFT_VALIDATION_CANDIDATES * 4, len(candidates)]
            for validate_limit in sorted(set(min(len(candidates), value) for value in validate_limits)):
                for score, source_a, source_b, render_target, render_donor, binary_cov in candidates[validated:validate_limit]:
                    coverages = rendered_coverage(render_target, render_donor)
                    if coverages is None:
                        continue
                    _, soft_cov = coverages
                    exact_score = (score[0], score[1], abs(float(soft_cov) - 0.30), score[3])
                    if best_valid is None or exact_score < best_valid[0]:
                        best_valid = (exact_score, source_a, source_b, render_target, render_donor, float(binary_cov), float(soft_cov))
                validated = validate_limit
                if best_valid is not None:
                    _, source_a, source_b, render_target, render_donor, binary_cov, soft_cov = best_valid
                    return source_a, source_b, render_target, render_donor, binary_cov, soft_cov
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source_manifest = concat_source_manifest()
    class_names, raw_items_by_class, skipped = load_concat_items(
        height=EXPORT_H,
        width=EXPORT_W,
        mask_threshold=MASK_THRESHOLD,
        require_exact_size=True,
    )
    class_names = {cid: name for cid, name in class_names.items() if int(cid) < CLASS_COUNT}
    raw_items_by_class = {cid: rows for cid, rows in raw_items_by_class.items() if int(cid) < CLASS_COUNT}
    if len(raw_items_by_class) != CLASS_COUNT:
        raise SystemExit(f"Expected {CLASS_COUNT} classes with source-mask records, got {len(raw_items_by_class)}")

    items_by_class: dict[int, list[ConcatItem]] = {}
    source_pool_rows: list[dict[str, Any]] = []
    for class_id, items in sorted(raw_items_by_class.items()):
        selected = select_pool(list(items), int(class_id))
        items_by_class[int(class_id)] = selected
        source_pool_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "validated_source_count": int(len(items)),
                "selected_source_count": int(len(selected)),
            }
        )

    rng = random.Random(SEED)
    usage: Counter[str] = Counter()
    usage_target: Counter[str] = Counter()
    usage_donor: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    started_at = utc_now()
    active_ids = sorted(items_by_class)
    pairs = [(a, b) for a in active_ids for b in active_ids if a != b]

    for a_id, b_id in tqdm(pairs, desc="cut-paste pairs", unit="pair"):
        pair_key = f"a{int(a_id):03d}_b{int(b_id):03d}"
        choice = choose_pair(items_by_class[int(a_id)], items_by_class[int(b_id)], usage, rng)
        if choice is None:
            rejection_rows.append(
                {
                    "pair_key": pair_key,
                    "class_a_id": int(a_id),
                    "class_b_id": int(b_id),
                    "class_a_name": class_names[int(a_id)],
                    "class_b_name": class_names[int(b_id)],
                    "reason": "no_valid_reuseaware_cutpaste_candidate",
                }
            )
            continue
        source_a, source_b, render_target, render_donor, binary_coverage, expected_soft_coverage = choice
        out_img, soft_coverage = render_pair(render_target, render_donor)
        if not (MIN_PASTE_COVERAGE <= float(soft_coverage) <= MAX_PASTE_COVERAGE):
            rejection_rows.append(
                {
                    "pair_key": pair_key,
                    "class_a_id": int(a_id),
                    "class_b_id": int(b_id),
                    "class_a_name": class_names[int(a_id)],
                    "class_b_name": class_names[int(b_id)],
                    "source_a_id": source_a.source_id,
                    "source_b_id": source_b.source_id,
                    "reason": "soft_coverage_outside_bounds",
                    "binary_coverage": float(binary_coverage),
                    "soft_coverage": float(soft_coverage),
                    "expected_soft_coverage": float(expected_soft_coverage),
                }
            )
            continue

        out_png = OUT / "samples" / f"{pair_key}_cutpaste.png"
        out_json = OUT / "samples" / f"{pair_key}_cutpaste.json"
        Image.fromarray(out_img, mode="RGB").save(out_png)
        usage[source_a.sha256] += 1
        usage[source_b.sha256] += 1
        usage_target[render_target.sha256] += 1
        usage_donor[render_donor.sha256] += 1
        active_classes = sorted([int(a_id), int(b_id)])
        meta = {
            "method": "cutpaste",
            "dataset_version": "cutpaste_reusecap10",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "image_path": str(out_png.relative_to(OUT)),
            "class_a_id": int(a_id),
            "class_b_id": int(b_id),
            "class_a_name": class_names[int(a_id)],
            "class_b_name": class_names[int(b_id)],
            "pair_key": pair_key,
            "target_source_id": source_a.source_id,
            "donor_source_id": source_b.source_id,
            "target_source_hash": source_a.sha256,
            "donor_source_hash": source_b.sha256,
            "target_source_ref": public_source_ref(source_a.image_path),
            "donor_source_ref": public_source_ref(source_b.image_path),
            "render_target_source_id": render_target.source_id,
            "render_donor_source_id": render_donor.source_id,
            "render_target_source_ref": public_source_ref(render_target.image_path),
            "render_donor_source_ref": public_source_ref(render_donor.image_path),
            "render_direction": "forward" if render_target.sha256 == source_a.sha256 else "reverse",
            "coverage": float(soft_coverage),
            "binary_candidate_coverage": float(binary_coverage),
            "active_classes": active_classes,
            "active_class_names": [class_names[class_id] for class_id in active_classes],
            "mask_source": "precomputed_u2net_manifest_mask",
            "mask_threshold": MASK_THRESHOLD,
            "soft_edges": True,
            "soft_edge_px": SOFT_EDGE_PX,
            "soft_edge_gauss": SOFT_EDGE_GAUSS,
            "max_paste_coverage": MAX_PASTE_COVERAGE,
            "min_paste_coverage": MIN_PASTE_COVERAGE,
            "min_mask_area": MIN_MASK_AREA,
            "source_pool_fraction": SOURCE_POOL_FRACTION,
            "source_pool_min_items": SOURCE_POOL_MIN_ITEMS,
            "selection_policy": "reuseaware_min_prospective_max_source_reuse",
            "seed": SEED,
            "run_started_at_utc": started_at,
        }
        out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(meta)

    (OUT / "dataset_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (OUT / "class_mapping.json").write_text(json.dumps({int(i): class_names[int(i)] for i in active_ids}, indent=2) + "\n", encoding="utf-8")

    selected_source_rows: list[dict[str, Any]] = []
    for class_id, selected in sorted(items_by_class.items()):
        for item in selected:
            selected_source_rows.append(
                {
                    "synthetic_family": "cut_paste",
                    "class_id": int(class_id),
                    "class_name": class_names[int(class_id)],
                    "source_ref": public_source_ref(item.image_path),
                    "source_id": item.source_id,
                    "sha256": item.sha256,
                    "selected_for_generation": 1,
                    "total_use_count": int(usage[item.sha256]),
                    "target_use_count": int(usage_target[item.sha256]),
                    "donor_use_count": int(usage_donor[item.sha256]),
                }
            )
    write_csv(
        OUT / "selected_cutpaste_source_pool.csv",
        selected_source_rows,
        [
            "synthetic_family",
            "class_id",
            "class_name",
            "source_ref",
            "source_id",
            "sha256",
            "selected_for_generation",
            "total_use_count",
            "target_use_count",
            "donor_use_count",
        ],
    )
    write_csv(OUT / "source_pool_summary.csv", source_pool_rows, ["class_id", "class_name", "validated_source_count", "selected_source_count"])
    write_csv(
        OUT / "rejections.csv",
        rejection_rows,
        [
            "pair_key",
            "class_a_id",
            "class_b_id",
            "class_a_name",
            "class_b_name",
            "source_a_id",
            "source_b_id",
            "reason",
            "binary_coverage",
            "soft_coverage",
            "expected_soft_coverage",
        ],
    )
    usage_values = [int(row["total_use_count"]) for row in selected_source_rows]
    summary = {
        "dataset_version": "cutpaste_reusecap10",
        "out_dir": str(OUT),
        "source_filter_skipped": skipped,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_pool_fraction": SOURCE_POOL_FRACTION,
        "source_pool_min_items": SOURCE_POOL_MIN_ITEMS,
        "selected_source_total": int(len(selected_source_rows)),
        "ordered_pairs_requested": int(len(pairs)),
        "generated_rows": int(len(rows)),
        "rejected_rows": int(len(rejection_rows)),
        "unique_ordered_pairs": int(len({(row["class_a_id"], row["class_b_id"]) for row in rows})),
        "unique_unordered_pairs": int(len({tuple(sorted((row["class_a_id"], row["class_b_id"]))) for row in rows})),
        "source_reuse_distribution": {str(k): int(v) for k, v in sorted(Counter(usage_values).items())},
        "source_reuse_min": int(min(usage_values)) if usage_values else 0,
        "source_reuse_max": int(max(usage_values)) if usage_values else 0,
        "source_reuse_mean": float(np.mean(usage_values)) if usage_values else 0.0,
        "nominal_ordered_nonself_count": NOMINAL_ORDERED_NON_SELF_COUNT,
        "thesis_detector_count": THESIS_DETECTOR_COUNT,
        "generated_count_matches_thesis": len(rows) == THESIS_DETECTOR_COUNT,
        "dataset_manifest": str(OUT / "dataset_manifest.jsonl"),
        "selected_source_pool_csv": str(OUT / "selected_cutpaste_source_pool.csv"),
        "rejections_csv": str(OUT / "rejections.csv"),
        "completed_at_utc": utc_now(),
    }
    (OUT / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (WORK / "training_history.json").write_text(json.dumps(CUTPASTE_TRAINING_HISTORY, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

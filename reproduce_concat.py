#!/usr/bin/env python3
"""Standalone reproduction of the reuse-capped non-overlap concat dataset."""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from appendix_data import (
    ConcatItem,
    concat_background_template,
    concat_reference_hashes,
    concat_source_manifest,
    load_binary_mask_crop,
    load_concat_items,
    load_rgb_crop,
    output_dir,
    public_source_ref,
    require,
    sha256_file,
    work_dir,
)


WORK = work_dir("concat")
OUT = output_dir("concat")

SEED = 20260527
OUT_H = 680
OUT_W = 340
CLASS_COUNT = 136
NOMINAL_UNORDERED_NON_SELF_COUNT = CLASS_COUNT * (CLASS_COUNT - 1) // 2
NOMINAL_REQUESTED_SAMPLE_COUNT = NOMINAL_UNORDERED_NON_SELF_COUNT * 2
THESIS_DETECTOR_COUNT = 17850
SAMPLES_PER_PAIR = 2
SOURCE_POOL_FRACTION = 0.10
SOURCE_POOL_MIN_ITEMS = 1
MIN_GAP_PX = 10
MAX_GAP_PX = 40
TOP_JITTER_PX = 24
BOTTOM_JITTER_PX = 24
MASK_THRESHOLD = 127
SCANNER_TAIL_BG_ROWS = 48
PLACEMENT_POLICY = "sampled_shifted_bbox_stack"


@dataclass(frozen=True)
class Placement:
    top_shift: int
    bottom_shift: int
    requested_gap_px: int
    actual_gap_px: int
    top_jitter_applied_px: int
    bottom_jitter_applied_px: int


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def select_source_pool(
    items_by_class: dict[int, list[ConcatItem]],
    class_names: dict[int, str],
) -> tuple[dict[int, list[ConcatItem]], list[dict[str, object]]]:
    selected_by_class: dict[int, list[ConcatItem]] = {}
    rows: list[dict[str, object]] = []
    for class_id, items in sorted(items_by_class.items()):
        limit = max(SOURCE_POOL_MIN_ITEMS, int(np.floor(SOURCE_POOL_FRACTION * len(items))))
        limit = min(len(items), max(1, limit))
        class_rng = random.Random(f"{SEED}:{int(class_id)}:concat_source_pool")
        selected = list(items)
        class_rng.shuffle(selected)
        selected = selected[:limit]
        selected_by_class[int(class_id)] = selected
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "validated_source_count": int(len(items)),
                "selected_source_count": int(len(selected)),
            }
        )
    return selected_by_class, rows


def sample_shifted_placement(top: ConcatItem, bottom: ConcatItem, rng: random.Random) -> Placement | None:
    top_h = int(top.y1) - int(top.y0)
    bottom_h = int(bottom.y1) - int(bottom.y0)
    if top_h <= 0 or bottom_h <= 0:
        return None

    max_top_extra = min(TOP_JITTER_PX, OUT_H - top_h - bottom_h - MIN_GAP_PX)
    if max_top_extra < 0:
        return None

    top_extra = rng.randint(0, max_top_extra) if max_top_extra > 0 else 0
    top_shift = -int(top.y0) + int(top_extra)
    top_end = top_h + int(top_extra)

    max_requested_gap = OUT_H - bottom_h - int(top_end)
    if max_requested_gap < MIN_GAP_PX:
        return None
    gap_upper = min(MAX_GAP_PX, int(max_requested_gap))
    requested_gap = rng.randint(MIN_GAP_PX, gap_upper) if gap_upper > MIN_GAP_PX else MIN_GAP_PX

    bottom_shift_base = int(top_end + requested_gap - int(bottom.y0))
    max_bottom_extra = max(0, min(BOTTOM_JITTER_PX, OUT_H - (bottom_h + int(top_end) + int(requested_gap))))
    bottom_extra = rng.randint(0, max_bottom_extra) if max_bottom_extra > 0 else 0
    bottom_shift = int(bottom_shift_base + bottom_extra)
    bottom_start = int(bottom.y0) + int(bottom_shift)
    actual_gap = int(bottom_start - top_end)
    if actual_gap < requested_gap:
        return None

    return Placement(
        top_shift=int(top_shift),
        bottom_shift=int(bottom_shift),
        requested_gap_px=int(requested_gap),
        actual_gap_px=int(actual_gap),
        top_jitter_applied_px=int(top_extra),
        bottom_jitter_applied_px=int(bottom_extra),
    )


def choose_pair(
    a_items: list[ConcatItem],
    b_items: list[ConcatItem],
    usage: Counter[str],
    rng: random.Random,
) -> tuple[ConcatItem, ConcatItem, Placement] | None:
    candidates: list[tuple[tuple[int, int, int, float], ConcatItem, ConcatItem]] = []
    for a_item in a_items:
        for b_item in b_items:
            if a_item.sha256 == b_item.sha256:
                continue
            for top, bottom in ((a_item, b_item), (b_item, a_item)):
                top_reuse = int(usage[top.sha256])
                bottom_reuse = int(usage[bottom.sha256])
                score = (
                    max(top_reuse, bottom_reuse) + 1,
                    top_reuse + bottom_reuse,
                    -int(top.area + bottom.area),
                    rng.random(),
                )
                candidates.append((score, top, bottom))
    for _, top, bottom in sorted(candidates, key=lambda row: row[0]):
        placement = sample_shifted_placement(top, bottom, rng)
        if placement is not None:
            return top, bottom, placement
    return None


def shift_array(arr: np.ndarray, dy: int) -> np.ndarray:
    out = np.zeros_like(arr)
    if dy >= 0:
        out[dy:] = arr[: arr.shape[0] - dy]
    else:
        out[:dy] = arr[-dy:]
    return out


def compose(top: ConcatItem, bottom: ConcatItem, placement: Placement, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_img = load_rgb_crop(top.image_path, height=OUT_H, width=OUT_W).astype(np.float32)
    bot_img = load_rgb_crop(bottom.image_path, height=OUT_H, width=OUT_W).astype(np.float32)
    top_mask = load_binary_mask_crop(top.mask_path, height=OUT_H, width=OUT_W, threshold=MASK_THRESHOLD)
    bot_mask = load_binary_mask_crop(bottom.mask_path, height=OUT_H, width=OUT_W, threshold=MASK_THRESHOLD)

    top_img = shift_array(top_img, placement.top_shift)
    bot_img = shift_array(bot_img, placement.bottom_shift)
    top_mask = shift_array(top_mask, placement.top_shift)
    bot_mask = shift_array(bot_mask, placement.bottom_shift)
    if int(top_mask.sum()) != top.area or int(bot_mask.sum()) != bottom.area:
        raise ValueError("mask clipped by placement")
    if np.any((top_mask > 0.5) & (bot_mask > 0.5)):
        raise ValueError("mask overlap")

    union_rows = np.any((top_mask > 0.5) | (bot_mask > 0.5), axis=1)
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)
    if union_rows.any():
        last = int(np.where(union_rows)[0].max())
        tail = min(OUT_H - 1, last + SCANNER_TAIL_BG_ROWS)
        canvas[: tail + 1] = bg[: tail + 1].astype(np.float32)
    canvas[top_mask > 0.5] = top_img[top_mask > 0.5]
    canvas[bot_mask > 0.5] = bot_img[bot_mask > 0.5]
    return np.clip(canvas, 0, 255).astype(np.uint8), top_mask, bot_mask


def main() -> None:
    source_manifest = concat_source_manifest()
    background_template = concat_background_template()
    require(background_template)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    rng = random.Random(SEED)
    forbidden_hashes = concat_reference_hashes()
    class_names, raw_items_by_class, skipped = load_concat_items(
        height=OUT_H,
        width=OUT_W,
        mask_threshold=MASK_THRESHOLD,
        require_exact_size=True,
    )
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    (WORK / "logs" / "source_filter_summary.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    class_names = {cid: name for cid, name in class_names.items() if int(cid) < CLASS_COUNT}
    raw_items_by_class = {cid: rows for cid, rows in raw_items_by_class.items() if int(cid) < CLASS_COUNT}
    items_by_class, source_pool_rows = select_source_pool(raw_items_by_class, class_names)
    active_ids = sorted(items_by_class)
    bg = np.asarray(Image.open(background_template).convert("RGB"), dtype=np.uint8)[:OUT_H, :OUT_W]

    manifest: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    usage: Counter[str] = Counter()
    top_usage: Counter[str] = Counter()
    bottom_usage: Counter[str] = Counter()
    pairs = list(itertools.combinations(active_ids, 2))
    rng.shuffle(pairs)
    for class_a, class_b in tqdm(pairs, desc="concat unordered pairs", unit="pair"):
        for sample_idx in range(SAMPLES_PER_PAIR):
            chosen = choose_pair(items_by_class[class_a], items_by_class[class_b], usage, rng)
            if chosen is None:
                failures.append(
                    {
                        "class_a": class_a,
                        "class_b": class_b,
                        "sample": sample_idx,
                        "reason": "no_nonoverlap_placement_found",
                    }
                )
                continue
            top, bottom, placement = chosen
            if top.sha256 in forbidden_hashes or bottom.sha256 in forbidden_hashes:
                failures.append(
                    {
                        "class_a": class_a,
                        "class_b": class_b,
                        "sample": sample_idx,
                        "reason": "reference_hash_overlap",
                    }
                )
                continue
            stem = f"a{class_a:03d}_b{class_b:03d}_s{sample_idx:02d}_concat"
            out_png = OUT / "samples" / f"{stem}.png"
            out_json = OUT / "samples" / f"{stem}.json"
            try:
                image, top_mask, bottom_mask = compose(top, bottom, placement, bg)
            except Exception as exc:
                failures.append({"class_a": class_a, "class_b": class_b, "sample": sample_idx, "reason": str(exc)})
                continue
            Image.fromarray(image).save(out_png)
            usage[top.sha256] += 1
            usage[bottom.sha256] += 1
            top_usage[top.sha256] += 1
            bottom_usage[bottom.sha256] += 1
            active_classes = sorted([int(class_a), int(class_b)])
            row = {
                "method": "concat_nonoverlap",
                "dataset_version": "concat_reusecap10",
                "image_path": str(out_png.relative_to(OUT)),
                "pair_key": f"a{class_a:03d}_b{class_b:03d}",
                "pair_sample_index": int(sample_idx),
                "ordered_pair": False,
                "class_a_id": int(class_a),
                "class_b_id": int(class_b),
                "active_classes": active_classes,
                "active_class_names": [class_names[cid] for cid in active_classes],
                "top_class_id": int(top.class_id),
                "bottom_class_id": int(bottom.class_id),
                "top_source_ref": public_source_ref(top.image_path),
                "bottom_source_ref": public_source_ref(bottom.image_path),
                "top_source_id": top.source_id,
                "bottom_source_id": bottom.source_id,
                "top_source_sha256": top.sha256,
                "bottom_source_sha256": bottom.sha256,
                "requested_gap_px": int(placement.requested_gap_px),
                "actual_gap_px": int(placement.actual_gap_px),
                "top_shift_px": int(placement.top_shift),
                "bottom_shift_px": int(placement.bottom_shift),
                "top_jitter_applied_px": int(placement.top_jitter_applied_px),
                "bottom_jitter_applied_px": int(placement.bottom_jitter_applied_px),
                "top_area_px": int(top_mask.sum()),
                "bottom_area_px": int(bottom_mask.sum()),
                "source_pool_fraction": SOURCE_POOL_FRACTION,
                "selection_policy": "reuseaware_min_prospective_max_source_reuse",
                "seed": SEED,
            }
            out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
            manifest.append(row)

    source_reuse_rows: list[dict[str, object]] = []
    for class_id in active_ids:
        for item in items_by_class[class_id]:
            source_reuse_rows.append(
                {
                    "source_hash": item.sha256,
                    "source_id": item.source_id,
                    "class_id": int(item.class_id),
                    "class_name": item.class_name,
                    "source_ref": public_source_ref(item.image_path),
                    "total_use_count": int(usage[item.sha256]),
                    "top_use_count": int(top_usage[item.sha256]),
                    "bottom_use_count": int(bottom_usage[item.sha256]),
                    "was_used": int(usage[item.sha256] > 0),
                }
            )

    (OUT / "dataset_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (OUT / "class_mapping.json").write_text(
        json.dumps({int(cid): class_names[int(cid)] for cid in active_ids}, indent=2),
        encoding="utf-8",
    )
    write_csv(
        OUT / "source_pool_summary.csv",
        source_pool_rows,
        ["class_id", "class_name", "validated_source_count", "selected_source_count"],
    )
    write_csv(
        OUT / "source_reuse_stats.csv",
        source_reuse_rows,
        [
            "source_hash",
            "source_id",
            "class_id",
            "class_name",
            "source_ref",
            "total_use_count",
            "top_use_count",
            "bottom_use_count",
            "was_used",
        ],
    )
    usage_values = [int(row["total_use_count"]) for row in source_reuse_rows]
    (OUT / "generation_config.json").write_text(
        json.dumps(
            {
                "dataset_version": "concat_reusecap10",
                "script": Path(__file__).name,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256_file(source_manifest),
                "background_template": str(background_template),
                "background_template_sha256": sha256_file(background_template),
                "seed": SEED,
                "out_h": OUT_H,
                "out_w": OUT_W,
                "samples_per_pair": SAMPLES_PER_PAIR,
                "pair_mode": "unordered_nonself",
                "nominal_unordered_nonself_count": NOMINAL_UNORDERED_NON_SELF_COUNT,
                "nominal_requested_sample_count": NOMINAL_REQUESTED_SAMPLE_COUNT,
                "thesis_detector_count": THESIS_DETECTOR_COUNT,
                "generation_limit": None,
                "count_scope": "thesis_detector_count_is_reference_count_not_generation_limit",
                "source_pool_fraction": SOURCE_POOL_FRACTION,
                "source_pool_min_items": SOURCE_POOL_MIN_ITEMS,
                "selection_policy": "reuseaware_min_prospective_max_source_reuse",
                "gap_policy": "random",
                "min_gap_px": MIN_GAP_PX,
                "max_gap_px": MAX_GAP_PX,
                "top_jitter_px": TOP_JITTER_PX,
                "bottom_jitter_px": BOTTOM_JITTER_PX,
                "placement_policy": PLACEMENT_POLICY,
                "same_class_allowed": False,
                "overlap_allowed": False,
                "class_count": CLASS_COUNT,
                "background_mode": "scanner",
                "scanner_tail_bg_rows": SCANNER_TAIL_BG_ROWS,
                "generated": len(manifest),
                "failed": len(failures),
                "generated_count_matches_thesis": len(manifest) == THESIS_DETECTOR_COUNT,
                "num_source_hashes_in_pool": len(source_reuse_rows),
                "num_unique_source_hashes_used": sum(1 for value in usage_values if value > 0),
                "source_reuse_max": max(usage_values) if usage_values else 0,
                "source_reuse_mean": float(np.mean(usage_values)) if usage_values else 0.0,
                "count_note": (
                    "Generation is not capped at 17,850. The script requests two samples "
                    "for each unordered non-self pair and records 17,850 only as the "
                    "reference count observed for the reusecap10 thesis detector run "
                    "after feasibility filtering."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "rejection_report.json").write_text(
        json.dumps(
            {
                "dataset_version": "concat_reusecap10",
                "num_records": len(failures),
                "reason_counts": dict(Counter(row["reason"] for row in failures)),
                "records": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} concat samples to {OUT}")


if __name__ == "__main__":
    main()

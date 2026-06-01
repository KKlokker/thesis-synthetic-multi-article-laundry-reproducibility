#!/usr/bin/env python3
"""Standalone scaffold for the thesis detector framework.

1. train one label-aware EfficientNetV2B0 embedding detector per synthetic family and seed;
2. score checkpoints with the normalized detector embedding and a KNN-margin score;
3. select the checkpoint using validation data only under FPR <= 0.02;
4. for the classifier-augmented readout, add the selected ordinary-classifier
   evidence scalar to the selected detector score;
5. select the operating threshold on the calibration validation split;
6. report final-test image-level binary metrics.

The same entrypoint also includes the no-synthetic novelty comparator. That
comparator trains only on real single-item source-label supervision with a
single-label class head and scores multi-item evidence as novelty relative to
the real-single reference bank.

Outputs are written under `outputs/standalone/detector_framework_reproduction/`
and per-seed checkpoints under `work/standalone/detector_framework/`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common import (
    FAMILIES,
    NO_SYNTHETIC_COMPARATOR,
    ORDINARY_CLASSIFIER_EVIDENCE_SOURCE,
    OUTPUTS,
    WORK,
    aggregate_worker_csvs,
    configure_tensorflow,
    ensure_dirs,
    load_detector_active_labels,
    load_detector_classifier_evidence_values,
    load_detector_records,
    load_no_synthetic_training_records,
    lookup_evidence,
    make_dataset,
    set_reproducible_seed,
    write_csv,
)


SEEDS = tuple(20261130 + repeat * 1009 for repeat in range(6))
STRESS_LIMITS = (0.002, 0.005, 0.010, 0.020)


@dataclass(frozen=True)
class DetectorConfig:
    architecture: str = "efficientnetv2b0"
    weights: str = "imagenet"
    image_size: int = 192
    image_decode_format: str = "png"
    batch_size: int = 64
    eval_batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 8e-6
    weight_decay: float = 3e-5
    unfreeze_last: int = 160
    head_dim: int = 256
    dropout: float = 0.35
    label_smoothing: float = 0.010
    source_label_count: int = 136
    hyp2_weight: float = 0.10
    hyp2_temperature: float = 0.07
    hyp2_margin: float = 0.35
    knn_k: int = 100
    selection_fpr_limit: float = 0.020


@dataclass(frozen=True)
class EvalPoint:
    threshold: float
    recall: float
    fpr: float
    precision: float
    tp: int
    fp: int
    tn: int
    fn: int


def hyp2_angular_label_overlap_loss(tf, cfg: DetectorConfig):
    """Appendix-local HYP2-style angular overlap loss for active-label embeddings.

    The thesis uses HYP2 as train-side representation supervision. This compact
    standalone implementation uses the same active-label contract: examples
    sharing at least one active label are pulled together in cosine space, while
    disjoint examples are separated by the configured angular margin.
    """

    margin = float(cfg.hyp2_margin)
    temperature = float(cfg.hyp2_temperature)

    def loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        z = tf.math.l2_normalize(tf.cast(y_pred, tf.float32), axis=-1)
        batch = tf.shape(labels)[0]
        overlap = tf.matmul(labels, labels, transpose_b=True)
        eye = tf.eye(batch, dtype=tf.float32)
        positive = tf.cast(overlap > 0.0, tf.float32) * (1.0 - eye)
        negative = tf.cast(overlap <= 0.0, tf.float32) * (1.0 - eye)
        sim = tf.matmul(z, z, transpose_b=True)
        pos_loss = tf.reduce_sum(positive * (1.0 - sim)) / tf.maximum(tf.reduce_sum(positive), 1.0)
        neg_loss = tf.reduce_sum(negative * tf.nn.softplus((sim - margin) / temperature)) / tf.maximum(tf.reduce_sum(negative), 1.0)
        return pos_loss + neg_loss

    return loss


def build_label_aware_detector(tf, cfg: DetectorConfig, *, include_hyp2: bool):
    inputs = tf.keras.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal", name="train_flip")(inputs)
    x = tf.keras.layers.RandomZoom(0.08, name="train_zoom")(x)
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=None if cfg.weights.lower() in {"", "none", "null"} else cfg.weights,
        include_preprocessing=True,
        pooling="avg",
        input_shape=(cfg.image_size, cfg.image_size, 3),
    )
    base.trainable = True
    for layer in base.layers[:-cfg.unfreeze_last]:
        layer.trainable = False
    x = base(x)
    h = tf.keras.layers.Dense(cfg.head_dim, activation="relu", name="embedding_dense")(x)
    h = tf.keras.layers.Dropout(cfg.dropout, name="embedding_dropout")(h)
    z = tf.keras.layers.Lambda(lambda value: tf.math.l2_normalize(value, axis=-1), name="detector_embedding", dtype="float32")(h)
    labels = tf.keras.layers.Dense(cfg.source_label_count, activation="sigmoid", dtype="float32", name="label_head")(z)
    model = tf.keras.Model(inputs, {"label_head": labels, "detector_embedding": z}, name="thesis_label_aware_efficientnetv2b0_detector")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses = {"label_head": tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg.label_smoothing)}
    loss_weights = {"label_head": 1.0}
    if include_hyp2 and cfg.hyp2_weight > 0:
        losses["detector_embedding"] = hyp2_angular_label_overlap_loss(tf, cfg)
        loss_weights["detector_embedding"] = float(cfg.hyp2_weight)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={"label_head": [tf.keras.metrics.BinaryAccuracy(name="active_label_acc")]},
    )
    return model


def build_no_synthetic_comparator(tf, cfg: DetectorConfig):
    inputs = tf.keras.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal", name="train_flip")(inputs)
    x = tf.keras.layers.RandomZoom(0.08, name="train_zoom")(x)
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=None if cfg.weights.lower() in {"", "none", "null"} else cfg.weights,
        include_preprocessing=True,
        pooling="avg",
        input_shape=(cfg.image_size, cfg.image_size, 3),
    )
    base.trainable = True
    for layer in base.layers[:-cfg.unfreeze_last]:
        layer.trainable = False
    x = base(x)
    h = tf.keras.layers.Dense(cfg.head_dim, activation="relu", name="embedding_dense")(x)
    h = tf.keras.layers.Dropout(cfg.dropout, name="embedding_dropout")(h)
    z = tf.keras.layers.Lambda(lambda value: tf.math.l2_normalize(value, axis=-1), name="detector_embedding", dtype="float32")(h)
    labels = tf.keras.layers.Dense(cfg.source_label_count, activation="softmax", dtype="float32", name="single_label_head")(z)
    model = tf.keras.Model(inputs, labels, name="thesis_no_synthetic_efficientnetv2b0_novelty_comparator")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="single_label_acc")],
    )
    return model


def embedding_model(tf, model):
    return tf.keras.Model(model.input, model.get_layer("detector_embedding").output, name="thesis_detector_embedding")


def make_active_label_dataset(tf, paths: list[str], labels: np.ndarray, cfg: DetectorConfig, *, training: bool, seed: int):
    ds = make_dataset(tf, paths, labels.astype(np.float32), cfg, training=training, seed=seed)
    return ds.map(
        lambda image, y: (image, {"label_head": y, "detector_embedding": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def predict_embeddings(tf, model, paths: list[str], cfg: DetectorConfig) -> np.ndarray:
    if not paths:
        return np.zeros((0, cfg.head_dim), dtype=np.float32)
    ds = make_dataset(tf, paths, [0.0] * len(paths), cfg, training=False, seed=0)
    emb = embedding_model(tf, model).predict(ds, verbose=0)
    emb = np.asarray(emb, dtype=np.float32)
    denom = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(denom, 1e-12)


def mean_topk_similarity(tf, query: np.ndarray, reference: np.ndarray, k: int, batch_size: int) -> list[float]:
    if len(query) == 0:
        return []
    if len(reference) == 0:
        raise ValueError("KNN reference bank is empty.")
    k_eff = min(int(k), int(reference.shape[0]))
    ref_t = tf.constant(reference.T.astype(np.float32), dtype=tf.float32)
    scores: list[float] = []
    for start in range(0, query.shape[0], max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), query.shape[0])
        sims = tf.matmul(tf.constant(query[start:stop], dtype=tf.float32), ref_t)
        top = tf.math.top_k(sims, k=k_eff).values
        scores.extend(float(v) for v in tf.reduce_mean(top, axis=1).numpy())
    return scores


def knn_margin_scores(
    tf,
    model,
    paths: list[str],
    neg_ref: np.ndarray,
    pos_ref: np.ndarray,
    cfg: DetectorConfig,
) -> list[float]:
    emb = predict_embeddings(tf, model, paths, cfg)
    pos = mean_topk_similarity(tf, emb, pos_ref, cfg.knn_k, cfg.eval_batch_size)
    neg = mean_topk_similarity(tf, emb, neg_ref, cfg.knn_k, cfg.eval_batch_size)
    return [float(p - n) for p, n in zip(pos, neg)]


def no_synthetic_novelty_scores(
    tf,
    model,
    paths: list[str],
    neg_ref: np.ndarray,
    cfg: DetectorConfig,
) -> list[float]:
    emb = predict_embeddings(tf, model, paths, cfg)
    single_sim = mean_topk_similarity(tf, emb, neg_ref, cfg.knn_k, cfg.eval_batch_size)
    return [float(-score) for score in single_sim]


def eval_at_threshold(neg_scores: list[float], pos_scores: list[float], threshold: float) -> EvalPoint:
    fp = sum(1 for score in neg_scores if score >= threshold)
    tp = sum(1 for score in pos_scores if score >= threshold)
    tn = len(neg_scores) - fp
    fn = len(pos_scores) - tp
    recall = tp / len(pos_scores) if pos_scores else 0.0
    fpr = fp / len(neg_scores) if neg_scores else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return EvalPoint(float(threshold), float(recall), float(fpr), float(precision), tp, fp, tn, fn)


def select_threshold(neg_scores: list[float], pos_scores: list[float], target_fpr: float) -> EvalPoint:
    if not neg_scores:
        return eval_at_threshold(neg_scores, pos_scores, -math.inf)
    thresholds = sorted({float(v) for v in neg_scores + pos_scores})
    thresholds.append(math.nextafter(max(thresholds), math.inf))
    allowed_fp = int(math.floor(float(target_fpr) * len(neg_scores) + 1e-12))
    best: EvalPoint | None = None
    for threshold in thresholds:
        point = eval_at_threshold(neg_scores, pos_scores, threshold)
        if point.fp > allowed_fp:
            continue
        if best is None or (point.recall, -point.fpr, point.threshold) > (best.recall, -best.fpr, best.threshold):
            best = point
    if best is None:
        return eval_at_threshold(neg_scores, pos_scores, math.nextafter(max(neg_scores), math.inf))
    return best


def point_fields(prefix: str, point: EvalPoint) -> dict[str, object]:
    return {
        f"{prefix}_recall": point.recall,
        f"{prefix}_fpr": point.fpr,
        f"{prefix}_precision": point.precision,
        f"{prefix}_tp": point.tp,
        f"{prefix}_fp": point.fp,
        f"{prefix}_tn": point.tn,
        f"{prefix}_fn": point.fn,
    }


def score_rows(
    family: str,
    worker_id: str,
    seed: int,
    split: str,
    paths: list[str],
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "worker_id": worker_id,
            "seed": seed,
            "split": split,
            "sample_index": idx,
            "path": path,
            "y_true": label,
            "score": float(score),
            "selected_threshold": float(threshold),
            "y_pred": int(float(score) >= float(threshold)),
            "score_minus_threshold": float(score) - float(threshold),
        }
        for idx, (path, label, score) in enumerate(zip(paths, labels, scores))
    ]


def validation_selection_key(row: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(row["model_selection_recall"]),
        -float(row["model_selection_fpr"]),
        float(row["model_selection_threshold"]),
        -int(row["epoch"]),
    )


def apply_classifier_augmented_readout(
    family: str,
    paths: list[str],
    raw_scores: list[float],
    evidence_values: dict[str, float],
) -> list[float]:
    """Apply the thesis-facing classifier-augmented readout.

    The selected ordinary-classifier evidence scalar and additive weight are
    selected outside this standalone script. The input export therefore
    contains the weighted scalar term that is added to the KNN-margin or
    no-synthetic novelty detector score during calibration and final scoring.
    """
    out: list[float] = []
    for path, raw_score in zip(paths, raw_scores):
        out.append(float(raw_score) + lookup_evidence(path, evidence_values))
    return out


def run_family_seed(tf, family: str, seed: int, cfg: DetectorConfig, out_dir: Path, *, classifier_readout: bool) -> dict[str, object]:
    worker_id = f"{family}_seed{seed}"
    records = load_detector_records(family)
    train_neg, train_neg_labels, _ = load_detector_active_labels(family, "train_singles", num_classes=cfg.source_label_count)
    train_pos, train_pos_labels, train_pos_active_ids = load_detector_active_labels(
        family, "train_synthetic_positives", num_classes=cfg.source_label_count
    )
    if not train_neg or not train_pos:
        raise SystemExit(f"{family} needs non-empty train_singles and train_synthetic_positives manifests.")

    model_sel_neg = records["model_selection_singles"]
    model_sel_pos = records["model_selection_positives"]
    cal_neg = records["calibration_singles"]
    cal_pos = records["calibration_positives"]
    final_neg = records["final_singles"]
    final_pos = records["final_positives"]

    set_reproducible_seed(tf, seed)
    model = build_label_aware_detector(tf, cfg, include_hyp2=True)
    train_paths = train_neg + train_pos
    train_labels = np.concatenate([train_neg_labels, train_pos_labels], axis=0)
    train_ds = make_active_label_dataset(tf, train_paths, train_labels, cfg, training=True, seed=seed)

    worker_dir = out_dir / worker_id
    ckpt_dir = WORK / "detector_framework" / worker_id / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_row: dict[str, object] | None = None
    best_neg_ref: np.ndarray | None = None
    best_pos_ref: np.ndarray | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.fit(train_ds, epochs=1, verbose=2)
        model.save_weights(str(ckpt_dir / f"epoch_{epoch:03d}.weights.h5"))
        neg_ref = predict_embeddings(tf, model, train_neg, cfg)
        pos_ref = predict_embeddings(tf, model, train_pos, cfg)
        model_sel_neg_scores = knn_margin_scores(tf, model, model_sel_neg, neg_ref, pos_ref, cfg)
        model_sel_pos_scores = knn_margin_scores(tf, model, model_sel_pos, neg_ref, pos_ref, cfg)
        model_sel_point = select_threshold(model_sel_neg_scores, model_sel_pos_scores, cfg.selection_fpr_limit)
        row = {
            "family": family,
            "worker_id": worker_id,
            "seed": seed,
            "epoch": epoch,
            "validation_fpr_ceiling": cfg.selection_fpr_limit,
            "model_selection_threshold": model_sel_point.threshold,
            **point_fields("model_selection", model_sel_point),
        }
        epoch_rows.append(row)
        if best_row is None or validation_selection_key(row) > validation_selection_key(best_row):
            best_epoch = epoch
            best_row = row
            best_neg_ref = neg_ref
            best_pos_ref = pos_ref

    if best_row is None or best_neg_ref is None or best_pos_ref is None:
        raise RuntimeError("No selected checkpoint was produced.")

    selected_weights = ckpt_dir / f"epoch_{best_epoch:03d}.weights.h5"
    model.load_weights(str(selected_weights))
    raw_model_sel_neg = knn_margin_scores(tf, model, model_sel_neg, best_neg_ref, best_pos_ref, cfg)
    raw_model_sel_pos = knn_margin_scores(tf, model, model_sel_pos, best_neg_ref, best_pos_ref, cfg)
    raw_cal_neg = knn_margin_scores(tf, model, cal_neg, best_neg_ref, best_pos_ref, cfg)
    raw_cal_pos = knn_margin_scores(tf, model, cal_pos, best_neg_ref, best_pos_ref, cfg)
    raw_final_neg = knn_margin_scores(tf, model, final_neg, best_neg_ref, best_pos_ref, cfg)
    raw_final_pos = knn_margin_scores(tf, model, final_pos, best_neg_ref, best_pos_ref, cfg)

    selected_method = "raw_detector_margin"
    if classifier_readout:
        all_eval_paths = sorted(set(model_sel_neg + model_sel_pos + cal_neg + cal_pos + final_neg + final_pos))
        evidence_values = load_detector_classifier_evidence_values(family, all_eval_paths)
        model_sel_neg_scores = apply_classifier_augmented_readout(family, model_sel_neg, raw_model_sel_neg, evidence_values)
        model_sel_pos_scores = apply_classifier_augmented_readout(family, model_sel_pos, raw_model_sel_pos, evidence_values)
        cal_neg_scores = apply_classifier_augmented_readout(family, cal_neg, raw_cal_neg, evidence_values)
        cal_pos_scores = apply_classifier_augmented_readout(family, cal_pos, raw_cal_pos, evidence_values)
        final_neg_scores = apply_classifier_augmented_readout(family, final_neg, raw_final_neg, evidence_values)
        final_pos_scores = apply_classifier_augmented_readout(family, final_pos, raw_final_pos, evidence_values)
        selected_method = "classifier_augmented_readout"
    else:
        model_sel_neg_scores = raw_model_sel_neg
        model_sel_pos_scores = raw_model_sel_pos
        cal_neg_scores = raw_cal_neg
        cal_pos_scores = raw_cal_pos
        final_neg_scores = raw_final_neg
        final_pos_scores = raw_final_pos

    cal_point = select_threshold(cal_neg_scores, cal_pos_scores, cfg.selection_fpr_limit)
    model_sel_at_cal = eval_at_threshold(model_sel_neg_scores, model_sel_pos_scores, cal_point.threshold)
    final_point = eval_at_threshold(final_neg_scores, final_pos_scores, cal_point.threshold)

    selected_row = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_checkpoint_path": str(selected_weights),
        "selected_threshold": cal_point.threshold,
        "score_method": selected_method,
        **point_fields("model_selection_at_calibration_threshold", model_sel_at_cal),
        **point_fields("calibration", cal_point),
        **point_fields("final", final_point),
    }
    final_metric = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_threshold": cal_point.threshold,
        "selection_fpr_limit": cfg.selection_fpr_limit,
        "score_method": selected_method,
        "test_recall": final_point.recall,
        "test_fpr": final_point.fpr,
        "test_precision": final_point.precision,
        "test_tp": final_point.tp,
        "test_fp": final_point.fp,
        "test_tn": final_point.tn,
        "test_fn": final_point.fn,
    }
    threshold_rows: list[dict[str, object]] = []
    for limit in STRESS_LIMITS:
        val_point = select_threshold(cal_neg_scores, cal_pos_scores, limit)
        out_point = eval_at_threshold(final_neg_scores, final_pos_scores, val_point.threshold)
        threshold_rows.append(
            {
                "family": family,
                "worker_id": worker_id,
                "seed": seed,
                "selected_epoch": best_epoch,
                "validation_fpr_ceiling": limit,
                "threshold": val_point.threshold,
                **point_fields("val", val_point),
                **point_fields("final", out_point),
                "score_method": selected_method,
            }
        )

    write_csv(worker_dir / "validation_epoch_metrics.csv", epoch_rows)
    write_csv(worker_dir / "selected_checkpoints.csv", [selected_row])
    write_csv(worker_dir / "final_test_metrics.csv", [final_metric])
    write_csv(worker_dir / "threshold_stress.csv", threshold_rows)
    write_csv(
        worker_dir / "model_selection_validation_scores.csv",
        score_rows(family, worker_id, seed, "model_selection_validation", model_sel_neg, [0] * len(model_sel_neg), model_sel_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "model_selection_validation", model_sel_pos, [1] * len(model_sel_pos), model_sel_pos_scores, cal_point.threshold),
    )
    write_csv(
        worker_dir / "calibration_validation_scores.csv",
        score_rows(family, worker_id, seed, "calibration_validation", cal_neg, [0] * len(cal_neg), cal_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "calibration_validation", cal_pos, [1] * len(cal_pos), cal_pos_scores, cal_point.threshold),
    )
    write_csv(
        worker_dir / "final_test_scores.csv",
        score_rows(family, worker_id, seed, "final_test", final_neg, [0] * len(final_neg), final_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "final_test", final_pos, [1] * len(final_pos), final_pos_scores, cal_point.threshold),
    )
    payload = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "config": asdict(cfg),
        "training_counts": {"real_singles": len(train_neg), "synthetic_positives": len(train_pos), "real_multi_train_count": 0},
        "active_label_training": {
            "source_label_count": cfg.source_label_count,
            "real_single_active_labels": 1,
            "synthetic_positive_active_label_min": min((len(ids) for ids in train_pos_active_ids), default=0),
            "synthetic_positive_active_label_max": max((len(ids) for ids in train_pos_active_ids), default=0),
            "hyp2_auxiliary_weight": cfg.hyp2_weight,
            "hyp2_temperature": cfg.hyp2_temperature,
            "hyp2_margin": cfg.hyp2_margin,
        },
        "selected_checkpoint": selected_row,
        "final_test_metrics": final_metric,
    }
    (worker_dir / "family_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final_metric


def run_no_synthetic_seed(tf, seed: int, cfg: DetectorConfig, out_dir: Path, *, classifier_readout: bool) -> dict[str, object]:
    family = NO_SYNTHETIC_COMPARATOR
    worker_id = f"{family}_seed{seed}"
    records = load_detector_records(family)
    train_paths, train_label_ids, class_names = load_no_synthetic_training_records()
    if not train_paths or not train_label_ids:
        raise SystemExit("no_synthetic_novelty needs train_singles.csv with real single-item class labels.")

    model_sel_neg = records["model_selection_singles"]
    model_sel_pos = records["model_selection_positives"]
    cal_neg = records["calibration_singles"]
    cal_pos = records["calibration_positives"]
    final_neg = records["final_singles"]
    final_pos = records["final_positives"]

    set_reproducible_seed(tf, seed)
    model = build_no_synthetic_comparator(tf, cfg)
    train_ds = make_dataset(tf, train_paths, train_label_ids, cfg, training=True, seed=seed, label_dtype="int32")

    worker_dir = out_dir / worker_id
    ckpt_dir = WORK / "detector_framework" / worker_id / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_rows: list[dict[str, object]] = []
    best_epoch = 0
    best_row: dict[str, object] | None = None
    best_neg_ref: np.ndarray | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.fit(train_ds, epochs=1, verbose=2)
        model.save_weights(str(ckpt_dir / f"epoch_{epoch:03d}.weights.h5"))
        neg_ref = predict_embeddings(tf, model, train_paths, cfg)
        model_sel_neg_scores = no_synthetic_novelty_scores(tf, model, model_sel_neg, neg_ref, cfg)
        model_sel_pos_scores = no_synthetic_novelty_scores(tf, model, model_sel_pos, neg_ref, cfg)
        model_sel_point = select_threshold(model_sel_neg_scores, model_sel_pos_scores, cfg.selection_fpr_limit)
        row = {
            "family": family,
            "worker_id": worker_id,
            "seed": seed,
            "epoch": epoch,
            "validation_fpr_ceiling": cfg.selection_fpr_limit,
            "model_selection_threshold": model_sel_point.threshold,
            **point_fields("model_selection", model_sel_point),
        }
        epoch_rows.append(row)
        if best_row is None or validation_selection_key(row) > validation_selection_key(best_row):
            best_epoch = epoch
            best_row = row
            best_neg_ref = neg_ref

    if best_row is None or best_neg_ref is None:
        raise RuntimeError("No selected no-synthetic checkpoint was produced.")

    selected_weights = ckpt_dir / f"epoch_{best_epoch:03d}.weights.h5"
    model.load_weights(str(selected_weights))
    raw_model_sel_neg = no_synthetic_novelty_scores(tf, model, model_sel_neg, best_neg_ref, cfg)
    raw_model_sel_pos = no_synthetic_novelty_scores(tf, model, model_sel_pos, best_neg_ref, cfg)
    raw_cal_neg = no_synthetic_novelty_scores(tf, model, cal_neg, best_neg_ref, cfg)
    raw_cal_pos = no_synthetic_novelty_scores(tf, model, cal_pos, best_neg_ref, cfg)
    raw_final_neg = no_synthetic_novelty_scores(tf, model, final_neg, best_neg_ref, cfg)
    raw_final_pos = no_synthetic_novelty_scores(tf, model, final_pos, best_neg_ref, cfg)

    selected_method = "no_synthetic_real_single_novelty"
    if classifier_readout:
        all_eval_paths = sorted(set(model_sel_neg + model_sel_pos + cal_neg + cal_pos + final_neg + final_pos))
        evidence_values = load_detector_classifier_evidence_values(family, all_eval_paths)
        model_sel_neg_scores = apply_classifier_augmented_readout(family, model_sel_neg, raw_model_sel_neg, evidence_values)
        model_sel_pos_scores = apply_classifier_augmented_readout(family, model_sel_pos, raw_model_sel_pos, evidence_values)
        cal_neg_scores = apply_classifier_augmented_readout(family, cal_neg, raw_cal_neg, evidence_values)
        cal_pos_scores = apply_classifier_augmented_readout(family, cal_pos, raw_cal_pos, evidence_values)
        final_neg_scores = apply_classifier_augmented_readout(family, final_neg, raw_final_neg, evidence_values)
        final_pos_scores = apply_classifier_augmented_readout(family, final_pos, raw_final_pos, evidence_values)
        selected_method = "no_synthetic_classifier_augmented_readout"
    else:
        model_sel_neg_scores = raw_model_sel_neg
        model_sel_pos_scores = raw_model_sel_pos
        cal_neg_scores = raw_cal_neg
        cal_pos_scores = raw_cal_pos
        final_neg_scores = raw_final_neg
        final_pos_scores = raw_final_pos

    cal_point = select_threshold(cal_neg_scores, cal_pos_scores, cfg.selection_fpr_limit)
    model_sel_at_cal = eval_at_threshold(model_sel_neg_scores, model_sel_pos_scores, cal_point.threshold)
    final_point = eval_at_threshold(final_neg_scores, final_pos_scores, cal_point.threshold)
    selected_row = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_checkpoint_path": str(selected_weights),
        "selected_threshold": cal_point.threshold,
        "score_method": selected_method,
        **point_fields("model_selection_at_calibration_threshold", model_sel_at_cal),
        **point_fields("calibration", cal_point),
        **point_fields("final", final_point),
    }
    final_metric = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_threshold": cal_point.threshold,
        "selection_fpr_limit": cfg.selection_fpr_limit,
        "score_method": selected_method,
        "test_recall": final_point.recall,
        "test_fpr": final_point.fpr,
        "test_precision": final_point.precision,
        "test_tp": final_point.tp,
        "test_fp": final_point.fp,
        "test_tn": final_point.tn,
        "test_fn": final_point.fn,
    }
    threshold_rows: list[dict[str, object]] = []
    for limit in STRESS_LIMITS:
        val_point = select_threshold(cal_neg_scores, cal_pos_scores, limit)
        out_point = eval_at_threshold(final_neg_scores, final_pos_scores, val_point.threshold)
        threshold_rows.append(
            {
                "family": family,
                "worker_id": worker_id,
                "seed": seed,
                "selected_epoch": best_epoch,
                "validation_fpr_ceiling": limit,
                "threshold": val_point.threshold,
                **point_fields("val", val_point),
                **point_fields("final", out_point),
                "score_method": selected_method,
            }
        )

    write_csv(worker_dir / "validation_epoch_metrics.csv", epoch_rows)
    write_csv(worker_dir / "selected_checkpoints.csv", [selected_row])
    write_csv(worker_dir / "final_test_metrics.csv", [final_metric])
    write_csv(worker_dir / "threshold_stress.csv", threshold_rows)
    write_csv(
        worker_dir / "model_selection_validation_scores.csv",
        score_rows(family, worker_id, seed, "model_selection_validation", model_sel_neg, [0] * len(model_sel_neg), model_sel_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "model_selection_validation", model_sel_pos, [1] * len(model_sel_pos), model_sel_pos_scores, cal_point.threshold),
    )
    write_csv(
        worker_dir / "calibration_validation_scores.csv",
        score_rows(family, worker_id, seed, "calibration_validation", cal_neg, [0] * len(cal_neg), cal_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "calibration_validation", cal_pos, [1] * len(cal_pos), cal_pos_scores, cal_point.threshold),
    )
    write_csv(
        worker_dir / "final_test_scores.csv",
        score_rows(family, worker_id, seed, "final_test", final_neg, [0] * len(final_neg), final_neg_scores, cal_point.threshold)
        + score_rows(family, worker_id, seed, "final_test", final_pos, [1] * len(final_pos), final_pos_scores, cal_point.threshold),
    )
    payload = {
        "family": family,
        "worker_id": worker_id,
        "seed": seed,
        "config": asdict(cfg),
        "training_counts": {
            "real_singles": len(train_paths),
            "synthetic_positives": 0,
            "single_item_classes": len(class_names),
            "real_multi_train_count": 0,
            "source_label_count": cfg.source_label_count,
            "hyp2_auxiliary_weight": 0.0,
        },
        "selected_checkpoint": selected_row,
        "final_test_metrics": final_metric,
    }
    (worker_dir / "family_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final_metric


def family_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        recalls = [float(row["test_recall"]) for row in family_rows]
        fprs = [float(row["test_fpr"]) for row in family_rows]
        out.append(
            {
                "family": family,
                "seed_count": len(family_rows),
                "mean_final_recall": float(np.mean(recalls)) if recalls else 0.0,
                "sd_final_recall": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
                "mean_final_fpr": float(np.mean(fprs)) if fprs else 0.0,
                "sd_final_fpr": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
                "seeds_final_fpr_le_2pct": sum(1 for value in fprs if value <= 0.020),
                "sum_final_tp": sum(int(row["test_tp"]) for row in family_rows),
                "sum_final_fp": sum(int(row["test_fp"]) for row in family_rows),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=",".join(FAMILIES), help="Comma-separated families or 'all'.")
    ap.add_argument("--seed-repeats", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--weights", default=DetectorConfig().weights, help="EfficientNetV2B0 weights: 'imagenet' or 'none'.")
    ap.add_argument("--out-dir", default=str(OUTPUTS / "detector_framework_reproduction"))
    ap.add_argument("--disable-classifier-augmented-readout", "--disable-classifier-readout", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    seed_repeats = int(args.seed_repeats)
    if seed_repeats < 1 or seed_repeats > len(SEEDS):
        raise SystemExit(f"--seed-repeats must be between 1 and {len(SEEDS)}")
    epochs = int(args.epochs)
    if epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    selected_seeds = list(SEEDS[:seed_repeats])
    families = list(FAMILIES) if args.families.strip().lower() == "all" else [p.strip() for p in args.families.split(",") if p.strip()]
    family_aliases = {"single_only": NO_SYNTHETIC_COMPARATOR, "gan0205": "gan"}
    families = [family_aliases.get(family, family) for family in families]
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise SystemExit(f"Unknown families: {unknown}")
    cfg = DetectorConfig(epochs=epochs, weights=str(args.weights))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (WORK / "detector_framework").mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "families": families,
                "seeds": selected_seeds,
                "classifier_augmented_readout": not bool(args.disable_classifier_augmented_readout),
                "ordinary_classifier_evidence_source": ORDINARY_CLASSIFIER_EVIDENCE_SOURCE,
                "real_multi_train_count": 0,
                "synthetic_families_mixed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tf = configure_tensorflow()
    final_rows: list[dict[str, object]] = []
    for family in families:
        for seed in selected_seeds:
            print(f"[detector] family={family} seed={seed}")
            if family == NO_SYNTHETIC_COMPARATOR:
                final_rows.append(run_no_synthetic_seed(tf, seed, cfg, out_dir, classifier_readout=not bool(args.disable_classifier_augmented_readout)))
            else:
                final_rows.append(run_family_seed(tf, family, seed, cfg, out_dir, classifier_readout=not bool(args.disable_classifier_augmented_readout)))
    aggregate_worker_csvs(
        out_dir,
        (
            "validation_epoch_metrics.csv",
            "selected_checkpoints.csv",
            "final_test_metrics.csv",
            "threshold_stress.csv",
            "model_selection_validation_scores.csv",
            "calibration_validation_scores.csv",
            "final_test_scores.csv",
        ),
    )
    write_csv(out_dir / "final_test_metrics.csv", final_rows)
    write_csv(out_dir / "family_summary.csv", family_summary(final_rows))
    metrics = {
        "family_count": len(families),
        "seed_count": seed_repeats,
        "classifier_augmented_readout": int(not bool(args.disable_classifier_augmented_readout)),
        "no_synthetic_comparator_count": int(NO_SYNTHETIC_COMPARATOR in families),
        "real_multi_train_count": 0,
        "same_framework_per_family": 1,
        "synthetic_families_mixed": 0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

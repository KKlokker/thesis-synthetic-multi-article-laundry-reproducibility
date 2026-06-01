#!/usr/bin/env python3
"""Standalone GAN reproduction for the thesis export.

The thesis GAN path is a single-label SPADE/pix2pix endpoint renderer followed
by deterministic two-source composition. It is not a native dual-output GAN.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

from appendix_data import (
    discover_class_image_mask_pairs,
    load_rgb_mask_minus_one_one,
    output_dir,
    public_source_ref,
    real_512_mask_root,
    relative_input_path,
    require,
    sha256_file,
    shared_background_template,
    work_dir,
)


WORK = work_dir("gan")
OUT = output_dir("gan")
REAL_DATA = real_512_mask_root()
BACKGROUND_TEMPLATE = shared_background_template()

SEED = 5101
CLASS_COUNT = 136
THESIS_DETECTOR_COUNT = CLASS_COUNT * (CLASS_COUNT - 1)
IMG_SIZE = 512
OUT_H = 680
OUT_W = 340

BASE_FILTERS = 32
CLASS_EMBED_DIM = 64
CLASS_MAP_CHANNELS = 16
BASE_EPOCHS = 45
BASE_BATCH = 2
BASE_G_LR = 3e-5
BASE_D_LR = 2e-5
LAMBDA_L1 = 8.0

PAIR_LONG_AXIS_PX = 286
CENTER_OFFSET_PX = 38
VERTICAL_OFFSET_PX = 4
PLACEMENT_JITTER_PX = 10
MIN_OVERLAP_FRAC = 0.14
MAX_OVERLAP_FRAC = 0.42
OVERLAP_ADJUST_STEPS = 18
OVERLAP_ADJUST_PX = 14
ENTANGLE_PERIOD_PX = 64.0
ENTANGLE_TRANSITION_SOFTNESS = 0.13
ENTANGLE_NOISE_WEIGHT = 0.7
EDGE_SOFTEN_RADIUS = 0.55
BOTTOM_BLACK_ROWS = 5

BASE_G_WEIGHTS = WORK / "spade_endpoint_generator.weights.h5"
TRAINING_HISTORY_PATH = WORK / "training_history.json"

GAN_TRAINING_HISTORY = [
    {
        "stage": "single_label_spade_endpoint_renderer",
        "role": "Train the 136-class mask-to-RGB endpoint renderer.",
        "epochs": BASE_EPOCHS,
        "checkpoint": BASE_G_WEIGHTS.name,
        "generator_arch": "spade_unet_endpoint",
        "base_filters": BASE_FILTERS,
        "class_embed_dim": CLASS_EMBED_DIM,
        "class_map_channels": CLASS_MAP_CHANNELS,
        "normalization": "batch plus SPADE affine modulation",
        "upsample_mode": "bilinear",
        "history_anchor": "SingleLabelGAN/SPADE endpoint renderer used by the final GAN export",
    },
    {
        "stage": "gan_ordered_pair_export",
        "role": "Render endpoint A and endpoint B separately, assign overlap ownership, composite on production background, and export detector-facing ordered pairs.",
        "render_mode": "single_label_endpoint_renderer_plus_two_source_compositor",
        "classes": CLASS_COUNT,
        "ordered_pairs": THESIS_DETECTOR_COUNT,
        "thesis_detector_count": THESIS_DETECTOR_COUNT,
        "output_height": OUT_H,
        "output_width": OUT_W,
        "overlap_priority_mode": "random_field",
        "background_mode": "shared_production_template",
    },
]


def discover_data() -> tuple[list[str], dict[int, list[tuple[Path, Path]]]]:
    return discover_class_image_mask_pairs(REAL_DATA, class_limit=CLASS_COUNT, min_classes=CLASS_COUNT)


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_rgb_mask_minus_one_one(image_path, mask_path, height=IMG_SIZE, width=IMG_SIZE, mask_channel=True)


class RandomGanSource:
    def __init__(self, files_by_class: dict[int, list[tuple[Path, Path]]], batch: int):
        self.files_by_class = files_by_class
        self.batch_size = batch
        self.class_ids = sorted(files_by_class)
        self.rng = random.Random(SEED)

    def single_batch(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        images, masks, labels = [], [], []
        for _ in range(self.batch_size):
            cid = self.rng.choice(self.class_ids)
            img, mask = load_pair(*self.rng.choice(self.files_by_class[cid]))
            images.append(img)
            masks.append(mask)
            labels.append(cid)
        return (
            tf.convert_to_tensor(np.stack(images), tf.float32),
            tf.convert_to_tensor(np.stack(masks), tf.float32),
            tf.convert_to_tensor(labels, tf.int32),
        )


@tf.keras.utils.register_keras_serializable(package="appendix")
class SPADE(tf.keras.layers.Layer):
    """SPADE-style per-channel affine modulation from a spatial condition map."""

    def __init__(self, channels: int, hidden_channels: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.shared = tf.keras.layers.Conv2D(hidden_channels, 3, padding="same", activation="relu")
        self.gamma = tf.keras.layers.Conv2D(channels, 3, padding="same")
        self.beta = tf.keras.layers.Conv2D(channels, 3, padding="same")

    def call(self, inputs, training=None):
        x, cond = inputs
        cond = tf.image.resize(cond, tf.shape(x)[1:3], method="nearest")
        h = self.shared(cond)
        normalized = self.norm(x, training=training)
        return normalized * (1.0 + self.gamma(h)) + self.beta(h)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "hidden_channels": self.hidden_channels})
        return cfg


def class_map_layer(class_ids, h: int, w: int, name: str):
    emb = tf.keras.layers.Embedding(CLASS_COUNT, CLASS_EMBED_DIM, name=f"{name}_embed")(class_ids)
    vec = tf.keras.layers.Dense(CLASS_MAP_CHANNELS, activation="relu", name=f"{name}_dense")(emb)
    vec = tf.keras.layers.Reshape((1, 1, CLASS_MAP_CHANNELS), name=f"{name}_reshape")(vec)
    return tf.keras.layers.Lambda(lambda t: tf.tile(t, [1, h, w, 1]), name=f"{name}_tile")(vec)


def down_block(x, ch: int, name: str):
    x = tf.keras.layers.Conv2D(ch, 4, strides=2, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.LeakyReLU(0.2, name=f"{name}_lrelu")(x)


def spade_res_block(x, cond, ch: int, name: str):
    in_ch = int(x.shape[-1])
    shortcut = x
    if in_ch != ch:
        shortcut = tf.keras.layers.Conv2D(ch, 1, padding="same", name=f"{name}_skip")(shortcut)
    h = SPADE(in_ch, name=f"{name}_spade1")([x, cond])
    h = tf.keras.layers.ReLU(name=f"{name}_relu1")(h)
    h = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_conv1")(h)
    h = SPADE(ch, name=f"{name}_spade2")([h, cond])
    h = tf.keras.layers.ReLU(name=f"{name}_relu2")(h)
    h = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_conv2")(h)
    return tf.keras.layers.Add(name=f"{name}_add")([shortcut, h])


def up_block(x, skip, cond, ch: int, name: str):
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_upsample")(x)
    x = tf.keras.layers.Concatenate(name=f"{name}_cat")([x, skip])
    return spade_res_block(x, cond, ch, name=f"{name}_spade_res")


def build_generator(name: str) -> tf.keras.Model:
    mask_in = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1), name="mask")
    class_in = tf.keras.Input((), dtype=tf.int32, name="class_id")
    class_map = class_map_layer(class_in, IMG_SIZE, IMG_SIZE, "gclass")
    cond = tf.keras.layers.Concatenate(name="g_condition_map")([mask_in, class_map])

    x0 = tf.keras.layers.Concatenate(name="g_input_concat")([mask_in, class_map])
    x0 = tf.keras.layers.Conv2D(BASE_FILTERS, 7, padding="same", use_bias=False, name="g_stem_conv")(x0)
    x0 = tf.keras.layers.BatchNormalization(name="g_stem_bn")(x0)
    x0 = tf.keras.layers.ReLU(name="g_stem_relu")(x0)

    d1 = down_block(x0, BASE_FILTERS * 2, "g_down1")
    d2 = down_block(d1, BASE_FILTERS * 4, "g_down2")
    d3 = down_block(d2, BASE_FILTERS * 8, "g_down3")
    d4 = down_block(d3, BASE_FILTERS * 8, "g_down4")

    b = spade_res_block(d4, cond, BASE_FILTERS * 8, "g_bottleneck1")
    b = spade_res_block(b, cond, BASE_FILTERS * 8, "g_bottleneck2")

    u3 = up_block(b, d3, cond, BASE_FILTERS * 8, "g_up3")
    u2 = up_block(u3, d2, cond, BASE_FILTERS * 4, "g_up2")
    u1 = up_block(u2, d1, cond, BASE_FILTERS * 2, "g_up1")
    u0 = up_block(u1, x0, cond, BASE_FILTERS, "g_up0")
    out = tf.keras.layers.Conv2D(3, 7, padding="same", activation="tanh", name="rgb")(u0)
    return tf.keras.Model([mask_in, class_in], out, name=name)


def build_discriminator() -> tf.keras.Model:
    image_in = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    mask_in = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1), name="mask")
    class_in = tf.keras.Input((), dtype=tf.int32, name="class_id")
    class_map = class_map_layer(class_in, IMG_SIZE, IMG_SIZE, "dclass")
    x = tf.keras.layers.Concatenate(name="disc_input")([image_in, mask_in, class_map])
    for idx, ch in enumerate([BASE_FILTERS * 2, BASE_FILTERS * 4, BASE_FILTERS * 8]):
        x = tf.keras.layers.Conv2D(ch, 4, strides=2, padding="same", use_bias=False, name=f"d{idx}_conv")(x)
        if idx:
            x = tf.keras.layers.BatchNormalization(name=f"d{idx}_bn")(x)
        x = tf.keras.layers.LeakyReLU(0.2, name=f"d{idx}_lrelu")(x)
    out = tf.keras.layers.Conv2D(1, 4, padding="same", name="patch")(x)
    return tf.keras.Model([image_in, mask_in, class_in], out, name="patch_discriminator")


def adversarial_loss(logits: tf.Tensor, target_real: bool) -> tf.Tensor:
    target = tf.ones_like(logits) if target_real else tf.zeros_like(logits)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, logits, from_logits=True))


def train_endpoint_renderer(source: RandomGanSource, steps_per_epoch: int) -> tf.keras.Model:
    generator = build_generator("spade_single_label_endpoint_renderer")
    if BASE_G_WEIGHTS.exists():
        generator.load_weights(BASE_G_WEIGHTS)
        return generator

    discriminator = build_discriminator()
    g_opt = tf.keras.optimizers.Adam(BASE_G_LR, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(BASE_D_LR, beta_1=0.5)
    WORK.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, BASE_EPOCHS + 1):
        g_losses, d_losses = [], []
        for _ in tqdm(range(steps_per_epoch), desc=f"SPADE endpoint epoch {epoch}", leave=False):
            real, mask, labels = source.single_batch()
            with tf.GradientTape() as gt, tf.GradientTape() as dt:
                fake = generator([mask, labels], training=True)
                real_logits = discriminator([real, mask, labels], training=True)
                fake_logits = discriminator([fake, mask, labels], training=True)
                g_adv = adversarial_loss(fake_logits, True)
                g_l1 = tf.reduce_mean(tf.abs((real - fake) * mask))
                g_loss = g_adv + LAMBDA_L1 * g_l1
                d_loss = 0.5 * (adversarial_loss(real_logits, True) + adversarial_loss(fake_logits, False))
            g_opt.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables), generator.trainable_variables))
            d_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables), discriminator.trainable_variables))
            g_losses.append(float(g_loss.numpy()))
            d_losses.append(float(d_loss.numpy()))
        print(f"endpoint epoch={epoch} g_loss={np.mean(g_losses):.5f} d_loss={np.mean(d_losses):.5f}")

    generator.save_weights(BASE_G_WEIGHTS)
    return generator


def load_detector_background() -> np.ndarray:
    require(BACKGROUND_TEMPLATE)
    img = Image.open(BACKGROUND_TEMPLATE).convert("RGB").resize((OUT_W, OUT_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def crop_to_mask(mask: np.ndarray) -> np.ndarray:
    mask2 = mask[..., 0] if mask.ndim == 3 else mask
    ys, xs = np.where(mask2 > 0.5)
    if ys.size == 0:
        raise ValueError("empty mask")
    return mask2[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]


def resize_mask_long_axis(mask: np.ndarray, long_axis: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = long_axis / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return (resized >= 0.5).astype(np.float32)


def place_mask(mask: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    h, w = mask.shape[:2]
    x0 = int(center_x - w // 2)
    y0 = int(center_y - h // 2)
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = w - max(0, x1 - IMG_SIZE), h - max(0, y1 - IMG_SIZE)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return canvas


def overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    overlap = float(np.sum((mask_a > 0.5) & (mask_b > 0.5)))
    denom = max(1.0, min(float(np.sum(mask_a > 0.5)), float(np.sum(mask_b > 0.5))))
    return overlap / denom


def place_controlled_overlap(mask_a_small: np.ndarray, mask_b_small: np.ndarray, rng: random.Random):
    jitter_ax = rng.randint(-PLACEMENT_JITTER_PX, PLACEMENT_JITTER_PX)
    jitter_bx = rng.randint(-PLACEMENT_JITTER_PX, PLACEMENT_JITTER_PX)
    jitter_y = rng.randint(-PLACEMENT_JITTER_PX, PLACEMENT_JITTER_PX)
    center_y = IMG_SIZE // 2 + VERTICAL_OFFSET_PX + jitter_y
    center_a = IMG_SIZE // 2 - CENTER_OFFSET_PX + jitter_ax
    base_b = IMG_SIZE // 2 + CENTER_OFFSET_PX + jitter_bx
    mask_a = place_mask(mask_a_small, center_a, center_y)

    target_mid = 0.5 * (MIN_OVERLAP_FRAC + MAX_OVERLAP_FRAC)
    best = None
    for step in range(-OVERLAP_ADJUST_STEPS, OVERLAP_ADJUST_STEPS + 1):
        center_b = base_b + step * OVERLAP_ADJUST_PX
        mask_b = place_mask(mask_b_small, center_b, center_y)
        frac = overlap_fraction(mask_a, mask_b)
        penalty = 0.0 if MIN_OVERLAP_FRAC <= frac <= MAX_OVERLAP_FRAC else min(abs(frac - MIN_OVERLAP_FRAC), abs(frac - MAX_OVERLAP_FRAC))
        score = penalty + 0.1 * abs(frac - target_mid)
        if best is None or score < best[0]:
            best = (score, frac, mask_b, center_b)
    assert best is not None
    return mask_a, best[2], {
        "overlap_fraction_of_smaller_mask": float(best[1]),
        "center_a_x": int(center_a),
        "center_b_x": int(best[3]),
        "center_y": int(center_y),
    }


def low_frequency_priority(overlap: np.ndarray, class_a: int, class_b: int, sample_idx: int) -> np.ndarray:
    if overlap.max() <= 0:
        return np.zeros_like(overlap, dtype=np.float32)
    seed = ((SEED * 1000003) + (class_a * 9176) + (class_b * 131) + (sample_idx * 7919)) & 0xFFFFFFFF
    rng = random.Random(seed)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    phase_a = rng.uniform(0.0, math.tau)
    phase_b = rng.uniform(0.0, math.tau)
    period = max(8.0, ENTANGLE_PERIOD_PX)
    field = (
        np.sin(((xx + 0.55 * yy) / period) + phase_a)
        + 0.72 * np.sin((((0.35 * xx) - yy) / (period * 0.73)) + phase_b)
    ).astype(np.float32)
    grid_h = max(4, int(round(IMG_SIZE / period)))
    grid_w = max(4, int(round(IMG_SIZE / period)))
    noise_values = np.array([rng.uniform(0.0, 255.0) for _ in range(grid_h * grid_w)], dtype=np.float32)
    noise = noise_values.reshape(grid_h, grid_w)
    noise_img = Image.fromarray(noise.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    noise_arr = (np.asarray(noise_img, dtype=np.float32) / 127.5) - 1.0
    field = field + ENTANGLE_NOISE_WEIGHT * noise_arr
    priority = 1.0 / (1.0 + np.exp(-field / max(0.03, ENTANGLE_TRANSITION_SOFTNESS)))
    return priority.astype(np.float32)


def soften(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    k = max(3, int(round(radius * 6.0)) | 1)
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), float(radius))


def render_endpoint(generator: tf.keras.Model, mask: np.ndarray, class_id: int) -> np.ndarray:
    mask_t = tf.convert_to_tensor(mask[None, ..., None], dtype=tf.float32)
    class_t = tf.constant([int(class_id)], dtype=tf.int32)
    pred = generator([mask_t, class_t], training=False)[0].numpy()
    return np.clip(pred, -1.0, 1.0).astype(np.float32)


def detector_composite(
    gen_a: np.ndarray,
    gen_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    class_a: int,
    class_b: int,
    sample_idx: int,
    bg: np.ndarray,
):
    overlap = mask_a * mask_b
    priority_a = low_frequency_priority(overlap, class_a, class_b, sample_idx)
    region_a = mask_a * (1.0 - mask_b) + overlap * priority_a
    region_b = mask_b * (1.0 - mask_a) + overlap * (1.0 - priority_a)
    union = np.maximum(mask_a, mask_b)
    region_a = soften(region_a, EDGE_SOFTEN_RADIUS) * union
    region_b = soften(region_b, EDGE_SOFTEN_RADIUS) * union
    total = np.clip(region_a + region_b, 0.0, 1.0) * union
    fg = region_a[..., None] * gen_a + region_b[..., None] * gen_b

    fg_rgb = np.clip((fg + 1.0) * 127.5, 0, 255).astype(np.float32)
    fg_detector = cv2.resize(fg_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    mask_detector = cv2.resize(total.astype(np.float32), (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    out = bg * (1.0 - mask_detector[..., None]) + fg_detector * mask_detector[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if BOTTOM_BLACK_ROWS > 0:
        out[-BOTTOM_BLACK_ROWS:, :, :] = 0
    return out, union, region_a, region_b


def render_pair(
    generator: tf.keras.Model,
    files_by_class: dict[int, list[tuple[Path, Path]]],
    a_id: int,
    b_id: int,
    sample_idx: int,
    rng: random.Random,
    bg: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    a_img, a_mask_path = rng.choice(files_by_class[a_id])
    b_img, b_mask_path = rng.choice(files_by_class[b_id])
    _ia, ma = load_pair(a_img, a_mask_path)
    _ib, mb = load_pair(b_img, b_mask_path)
    ma = resize_mask_long_axis(crop_to_mask(ma), PAIR_LONG_AXIS_PX)
    mb = resize_mask_long_axis(crop_to_mask(mb), PAIR_LONG_AXIS_PX)
    mask_a, mask_b, placement = place_controlled_overlap(ma, mb, rng)
    gen_a = render_endpoint(generator, mask_a, a_id)
    gen_b = render_endpoint(generator, mask_b, b_id)
    image, _union, _region_a, _region_b = detector_composite(gen_a, gen_b, mask_a, mask_b, a_id, b_id, sample_idx, bg)
    sources = {
        "source_a_ref": public_source_ref(a_img),
        "source_b_ref": public_source_ref(b_img),
        "source_a_mask_ref": public_source_ref(a_mask_path),
        "source_b_mask_ref": public_source_ref(b_mask_path),
    }
    return image, {**sources, **placement}


def export_dataset(class_names: list[str], files_by_class: dict[int, list[tuple[Path, Path]]], generator: tf.keras.Model) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DOUBLEPICK").mkdir(exist_ok=True)
    rng = random.Random(SEED)
    bg = load_detector_background()
    background_ref = relative_input_path(BACKGROUND_TEMPLATE)
    background_sha = sha256_file(BACKGROUND_TEMPLATE)
    manifest = []
    sample_idx = 0
    for a_id in tqdm(range(CLASS_COUNT), desc="GAN ordered pairs", unit="class"):
        for b_id in range(CLASS_COUNT):
            if a_id == b_id:
                continue
            image, placement = render_pair(generator, files_by_class, a_id, b_id, sample_idx, rng, bg)
            stem = f"gan_a{a_id:03d}_b{b_id:03d}"
            out_png = OUT / "DOUBLEPICK" / f"{stem}.png"
            Image.fromarray(image).save(out_png)
            row = {
                "method": "gan",
                "render_mode": "single_label_endpoint_renderer_plus_two_source_compositor",
                "image_path": str(out_png.relative_to(OUT)),
                "class_a_id": a_id,
                "class_b_id": b_id,
                "class_a_name": class_names[a_id],
                "class_b_name": class_names[b_id],
                "ordered_pair": True,
                "active_classes": [a_id, b_id],
                "detector_label_active_set": sorted([a_id, b_id]),
                "canvas_height": IMG_SIZE,
                "canvas_width": IMG_SIZE,
                "output_height": OUT_H,
                "output_width": OUT_W,
                "placement_long_axis_px": PAIR_LONG_AXIS_PX,
                "center_offset_px": CENTER_OFFSET_PX,
                "vertical_offset_px": VERTICAL_OFFSET_PX,
                "overlap_priority_mode": "random_field",
                "bottom_black_rows": BOTTOM_BLACK_ROWS,
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
                "generator_arch": "spade_unet_endpoint",
                "base_filters": BASE_FILTERS,
                "class_embed_dim": CLASS_EMBED_DIM,
                "class_map_channels": CLASS_MAP_CHANNELS,
                **placement,
            }
            out_png.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            manifest.append(row)
            sample_idx += 1
    (OUT / "dataset_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest), encoding="utf-8")
    (OUT / "class_mapping.json").write_text(json.dumps({i: n for i, n in enumerate(class_names)}, indent=2), encoding="utf-8")
    (OUT / "generation_config.json").write_text(
        json.dumps(
            {
                "method": "gan",
                "render_mode": "single_label_endpoint_renderer_plus_two_source_compositor",
                "classes": CLASS_COUNT,
                "ordered_pairs": THESIS_DETECTOR_COUNT,
                "thesis_detector_count": THESIS_DETECTOR_COUNT,
                "count_scope": "final_136_class_detector_run",
                "base_epochs": BASE_EPOCHS,
                "base_batch": BASE_BATCH,
                "generator_arch": "spade_unet_endpoint",
                "base_filters": BASE_FILTERS,
                "class_embed_dim": CLASS_EMBED_DIM,
                "class_map_channels": CLASS_MAP_CHANNELS,
                "g_upsample_mode": "bilinear",
                "norm_type": "batch",
                "output_height": OUT_H,
                "output_width": OUT_W,
                "bottom_black_rows": BOTTOM_BLACK_ROWS,
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
                "seed": SEED,
                "generated": len(manifest),
                "generated_count_matches_thesis": len(manifest) == THESIS_DETECTOR_COUNT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} GAN samples to {OUT}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    class_names, files_by_class = discover_data()
    total_files = sum(len(v) for v in files_by_class.values())
    source = RandomGanSource(files_by_class, BASE_BATCH)
    steps_per_epoch = max(1, total_files // BASE_BATCH)
    generator = train_endpoint_renderer(source, steps_per_epoch)
    TRAINING_HISTORY_PATH.write_text(json.dumps(GAN_TRAINING_HISTORY, indent=2), encoding="utf-8")
    export_dataset(class_names, files_by_class, generator)


if __name__ == "__main__":
    main()

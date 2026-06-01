#!/usr/bin/env python3
"""Standalone VAE plus latent-diffusion reproduction for the latent datasets."""

from __future__ import annotations

import json
import math
import random
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

from appendix_data import discover_class_image_files, load_rgb_minus_one_one, output_dir, real_340_root, work_dir


WORK = work_dir("latent_diffusion")
OUT_ROOT = output_dir("")
REAL_DATA = real_340_root()

SEED = 1337
IMG_H = 680
IMG_W = 340
LATENT_CH = 6
CLASS_COUNT = 136
VAE_COMPRESSION = 4
BASE_CH = 256
COND_DIM = 1280
VAE_EPOCHS = 20
VAE_STEPS_PER_EPOCH = 250
VAE_LR = 6e-5
VAE_KL_BETA = 5e-4
VAE_FREE_BITS = 0.02
VAE_KL_WARMUP_STEPS = 40000
DIFF_EPOCHS = 30
DIFF_STEPS_PER_EPOCH = 1000
DIFF_LR = 5e-6
CLASS_DROPOUT_P = 0.1
EMA_DECAY = 0.9998
NUM_RES_BLOCKS = 5
DIFFUSION_STEPS = 1000
SWITCH_SAMPLER_STEPS = 50
SWEEP_SAMPLER_STEPS = 200
SWITCH_GUIDANCE = 2.0
SWEEP_GUIDANCE = 5.2
LATENT_CLIP = 3.4
# Diagnostic transition sweep only; the detector-facing latent export uses the
# full ordered non-self grid before quality filtering.
SWEEP_PAIR_FRACTION = 0.25
SWITCH_NOMINAL_ORDERED_COUNT = CLASS_COUNT * (CLASS_COUNT - 1)
SWITCH_THESIS_FILTERED_COUNT = 16080
THESIS_DETECTOR_COUNT = SWITCH_THESIS_FILTERED_COUNT

VAE_WEIGHTS = WORK / "vae.weights.h5"
DIFF_WEIGHTS = WORK / "latent_diffusion.weights.h5"
LATENT_STATS = WORK / "latent_stats.npz"
TRAINING_HISTORY_PATH = WORK / "training_history.json"

QUALITY_THRESHOLDS = {
    "foreground_threshold": 12.0,
    "near_black_threshold": 8.0,
    "near_white_threshold": 247.0,
    "edge_threshold": 10.0,
    "null_min_foreground_frac": 0.18,
    "null_min_entropy": 1.70,
    "null_min_edge_density": 0.005,
    "low_detail_min_foreground_frac": 0.25,
    "low_detail_min_entropy": 2.40,
    "low_detail_min_edge_density": 0.010,
    "low_detail_min_lap_var": 14.0,
}
FILTER_KEEP_CATEGORIES = {"usable"}

LATENT_TRAINING_HISTORY = [
    {
        "stage": "vae_training",
        "role": "Train the image autoencoder used to define latent space.",
        "epochs": VAE_EPOCHS,
        "steps_per_epoch": VAE_STEPS_PER_EPOCH,
        "kl_beta": VAE_KL_BETA,
        "kl_free_bits": VAE_FREE_BITS,
        "kl_warmup_steps": VAE_KL_WARMUP_STEPS,
        "checkpoint": VAE_WEIGHTS.name,
    },
    {
        "stage": "latent_statistics",
        "role": "Compute latent mean and standard deviation used for diffusion normalization.",
        "output": LATENT_STATS.name,
    },
    {
        "stage": "single_label_latent_diffusion",
        "role": "Train the class-conditional latent diffusion model from single-label data.",
        "epochs": DIFF_EPOCHS,
        "steps_per_epoch": DIFF_STEPS_PER_EPOCH,
        "objective": "velocity prediction",
        "cosine_schedule_steps": DIFFUSION_STEPS,
        "classifier_free_dropout": CLASS_DROPOUT_P,
        "ema_decay": EMA_DECAY,
        "checkpoint": DIFF_WEIGHTS.name,
    },
    {
        "stage": "two_label_conditioned_sampling",
        "role": "Generate two-item datasets by mixing two class conditions during sampling; no separate two-label latent model was recovered.",
        "outputs": ["latent_diffusion", "latent_diffusion_sweep"],
    },
    {
        "stage": "fixed_quality_screen_filtering",
        "role": "Apply the recovered latent-diffusion quality screen and write the detector-facing keep manifest.",
        "quality_screen": "luminance foreground occupancy, luminance entropy, edge density, and Laplacian variance",
        "keep_categories": sorted(FILTER_KEEP_CATEGORIES),
        "reject_categories": ["low_detail", "null_or_tiny"],
        "nominal_ordered_nonself_count": SWITCH_NOMINAL_ORDERED_COUNT,
        "thesis_detector_count": SWITCH_THESIS_FILTERED_COUNT,
        "historical_sources": [
            "scripts/analyze_latent_diffusion_acceptance.py",
            "scripts/audit_latent_diffusion_acceptance_web.py",
            "scripts/calibrate_latent_diffusion_acceptance_with_audit.py",
            "scripts/move_latent_diffusion_unusable_samples.py",
            "analysis_outputs/latent_diffusion_acceptance_20260414/latent_diffusion_training_quality_error_estimate_v2_20260414.json",
        ],
    },
]


def discover_data() -> tuple[list[str], dict[int, list[Path]]]:
    return discover_class_image_files(REAL_DATA, class_limit=CLASS_COUNT, min_classes=CLASS_COUNT)


def load_image(path: Path) -> np.ndarray:
    return load_rgb_minus_one_one(path, height=IMG_H, width=IMG_W)


def from_minus_one_one(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value((tf.cast(x, tf.float32) + 1.0) * 0.5, 0.0, 1.0)


def lab_from_rgb01(x01: tf.Tensor) -> tf.Tensor:
    x01 = tf.cast(tf.clip_by_value(x01, 0.0, 1.0), tf.float32)
    linear = tf.where(x01 <= 0.04045, x01 / 12.92, tf.pow((x01 + 0.055) / 1.055, 2.4))
    rgb_to_xyz = tf.constant(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        tf.float32,
    )
    xyz = tf.tensordot(linear, rgb_to_xyz, axes=[[3], [1]])
    x = tf.clip_by_value(xyz[..., 0] / 0.95047, 1e-8, 10.0)
    y = tf.clip_by_value(xyz[..., 1], 1e-8, 10.0)
    z = tf.clip_by_value(xyz[..., 2] / 1.08883, 1e-8, 10.0)
    eps = tf.constant(216.0 / 24389.0, tf.float32)
    k = tf.constant(24389.0 / 27.0, tf.float32)

    def f(v: tf.Tensor) -> tf.Tensor:
        return tf.where(v > eps, tf.pow(v, 1.0 / 3.0), (k * v + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return tf.stack([lightness / 100.0, a / 110.0, b / 110.0], axis=-1)


def sobel_edges(img01: tf.Tensor) -> tf.Tensor:
    gray = tf.reduce_mean(img01, axis=-1, keepdims=True)
    sobel_x = tf.reshape(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32), [3, 3, 1, 1])
    sobel_y = tf.reshape(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32), [3, 3, 1, 1])
    edge_x = tf.nn.conv2d(gray, sobel_x, strides=1, padding="SAME")
    edge_y = tf.nn.conv2d(gray, sobel_y, strides=1, padding="SAME")
    return tf.sqrt(tf.square(edge_x) + tf.square(edge_y) + 1e-8)


class RandomImageSource:
    def __init__(self, files_by_class: dict[int, list[Path]], batch_size: int):
        self.files_by_class = files_by_class
        self.batch_size = batch_size
        self.class_ids = [cid for cid, files in files_by_class.items() if files]
        self.rng = random.Random(SEED)
        if not self.class_ids:
            raise SystemExit("No training images found")

    def batch(self) -> tuple[tf.Tensor, tf.Tensor]:
        xs = []
        ys = []
        for _ in range(self.batch_size):
            cid = self.rng.choice(self.class_ids)
            xs.append(load_image(self.rng.choice(self.files_by_class[cid])))
            ys.append(cid)
        return tf.convert_to_tensor(np.stack(xs), tf.float32), tf.convert_to_tensor(ys, tf.int32)


def conv_block(x: tf.Tensor, ch: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_c1")(x)
    x = tf.keras.layers.GroupNormalization(groups=8, name=f"{name}_gn1")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_sw1")(x)
    x = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_c2")(x)
    x = tf.keras.layers.GroupNormalization(groups=8, name=f"{name}_gn2")(x)
    return tf.keras.layers.Activation("swish", name=f"{name}_sw2")(x)


def build_encoder() -> tf.keras.Model:
    inp = tf.keras.Input((IMG_H, IMG_W, 3))
    x = conv_block(inp, 64, "enc0")
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="swish", name="enc_down1")(x)
    x = conv_block(x, 128, "enc1")
    x = tf.keras.layers.Conv2D(192, 3, strides=2, padding="same", activation="swish", name="enc_down2")(x)
    x = conv_block(x, 192, "enc2")
    mean = tf.keras.layers.Conv2D(
        LATENT_CH,
        1,
        padding="same",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        name="z_mean",
    )(x)
    logvar_raw = tf.keras.layers.Conv2D(
        LATENT_CH,
        1,
        padding="same",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
        bias_initializer=tf.keras.initializers.Constant(-1.0),
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        name="z_logvar_raw",
    )(x)
    logvar = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -2.0, 0.0), name="z_logvar")(logvar_raw)
    return tf.keras.Model(inp, [mean, logvar], name="latent_vae_encoder")


def build_decoder() -> tf.keras.Model:
    inp = tf.keras.Input((IMG_H // VAE_COMPRESSION, IMG_W // VAE_COMPRESSION, LATENT_CH))
    x = tf.keras.layers.Conv2D(192, 3, padding="same", activation="swish", name="dec_in")(inp)
    x = conv_block(x, 192, "dec0")
    x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="swish", name="dec_up1")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="swish", name="dec_c1")(x)
    x = conv_block(x, 128, "dec1")
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="swish", name="dec_up2")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="swish", name="dec_c2")(x)
    x = conv_block(x, 64, "dec2")
    out = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh", name="rgb")(x)
    return tf.keras.Model(inp, out, name="latent_vae_decoder")


def timestep_embedding(t: tf.Tensor, dim: int) -> tf.Tensor:
    half = dim // 2
    freqs = tf.exp(-math.log(10000.0) * tf.range(half, dtype=tf.float32) / float(half - 1))
    args = tf.cast(t[:, None], tf.float32) * freqs[None, :]
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


def res_block(x: tf.Tensor, cond: tf.Tensor, ch: int, name: str) -> tf.Tensor:
    skip = x
    if x.shape[-1] != ch:
        skip = tf.keras.layers.Conv2D(ch, 1, padding="same", name=f"{name}_skip")(skip)
    h = tf.keras.layers.GroupNormalization(groups=8, name=f"{name}_gn1")(x)
    h = tf.keras.layers.Activation("swish", name=f"{name}_sw1")(h)
    h = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_c1")(h)
    c = tf.keras.layers.Dense(ch, name=f"{name}_cond")(cond)
    h = h + c[:, None, None, :]
    h = tf.keras.layers.GroupNormalization(groups=8, name=f"{name}_gn2")(h)
    h = tf.keras.layers.Activation("swish", name=f"{name}_sw2")(h)
    h = tf.keras.layers.Conv2D(ch, 3, padding="same", name=f"{name}_c2")(h)
    return skip + h


def res_block_stack(x: tf.Tensor, cond: tf.Tensor, ch: int, name: str) -> tf.Tensor:
    for idx in range(NUM_RES_BLOCKS):
        x = res_block(x, cond, ch, f"{name}_{idx}")
    return x


def build_diffusion(num_classes: int) -> tf.keras.Model:
    z_in = tf.keras.Input((IMG_H // VAE_COMPRESSION, IMG_W // VAE_COMPRESSION, LATENT_CH), name="z_t")
    t_in = tf.keras.Input((), dtype=tf.int32, name="t")
    cond_in = tf.keras.Input((num_classes,), name="class_condition")
    t_emb = timestep_embedding(t_in, COND_DIM)
    c_emb = tf.keras.layers.Dense(COND_DIM, use_bias=False, name="class_embedding")(cond_in)
    cond = tf.keras.layers.Dense(COND_DIM, activation="swish", name="cond_merge")(t_emb + c_emb)

    h1 = tf.keras.layers.Conv2D(BASE_CH, 3, padding="same", name="in_conv")(z_in)
    h1 = res_block_stack(h1, cond, BASE_CH, "down0")
    d1 = tf.keras.layers.Conv2D(BASE_CH * 2, 3, strides=2, padding="same", name="downsample1")(h1)
    h2 = res_block_stack(d1, cond, BASE_CH * 2, "down1")
    d2 = tf.keras.layers.Conv2D(BASE_CH * 4, 3, strides=2, padding="same", name="downsample2")(h2)
    h3 = res_block_stack(d2, cond, BASE_CH * 4, "down2")
    mid = res_block_stack(h3, cond, BASE_CH * 4, "mid")

    u1 = tf.keras.layers.Conv2DTranspose(BASE_CH * 2, 4, strides=2, padding="same", name="upsample1")(mid)
    u1 = tf.keras.layers.Resizing(int(h2.shape[1]), int(h2.shape[2]), interpolation="bilinear", name="upsample1_match_skip")(u1)
    u1 = tf.keras.layers.Concatenate(name="skip_cat1")([u1, h2])
    u1 = res_block_stack(u1, cond, BASE_CH * 2, "up1")

    u2 = tf.keras.layers.Conv2DTranspose(BASE_CH, 4, strides=2, padding="same", name="upsample2")(u1)
    u2 = tf.keras.layers.Resizing(int(h1.shape[1]), int(h1.shape[2]), interpolation="bilinear", name="upsample2_match_skip")(u2)
    u2 = tf.keras.layers.Concatenate(name="skip_cat2")([u2, h1])
    u2 = res_block_stack(u2, cond, BASE_CH, "up0")

    out = tf.keras.layers.GroupNormalization(groups=8, name="out_gn")(u2)
    out = tf.keras.layers.Activation("swish", name="out_sw")(out)
    out = tf.keras.layers.Conv2D(
        LATENT_CH,
        3,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="velocity",
    )(out)
    return tf.keras.Model([z_in, t_in, cond_in], out, name="latent_velocity_predictor")


def sample_vae(mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    eps = tf.random.normal(tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * eps


def vae_training_loss(
    images: tf.Tensor,
    recon: tf.Tensor,
    mean: tf.Tensor,
    logvar: tf.Tensor,
    global_step: int,
) -> tf.Tensor:
    kl_beta = VAE_KL_BETA * min(1.0, float(global_step + 1) / float(VAE_KL_WARMUP_STEPS))
    kl_per_dim = -0.5 * (1.0 + logvar - tf.square(mean) - tf.exp(logvar))
    kl_by_channel = tf.reduce_mean(kl_per_dim, axis=[0, 1, 2])
    kl = tf.reduce_sum(tf.nn.relu(kl_by_channel - tf.cast(VAE_FREE_BITS, kl_by_channel.dtype)))

    x01 = from_minus_one_one(images)
    recon01 = from_minus_one_one(recon)
    lab = lab_from_rgb01(x01)
    lab_recon = lab_from_rgb01(recon01)
    hsv = tf.image.rgb_to_hsv(x01)
    hsv_recon = tf.image.rgb_to_hsv(recon01)

    saturation_mask = tf.pow(hsv[..., 1:2], 1.2)
    saturation_mask /= tf.reduce_mean(saturation_mask, axis=[1, 2, 3], keepdims=True) + 1e-6
    saturation_mask = tf.clip_by_value(saturation_mask, 0.0, 3.0)

    recon_mse = tf.reduce_mean(tf.square(images - recon))
    recon_l1 = tf.reduce_mean(tf.abs(x01 - recon01))
    chroma_l1 = tf.reduce_mean(tf.abs(lab[..., 1:] - lab_recon[..., 1:]))
    sat_chroma_l1 = tf.reduce_mean(tf.abs(lab[..., 1:] - lab_recon[..., 1:]) * saturation_mask)

    hue_diff = tf.abs(hsv[..., 0] - hsv_recon[..., 0])
    hue_loss = tf.reduce_mean(tf.minimum(hue_diff, 1.0 - hue_diff))
    sv_loss = tf.reduce_mean(tf.abs(hsv[..., 1:] - hsv_recon[..., 1:]))
    hsv_l1 = hue_loss + 0.6 * sv_loss

    mean_color_l1 = tf.reduce_mean(
        tf.abs(tf.reduce_mean(x01, axis=[1, 2]) - tf.reduce_mean(recon01, axis=[1, 2]))
    )
    color_std_l1 = tf.reduce_mean(
        tf.abs(tf.math.reduce_std(x01, axis=[1, 2]) - tf.math.reduce_std(recon01, axis=[1, 2]))
    )
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(x01, recon01, max_val=1.0))

    edges_x = sobel_edges(x01)
    edges_recon = sobel_edges(recon01)
    edge_loss = tf.reduce_mean(tf.abs(edges_x - edges_recon))
    edge_mask = edges_x / (tf.reduce_mean(edges_x, axis=[1, 2, 3], keepdims=True) + 1e-6)
    edge_mask = tf.clip_by_value(edge_mask, 0.0, 5.0)
    edge_chroma_l1 = tf.reduce_mean(tf.abs(lab[..., 1:] - lab_recon[..., 1:]) * edge_mask)

    mean_mu = tf.reduce_mean(tf.abs(mean))
    return (
        0.55 * tf.cast(recon_mse, tf.float32)
        + 0.2 * tf.cast(recon_l1, tf.float32)
        + 2.0 * tf.cast(chroma_l1, tf.float32)
        + 1.0 * tf.cast(hsv_l1, tf.float32)
        + 0.6 * tf.cast(mean_color_l1, tf.float32)
        + 0.3 * tf.cast(ssim_loss, tf.float32)
        + 0.2 * tf.cast(color_std_l1, tf.float32)
        + 0.4 * tf.cast(edge_loss, tf.float32)
        + 0.6 * tf.cast(edge_chroma_l1, tf.float32)
        + 0.3 * tf.cast(sat_chroma_l1, tf.float32)
        + tf.cast(kl_beta, tf.float32) * tf.cast(kl, tf.float32)
        + 1e-2 * tf.cast(mean_mu, tf.float32)
    )


def train_vae(source: RandomImageSource) -> tuple[tf.keras.Model, tf.keras.Model]:
    encoder = build_encoder()
    decoder = build_decoder()
    opt = tf.keras.optimizers.Adam(VAE_LR)
    WORK.mkdir(parents=True, exist_ok=True)
    if VAE_WEIGHTS.exists():
        encoder, decoder = build_encoder(), build_decoder()
        combined = tf.keras.Model(encoder.input, decoder(encoder(encoder.input)[0]))
        combined.load_weights(VAE_WEIGHTS)
        return encoder, decoder

    global_step = 0
    for epoch in range(1, VAE_EPOCHS + 1):
        losses = []
        for _ in tqdm(range(VAE_STEPS_PER_EPOCH), desc=f"vae epoch {epoch}", leave=False):
            images, _labels = source.batch()
            with tf.GradientTape() as tape:
                mean, logvar = encoder(images, training=True)
                z = sample_vae(mean, logvar)
                recon = decoder(z, training=True)
                loss = vae_training_loss(images, recon, mean, logvar, global_step)
            vars_ = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, vars_)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            opt.apply_gradients(zip(grads, vars_))
            global_step += 1
            losses.append(float(loss.numpy()))
        print(f"vae epoch={epoch} loss={np.mean(losses):.6f}")

    inp = tf.keras.Input((IMG_H, IMG_W, 3))
    mean, _logvar = encoder(inp)
    combined = tf.keras.Model(inp, decoder(mean))
    combined.save_weights(VAE_WEIGHTS)
    return encoder, decoder


def diffusion_alpha_bar() -> tf.Tensor:
    t = tf.linspace(0.0, 1.0, DIFFUSION_STEPS + 1)
    schedule_offset = 0.008
    alpha_bar = tf.cos((t + schedule_offset) / (1.0 + schedule_offset) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[:1]
    return tf.cast(alpha_bar, tf.float32)


def one_hot(labels: tf.Tensor, num_classes: int) -> tf.Tensor:
    return tf.one_hot(tf.cast(labels, tf.int32), num_classes, dtype=tf.float32)


def compute_latent_stats(encoder: tf.keras.Model, source: RandomImageSource) -> tuple[np.ndarray, np.ndarray]:
    if LATENT_STATS.exists():
        data = np.load(LATENT_STATS)
        return data["mean"], data["std"]
    chunks = []
    for _ in tqdm(range(200), desc="latent statistics", leave=False):
        images, _labels = source.batch()
        mean, _ = encoder(images, training=False)
        chunks.append(mean.numpy())
    arr = np.concatenate(chunks, axis=0)
    mean = arr.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    std = np.maximum(arr.std(axis=(0, 1, 2), keepdims=True), 1e-4).astype(np.float32)
    np.savez(LATENT_STATS, mean=mean, std=std)
    return mean, std


def train_diffusion(
    encoder: tf.keras.Model,
    source: RandomImageSource,
    num_classes: int,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
) -> tf.keras.Model:
    model = build_diffusion(num_classes)
    if DIFF_WEIGHTS.exists():
        model.load_weights(DIFF_WEIGHTS)
        return model
    opt = tf.keras.optimizers.Adam(DIFF_LR)
    alpha_bar = diffusion_alpha_bar()
    latent_mean_t = tf.convert_to_tensor(latent_mean, tf.float32)
    latent_std_t = tf.convert_to_tensor(latent_std, tf.float32)
    ema_vars = [tf.Variable(var.read_value(), trainable=False) for var in model.trainable_variables]

    for epoch in range(1, DIFF_EPOCHS + 1):
        losses = []
        for _ in tqdm(range(DIFF_STEPS_PER_EPOCH), desc=f"diffusion epoch {epoch}", leave=False):
            images, labels = source.batch()
            with tf.GradientTape() as tape:
                z_mean, _ = encoder(images, training=False)
                z0 = tf.clip_by_value((z_mean - latent_mean_t) / latent_std_t, -LATENT_CLIP, LATENT_CLIP)
                noise = tf.random.normal(tf.shape(z0))
                t = tf.random.uniform((tf.shape(z0)[0],), 1, DIFFUSION_STEPS + 1, dtype=tf.int32)
                ab = tf.gather(alpha_bar, t)[:, None, None, None]
                sqrt_ab = tf.sqrt(ab)
                sqrt_one_minus_ab = tf.sqrt(1.0 - ab)
                zt = sqrt_ab * z0 + sqrt_one_minus_ab * noise
                velocity_target = sqrt_ab * noise - sqrt_one_minus_ab * z0
                cond = one_hot(labels, num_classes)
                drop = tf.random.uniform((tf.shape(labels)[0], 1), dtype=tf.float32) < CLASS_DROPOUT_P
                cond = tf.where(drop, tf.zeros_like(cond), cond)
                pred = model([zt, t, cond], training=True)
                loss = tf.reduce_mean(tf.square(pred - velocity_target))
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            for ema_var, var in zip(ema_vars, model.trainable_variables):
                ema_var.assign(EMA_DECAY * ema_var + (1.0 - EMA_DECAY) * var)
            losses.append(float(loss.numpy()))
        print(f"diffusion epoch={epoch} loss={np.mean(losses):.6f}")
    for var, ema_var in zip(model.trainable_variables, ema_vars):
        var.assign(ema_var)
    model.save_weights(DIFF_WEIGHTS)
    return model


def make_condition(a_id: int, b_id: int, class_b_weight: float, num_classes: int) -> tf.Tensor:
    cond = np.zeros((1, num_classes), dtype=np.float32)
    cond[0, a_id] = 1.0 - class_b_weight
    cond[0, b_id] = class_b_weight
    return tf.convert_to_tensor(cond, tf.float32)


def sample_latent(
    model: tf.keras.Model,
    decoder: tf.keras.Model,
    num_classes: int,
    a_id: int,
    b_id: int,
    sampler_steps: int,
    guidance: float,
    transition_start: float,
    transition_duration: float,
    final_class_a_weight: float,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    seed: int,
) -> np.ndarray:
    alpha_bar = diffusion_alpha_bar()
    timesteps = np.linspace(DIFFUSION_STEPS, 1, sampler_steps, dtype=np.int32)
    timesteps = np.concatenate([timesteps, np.array([0], dtype=np.int32)])
    rng = tf.random.Generator.from_seed(seed)
    shape = (1, IMG_H // VAE_COMPRESSION, IMG_W // VAE_COMPRESSION, LATENT_CH)
    z = rng.normal(shape)
    uncond = tf.zeros((1, num_classes), tf.float32)

    for step_idx in range(sampler_steps):
        t_val = int(timesteps[step_idx])
        t_next = int(timesteps[step_idx + 1])
        frac = step_idx / max(1, sampler_steps - 1)
        if transition_duration <= 0:
            class_a_weight = final_class_a_weight if frac >= transition_start else 1.0
        else:
            mix = np.clip((frac - transition_start) / transition_duration, 0.0, 1.0)
            class_a_weight = 1.0 + (final_class_a_weight - 1.0) * mix
        class_b_weight = 1.0 - float(class_a_weight)
        t = tf.convert_to_tensor([int(t_val)], tf.int32)
        cond = make_condition(a_id, b_id, class_b_weight, num_classes)
        v_cond = model([z, t, cond], training=False)
        v_uncond = model([z, t, uncond], training=False)
        velocity = v_uncond + guidance * (v_cond - v_uncond)

        ab = tf.reshape(tf.gather(alpha_bar, t_val), [1, 1, 1, 1])
        ab_next = tf.reshape(tf.gather(alpha_bar, t_next), [1, 1, 1, 1])
        sqrt_ab = tf.sqrt(ab)
        sqrt_one = tf.sqrt(tf.maximum(1.0 - ab, 1e-8))
        sqrt_ab_next = tf.sqrt(ab_next)
        sqrt_one_next = tf.sqrt(tf.maximum(1.0 - ab_next, 1e-8))
        z0 = sqrt_ab * z - sqrt_one * velocity
        eps = sqrt_one * z + sqrt_ab * velocity
        z = sqrt_ab_next * z0 + sqrt_one_next * eps
        z = tf.clip_by_value(tf.where(tf.math.is_finite(z), z, tf.zeros_like(z)), -10.0, 10.0)

    z = z * tf.convert_to_tensor(latent_std, tf.float32) + tf.convert_to_tensor(latent_mean, tf.float32)
    z = tf.clip_by_value(z, -LATENT_CLIP, LATENT_CLIP)
    img = decoder(z, training=False)[0].numpy()
    img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img


def latent_image_metrics(png_path: Path) -> dict[str, float]:
    with Image.open(png_path) as im:
        rgb = im.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32)

    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    luma_u8 = np.clip(luma, 0, 255).astype(np.uint8)
    hist = np.bincount(luma_u8.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(float(hist.sum()), 1.0)
    nz = prob > 0
    entropy = float(-(prob[nz] * np.log2(prob[nz])).sum())

    p01, p99 = np.percentile(luma, [1, 99])
    dy = np.abs(np.diff(luma, axis=0))
    dx = np.abs(np.diff(luma, axis=1))
    edge_count = int(
        (dy > QUALITY_THRESHOLDS["edge_threshold"]).sum()
        + (dx > QUALITY_THRESHOLDS["edge_threshold"]).sum()
    )
    edge_total = int(dy.size + dx.size)

    if luma.shape[0] >= 3 and luma.shape[1] >= 3:
        center = luma[1:-1, 1:-1]
        lap = (
            4.0 * center
            - luma[:-2, 1:-1]
            - luma[2:, 1:-1]
            - luma[1:-1, :-2]
            - luma[1:-1, 2:]
        )
        lap_var = float(lap.var())
    else:
        lap_var = 0.0

    return {
        "width": float(width),
        "height": float(height),
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "p01_luma": float(p01),
        "p99_luma": float(p99),
        "dyn_p99_p01": float(p99 - p01),
        "entropy_luma": entropy,
        "foreground_frac": float((luma > QUALITY_THRESHOLDS["foreground_threshold"]).mean()),
        "near_black_frac": float((luma < QUALITY_THRESHOLDS["near_black_threshold"]).mean()),
        "near_white_frac": float((luma > QUALITY_THRESHOLDS["near_white_threshold"]).mean()),
        "grad_mean": float((dy.sum() + dx.sum()) / max(edge_total, 1)),
        "edge_density": float(edge_count / max(edge_total, 1)),
        "lap_var": lap_var,
    }


def classify_latent_quality(metrics: dict[str, float]) -> tuple[str, str]:
    null_reasons = []
    if metrics["foreground_frac"] < QUALITY_THRESHOLDS["null_min_foreground_frac"]:
        null_reasons.append("foreground_frac")
    if metrics["entropy_luma"] < QUALITY_THRESHOLDS["null_min_entropy"]:
        null_reasons.append("entropy_luma")
    if metrics["edge_density"] < QUALITY_THRESHOLDS["null_min_edge_density"]:
        null_reasons.append("edge_density")
    if null_reasons:
        return "null_or_tiny", ";".join(null_reasons)

    low_reasons = []
    if metrics["foreground_frac"] < QUALITY_THRESHOLDS["low_detail_min_foreground_frac"]:
        low_reasons.append("foreground_frac")
    if metrics["entropy_luma"] < QUALITY_THRESHOLDS["low_detail_min_entropy"]:
        low_reasons.append("entropy_luma")
    if metrics["edge_density"] < QUALITY_THRESHOLDS["low_detail_min_edge_density"]:
        low_reasons.append("edge_density")
    if metrics["lap_var"] < QUALITY_THRESHOLDS["low_detail_min_lap_var"]:
        low_reasons.append("lap_var")
    if low_reasons:
        return "low_detail", ";".join(low_reasons)

    return "usable", ""


def latent_quality_score(metrics: dict[str, float]) -> float:
    margins = [
        (metrics["foreground_frac"] - QUALITY_THRESHOLDS["low_detail_min_foreground_frac"])
        / QUALITY_THRESHOLDS["low_detail_min_foreground_frac"],
        (metrics["entropy_luma"] - QUALITY_THRESHOLDS["low_detail_min_entropy"])
        / QUALITY_THRESHOLDS["low_detail_min_entropy"],
        (metrics["edge_density"] - QUALITY_THRESHOLDS["low_detail_min_edge_density"])
        / QUALITY_THRESHOLDS["low_detail_min_edge_density"],
        (metrics["lap_var"] - QUALITY_THRESHOLDS["low_detail_min_lap_var"])
        / QUALITY_THRESHOLDS["low_detail_min_lap_var"],
    ]
    return float(min(margins))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_filter_status(name: str, out_dir: Path, manifest: list[dict[str, object]]) -> None:
    info: dict[str, object] = {
        "dataset": name,
        "class_count": CLASS_COUNT,
        "count_scope": "final_136_class_detector_run",
        "nominal_generated": len(manifest),
        "nominal_ordered_nonself_count": SWITCH_NOMINAL_ORDERED_COUNT,
    }
    if name != "latent_diffusion":
        info["filter_status"] = "not_applicable"
        (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return

    filtered: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}

    for row in manifest:
        image_path = Path(str(row["image_path"]))
        sample_id = image_path.stem
        metrics = latent_image_metrics(out_dir / image_path)
        category, reject_reasons = classify_latent_quality(metrics)
        keep_sample = category in FILTER_KEEP_CATEGORIES
        category_counts[category] = category_counts.get(category, 0) + 1
        decision_key = "kept" if keep_sample else "rejected"
        decision_counts[decision_key] = decision_counts.get(decision_key, 0) + 1
        quality_rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "class_a_id": row["class_a_id"],
                "class_b_id": row["class_b_id"],
                "quality_category": category,
                "reject_reasons": reject_reasons,
                "audit_quality_score": f"{latent_quality_score(metrics):.12g}",
                "decision": "keep" if keep_sample else "reject",
                **{key: f"{value:.12g}" for key, value in metrics.items()},
            }
        )
        if keep_sample:
            filtered.append(row)
        else:
            rejected.append(row)

    quality_csv = out_dir / "latent_quality_screen.csv"
    fieldnames = [
        "sample_id",
        "image_path",
        "class_a_id",
        "class_b_id",
        "quality_category",
        "reject_reasons",
        "audit_quality_score",
        "decision",
        "width",
        "height",
        "mean_luma",
        "std_luma",
        "p01_luma",
        "p99_luma",
        "dyn_p99_p01",
        "entropy_luma",
        "foreground_frac",
        "near_black_frac",
        "near_white_frac",
        "grad_mean",
        "edge_density",
        "lap_var",
    ]
    write_csv_rows(quality_csv, quality_rows, fieldnames)

    filtered_manifest = out_dir / "dataset_manifest_detector_filtered.jsonl"
    rejected_manifest = out_dir / "dataset_manifest_detector_rejected.jsonl"
    filtered_manifest.write_text("".join(json.dumps(r) + "\n" for r in filtered), encoding="utf-8")
    rejected_manifest.write_text("".join(json.dumps(r) + "\n" for r in rejected), encoding="utf-8")

    info.update(
        {
            "filter_status": "fixed_quality_screen_applied",
            "filter_method": "Recovered quality screen: keep usable; reject low_detail and null_or_tiny.",
            "quality_thresholds": QUALITY_THRESHOLDS,
            "quality_category_counts": dict(sorted(category_counts.items())),
            "filter_decision_counts": dict(sorted(decision_counts.items())),
            "thesis_detector_filtered_count": THESIS_DETECTOR_COUNT,
            "filtered_generated": len(filtered),
            "rejected_generated": len(rejected),
            "quality_screen_csv": quality_csv.name,
            "filtered_manifest": filtered_manifest.name,
            "rejected_manifest": rejected_manifest.name,
            "filtered_count_matches_thesis": len(filtered) == THESIS_DETECTOR_COUNT,
            "count_note": (
                "The thesis detector-run count is 16,080 latent-diffusion labels after "
                "the recovered quality screen removes low-detail or unusable generated rows from the "
                "nominal 18,360 ordered non-self export."
            ),
        }
    )
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def export_dataset(
    name: str,
    class_names: list[str],
    model: tf.keras.Model,
    decoder: tf.keras.Model,
    sampler_steps: int,
    guidance: float,
    transition_start: float,
    transition_duration: float,
    final_class_a_weight: float,
    pair_fraction: float,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
) -> None:
    out_dir = OUT_ROOT / name
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    class_ids = list(range(len(class_names)))
    pairs = [(a, b) for a in class_ids for b in class_ids if a != b]
    if pair_fraction < 1.0:
        rng = np.random.default_rng(SEED)
        count = max(1, int(math.floor(len(pairs) * pair_fraction + 0.5)))
        chosen = rng.choice(len(pairs), size=count, replace=False)
        chosen.sort()
        pairs = [pairs[int(i)] for i in chosen]

    manifest = []
    for idx, (a_id, b_id) in enumerate(tqdm(pairs, desc=name, unit="sample")):
        img = sample_latent(
            model,
            decoder,
            len(class_names),
            a_id,
            b_id,
            sampler_steps,
            guidance,
            transition_start,
            transition_duration,
            final_class_a_weight,
            latent_mean,
            latent_std,
            SEED + idx,
        )
        stem = f"a{a_id:03d}_b{b_id:03d}_{name}"
        out_png = out_dir / "samples" / f"{stem}.png"
        Image.fromarray(img).save(out_png)
        row = {
            "method": "latent_diffusion",
            "dataset": name,
            "image_path": str(out_png.relative_to(out_dir)),
            "class_a_id": a_id,
            "class_b_id": b_id,
            "class_a_name": class_names[a_id],
            "class_b_name": class_names[b_id],
            "active_classes": [a_id, b_id],
            "sampler_steps": sampler_steps,
            "guidance": guidance,
            "transition_start": transition_start,
            "transition_duration": transition_duration,
            "final_class_a_weight": final_class_a_weight,
            "final_class_b_weight": 1.0 - final_class_a_weight,
            "seed": SEED + idx,
        }
        out_png.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        manifest.append(row)
    (out_dir / "dataset_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest), encoding="utf-8")
    (out_dir / "class_mapping.json").write_text(json.dumps({i: n for i, n in enumerate(class_names)}, indent=2), encoding="utf-8")
    write_filter_status(name, out_dir, manifest)
    print(f"Wrote {len(manifest)} samples to {out_dir}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    class_names, files_by_class = discover_data()
    source = RandomImageSource(files_by_class, batch_size=6)
    encoder, decoder = train_vae(source)
    latent_mean, latent_std = compute_latent_stats(encoder, source)
    diffusion = train_diffusion(encoder, source, len(class_names), latent_mean, latent_std)
    TRAINING_HISTORY_PATH.write_text(json.dumps(LATENT_TRAINING_HISTORY, indent=2), encoding="utf-8")

    export_dataset(
        "latent_diffusion",
        class_names,
        diffusion,
        decoder,
        sampler_steps=SWITCH_SAMPLER_STEPS,
        guidance=SWITCH_GUIDANCE,
        transition_start=0.02,
        transition_duration=0.0,
        final_class_a_weight=0.0,
        pair_fraction=1.0,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    export_dataset(
        "latent_diffusion_sweep",
        class_names,
        diffusion,
        decoder,
        sampler_steps=SWEEP_SAMPLER_STEPS,
        guidance=SWEEP_GUIDANCE,
        transition_start=0.0,
        transition_duration=0.015,
        final_class_a_weight=0.0,
        pair_fraction=SWEEP_PAIR_FRACTION,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )


if __name__ == "__main__":
    main()

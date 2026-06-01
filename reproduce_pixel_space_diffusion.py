#!/usr/bin/env python3
"""Compact standalone pixel-space diffusion reproduction for the thesis export."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from appendix_data import (
    discover_class_image_mask_pairs,
    load_rgb_mask_minus_one_one,
    output_dir,
    pixel_openai_checkpoint,
    public_source_ref,
    real_512_mask_root,
    relative_input_path,
    require,
    sha256_file,
    shared_background_template,
    work_dir,
)


WORK = work_dir("pixel_space_diffusion")
REAL_DATA = real_512_mask_root()
OPENAI_PRETRAINED = pixel_openai_checkpoint()
BACKGROUND_TEMPLATE = shared_background_template()
RAW_OUT = output_dir("pixel_space_diffusion_raw")
FINAL_OUT = output_dir("pixel_space_diffusion")
LAYOUT_DIR = WORK / "layouts"

SINGLE_LABEL_MODEL_PATH = WORK / "stage1_single_label_rgb_anchor.pt"
PAIR_REPLAY_MODEL_PATH = WORK / "stage2_mixed_pair_replay_slot_model.pt"
MODEL_PATH = WORK / "stage3_pair_slot_model093408_equivalent.pt"
TRAINING_HISTORY_PATH = WORK / "training_history.json"

SEED = 2404
CLASS_COUNT = 136
THESIS_DETECTOR_COUNT = CLASS_COUNT * (CLASS_COUNT - 1)
IMG_SIZE = 512
OUT_H = 680
OUT_W = 340
RGB_CHANNELS = 3
SLOT_CHANNELS = 6
COND_CHANNELS = 2
BATCH_SIZE = 8

SINGLE_LABEL_STEPS = 93000
PAIR_REPLAY_STEPS = 2501
PAIR_REFINEMENT_STEPS = 408
SINGLE_LABEL_LR = 2e-5
PAIR_REPLAY_LR = 7e-7
PAIR_REFINEMENT_LR = 1.25e-7
DIFFUSION_STEPS = 1000
SAMPLE_STEPS = 10
MASK_LOSS_SCALE = 3.0

PAIR_LONG_AXIS_PX = 286
CENTER_OFFSET_PX = 38
VERTICAL_OFFSET_PX = 4
JITTER_PX = 10
OMEGA_PERIOD_PX = 64.0
OMEGA_TRANSITION_SOFTNESS = 0.13
OMEGA_NOISE_WEIGHT = 0.7

MERGE_LOSS_WEIGHT = 0.25
OCC_LOSS_WEIGHT = 0.05
OWNER_LOSS_WEIGHT = 0.05
PAIRSTRUCT_LOSS_WEIGHT = 0.05

PIXEL_TRAINING_HISTORY = [
    {
        "stage": "single_label_rgb_anchor",
        "local_checkpoint": SINGLE_LABEL_MODEL_PATH.name,
        "resume_source": "input/pixel/openai_pretrained.pt",
        "history_anchor": "openAIDiffusionData model093000.pt",
        "pair_probability": 0.0,
        "target": "single RGB image with single garment class",
        "data_channels": RGB_CHANNELS,
        "mask_loss_scale": MASK_LOSS_SCALE,
        "steps": SINGLE_LABEL_STEPS,
        "lr": SINGLE_LABEL_LR,
    },
    {
        "stage": "mixed_pair_with_single_replay",
        "local_checkpoint": PAIR_REPLAY_MODEL_PATH.name,
        "resume_source": SINGLE_LABEL_MODEL_PATH.name,
        "history_anchor": "phase2_mixed_pair_with_replay model095752.pt",
        "pair_probability": 0.5,
        "ordered_pair_condition": True,
        "slot_target_channels": SLOT_CHANNELS,
        "thesis_detector_count": THESIS_DETECTOR_COUNT,
        "steps": PAIR_REPLAY_STEPS,
        "lr": PAIR_REPLAY_LR,
    },
    {
        "stage": "pair_only_slot_refinement",
        "local_checkpoint": MODEL_PATH.name,
        "resume_source": PAIR_REPLAY_MODEL_PATH.name,
        "history_anchor": "selected model093408.pt pair/slot refinement chain",
        "pair_probability": 1.0,
        "slot_dual_output_head": True,
        "slot_merge_output_head": True,
        "slot_occ_output_head": True,
        "slot_owner_output_head": True,
        "slot_pairstruct_output_head": True,
        "thesis_detector_count": THESIS_DETECTOR_COUNT,
        "steps": PAIR_REFINEMENT_STEPS,
        "lr": PAIR_REFINEMENT_LR,
    },
]


def discover_data() -> tuple[list[str], dict[int, list[tuple[Path, Path]]]]:
    return discover_class_image_mask_pairs(REAL_DATA, class_limit=CLASS_COUNT, min_classes=CLASS_COUNT)


def load_detector_background() -> np.ndarray:
    require(BACKGROUND_TEMPLATE)
    image = Image.open(BACKGROUND_TEMPLATE).convert("RGB").resize((OUT_W, OUT_H), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def load_rgb_mask(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_rgb_mask_minus_one_one(image_path, mask_path, height=IMG_SIZE, width=IMG_SIZE, mask_channel=False)


def crop_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        raise ValueError("empty mask")
    return mask[int(ys.min()): int(ys.max()) + 1, int(xs.min()): int(xs.max()) + 1]


def resize_long_axis(mask: np.ndarray, long_axis: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = long_axis / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return (out >= 0.5).astype(np.float32)


def place(mask: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    h, w = mask.shape[:2]
    x0, y0 = center_x - w // 2, center_y - h // 2
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = w - max(0, x1 - IMG_SIZE), h - max(0, y1 - IMG_SIZE)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return canvas


def ownership_priority(overlap: np.ndarray, class_a: int, class_b: int, sample_idx: int) -> np.ndarray:
    if overlap.max() <= 0:
        return np.zeros_like(overlap, dtype=np.float32)
    seed = ((SEED * 1000003) + (class_a * 9176) + (class_b * 131) + (sample_idx * 7919)) & 0xFFFFFFFF
    rng = random.Random(seed)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    phase_a = rng.uniform(0.0, math.tau)
    phase_b = rng.uniform(0.0, math.tau)
    period = max(8.0, OMEGA_PERIOD_PX)
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
    field = field + OMEGA_NOISE_WEIGHT * noise_arr
    return (1.0 / (1.0 + np.exp(-field / max(0.03, OMEGA_TRANSITION_SOFTNESS)))).astype(np.float32)


def ownership_from_masks(mask_a: np.ndarray, mask_b: np.ndarray, class_a: int, class_b: int, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    mask_a = (mask_a > 0.5).astype(np.float32)
    mask_b = (mask_b > 0.5).astype(np.float32)
    overlap = mask_a * mask_b
    priority = ownership_priority(overlap, class_a, class_b, sample_idx)
    pos = mask_a * (1.0 - mask_b) + overlap * priority
    neg = mask_b * (1.0 - mask_a) + overlap * (1.0 - priority)
    u = np.maximum(mask_a, mask_b).astype(np.float32)
    omega = np.zeros_like(u, dtype=np.float32)
    omega[pos > 0.5] = 1.0
    omega[neg > 0.5] = -1.0
    return u, omega


def build_layouts(class_names: list[str], data: dict[int, list[tuple[Path, Path]]]) -> list[dict[str, object]]:
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    idx = 0
    for a_id in range(CLASS_COUNT):
        for b_id in range(CLASS_COUNT):
            if a_id == b_id:
                continue
            ia, ma_path = rng.choice(data[a_id])
            ib, mb_path = rng.choice(data[b_id])
            _xa, ma = load_rgb_mask(ia, ma_path)
            _xb, mb = load_rgb_mask(ib, mb_path)
            ma = resize_long_axis(crop_mask(ma), PAIR_LONG_AXIS_PX)
            mb = resize_long_axis(crop_mask(mb), PAIR_LONG_AXIS_PX)
            cy = IMG_SIZE // 2 + VERTICAL_OFFSET_PX + rng.randint(-JITTER_PX, JITTER_PX)
            mask_a = place(ma, IMG_SIZE // 2 - CENTER_OFFSET_PX + rng.randint(-JITTER_PX, JITTER_PX), cy)
            mask_b = place(mb, IMG_SIZE // 2 + CENTER_OFFSET_PX + rng.randint(-JITTER_PX, JITTER_PX), cy)
            u, omega = ownership_from_masks(mask_a, mask_b, a_id, b_id, idx)
            np.savez_compressed(LAYOUT_DIR / f"layout_{idx:06d}.npz", u=u, omega=omega, mask_a=mask_a, mask_b=mask_b)
            rows.append(
                {
                    "index": idx,
                    "class_a_id": a_id,
                    "class_b_id": b_id,
                    "y_pair": [a_id, b_id],
                    "class_a_name": class_names[a_id],
                    "class_b_name": class_names[b_id],
                    "layout_file": f"layout_{idx:06d}.npz",
                    "source_a_ref": public_source_ref(ia),
                    "source_a_mask_ref": public_source_ref(ma_path),
                    "source_b_ref": public_source_ref(ib),
                    "source_b_mask_ref": public_source_ref(mb_path),
                    "ownership_map": "omega in {-1,0,+1}, +1=slot_a, -1=slot_b",
                }
            )
            idx += 1
    (LAYOUT_DIR / "layout_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (LAYOUT_DIR / "class_map_136.json").write_text(json.dumps({name: i for i, name in enumerate(class_names)}, indent=2), encoding="utf-8")
    background = np.clip(load_detector_background(), 0, 255).astype(np.uint8)
    Image.fromarray(background).save(LAYOUT_DIR / "background_template.png")
    return rows


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_ch, out_ch)
        self.g1 = nn.GroupNorm(8, out_ch)
        self.g2 = nn.GroupNorm(8, out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.g1(self.c1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.c2(F.silu(self.g2(h)))
        return h + self.skip(x)


class SingleLabelRGBUNet(nn.Module):
    def __init__(self, classes: int = CLASS_COUNT, base: int = 128, cond_ch: int = 512):
        super().__init__()
        self.cond_ch = int(cond_ch)
        self.class_emb = nn.Embedding(classes, cond_ch)
        self.time_mlp = nn.Sequential(nn.Linear(cond_ch, cond_ch), nn.SiLU(), nn.Linear(cond_ch, cond_ch))
        self.in_conv = nn.Conv2d(RGB_CHANNELS, base, 3, padding=1)
        self.r1 = ResBlock(base, base, cond_ch)
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.r2 = ResBlock(base * 2, base * 2, cond_ch)
        self.mid = ResBlock(base * 2, base * 2, cond_ch)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.r3 = ResBlock(base * 2, base, cond_ch)
        self.noise_head = nn.Conv2d(base, RGB_CHANNELS, 3, padding=1)

    def timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(1, half - 1))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def condition(self, label: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.class_emb(label) + self.time_mlp(self.timestep_embedding(t, self.cond_ch))

    def forward(self, x_rgb: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cond = self.condition(label, t)
        h1 = self.r1(self.in_conv(x_rgb), cond)
        h2 = self.r2(self.down(h1), cond)
        h2 = self.mid(h2, cond)
        up = self.up(h2)
        h = self.r3(torch.cat([up, h1], dim=1), cond)
        return self.noise_head(h)


class SlotPixelUNet(nn.Module):
    def __init__(self, classes: int = CLASS_COUNT, base: int = 128, cond_ch: int = 512):
        super().__init__()
        if cond_ch % 2:
            raise ValueError("cond_ch must be even for ordered pair embeddings")
        self.cond_ch = int(cond_ch)
        self.class_emb = nn.Embedding(classes, cond_ch // 2)
        self.pair_proj = nn.Linear(cond_ch, cond_ch)
        self.time_mlp = nn.Sequential(nn.Linear(cond_ch, cond_ch), nn.SiLU(), nn.Linear(cond_ch, cond_ch))
        self.in_conv = nn.Conv2d(SLOT_CHANNELS + COND_CHANNELS, base, 3, padding=1)
        self.r1 = ResBlock(base, base, cond_ch)
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.r2 = ResBlock(base * 2, base * 2, cond_ch)
        self.mid = ResBlock(base * 2, base * 2, cond_ch)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.r3 = ResBlock(base * 2, base, cond_ch)
        self.noise_head = nn.Conv2d(base, SLOT_CHANNELS, 3, padding=1)
        self.slot_merge_head = nn.Conv2d(base, 3, 3, padding=1)
        self.slot_occ_head = nn.Conv2d(base, 2, 3, padding=1)
        self.slot_owner_head = nn.Conv2d(base, 3, 3, padding=1)
        self.slot_pairstruct_head = nn.Conv2d(base, 3, 3, padding=1)

    def timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(1, half - 1))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def pair_condition(self, label_a: torch.Tensor, label_b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ordered_pair = torch.cat([self.class_emb(label_a), self.class_emb(label_b)], dim=1)
        return self.pair_proj(ordered_pair) + self.time_mlp(self.timestep_embedding(t, self.cond_ch))

    def forward(
        self,
        x_slots: torch.Tensor,
        u: torch.Tensor,
        omega: torch.Tensor,
        t: torch.Tensor,
        label_a: torch.Tensor,
        label_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.pair_condition(label_a, label_b, t)
        inp = torch.cat([x_slots, u, omega], dim=1)
        h1 = self.r1(self.in_conv(inp), cond)
        h2 = self.r2(self.down(h1), cond)
        h2 = self.mid(h2, cond)
        up = self.up(h2)
        h = self.r3(torch.cat([up, h1], dim=1), cond)
        extra = {
            "slot_merge_rgb": torch.tanh(self.slot_merge_head(h)),
            "slot_occ_logits": self.slot_occ_head(h),
            "slot_owner_logits": self.slot_owner_head(h),
            "slot_pairstruct_logits": self.slot_pairstruct_head(h),
        }
        return self.noise_head(h), extra


def load_openai_start(model: nn.Module, report_name: str = "openai_checkpoint_load_report.json") -> dict[str, int]:
    if not OPENAI_PRETRAINED.exists():
        return {"loaded": 0, "available": 0}
    state = torch.load(OPENAI_PRETRAINED, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    used_state_keys = set(compatible)
    state_by_shape: dict[tuple[int, ...], list[tuple[str, torch.Tensor]]] = {}
    for key, value in state.items():
        if key in used_state_keys:
            continue
        state_by_shape.setdefault(tuple(value.shape), []).append((key, value))
    shape_loaded = 0
    for key, value in own.items():
        if key in compatible:
            continue
        candidates = state_by_shape.get(tuple(value.shape), [])
        while candidates:
            src_key, src_value = candidates.pop(0)
            if src_key not in used_state_keys:
                compatible[key] = src_value
                used_state_keys.add(src_key)
                shape_loaded += 1
                break
    own.update(compatible)
    model.load_state_dict(own)
    report = {"loaded": len(compatible), "shape_loaded": shape_loaded, "available": len(state)}
    (WORK / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def initialise_slot_from_rgb_anchor(slot_model: nn.Module, rgb_model: nn.Module) -> dict[str, int]:
    rgb_state = rgb_model.state_dict()
    slot_state = slot_model.state_dict()
    compatible = {k: v for k, v in rgb_state.items() if k in slot_state and tuple(v.shape) == tuple(slot_state[k].shape)}
    slot_state.update(compatible)
    slot_model.load_state_dict(slot_state)
    report = {"loaded": len(compatible), "rgb_anchor_tensors": len(rgb_state), "slot_tensors": len(slot_state)}
    (WORK / "rgb_anchor_to_slot_initialisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def schedule(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def torch_side_masks(u: torch.Tensor, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support = (u > 0.5).to(dtype=u.dtype)
    pos = ((omega > 0.5).to(dtype=u.dtype) * support).clamp(0.0, 1.0)
    neg = ((omega < -0.5).to(dtype=u.dtype) * support).clamp(0.0, 1.0)
    neutral = (support * (1.0 - pos) * (1.0 - neg)).clamp(0.0, 1.0)
    return support, pos, neg, neutral


def structure_target(u: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    support, pos, neg, _neutral = torch_side_masks(u, omega)
    target = torch.full((u.shape[0], u.shape[2], u.shape[3]), 2, dtype=torch.long, device=u.device)
    target[(support[:, 0] > 0.5) & (pos[:, 0] > 0.5)] = 0
    target[(support[:, 0] > 0.5) & (neg[:, 0] > 0.5)] = 1
    return target


class TrainSource:
    def __init__(self, data: dict[int, list[tuple[Path, Path]]]):
        self.data = data
        self.rng = random.Random(SEED)
        self.class_ids = sorted(data)
        self.sample_idx = 0

    def rgb_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, masks, labels = [], [], []
        for _ in range(batch_size):
            label = self.rng.choice(self.class_ids)
            image_path, mask_path = self.rng.choice(self.data[label])
            x, mask = load_rgb_mask(image_path, mask_path)
            images.append(x.transpose(2, 0, 1))
            masks.append(mask[None])
            labels.append(label)
        return (
            torch.tensor(np.stack(images), dtype=torch.float32, device=device),
            torch.tensor(np.stack(masks), dtype=torch.float32, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
        )

    def batch(
        self,
        batch_size: int,
        device: torch.device,
        *,
        pair_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, u_rows, omega_rows, labels_a, labels_b = [], [], [], [], []
        for _ in range(batch_size):
            pair_sample = self.rng.random() < float(pair_probability)
            if pair_sample:
                a, b = self.rng.sample(self.class_ids, 2)
            else:
                a = b = self.rng.choice(self.class_ids)
            ia, ma_path = self.rng.choice(self.data[a])
            xa, ma = load_rgb_mask(ia, ma_path)
            if pair_sample:
                ib, mb_path = self.rng.choice(self.data[b])
                xb, mb = load_rgb_mask(ib, mb_path)
                u, omega = ownership_from_masks(ma, mb, a, b, self.sample_idx)
            else:
                xb = xa
                u = ma.astype(np.float32)
                sign = 1.0 if self.rng.random() >= 0.5 else -1.0
                omega = u * sign
            slot_rgb = np.concatenate([xa.transpose(2, 0, 1), xb.transpose(2, 0, 1)], axis=0)
            slots.append(slot_rgb)
            u_rows.append(u[None])
            omega_rows.append(omega[None])
            labels_a.append(a)
            labels_b.append(b)
            self.sample_idx += 1
        return (
            torch.tensor(np.stack(slots), dtype=torch.float32, device=device),
            torch.tensor(np.stack(u_rows), dtype=torch.float32, device=device),
            torch.tensor(np.stack(omega_rows), dtype=torch.float32, device=device),
            torch.tensor(labels_a, dtype=torch.long, device=device),
            torch.tensor(labels_b, dtype=torch.long, device=device),
        )


def load_checkpoint_if_compatible(model: nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False
    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print(f"Skipping incompatible checkpoint {checkpoint_path}: {exc}")
        return False
    model.eval()
    return True


def train_rgb_anchor(
    *,
    model: SingleLabelRGBUNet,
    source: TrainSource,
    device: torch.device,
    checkpoint_path: Path,
    steps: int,
    lr: float,
) -> SingleLabelRGBUNet:
    if load_checkpoint_if_compatible(model, checkpoint_path, device):
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    _betas, _alphas, alpha_bar = schedule(device)
    model.train()
    for step in tqdm(range(1, int(steps) + 1), desc="stage1 single-label RGB anchor", unit="step"):
        x0, mask, labels = source.rgb_batch(BATCH_SIZE, device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), device=device)
        ab = alpha_bar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        pred_noise = model(xt, t, labels)
        err = (pred_noise - noise).pow(2)
        weight = 1.0 + MASK_LOSS_SCALE * mask
        loss = (err * weight).sum() / weight.expand_as(err).sum().clamp(min=1e-8)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"stage1 single-label RGB anchor step={step} loss={float(loss.item()):.6f}")
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model


def train_stage(
    *,
    model: SlotPixelUNet,
    source: TrainSource,
    device: torch.device,
    checkpoint_path: Path,
    stage_name: str,
    steps: int,
    lr: float,
    pair_probability: float,
) -> SlotPixelUNet:
    if load_checkpoint_if_compatible(model, checkpoint_path, device):
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    _betas, _alphas, alpha_bar = schedule(device)
    model.train()
    for step in tqdm(range(1, int(steps) + 1), desc=stage_name, unit="step"):
        x0, u, omega, a, b = source.batch(BATCH_SIZE, device, pair_probability=pair_probability)
        noise = torch.randn_like(x0)
        t = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), device=device)
        ab = alpha_bar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        pred_noise, extra = model(xt, u, omega, t, a, b)
        loss = F.mse_loss(pred_noise, noise)

        support, pos, neg, _neutral = torch_side_masks(u, omega)
        target_merge = x0[:, 0:3] * pos + x0[:, 3:6] * neg
        merge_loss = F.l1_loss(extra["slot_merge_rgb"] * support, target_merge * support)
        occ_target = torch.cat([pos, neg], dim=1)
        occ_loss = F.binary_cross_entropy_with_logits(extra["slot_occ_logits"], occ_target)
        side_target = structure_target(u, omega)
        owner_loss = F.cross_entropy(extra["slot_owner_logits"], side_target)
        pairstruct_loss = F.cross_entropy(extra["slot_pairstruct_logits"], side_target)
        loss = (
            loss
            + MERGE_LOSS_WEIGHT * merge_loss
            + OCC_LOSS_WEIGHT * occ_loss
            + OWNER_LOSS_WEIGHT * owner_loss
            + PAIRSTRUCT_LOSS_WEIGHT * pairstruct_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"{stage_name} step={step} loss={float(loss.item()):.6f}")
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model


def train_model(data: dict[int, list[tuple[Path, Path]]]) -> SlotPixelUNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = TrainSource(data)
    WORK.mkdir(parents=True, exist_ok=True)

    rgb_anchor = SingleLabelRGBUNet().to(device)
    load_openai_start(rgb_anchor, "openai_rgb_anchor_load_report.json")
    rgb_anchor = train_rgb_anchor(
        model=rgb_anchor,
        source=source,
        device=device,
        checkpoint_path=SINGLE_LABEL_MODEL_PATH,
        steps=SINGLE_LABEL_STEPS,
        lr=SINGLE_LABEL_LR,
    )

    model = SlotPixelUNet().to(device)
    initialise_slot_from_rgb_anchor(model, rgb_anchor)
    model = train_stage(
        model=model,
        source=source,
        device=device,
        checkpoint_path=PAIR_REPLAY_MODEL_PATH,
        stage_name="stage2 mixed pair with single replay",
        steps=PAIR_REPLAY_STEPS,
        lr=PAIR_REPLAY_LR,
        pair_probability=0.5,
    )
    model = train_stage(
        model=model,
        source=source,
        device=device,
        checkpoint_path=MODEL_PATH,
        stage_name="stage3 pair-only slot refinement",
        steps=PAIR_REFINEMENT_STEPS,
        lr=PAIR_REFINEMENT_LR,
        pair_probability=1.0,
    )
    TRAINING_HISTORY_PATH.write_text(json.dumps(PIXEL_TRAINING_HISTORY, indent=2), encoding="utf-8")
    return model


def detector_render(merge_rgb: np.ndarray, support: np.ndarray, background: np.ndarray) -> np.ndarray:
    merge = np.clip((merge_rgb + 1.0) * 127.5, 0, 255).astype(np.float32)
    support = support.astype(np.float32)
    rgb = cv2.resize(merge, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(support, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)
    out = background * (1.0 - mask[..., None]) + rgb * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


@torch.no_grad()
def sample(
    model: SlotPixelUNet,
    u_np: np.ndarray,
    omega_np: np.ndarray,
    a_id: int,
    b_id: int,
    seed: int,
    background: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    _betas, _alphas, alpha_bar = schedule(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, SLOT_CHANNELS, IMG_SIZE, IMG_SIZE), generator=gen, device=device)
    u = torch.tensor(u_np[None, None], dtype=torch.float32, device=device)
    omega = torch.tensor(omega_np[None, None], dtype=torch.float32, device=device)
    a = torch.tensor([a_id], dtype=torch.long, device=device)
    b = torch.tensor([b_id], dtype=torch.long, device=device)
    indices = np.linspace(DIFFUSION_STEPS - 1, 0, SAMPLE_STEPS, dtype=np.int64)
    for i, t_val in enumerate(indices):
        t = torch.tensor([int(t_val)], dtype=torch.long, device=device)
        eps, _extra = model(x, u, omega, t, a, b)
        ab = alpha_bar[t_val]
        x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
        x0 = torch.clamp(x0, -1.0, 1.0)
        if i == len(indices) - 1:
            x = x0
        else:
            prev_t = int(indices[i + 1])
            ab_prev = alpha_bar[prev_t]
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
    t0 = torch.zeros((1,), dtype=torch.long, device=device)
    _eps, extra = model(x, u, omega, t0, a, b)
    merge = extra["slot_merge_rgb"][0].detach().cpu().numpy().transpose(1, 2, 0)
    support = (u[0, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
    return detector_render(merge, support, background)


def generate_raw(model: SlotPixelUNet, layouts: list[dict[str, object]], class_names: list[str]) -> None:
    gen_dir = RAW_OUT / "generated"
    layout_export = RAW_OUT / "layouts"
    gen_dir.mkdir(parents=True, exist_ok=True)
    layout_export.mkdir(parents=True, exist_ok=True)
    background = load_detector_background()
    background_ref = relative_input_path(BACKGROUND_TEMPLATE)
    background_sha = sha256_file(BACKGROUND_TEMPLATE)
    (RAW_OUT / "class_mapping.json").write_text(json.dumps({i: n for i, n in enumerate(class_names)}, indent=2), encoding="utf-8")
    (RAW_OUT / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset": "pixel_space_diffusion_raw",
                "class_count": CLASS_COUNT,
                "samples": len(layouts),
                "ordered_pairs": THESIS_DETECTOR_COUNT,
                "thesis_detector_count": THESIS_DETECTOR_COUNT,
                "count_scope": "final_136_class_detector_run",
                "generated_count_matches_thesis": len(layouts) == THESIS_DETECTOR_COUNT,
                "detector_ingestion": "ordered pair rows used as binary positive training examples",
                "render_mode": "ordered_pair_slot_diffusion_learned_merge",
                "single_label_anchor_channels": RGB_CHANNELS,
                "slot_target_channels": SLOT_CHANNELS,
                "layout_condition_channels": COND_CHANNELS,
                "output_height": OUT_H,
                "output_width": OUT_W,
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (layout_export / "layout_index.jsonl").write_text((LAYOUT_DIR / "layout_index.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (layout_export / "class_map_136.json").write_text((LAYOUT_DIR / "class_map_136.json").read_text(encoding="utf-8"), encoding="utf-8")
    Image.open(LAYOUT_DIR / "background_template.png").save(layout_export / "background_template.png")

    for row in tqdm(layouts, desc="pixel raw generate", unit="sample"):
        idx = int(row["index"])
        data = np.load(LAYOUT_DIR / str(row["layout_file"]))
        image = sample(
            model,
            data["u"],
            data["omega"],
            int(row["class_a_id"]),
            int(row["class_b_id"]),
            SEED + idx,
            background,
        )
        stem = f"mix_{idx:06d}"
        out_png = gen_dir / f"{stem}.png"
        Image.fromarray(image).save(out_png)
        meta = dict(row)
        meta.update(
            {
                "method": "pixel_space_diffusion_pair_inpaint_slot",
                "render_mode": "ordered_pair_slot_diffusion_learned_merge",
                "files": {"composite": out_png.name},
                "checkpoint": str(MODEL_PATH),
                "slot_target": "[xA; xB]",
                "condition": "[u; omega] plus ordered y_pair",
                "ordered_pair": True,
                "detector_training_label": "binary_positive_multi_item",
                "detector_binary_target": 1,
                "output_height": OUT_H,
                "output_width": OUT_W,
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
                "training_history": "single_label_rgb_anchor_to_pair_refined",
            }
        )
        out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def postprocess_image(image: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, float]:
    h, _w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(keep, [largest], -1, 255, cv2.FILLED)
    removed = ((mask > 0) & (keep == 0)).mean()
    out = image.copy()
    out[keep == 0] = background[keep == 0]
    rows = np.where(np.any(keep > 0, axis=1))[0]
    if rows.size:
        bg_start = min(h, int(rows.max()) + 5)
        out[bg_start:] = background[bg_start:]
    return out, float(removed)


def postprocess_final() -> None:
    src = RAW_OUT / "generated"
    dst = FINAL_OUT / "generated"
    dst.mkdir(parents=True, exist_ok=True)
    background = load_detector_background()
    background_ref = relative_input_path(BACKGROUND_TEMPLATE)
    background_sha = sha256_file(BACKGROUND_TEMPLATE)
    for rel in ["class_mapping.json", "dataset_info.json", "layouts/layout_index.jsonl", "layouts/class_map_136.json", "layouts/background_template.png"]:
        src_file = RAW_OUT / rel
        if src_file.exists():
            dst_file = FINAL_OUT / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if src_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                Image.open(src_file).save(dst_file)
            else:
                dst_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
    summary = []
    for png in tqdm(sorted(src.glob("mix_*.png")), desc="pixel-space diffusion postprocess", unit="sample"):
        image = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
        after, removed = postprocess_image(image, background)
        out_png = dst / png.name
        Image.fromarray(after).save(out_png)
        meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        meta["postprocess"] = "standalone_minimal"
        meta["postprocess_bg_ghost_removed_area_frac"] = removed
        meta["postprocess_output"] = str(out_png)
        meta["background_template"] = background_ref
        meta["background_template_sha256"] = background_sha
        meta["background_mode"] = "shared_production_template"
        out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary.append({"sample_id": png.stem, "removed_area_frac": removed})
    (FINAL_OUT / "postprocess_summary.json").write_text(
        json.dumps(
            {
                "processed": len(summary),
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
                "entries": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (FINAL_OUT / "dataset_info.json").write_text(
        json.dumps(
            {
                "dataset": "pixel_space_diffusion",
                "class_count": CLASS_COUNT,
                "samples": len(summary),
                "ordered_pairs": THESIS_DETECTOR_COUNT,
                "thesis_detector_count": THESIS_DETECTOR_COUNT,
                "count_scope": "final_136_class_detector_run",
                "processed_count_matches_thesis": len(summary) == THESIS_DETECTOR_COUNT,
                "detector_ingestion": "ordered pair rows used as binary positive training examples",
                "render_mode": "ordered_pair_slot_diffusion_learned_merge",
                "postprocess": "standalone_minimal",
                "single_label_anchor_channels": RGB_CHANNELS,
                "slot_target_channels": SLOT_CHANNELS,
                "layout_condition_channels": COND_CHANNELS,
                "output_height": OUT_H,
                "output_width": OUT_W,
                "background_template": background_ref,
                "background_template_sha256": background_sha,
                "background_mode": "shared_production_template",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(summary)} postprocessed samples to {FINAL_OUT}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    require(OPENAI_PRETRAINED)
    WORK.mkdir(parents=True, exist_ok=True)
    class_names, data = discover_data()
    layouts = build_layouts(class_names, data)
    model = train_model(data)
    generate_raw(model, layouts, class_names)
    postprocess_final()


if __name__ == "__main__":
    main()

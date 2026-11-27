"""Self-contained PyTorch FP32 + W8A8 QAT trainer for OmniShift pure-convolutional VFI (`omnishift_vfi.py`).

Dataset format:
  Pass `--data-npy <triplets.npy>` containing uint8 BGR frame triplets of shape `[N, 3, H, W, 3]`
  (`[:, 0]` = frame A, `[:, 1]` = midpoint ground truth, `[:, 2]` = frame B), or omit `--data-npy`
  to run a synthetic smoke-training verification pass.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from omnishift_vfi import OmniShiftVFIv7


def charbonnier(diff: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(diff * diff + eps * eps)


def laplacian_pyramid_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 4) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    cur_p, cur_t = pred, target
    weight = 1.0
    for lvl in range(levels):
        if lvl < levels - 1:
            down_p = F.avg_pool2d(cur_p, 2, 2)
            down_t = F.avg_pool2d(cur_t, 2, 2)
            up_p = F.interpolate(down_p, size=cur_p.shape[-2:], mode="bilinear", align_corners=False)
            up_t = F.interpolate(down_t, size=cur_t.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + weight * charbonnier((cur_p - up_p) - (cur_t - up_t), eps=1e-3).mean()
            cur_p, cur_t = down_p, down_t
            weight *= 0.5
        else:
            loss = loss + weight * charbonnier(cur_p - cur_t, eps=1e-3).mean()
    return loss


def census_loss_3x3(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p_y = 0.114 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.299 * pred[:, 2:3]
    t_y = 0.114 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.299 * target[:, 2:3]
    p_pad = F.pad(p_y, (1, 1, 1, 1), mode="replicate")
    t_pad = F.pad(t_y, (1, 1, 1, 1), mode="replicate")
    _, _, h, w = p_y.shape
    diffs_p, diffs_t = [], []
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            diffs_p.append(p_pad[:, :, dy : dy + h, dx : dx + w] - p_y)
            diffs_t.append(t_pad[:, :, dy : dy + h, dx : dx + w] - t_y)
    dp = torch.cat(diffs_p, dim=1)
    dt = torch.cat(diffs_t, dim=1)
    sp = dp / torch.sqrt(dp * dp + 0.015 * 0.015)
    st = dt / torch.sqrt(dt * dt + 0.015 * 0.015)
    return charbonnier(sp - st, eps=1e-3).mean()


def sample_batch(tri: np.ndarray | None, batch_size: int, crop: int, rng: np.random.Generator, device: torch.device):
    if tri is None:
        a = torch.rand(batch_size, 4, crop, crop, device=device)
        shift = int(rng.integers(-12, 13))
        b = torch.roll(a, shifts=shift * 2, dims=3)
        m = torch.roll(a, shifts=shift, dims=3)
        a[:, 3] = 1.0
        b[:, 3] = 1.0
        m[:, 3] = 1.0
        return a, m, b
    idxs = rng.integers(0, tri.shape[0], size=batch_size)
    batch = tri[idxs]  # [B, 3, H, W, 3]
    _, _, h, w, _ = batch.shape
    y0 = int(rng.integers(0, max(1, h - crop + 1)))
    x0 = int(rng.integers(0, max(1, w - crop + 1)))
    sub = torch.from_numpy(batch[:, :, y0 : y0 + crop, x0 : x0 + crop, :]).to(device=device, dtype=torch.float32) / 255.0
    sub = sub.permute(0, 1, 4, 2, 3).contiguous()  # [B, 3, 3, H, W]
    alpha = torch.ones((batch_size, 1, crop, crop), device=device, dtype=torch.float32)
    return (
        torch.cat([sub[:, 0], alpha], dim=1),
        torch.cat([sub[:, 1], alpha], dim=1),
        torch.cat([sub[:, 2], alpha], dim=1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-npy", type=Path, default=None, help="Optional [N,3,H,W,3] uint8 BGR triplets .npy")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/omnishift_sm8750"))
    ap.add_argument("--tier", type=str, default="medium", choices=("small", "medium", "large"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--qat-start", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tri = np.load(args.data_npy, mmap_mode="r") if args.data_npy else None
    rng = np.random.default_rng(20261127)

    model = OmniShiftVFIv7(tier=args.tier).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.05)

    t0 = time.perf_counter()
    qat_enabled = False
    for step in range(1, args.steps + 1):
        if step == args.qat_start and not qat_enabled:
            qat_enabled = True
            print(f"[step {step}] Enabling Stage-2 W8/UFIXED8 QAT...", flush=True)

        a, gt, b = sample_batch(tri, args.batch_size, args.crop, rng, dev)
        phase = float(rng.choice([1.0 / 3.0, 0.5, 2.0 / 3.0]))
        if abs(phase - 0.5) > 1e-3:
            # Linear blend fallback when training on 3-frame triplets; 5-frame quintets supervise exact 1/3 & 2/3
            gt_phase = gt if abs(phase - 0.5) < 1e-3 else ((1.0 - phase) * a + phase * b)
        else:
            gt_phase = gt

        pred, aux = model.forward_train(a, b, phase=phase, qat=qat_enabled)
        loss_rec = charbonnier(pred[:, :3] - gt_phase[:, :3]).mean()
        loss_pyr = laplacian_pyramid_loss(pred[:, :3], gt_phase[:, :3])
        loss_cen = census_loss_3x3(pred[:, :3], gt_phase[:, :3])
        wa, wb = aux["wa_bgr"], aux["wb_bgr"]
        loss_ep = torch.minimum(charbonnier(wa - gt_phase[:, :3]), charbonnier(wb - gt_phase[:, :3])).mean()
        loss = loss_rec + 0.35 * loss_pyr + 0.15 * loss_cen + 0.25 * loss_ep

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            for p_e, p_m in zip(ema.parameters(), model.parameters()):
                p_e.mul_(0.999).add_(p_m, alpha=0.001)

        if step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            print(f"[step {step}/{args.steps}] loss={loss.item():.5f} lr={sched.get_last_lr()[0]:.2e} ({elapsed:.1f}s)", flush=True)

    ckpt_path = args.out_dir / "omnishift_vfi_sm8750.pt"
    torch.save({"step": args.steps, "tier": args.tier, "ema_state_dict": ema.state_dict()}, ckpt_path)
    (args.out_dir / "train_summary.json").write_text(
        json.dumps({"steps": args.steps, "tier": args.tier, "checkpoint": str(ckpt_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved trained checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()

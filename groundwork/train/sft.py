"""Supervised fine-tuning on the assembled mix (assistant tokens only).

Single GPU: ``groundwork train --config cfg.yaml``
Multi GPU:  ``accelerate launch -m groundwork.cli train --config cfg.yaml``
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

from ..config import Config
from ..io import append_jsonl, ensure_dir, read_jsonl, write_json
from ..models.hf import load_hf
from .encode import IGNORE, ChatEncoder, collate


def _freeze(model, names: list[str]) -> int:
    frozen = 0
    for pname, p in model.named_parameters():
        parts = pname.split(".")
        if any(n in parts for n in names):
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen


def _apply_lora(model, cfg: Config):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("train.lora.enabled needs `pip install peft`") from e
    lc = cfg.train.lora
    model = get_peft_model(model, LoraConfig(
        r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout, target_modules=list(lc.target_modules),
        task_type="CAUSAL_LM",
    ))
    _freeze(model, cfg.train.freeze_modules)  # keep adapters off frozen towers
    return model


def _lr_lambda(scheduler: str, warmup: int, total: int):
    def fn(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        if scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def token_ce_sum(logits, labels, chunk: int = 1024):
    """Sum of next-token cross-entropy over labelled positions, computed in float32 chunks."""
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for s in range(0, shift_logits.shape[1], chunk):
        lg = shift_logits[:, s : s + chunk].float()
        lb = shift_labels[:, s : s + chunk]
        total = total + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb.reshape(-1), ignore_index=IGNORE, reduction="sum")
    return total


def train_sft(cfg: Config) -> str:
    import torch

    tc = cfg.train
    if not tc.base_model:
        raise ValueError("train.base_model is required (path or hub id of the model to fine-tune)")
    data_path = tc.data or str(Path(cfg.data.work_dir) / "sft.jsonl")
    random.seed(tc.seed)
    torch.manual_seed(tc.seed)

    accelerator = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        accelerator = Accelerator()
    is_main = accelerator is None or accelerator.is_main_process
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok, proc = load_hf(tc.base_model, tc.dtype, "none", tc.trust_remote_code, tc.multimodal, tc.loader)
    frozen = _freeze(model, tc.freeze_modules)
    if tc.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
    if tc.lora.enabled:
        if tc.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = _apply_lora(model, cfg)
    if accelerator is None:
        model.to(device)

    encoder = ChatEncoder(tok, proc, tc.multimodal, tc.max_length, cfg.data.media_root, cfg.data.video_frames, tc.tool_role_as_user)
    rows = read_jsonl(data_path)
    examples = [e for e in (encoder.encode(r["messages"]) for r in rows) if e is not None]
    if not examples:
        raise ValueError(f"No trainable examples in {data_path} (encoder stats: {encoder.stats})")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if is_main:
        print(f"[train] {len(examples)} examples (encoder: {encoder.stats}); trainable params {n_trainable:,}; frozen {frozen:,}")

    loader = torch.utils.data.DataLoader(
        examples, batch_size=tc.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(tc.seed),
    )
    optim = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.95))
    updates_per_epoch = math.ceil(len(loader) / tc.grad_accum)
    total_updates = max(1, updates_per_epoch * tc.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda(tc.scheduler, int(tc.warmup_ratio * total_updates), total_updates))
    if accelerator is not None:
        model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)

    out_dir = ensure_dir(tc.output_dir)
    log_path = out_dir / "train_log.jsonl"
    use_autocast = device.type == "cuda" and next(model.parameters()).dtype == torch.float32
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(tc.epochs):
        batches = iter(loader)
        while True:
            group = []
            for _ in range(tc.grad_accum):
                try:
                    group.append(next(batches))
                except StopIteration:
                    break
            if not group:
                break
            n_tokens = sum(int((b["labels"][:, 1:] != IGNORE).sum()) for b in group)
            loss_total = 0.0
            for k, batch in enumerate(group):
                batch = {key: (v.to(device) if hasattr(v, "to") else v) for key, v in batch.items()}
                labels = batch.pop("labels")
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else torch.autocast("cpu", enabled=False)
                sync_ok = accelerator is None or k == len(group) - 1
                no_sync = accelerator.no_sync(model) if (accelerator is not None and not sync_ok) else _null()
                with no_sync, ctx:
                    logits = model(**batch).logits
                    loss = token_ce_sum(logits, labels) / max(1, n_tokens)
                    (accelerator.backward(loss) if accelerator else loss.backward())
                loss_total += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if is_main and (step % tc.log_every_steps == 0 or step == 1 or step == total_updates):
                rec = {"step": step, "epoch": epoch, "loss": round(loss_total, 5), "lr": sched.get_last_lr()[0],
                       "grad_norm": round(float(grad_norm), 4), "tokens": n_tokens, "elapsed_s": round(time.time() - t0, 1)}
                print(f"[train] {rec}", flush=True)
                append_jsonl(log_path, rec)
            if tc.save_every_steps and step % tc.save_every_steps == 0:
                _save(model, tok, proc, out_dir / f"checkpoint-{step}", accelerator)
    _save(model, tok, proc, out_dir, accelerator)
    if is_main:
        write_json(out_dir / "groundwork_train.json", {"config": cfg.to_dict()["train"], "examples": len(examples), "updates": step})
        print(f"[train] saved to {out_dir}")
    return str(out_dir)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _save(model, tok, proc, path, accelerator) -> None:
    path = ensure_dir(path)
    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(path, save_function=accelerator.save)
    else:
        model.save_pretrained(path)
    if accelerator is None or accelerator.is_main_process:
        (proc or tok).save_pretrained(path)

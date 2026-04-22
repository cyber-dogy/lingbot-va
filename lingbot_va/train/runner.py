from __future__ import annotations

import time
from typing import Any

import torch

from lingbot_va.config import ExperimentConfig, save_config

from .builders import (
    build_dataloaders,
    build_model,
    build_optimizer,
    build_scheduler,
    move_batch_to_device,
    seed_everything,
)
from .checkpoints import load_resume_state, save_checkpoint
from .eval import evaluate_model_on_loader, write_summary_json


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("空列表无法求平均值。")
    return float(sum(values) / len(values))


def train_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    cfg.validate()
    device = torch.device(cfg.device)
    seed_everything(cfg.seed)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.periodic_ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    _, _, dataloader_train, dataloader_valid = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(dataloader_train))
    resume_state = load_resume_state(cfg, model, optimizer, scheduler)

    start_epoch = int(resume_state["start_epoch"])
    global_step = int(resume_state["global_step"])
    best_metric = resume_state["best_metric"]
    history = list(resume_state["history"])
    run_started_at = time.perf_counter()

    for epoch in range(start_epoch, cfg.train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_total_losses: list[float] = []
        train_latent_losses: list[float] = []
        train_action_losses: list[float] = []
        epoch_started_at = time.perf_counter()

        for batch_idx, batch_cpu in enumerate(dataloader_train):
            batch = move_batch_to_device(batch_cpu, device=device)
            loss_dict = model.compute_loss_dict(batch)
            loss = loss_dict["loss_total"] / max(1, cfg.grad_accum_steps)
            loss.backward()

            should_step = ((batch_idx + 1) % max(1, cfg.grad_accum_steps) == 0) or (
                batch_idx == len(dataloader_train) - 1
            )
            if should_step:
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            raw_total_loss = float(loss_dict["loss_total"].detach().cpu())
            raw_latent_loss = float(loss_dict["loss_latent"].detach().cpu())
            raw_action_loss = float(loss_dict["loss_action"].detach().cpu())
            train_total_losses.append(raw_total_loss)
            train_latent_losses.append(raw_latent_loss)
            train_action_losses.append(raw_action_loss)

            if ((batch_idx + 1) % cfg.print_every) == 0:
                print(
                    f"[train] epoch={epoch + 1}/{cfg.train_epochs} "
                    f"step={batch_idx + 1}/{len(dataloader_train)} "
                    f"loss={raw_total_loss:.6f} "
                    f"latent={raw_latent_loss:.6f} "
                    f"action={raw_action_loss:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_summary = {
            "loss_total": _mean(train_total_losses),
            "loss_latent": _mean(train_latent_losses),
            "loss_action": _mean(train_action_losses),
            "num_batches": len(train_total_losses),
            "epoch_time_sec": time.perf_counter() - epoch_started_at,
        }

        valid_summary = None
        if ((epoch + 1) % cfg.val_every_epochs) == 0:
            valid_summary = evaluate_model_on_loader(model, dataloader_valid, device=device)

        epoch_row = {
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "train": train_summary,
            "valid": valid_summary,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_row)

        current_metric = None if valid_summary is None else float(valid_summary["loss_total"])
        if current_metric is not None and (best_metric is None or current_metric < best_metric):
            best_metric = current_metric
            save_checkpoint(
                path=cfg.best_ckpt_path,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                history=history,
            )

        save_checkpoint(
            path=cfg.latest_ckpt_path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            history=history,
        )

        if ((epoch + 1) % cfg.checkpoint_every_epochs) == 0:
            save_checkpoint(
                path=cfg.periodic_ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                history=history,
            )

        print(
            f"[epoch] epoch={epoch + 1}/{cfg.train_epochs} "
            f"train_loss={train_summary['loss_total']:.6f} "
            f"valid_loss={None if valid_summary is None else round(valid_summary['loss_total'], 6)}"
        )

    summary = {
        "run_name": cfg.run_name,
        "device": cfg.device,
        "best_metric": best_metric,
        "epochs": cfg.train_epochs,
        "global_step": global_step,
        "wall_time_sec": time.perf_counter() - run_started_at,
        "history": history,
        "latest_ckpt_path": str(cfg.latest_ckpt_path),
        "best_ckpt_path": str(cfg.best_ckpt_path),
    }
    write_summary_json(summary, cfg.summary_path)
    return summary

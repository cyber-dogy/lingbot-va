#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset, get_relative_pose
from wan_va.utils import init_logger, logger
from wan_va.utils.utils import executor as save_executor
from wan_va.wan_va_server import VA_Server


CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LEFT_KEY = "observation.images.cam_left_wrist"
CAM_RIGHT_KEY = "observation.images.cam_right_wrist"
OBS_CAM_KEYS = [CAM_HIGH_KEY, CAM_LEFT_KEY, CAM_RIGHT_KEY]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export offline LingBot-VA demo videos for local RoboTwin tasks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT_DIR / "data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500",
        help="Root directory that contains downloaded RoboTwin task repos.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "models/lingbot-va-posttrain-robotwin",
        help="Checkpoint directory that contains transformer / vae / tokenizer / text_encoder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "eval_out/robotwin_task_demos/default_run",
        help="Directory used to save videos, charts and summaries.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="robotwin",
        choices=sorted(VA_CONFIGS.keys()),
        help="Config name used to instantiate the inference server.",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default="",
        help="Comma separated task repo names. Empty means all local tasks under dataset-root.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit exported task count. 0 means no limit.",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Which valid segment inside each repo to export.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=6,
        # 在当前 12 fps 导出设置下，6 chunks 大约对应 3.7 秒，正好落在 3-5 秒短视频区间。
        help="How many autoregressive chunks to infer for each task. Default 6 chunks gives a 3-5 second short demo on RoboTwin.",
    )
    parser.add_argument(
        "--attn-mode",
        type=str,
        default="flashattn",
        choices=["torch", "flashattn"],
        help="Inference attention kernel used by the transformer.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Video CFG scale. Local exporter defaults to 1.0 to fit 24GB single-GPU inference.",
    )
    parser.add_argument(
        "--action-guidance-scale",
        type=float,
        default=1.0,
        help="Action CFG scale. Local exporter defaults to 1.0 to reduce cache size and runtime.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=0,
        help="Override video diffusion steps. 0 keeps the config default.",
    )
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=0,
        help="Override action diffusion steps. 0 keeps the config default.",
    )
    parser.add_argument(
        "--save-debug-cache",
        action="store_true",
        help="Keep server-side latent/action debug dumps under output-root/_server_cache.",
    )
    return parser.parse_args()


def to_uint8_frames(video):
    video = np.asarray(video)
    if video.dtype == np.uint8:
        return video
    if np.issubdtype(video.dtype, np.floating):
        clipped = np.clip(video, 0.0, 1.0)
        return (clipped * 255.0).round().astype(np.uint8)
    return np.clip(video, 0, 255).astype(np.uint8)


def resize_frame(frame, width, height):
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def compose_robotwin_tshape_frame(frame_dict):
    high = resize_frame(frame_dict[CAM_HIGH_KEY], 320, 256)
    left = resize_frame(frame_dict[CAM_LEFT_KEY], 160, 128)
    right = resize_frame(frame_dict[CAM_RIGHT_KEY], 160, 128)
    return np.concatenate([np.concatenate([left, right], axis=1), high], axis=0)


def load_video_frames(video_path):
    return [np.asarray(frame) for frame in iio.imiter(video_path, plugin="pyav")]


def build_task_list(dataset_root, task_names, max_tasks):
    if task_names:
        requested_names = [name.strip() for name in task_names.split(",") if name.strip()]
        task_dirs = [dataset_root / name for name in requested_names]
    else:
        task_dirs = sorted(
            path for path in dataset_root.iterdir()
            if path.is_dir() and (path / "meta" / "info.json").is_file()
        )

    if max_tasks > 0:
        task_dirs = task_dirs[:max_tasks]

    missing_dirs = [str(path) for path in task_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Task repos not found: {missing_dirs}")

    if not task_dirs:
        raise FileNotFoundError(f"No task repos found under {dataset_root}")
    return task_dirs


def load_task_context(repo_path, config, segment_index):
    dataset = LatentLeRobotDataset(str(repo_path), config)
    if len(dataset.new_metas) <= segment_index:
        raise IndexError(
            f"segment-index={segment_index} is out of range for {repo_path.name}, valid={len(dataset.new_metas)}"
        )

    meta = dataset.new_metas[segment_index]
    episode_index = meta["episode_index"]
    episode_chunk = dataset.meta.get_episode_chunk(episode_index)

    video_paths = {
        key: repo_path / dataset.meta.get_video_file_path(episode_index, key)
        for key in OBS_CAM_KEYS
    }

    start_frame = meta["start_frame"]
    end_frame = meta["end_frame"]
    global_start = dataset._get_global_idx(episode_index, start_frame)
    global_end = dataset._get_global_idx(episode_index, end_frame)
    hf_frames = dataset._get_range_hf_data(global_start, global_end)
    raw_actions = hf_frames["action"].detach().cpu().numpy()

    episode_videos = {
        key: load_video_frames(path)
        for key, path in video_paths.items()
    }

    prompt = meta.get("action_text") or meta["tasks"][0]
    return {
        "dataset": dataset,
        "meta": meta,
        "episode_chunk": episode_chunk,
        "prompt": prompt,
        "raw_actions": raw_actions,
        "episode_videos": episode_videos,
    }


def build_init_obs(task_context):
    start_frame = task_context["meta"]["start_frame"]
    first_obs = {
        key: task_context["episode_videos"][key][start_frame]
        for key in OBS_CAM_KEYS
    }
    return {"obs": [first_obs]}


def build_gt_video(task_context, pred_frame_count):
    start_frame = task_context["meta"]["start_frame"]
    gt_frames = []
    for offset in range(pred_frame_count):
        frame_id = start_frame + offset
        if frame_id >= len(task_context["episode_videos"][CAM_HIGH_KEY]):
            break
        gt_frames.append(
            compose_robotwin_tshape_frame(
                {key: task_context["episode_videos"][key][frame_id] for key in OBS_CAM_KEYS}
            )
        )
    return np.asarray(gt_frames, dtype=np.uint8)


def build_gt_actions(task_context, pred_action_steps):
    raw_actions = task_context["raw_actions"]
    left_action = get_relative_pose(raw_actions[:, :7]).numpy()
    right_action = get_relative_pose(raw_actions[:, 8:15]).numpy()
    gt_actions = np.concatenate(
        [left_action, raw_actions[:, 7:8], right_action, raw_actions[:, 15:16]],
        axis=1,
    )
    # 训练和推理都把第一个 latent frame 对齐到 16 个 action slots，这里沿用同一套对齐规则。
    gt_actions = np.pad(
        gt_actions,
        pad_width=((16, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return gt_actions[:pred_action_steps]


def flatten_pred_actions(pred_actions):
    return np.transpose(pred_actions, (1, 2, 0)).reshape(-1, pred_actions.shape[0])


def run_task_inference(server, prompt, init_obs, num_chunks):
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    server._reset(prompt=prompt)

    pred_latent_list = []
    pred_action_list = []
    for chunk_id in range(num_chunks):
        frame_st_id = chunk_id * server.job_config.frame_chunk_size
        actions_np, latents = server._infer(init_obs, frame_st_id=frame_st_id)
        pred_action_list.append(np.asarray(actions_np))
        pred_latent_list.append(latents.detach().cpu())

    pred_latents = torch.cat(pred_latent_list, dim=2)
    server.transformer.clear_cache(server.cache_name)
    server.streaming_vae.clear_cache()
    if server.streaming_vae_half is not None:
        server.streaming_vae_half.clear_cache()

    if server.enable_offload:
        # 导出脚本优先保证单卡能跑通：先把 transformer 挪开，再把 VAE 挪回 GPU 做解码。
        server.transformer.to("cpu")
        torch.cuda.empty_cache()
        server.vae = server.vae.to(server.device).to(server.dtype)

    pred_video = server.decode_one_video(pred_latents.to(server.device), output_type="np")[0]

    if server.enable_offload:
        server.vae = server.vae.to("cpu")
        server.transformer.to(server.device).to(server.dtype)
        torch.cuda.empty_cache()

    pred_video = to_uint8_frames(pred_video)
    pred_actions = np.concatenate(pred_action_list, axis=1)
    return pred_video, pred_actions


def build_comparison_video(gt_video, pred_video):
    compare_length = min(len(gt_video), len(pred_video))
    comparison_frames = []
    for frame_id in range(compare_length):
        gt_frame = gt_video[frame_id]
        pred_frame = resize_frame(pred_video[frame_id], gt_frame.shape[1], gt_frame.shape[0])
        comparison_frames.append(np.concatenate([gt_frame, pred_frame], axis=0))
    return np.asarray(comparison_frames, dtype=np.uint8)


def compute_video_metrics(gt_video, pred_video):
    compare_length = min(len(gt_video), len(pred_video))
    if compare_length == 0:
        raise RuntimeError("No overlapping GT/pred video frames to compare.")
    gt_eval = gt_video[:compare_length].astype(np.float32) / 255.0
    pred_eval = pred_video[:compare_length].astype(np.float32) / 255.0
    diff = pred_eval - gt_eval
    return {
        "video_frames_used": int(compare_length),
        "video_mse": float(np.mean(diff ** 2)),
        "video_mae": float(np.mean(np.abs(diff))),
    }


def compute_action_metrics(gt_actions, pred_actions):
    compare_length = min(len(gt_actions), len(pred_actions))
    if compare_length == 0:
        raise RuntimeError("No overlapping GT/pred action steps to compare.")
    gt_eval = gt_actions[:compare_length].astype(np.float32)
    pred_eval = pred_actions[:compare_length].astype(np.float32)
    diff = pred_eval - gt_eval
    return {
        "action_steps_used": int(compare_length),
        "action_mse": float(np.mean(diff ** 2)),
        "action_mae": float(np.mean(np.abs(diff))),
    }


def save_video(video, output_path, fps):
    imageio.mimsave(output_path, list(video), fps=fps)


def save_action_plot(gt_actions, pred_actions, output_path, task_name):
    groups = [
        ("Left XYZ + Grip", [0, 1, 2, 7], ["left_x", "left_y", "left_z", "left_grip"]),
        ("Left Quaternion", [3, 4, 5, 6], ["left_qx", "left_qy", "left_qz", "left_qw"]),
        ("Right XYZ + Grip", [8, 9, 10, 15], ["right_x", "right_y", "right_z", "right_grip"]),
        ("Right Quaternion", [11, 12, 13, 14], ["right_qx", "right_qy", "right_qz", "right_qw"]),
    ]

    compare_length = min(len(gt_actions), len(pred_actions))
    x_axis = np.arange(compare_length)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=120, sharex=True)
    for axis, (title, dim_ids, labels) in zip(axes.flatten(), groups):
        for dim_id, label in zip(dim_ids, labels):
            axis.plot(x_axis, gt_actions[:compare_length, dim_id], label=f"gt_{label}", linewidth=1.6)
            axis.plot(
                x_axis,
                pred_actions[:compare_length, dim_id],
                linestyle="--",
                label=f"pred_{label}",
                linewidth=1.2,
            )
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize="x-small", ncol=2)

    axes[1, 0].set_xlabel("Action Step")
    axes[1, 1].set_xlabel("Action Step")
    fig.suptitle(f"{task_name} action comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(summary_rows, output_path):
    success_rows = [row for row in summary_rows if row["status"] == "success"]
    if not success_rows:
        return

    task_names = [row["task_name"] for row in success_rows]
    video_mse = [row["video_mse"] for row in success_rows]
    action_mse = [row["action_mse"] for row in success_rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=120, sharex=True)
    axes[0].bar(task_names, video_mse, color="#4C78A8")
    axes[0].set_ylabel("Video MSE")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(task_names, action_mse, color="#F58518")
    axes[1].set_ylabel("Action MSE")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_json(data, output_path):
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_one_task(server, repo_path, config, args):
    task_name = repo_path.name
    task_output_dir = args.output_root / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    task_context = load_task_context(repo_path, config, args.segment_index)
    prompt = task_context["prompt"]
    init_obs = build_init_obs(task_context)

    pred_video, pred_actions = run_task_inference(
        server=server,
        prompt=prompt,
        init_obs=init_obs,
        num_chunks=args.num_chunks,
    )
    pred_video = to_uint8_frames(pred_video)
    pred_actions_flat = flatten_pred_actions(pred_actions)

    gt_video = build_gt_video(task_context, pred_frame_count=len(pred_video))
    gt_actions = build_gt_actions(task_context, pred_action_steps=len(pred_actions_flat))
    comparison_video = build_comparison_video(gt_video, pred_video)

    video_metrics = compute_video_metrics(gt_video, pred_video)
    action_metrics = compute_action_metrics(gt_actions, pred_actions_flat)

    metadata = {
        "task_name": task_name,
        "prompt": prompt,
        "episode_index": int(task_context["meta"]["episode_index"]),
        "episode_chunk": int(task_context["episode_chunk"]),
        "segment_index": int(args.segment_index),
        "start_frame": int(task_context["meta"]["start_frame"]),
        "end_frame": int(task_context["meta"]["end_frame"]),
        "num_chunks": int(args.num_chunks),
        "pred_video_frames": int(len(pred_video)),
        "pred_action_steps": int(len(pred_actions_flat)),
        "infer_attn_mode": args.attn_mode,
        "guidance_scale": float(args.guidance_scale),
        "action_guidance_scale": float(args.action_guidance_scale),
        "num_inference_steps": int(server.job_config.num_inference_steps),
        "action_num_inference_steps": int(server.job_config.action_num_inference_steps),
        **video_metrics,
        **action_metrics,
    }

    save_video(pred_video, task_output_dir / "prediction.mp4", fps=12)
    save_video(gt_video[:video_metrics["video_frames_used"]], task_output_dir / "ground_truth.mp4", fps=12)
    save_video(comparison_video, task_output_dir / "comparison.mp4", fps=12)
    np.save(task_output_dir / "prediction_actions.npy", pred_actions_flat)
    np.save(task_output_dir / "ground_truth_actions.npy", gt_actions[:action_metrics["action_steps_used"]])
    (task_output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    save_json(metadata, task_output_dir / "metrics.json")
    save_action_plot(
        gt_actions[:action_metrics["action_steps_used"]],
        pred_actions_flat[:action_metrics["action_steps_used"]],
        task_output_dir / "action_compare.png",
        task_name=task_name,
    )

    return metadata


def cleanup_debug_cache(output_root):
    debug_cache_root = output_root / "_server_cache"
    if not debug_cache_root.exists():
        return
    for path in sorted(debug_cache_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this exporter.")
    torch.cuda.set_device(0)

    init_logger()

    task_dirs = build_task_list(args.dataset_root, args.task_names, args.max_tasks)
    logger.info("Found %d task repos to export under %s", len(task_dirs), args.dataset_root)

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.wan22_pretrained_model_name_or_path = str(args.model_path)
    config.infer_attn_mode = args.attn_mode
    config.guidance_scale = args.guidance_scale
    config.action_guidance_scale = args.action_guidance_scale
    if args.num_inference_steps > 0:
        config.num_inference_steps = args.num_inference_steps
    if args.action_num_inference_steps > 0:
        config.action_num_inference_steps = args.action_num_inference_steps
    config.aux_vae_device = "cpu"
    config.enable_text_encoder_offload = True
    config.save_root = str(args.output_root / "_server_cache")
    config.dataset_path = str(args.dataset_root.parent)
    config.empty_emb_path = str(args.dataset_root.parent / "empty_emb.pt")
    config.cfg_prob = 0.0
    # 本地导出保留主 VAE 在 GPU，只把 text encoder 和辅助 wrist VAE 放 CPU，速度和显存更平衡。
    config.enable_offload = False

    server = VA_Server(config)
    summary_rows = []

    for task_dir in task_dirs:
        task_name = task_dir.name
        logger.info("Exporting task demo for %s", task_name)
        try:
            summary = export_one_task(server, task_dir, config, args)
            summary["status"] = "success"
            summary_rows.append(summary)
            logger.info(
                "Finished %s: video_mse=%.6f action_mse=%.6f",
                task_name,
                summary["video_mse"],
                summary["action_mse"],
            )
        except Exception as exc:
            # 批量导出按任务隔离失败，避免单个坏 repo 让整批可视化结果白跑。
            error_row = {
                "task_name": task_name,
                "status": "failed",
                "video_mse": "",
                "action_mse": "",
                "error": str(exc),
            }
            summary_rows.append(error_row)
            logger.exception("Failed to export task demo for %s", task_name)
        finally:
            torch.cuda.empty_cache()

    summary_csv_path = args.output_root / "summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    save_json(summary_rows, args.output_root / "summary.json")
    save_summary_plot(summary_rows, args.output_root / "summary_metrics.png")

    if not args.save_debug_cache:
        save_executor.shutdown(wait=True)
        cleanup_debug_cache(args.output_root)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    logger.info(
        "Export finished: %d/%d tasks succeeded. Outputs saved to %s",
        success_count,
        len(summary_rows),
        args.output_root,
    )


if __name__ == "__main__":
    main()

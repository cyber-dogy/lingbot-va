import sys
from pathlib import Path

import argparse
import os
import time

import cv2
import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from lerobot.datasets.utils import write_json
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LINGBOT_ROOT = Path(__file__).resolve().parents[2]
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.append(str(LINGBOT_ROOT))

DEFAULT_LIBERO_HEADLESS_TOOLS_ROOT = PROJECT_ROOT / "autodl_unplug_charger_transformer_fm" / "libero" / "headless_tools"
LIBERO_HEADLESS_TOOLS_ROOT = Path(
    os.environ.get("LIBERO_HEADLESS_TOOLS_ROOT", str(DEFAULT_LIBERO_HEADLESS_TOOLS_ROOT))
)
if str(LIBERO_HEADLESS_TOOLS_ROOT) not in sys.path:
    sys.path.append(str(LIBERO_HEADLESS_TOOLS_ROOT))

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from rollout_artifacts import build_eval_artifact_paths, save_action_trace_artifact


def build_video_frames(real_obs_list, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        return []

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)

    return [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    final_frames = build_video_frames(real_obs_list, video_names=video_names)
    if not final_frames:
        print("❌ No real observation frames")
        return

    print(f"Saving video: {len(final_frames)} frames...")

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, artifact_dir=None, save_rollout_video=True, save_action_trace=True, fps=60):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    task = benchmark_instance.get_task(task_idx)
    prompt = task.language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)
    initial_state = init_states[episode_idx % init_states.shape[0]]

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, initial_state)

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    executed_actions = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                executed_actions.append(np.asarray(ee_action, dtype=np.float32).copy())
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    # 统一把评测产物写到同一目录结构，后面筛成功案例和重放 trace 会更方便。
    action_trace_path, rollout_video_path = build_eval_artifact_paths(
        model_tag="lingbot_va",
        suite_name=libero_benchmark,
        task_id=task_idx,
        episode_idx=episode_idx,
        success=done,
        task_name=task.name,
        output_root=artifact_dir,
    )

    if save_rollout_video:
        save_video(
            real_obs_list=full_obs_list,
            save_path=rollout_video_path,
            fps=fps,
            video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
        )

    if save_action_trace and len(executed_actions) > 0:
        save_action_trace_artifact(
            action_trace_path,
            executed_actions,
            initial_state=initial_state,
            metadata={
                "model_tag": "lingbot_va",
                "suite_name": libero_benchmark,
                "task_id": task_idx,
                "task_name": task.name,
                "instruction": prompt,
                "episode_idx": episode_idx,
                "success": bool(done),
            },
        )
        print(f"✅ Action trace saved to: {action_trace_path}")
    elif save_action_trace:
        print("⚠️ Skip saving action trace because executed_actions is empty")

    cur_env.close()
    return done


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
    artifact_dir=None,
    disable_rollout_video=False,
    disable_action_trace=False,
    fps=60,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                artifact_dir=artifact_dir,
                save_rollout_video=not disable_rollout_video,
                save_action_trace=not disable_action_trace,
                fps=fps,
            )
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="统一保存 rollout 视频和 action trace 的目录；为空时默认写到 autodl_unplug_charger_transformer_fm/libero/headless_tools/output/evals/lingbot_va",
    )
    parser.add_argument(
        "--disable-rollout-video",
        action="store_true",
        help="只跑评测，不保存 rollout 视频",
    )
    parser.add_argument(
        "--disable-action-trace",
        action="store_true",
        help="只保存视频，不保存 action trace",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="rollout 视频帧率",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()

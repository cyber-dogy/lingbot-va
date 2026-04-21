# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_libero_train_cfg import va_libero_train_cfg


va_libero_train_4070_cfg = EasyDict(__name__="Config: VA libero train 4070")
va_libero_train_4070_cfg.update(va_libero_train_cfg)
va_libero_train_4070_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser(
    os.getenv("LINGBOT_VA_MODEL_DIR", "~/models/lingbot-va/libero")
)
va_libero_train_4070_cfg.dataset_path = os.path.expanduser(
    os.getenv("LINGBOT_VA_DATASET_DIR", "~/datasets/libero_lerobot")
)
va_libero_train_4070_cfg.empty_emb_path = os.path.join(
    va_libero_train_4070_cfg.dataset_path, "empty_emb.pt"
)
va_libero_train_4070_cfg.enable_wandb = False
va_libero_train_4070_cfg.load_worker = int(os.getenv("LINGBOT_VA_LOAD_WORKERS", "4"))
va_libero_train_4070_cfg.gradient_accumulation_steps = int(
    os.getenv("LINGBOT_VA_GRAD_ACCUM_STEPS", "16")
)
va_libero_train_4070_cfg.num_steps = int(os.getenv("LINGBOT_VA_NUM_STEPS", "1000"))

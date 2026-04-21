# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_libero_cfg import va_libero_cfg


va_libero_4070_cfg = EasyDict(__name__="Config: VA libero 4070")
va_libero_4070_cfg.update(va_libero_cfg)
va_libero_4070_cfg.enable_offload = True
va_libero_4070_cfg.port = int(os.getenv("LINGBOT_VA_PORT", "29056"))
va_libero_4070_cfg.save_root = os.path.expanduser(
    os.getenv("LINGBOT_VA_SAVE_ROOT", "outputs/libero_4070")
)
va_libero_4070_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser(
    os.getenv("LINGBOT_VA_MODEL_DIR", "~/models/lingbot-va/libero")
)

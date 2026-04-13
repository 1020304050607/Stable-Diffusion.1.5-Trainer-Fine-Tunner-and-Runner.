# config.py
import os
from pathlib import Path

DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "output")

DEFAULT_RESOLUTION = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_STEPS = 16000
DEFAULT_SAVE_EVERY = 2000
DEFAULT_LR = 3e-5
DEFAULT_RANK = 64
DEFAULT_ALPHA = 128

MODEL_PRESETS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-dev",
}

IMAGE_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
VIDEO_EXT = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

VRAM_PRESETS = {
    "6GB":  {"res": 512, "bs": 1,  "ga": 32, "rank": 16},
    "8GB":  {"res": 512, "bs": 1,  "ga": 16, "rank": 32},
    "12GB": {"res": 768, "bs": 2,  "ga": 8,  "rank": 64},
    "16GB": {"res": 768, "bs": 3,  "ga": 4,  "rank": 128},
    "24GB": {"res": 1024,"bs": 4,  "ga": 4,  "rank": 128},
}

def get_default_config():
    """Return a default configuration dictionary"""
    return {
        "data_dir": DEFAULT_DATA_DIR,
        "checkpoint_dir": DEFAULT_OUTPUT_DIR,
        "resolution": DEFAULT_RESOLUTION,
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_steps": DEFAULT_STEPS,
        "save_every": DEFAULT_SAVE_EVERY,
        "lr": DEFAULT_LR,
        "rank": DEFAULT_RANK,
        "alpha": DEFAULT_ALPHA,
        "model_id": MODEL_PRESETS["sd15"],
    }

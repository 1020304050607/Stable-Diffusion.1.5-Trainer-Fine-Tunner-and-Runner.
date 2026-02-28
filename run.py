import os
import sys
import subprocess

os.system("")
R    = "\033[91m"
G    = "\033[92m"
Y    = "\033[93m"
C    = "\033[96m"
W    = "\033[97m"
DIM  = "\033[2m"
RST  = "\033[0m"
BOLD = "\033[1m"

def clr(t, c):
    return f"{c}{t}{RST}"

def section(title):
    print(f"\n{clr('─'*60, DIM)}")
    print(f"  {clr('▸', C)} {clr(title, BOLD)}")
    print(clr('─'*60, DIM))


PACKAGES = [
    ("tqdm",         "tqdm"),
    ("PIL",          "Pillow"),
    ("safetensors",  "safetensors"),
    ("accelerate",   "accelerate"),
    ("peft",         "peft"),
    ("cv2",          "opencv-python"),
    ("transformers", "transformers"),
    ("diffusers",    "diffusers[torch]"),
    ("torchvision",  "torchvision --index-url https://download.pytorch.org/whl/cu121"),
    ("torch",        "torch --index-url https://download.pytorch.org/whl/cu121"),
]


def _can_import(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _install_with_progress(label, pip_args):
    import re as _re
    import time as _time
    import threading

    cmd = [sys.executable, "-m", "pip", "install",
           "--no-warn-script-location"] + pip_args

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    bar_width  = 28
    last_pct   = -1
    last_label = label
    speed_str  = ""
    start      = _time.time()
    last_bytes = [0.0]

    pct_re  = _re.compile(r"(\d+)%")
    size_re = _re.compile(r"([\d.]+)\s*(MB|KB|GB)", _re.IGNORECASE)
    pkg_re  = _re.compile(r"Downloading\s+([\w.\-]+(?:\.whl|\.tar\.gz)?)")

    spinning = [True]
    frames   = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    tick     = [0]

    def _spin():
        while spinning[0]:
            bar   = "░" * bar_width
            col   = DIM
            sys.stdout.write(
                f"\r  {clr(frames[tick[0] % len(frames)], C)} "
                f"{clr(bar, col)}   0%  "
                f"{clr(last_label[:32], W)}  {clr('waiting...', DIM)}   "
            )
            sys.stdout.flush()
            tick[0] += 1
            _time.sleep(0.09)

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    for line in proc.stdout:
        line = line.strip()

        pkg_m = pkg_re.search(line)
        if pkg_m:
            last_label = pkg_m.group(1)
            spinning[0] = False
            spin_thread.join()

        pct_m = pct_re.search(line)
        if pct_m:
            pct = int(pct_m.group(1))
            if pct != last_pct:
                last_pct = pct
                filled   = int(bar_width * pct / 100)
                bar      = "█" * filled + "░" * (bar_width - filled)
                col      = G if pct > 80 else C if pct > 40 else Y
                sys.stdout.write(
                    f"\r  {clr('↓', C)} "
                    f"{clr(bar, col)} {clr(f'{pct:3d}%', col)}  "
                    f"{clr(last_label[:32], W)}  {clr(speed_str, DIM)}   "
                )
                sys.stdout.flush()

        size_matches = size_re.findall(line)
        if size_matches:
            try:
                val, unit = size_matches[-1]
                val = float(val)
                if unit.upper() == "GB":
                    val *= 1024.0
                elif unit.upper() == "KB":
                    val /= 1024.0
                elapsed   = max(_time.time() - start, 0.001)
                speed     = val / elapsed
                speed_str = f"{speed:.1f} MB/s" if speed >= 1 else f"{speed*1024:.0f} KB/s"
            except Exception:
                pass

    spinning[0] = False
    spin_thread.join()
    proc.wait()

    if proc.returncode == 0:
        sys.stdout.write(f"\r  {clr('✓', G)} {clr(label, W)} ready{' '*55}\n")
    else:
        sys.stdout.write(f"\r  {clr('!', Y)} {clr(label, W)} may have issues — continuing{' '*40}\n")
    sys.stdout.flush()


def bootstrap():
    section("Checking Dependencies")
    print(f"  {clr('Scanning for missing packages...', DIM)}\n")

    missing = [(lbl, pip) for mod, pip in PACKAGES
               for lbl in [pip.split()[0]]
               if not _can_import(mod)]

    missing = []
    for mod, pip in PACKAGES:
        if not _can_import(mod):
            missing.append((mod.replace("PIL", "Pillow").replace("cv2", "opencv-python"), pip))

    if not missing:
        print(f"  {clr('✓', G)} All dependencies already installed.\n")
        return

    print(f"  {clr(f'Found {len(missing)} package(s) to install:', Y)}\n")
    for lbl, _ in missing:
        print(f"    {clr('·', DIM)} {lbl}")
    print()

    for label, pip_spec in missing:
        _install_with_progress(label, pip_spec.split())

    print(f"\n  {clr('✓', G)} All packages ready.\n")


bootstrap()


import glob
import re
import time
import json
import warnings
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TRANSFORMERS_VERBOSITY"]        = "error"
os.environ["DIFFUSERS_VERBOSITY"]           = "error"
os.environ["TOKENIZERS_PARALLELISM"]        = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def vram_str():
    if not torch.cuda.is_available():
        return "CPU"
    free   = torch.cuda.mem_get_info()[0] / 1024**3
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    used   = total - free
    n      = 20
    filled = int(n * used / total)
    bar    = "█" * filled + "░" * (n - filled)
    col    = G if used / total < .7 else Y if used / total < .9 else R
    return f"{clr(bar, col)}  {clr(f'{used:.1f}', col)}/{total:.0f}GB"


def print_header():
    clear()
    print(clr("╔══════════════════════════════════════════════════════════╗", C))
    print(clr("║", C) + clr("      SD LoRA Trainer  ·  Train + Fine-Tune               ", BOLD) + clr("║", C))
    print(clr("║", C) + clr("      RTX Ready  ·  Training  Fine-Tuning  Generation     ", DIM)  + clr("║", C))
    print(clr("╚══════════════════════════════════════════════════════════╝", C))
    print()
    if torch.cuda.is_available():
        print(f"  {clr('GPU ', DIM)} {torch.cuda.get_device_properties(0).name}")
        print(f"  {clr('VRAM', DIM)} {vram_str()}")
    print()


MAX_FRAMES_PER_VIDEO = 10
VIDEO_EXTENSIONS     = ["mp4", "mov", "avi", "mkv", "webm"]


class LatentCacheDataset(Dataset):
    def __init__(self, folder, tokenizer, vae, resolution, checkpoint_dir):
        self.tokenizer  = tokenizer
        self.cache_file = os.path.join(checkpoint_dir, "latent_cache.pt")
        self.meta_file  = os.path.join(checkpoint_dir, "dataset_meta.json")
        self.resolution = resolution

        self.image_paths = []
        for ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            self.image_paths += glob.glob(os.path.join(folder, f"*.{ext}"))

        self.video_paths = []
        if CV2_AVAILABLE:
            for ext in VIDEO_EXTENSIONS:
                self.video_paths += glob.glob(os.path.join(folder, f"*.{ext}"))

        self.captions = {}
        for txt in glob.glob(os.path.join(folder, "*.txt")):
            name = os.path.splitext(os.path.basename(txt))[0]
            try:
                with open(txt, "r", encoding="utf-8") as f:
                    self.captions[name] = f.read().strip()
            except Exception:
                continue

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.data = self._prepare_latents(vae)

    def extract_frames(self, video_path):
        frames = []
        if not CV2_AVAILABLE:
            return frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return frames
        step = max(1, total_frames // MAX_FRAMES_PER_VIDEO)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            except Exception:
                continue
            if len(frames) >= MAX_FRAMES_PER_VIDEO:
                break
        cap.release()
        return frames

    def _prepare_latents(self, vae):
        total_items = len(self.image_paths) + len(self.video_paths)
        if os.path.exists(self.cache_file) and os.path.exists(self.meta_file):
            with open(self.meta_file) as f:
                meta = json.load(f)
            if meta.get("count") == total_items:
                print(f"  {clr('✓', G)} Loaded cached latents — skipping VAE encode")
                return torch.load(self.cache_file)

        print(f"  {clr('→', C)} Encoding VAE latents...")
        print(
    f" {clr('ℹ', C)} "
    f"{clr('Don’t worry if it looks stuck, speed depends on your GPU’s VRAM', DIM)}\n")
        vae.eval()
        data = []

        for path in tqdm(self.image_paths, desc="  Images", unit="img",
                         dynamic_ncols=True, colour="cyan"):
            try:
                img    = Image.open(path).convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(device, non_blocking=True)
                with torch.no_grad():
                    latent = vae.encode(tensor).latent_dist.sample() * 0.18215
                data.append({
                    "latent": latent.squeeze(0).cpu(),
                    "caption": self.captions.get(
                        os.path.splitext(os.path.basename(path))[0],
                        os.path.basename(path)
                    )
                })
            except Exception:
                print(f"  {clr('!', Y)} Skipping corrupted image: {path}")
                continue

        if CV2_AVAILABLE and self.video_paths:
            for path in tqdm(self.video_paths, desc="  Videos", unit="vid",
                             dynamic_ncols=True, colour="cyan"):
                try:
                    frames = self.extract_frames(path)
                    if not frames:
                        print(f"  {clr('!', Y)} Skipping corrupted video: {path}")
                        continue
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    caption   = self.captions.get(base_name, base_name)
                    for idx, frame in enumerate(frames):
                        try:
                            tensor = self.transform(frame).unsqueeze(0).to(device, non_blocking=True)
                            with torch.no_grad():
                                latent = vae.encode(tensor).latent_dist.sample() * 0.18215
                            data.append({
                                "latent": latent.squeeze(0).cpu(),
                                "caption": f"{caption}, frame {idx}"
                            })
                        except Exception:
                            continue
                except Exception:
                    print(f"  {clr('!', Y)} Skipping corrupted video: {path}")
                    continue

        torch.save(data, self.cache_file)
        with open(self.meta_file, "w") as f:
            json.dump({"count": total_items}, f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        tokens = self.tokenizer(
            item["caption"], padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        )
        return {
            "latent":         item["latent"],
            "input_ids":      tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0)
        }


class ImageCaptionDataset(Dataset):
    def __init__(self, folder, tokenizer, resolution):
        self.paths = []
        for ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            self.paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        print(f"  {clr('✓', G)} Loaded {clr(str(len(self.paths)), W)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))
        image   = self.transform(image)
        caption = os.path.splitext(os.path.basename(path))[0]
        txt     = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt):
            try:
                with open(txt, encoding="utf-8") as f:
                    caption = f.read().strip()
            except Exception:
                pass
        tokens = self.tokenizer(
            caption, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        return image, tokens.input_ids[0]


def find_latest_checkpoint(folder):
    hits = []
    for pattern in ["step_*.pt", "step_*.safetensors"]:
        for c in glob.glob(os.path.join(folder, pattern)):
            m = re.search(r"step_(\d+)\.(pt|safetensors)$", os.path.basename(c))
            if m:
                hits.append((c, int(m.group(1))))
    if not hits:
        return None, 0
    hits.sort(key=lambda x: x[1])
    return hits[-1]


def find_all_weights(folder):
    found = []
    for f in (glob.glob(os.path.join(folder, "*.pt")) +
              glob.glob(os.path.join(folder, "*.safetensors"))):
        found.append(f)
    return sorted(found)


def load_weights_into_unet(unet, path):
    ext    = Path(path).suffix.lower()
    state  = load_file(path) if ext == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    result = unet.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.missing_keys)
    print(f"  {clr('✓', G)} Loaded {loaded}/{len(state)} tensors from {clr(Path(path).name, W)}")


def save_checkpoint_st(unet, text_encoder, folder, step):
    os.makedirs(folder, exist_ok=True)
    lora_state = {k: v for k, v in unet.state_dict().items() if "lora_" in k}
    lora_state.update({k: v for k, v in text_encoder.state_dict().items() if "lora_" in k})
    save_file(lora_state, os.path.join(folder, f"step_{step}.safetensors"))
    print(f"  {clr('💾', Y)} Saved step_{step}.safetensors")


def save_checkpoint_pt(unet, folder, step):
    os.makedirs(folder, exist_ok=True)
    pt_path    = os.path.join(folder, f"step_{step}.pt")
    st_path    = os.path.join(folder, f"step_{step}.safetensors")
    state      = unet.state_dict()
    torch.save(state, pt_path)
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if lora_state:
        save_file(lora_state, st_path)
    print(f"  {clr('💾', Y)} Saved step_{step}.pt + step_{step}.safetensors")


def run_finetune(cfg):
    section("Loading Models  [Fine-Tune Mode]")

    data_dir       = cfg["data_dir"]
    checkpoint_dir = cfg["checkpoint_dir"]
    model_id       = cfg["model_id"]
    max_steps      = int(cfg["max_steps"])
    save_every     = int(cfg["save_every"])
    batch_size     = int(cfg["batch_size"])
    grad_accum     = int(cfg["grad_accum"])
    resolution     = int(cfg["resolution"])
    unet_lr        = float(cfg["unet_lr"])
    text_lr        = float(cfg["text_lr"])
    unet_rank      = int(cfg["unet_rank"])
    unet_alpha     = int(cfg["unet_alpha"])
    te_rank        = int(cfg["te_rank"])
    te_alpha       = int(cfg["te_alpha"])
    dropout        = float(cfg["dropout"])
    grad_ckpt      = bool(cfg.get("gradient_checkpointing", True))
    existing_lora  = cfg.get("existing_lora")

    unet_targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    te_targets   = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"  {clr('Base model', DIM)} {model_id}")
    print(f"  {clr('Data dir  ', DIM)} {data_dir}")
    print(f"  {clr('Output    ', DIM)} {checkpoint_dir}")

    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler    = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    for p in vae.parameters():          p.requires_grad = False
    for p in unet.parameters():         p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False

    unet = get_peft_model(unet, LoraConfig(
        r=unet_rank, lora_alpha=unet_alpha,
        target_modules=unet_targets,
        lora_dropout=dropout, bias="none"
    ))
    text_encoder = get_peft_model(text_encoder, LoraConfig(
        r=te_rank, lora_alpha=te_alpha,
        target_modules=te_targets,
        lora_dropout=dropout, bias="none"
    ))

    if existing_lora and os.path.exists(existing_lora):
        print(f"  {clr('→', C)} Loading existing LoRA weights...")
        state = load_file(existing_lora)
        unet.load_state_dict(state, strict=False)
        text_encoder.load_state_dict(state, strict=False)

    trainable = (
        sum(p.numel() for p in unet.parameters() if p.requires_grad) +
        sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    )
    print(f"  {clr('✓', G)} Trainable LoRA params: {clr(f'{trainable:,}', W)}")

    if grad_ckpt:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(text_encoder.parameters()),
        lr=unet_lr, betas=(0.9, 0.99), weight_decay=0.01
    )

    section("Loading Dataset")
    print(f" {clr('ℹ', C)} {clr('Don’t worry if it looks stuck, speed depends on your GPU’s VRAM', DIM)}\n")
    dataset    = LatentCacheDataset(data_dir, tokenizer, vae, resolution, checkpoint_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    scaler      = torch.cuda.amp.GradScaler()
    global_step = 0

    section("Training")
    print(f"  {clr('Steps', DIM)} 0 → {max_steps}  ({clr(str(max_steps), W)} total)\n")

    pbar       = tqdm(total=max_steps, desc="  Fine-Tune", unit="step",
                      dynamic_ncols=True, colour="cyan")
    start_time = time.time()

    while global_step < max_steps:
        for batch in dataloader:
            latents   = batch["latent"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                enc_hidden    = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
                noise         = torch.randn_like(latents)
                timesteps     = torch.randint(0, scheduler.config.num_train_timesteps,
                                              (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                noise_pred    = unet(noisy_latents, timesteps, enc_hidden).sample
                loss          = torch.nn.functional.mse_loss(noise_pred, noise) / grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            elapsed = time.time() - start_time
            eta_sec = (elapsed / global_step) * (max_steps - global_step) if global_step > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_sec)))
            mem     = torch.cuda.memory_allocated() / 1024**3

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum:.4f}",
                VRAM=f"{mem:.2f}GB",
                ETA=eta_str
            )

            if global_step % save_every == 0:
                pbar.clear()
                save_checkpoint_st(unet, text_encoder, checkpoint_dir, global_step)

            if global_step >= max_steps:
                break

    pbar.close()

    section("Saving Final Model")
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path  = os.path.join(checkpoint_dir, f"finetuned_final_{ts}.safetensors")
    final_state = {k: v for k, v in unet.state_dict().items() if "lora_" in k}
    final_state.update({k: v for k, v in text_encoder.state_dict().items() if "lora_" in k})
    save_file(final_state, final_path)
    print(f"  {clr('✓', G)} {final_path}")
    print(f"\n  {clr('Done!', G)}\n")


def run_training(cfg):
    section("Loading Models  [Training Mode]")

    data_dir       = cfg["data_dir"]
    checkpoint_dir = cfg["checkpoint_dir"]
    model_id       = cfg["model_id"]
    resolution     = int(cfg["resolution"])
    batch_size     = int(cfg["batch_size"])
    max_steps      = int(cfg["max_steps"])
    save_every     = int(cfg["save_every"])
    lr             = float(cfg["lr"])
    rank           = int(cfg["rank"])
    alpha          = int(cfg["alpha"])
    weights        = cfg.get("weights_path")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"  {clr('Base model', DIM)} {model_id}")
    print(f"  {clr('Data dir  ', DIM)} {data_dir}")
    print(f"  {clr('Output    ', DIM)} {checkpoint_dir}")
    if weights:
        print(f"  {clr('Weights   ', DIM)} {weights}")

    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler    = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    for p in unet.parameters():         p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False
    for p in vae.parameters():          p.requires_grad = False

    target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    unet = get_peft_model(unet, LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05, bias="none"
    ))

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  {clr('✓', G)} Trainable params: {clr(f'{trainable:,}', W)}")

    latest_ckpt, completed = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        print(f"\n  {clr('↺', Y)} Resuming from step_{completed}")
        load_weights_into_unet(unet, latest_ckpt)
    elif weights:
        print(f"\n  {clr('→', C)} Loading weights: {Path(weights).name}")
        load_weights_into_unet(unet, weights)
    else:
        print(f"\n  {clr('→', C)} Starting fresh from base model")

    if completed >= max_steps:
        print(f"\n  {clr('✓', G)} Already at {completed}/{max_steps} — nothing to do!")
        input("\n  Press Enter...")
        return

    try:
        unet.enable_xformers_memory_efficient_attention()
        print(f"  {clr('✓', G)} xformers enabled")
    except Exception:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print(f"  {clr('✓', G)} Flash attention enabled")
        except Exception:
            pass

    section("Loading Dataset")
    print(f" {clr('ℹ', C)} {clr('Don’t worry if it looks stuck, speed depends on your GPU’s VRAM', DIM)}\n")
    dataset = ImageCaptionDataset(data_dir, tokenizer, resolution)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    section("Training")
    print(f"  {clr('Steps', DIM)} {completed} → {max_steps}  ({clr(str(max_steps - completed), W)} remaining)\n")

    global_step = completed
    start_time  = time.time()
    pbar        = tqdm(total=max_steps, initial=completed, desc="  Training",
                       unit="step", dynamic_ncols=True, colour="cyan")

    while global_step < max_steps:
        for images, input_ids in loader:
            step_start = time.time()
            images     = images.to(device)
            input_ids  = input_ids.to(device)

            with torch.no_grad():
                latents               = vae.encode(images).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1

            elapsed  = time.time() - start_time
            done     = global_step - completed
            eta_sec  = (elapsed / done) * (max_steps - global_step) if done > 0 else 0
            eta_str  = str(timedelta(seconds=int(eta_sec)))
            mem      = torch.cuda.memory_allocated() / 1024**3
            step_t   = time.time() - step_start

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                step=f"{step_t:.2f}s",
                VRAM=f"{mem:.2f}GB",
                ETA=eta_str
            )

            if global_step % save_every == 0:
                pbar.clear()
                save_checkpoint_pt(unet, checkpoint_dir, global_step)

            if global_step >= max_steps:
                break

    pbar.close()

    section("Saving Final Model")
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_pt  = os.path.join(checkpoint_dir, f"final_{ts}.pt")
    final_st  = os.path.join(checkpoint_dir, f"final_{ts}.safetensors")
    state     = unet.state_dict()
    torch.save(state, final_pt)
    lora_only = {k: v for k, v in state.items() if "lora_" in k}
    if lora_only:
        save_file(lora_only, final_st)
    print(f"  {clr('✓', G)} {final_pt}")
    print(f"  {clr('✓', G)} {final_st}")
    print(f"\n  {clr('Done!', G)}\n")


def run_generate(cfg):
    section("Image Generation")

    model_id   = cfg["model_id"]
    lora_path  = cfg.get("lora_path")
    output_dir = Path(cfg["output_dir"])
    prompt     = cfg["prompt"]
    num_images = int(cfg["num_images"])
    steps      = int(cfg.get("steps", 30))
    guidance   = float(cfg.get("guidance", 7.5))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {clr('Base model', DIM)} {model_id}")
    if lora_path:
        print(f"  {clr('LoRA      ', DIM)} {lora_path}")
    print(f"  {clr('Output    ', DIM)} {output_dir}")
    print(f"  {clr('Prompt    ', DIM)} {prompt}")
    print(f"  {clr('Images    ', DIM)} {num_images}  |  {clr('Steps', DIM)} {steps}  |  {clr('CFG', DIM)} {guidance}\n")

    print(f"  {clr('→', C)} Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    if lora_path and os.path.exists(lora_path):
        print(f"  {clr('→', C)} Applying LoRA weights...")
        lora_state = load_file(lora_path)
        pipe.unet.load_state_dict(lora_state, strict=False)
        print(f"  {clr('✓', G)} LoRA applied")
    elif lora_path:
        print(f"  {clr('!', Y)} LoRA file not found — running base model only")

    print(f"  {clr('→', C)} Generating {num_images} image(s)...\n")

    for i in tqdm(range(1, num_images + 1), desc="  Generating", unit="img",
                  dynamic_ncols=True, colour="cyan"):
        image     = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
        file_path = output_dir / f"dbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
        image.save(file_path)

    print(f"\n  {clr('✓', G)} Done! Saved to {clr(str(output_dir.resolve()), W)}\n")


HF_MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "dreamlike-art/dreamlike-photoreal-2.0",
    "SG161222/Realistic_Vision_V5.1_noVAE",
]

PRESETS = {
    "1": ("Higher Resolution  512→768",              dict(resolution=768,  batch_size=2)),
    "2": ("Higher Resolution  512→1024",             dict(resolution=1024, batch_size=1)),
    "3": ("Higher Rank  128→192",                    dict(rank=192, alpha=384)),
    "4": ("More Steps  +8000",                       dict(max_steps=24000)),
    "5": ("Faster Training  (rank↓ batch↑)",         dict(rank=64, alpha=128, batch_size=8)),
    "6": ("Quality Pack  768 + rank192 + 20k steps", dict(resolution=768, rank=192, alpha=384, max_steps=20000, batch_size=2)),
}


def ask(label, default, cast=str):
    v = input(f"  {clr(label, C)} [{clr(str(default), W)}]: ").strip()
    if not v:
        return default
    try:
        return cast(v)
    except Exception:
        return default


def pick_data_dir(default=""):
    section("Training Data Folder")
    print(f"  {clr('Enter the path to your image/video dataset folder.', DIM)}")
    if default:
        print(f"  {clr('Last used:', DIM)} {default}\n")
    while True:
        path = input(f"  {clr('Path', C)} [{clr(default or 'required', W)}]: ").strip()
        if not path and default:
            path = default
        if not path:
            print(f"  {clr('A data directory is required.', R)}")
            continue
        path = path.strip('"').strip("'")
        if not os.path.isdir(path):
            print(f"  {clr('Directory not found. Try again.', R)}")
            continue
        img_count = sum(
            len(glob.glob(os.path.join(path, f"*.{ext}")))
            for ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
        )
        vid_count = sum(
            len(glob.glob(os.path.join(path, f"*.{ext}")))
            for ext in VIDEO_EXTENSIONS
        ) if CV2_AVAILABLE else 0
        print(f"  {clr('✓', G)} Found {clr(str(img_count), W)} images, {clr(str(vid_count), W)} videos")
        return path


def pick_output_dir(default="", label="Output / Checkpoint Folder"):
    section(label)
    print(f"  {clr('Where to save checkpoints and final weights.', DIM)}")
    if default:
        print(f"  {clr('Last used:', DIM)} {default}\n")
    path = input(f"  {clr('Path', C)} [{clr(default or 'required', W)}]: ").strip()
    if not path and default:
        return default
    path = path.strip('"').strip("'")
    if path:
        os.makedirs(path, exist_ok=True)
        print(f"  {clr('✓', G)} Folder ready")
        return path
    return default


def pick_model():
    section("Select SD Base Model")
    print(f"  {clr('The base Stable Diffusion model to use.', DIM)}\n")
    for i, m in enumerate(HF_MODELS, 1):
        print(f"  {clr(str(i), C)}.  {m}")
    print(f"  {clr('c', C)}.  Custom HuggingFace ID")
    print()
    while True:
        ch = input(f"  {clr('Model', C)} [{clr('1', W)}]: ").strip() or "1"
        if ch == "c":
            v = input(f"  {clr('HF ID', C)}: ").strip()
            if v:
                return v
        elif ch.isdigit() and 1 <= int(ch) <= len(HF_MODELS):
            return HF_MODELS[int(ch) - 1]
        print(f"  {clr('Invalid.', R)}")


def pick_lora_for_generation(search_dirs):
    section("Select LoRA Weights")
    print(f"  {clr('Pick a .safetensors file to apply to the model.', DIM)}")
    print(f"  {clr('Searches your output folders and current directory.', DIM)}\n")

    files    = []
    searched = list(search_dirs) + [os.getcwd()]
    for d in searched:
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.safetensors")):
                if f not in files:
                    files.append(f)

    if not files:
        print(f"  {clr('No .safetensors files found in known folders.', DIM)}")
        custom = input(f"  {clr('Enter full path manually (or Enter to skip)', C)}: ").strip().strip('"').strip("'")
        return custom if custom and os.path.exists(custom) else None

    print(f"  {clr('0', C)}.  None — use base model only")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f) / 1024**2
        print(f"  {clr(str(i), C)}.  {Path(f).name}  {clr(f'({size:.0f} MB)', DIM)}  {clr(str(Path(f).parent), DIM)}")
    print(f"  {clr('m', C)}.  Enter path manually")
    print()

    while True:
        ch = input(f"  {clr('Choice', C)} [{clr('0', W)}]: ").strip() or "0"
        if ch == "0":
            return None
        if ch == "m":
            custom = input(f"  {clr('Full path', C)}: ").strip().strip('"').strip("'")
            if custom and os.path.exists(custom):
                return custom
            print(f"  {clr('File not found.', R)}")
        elif ch.isdigit() and 1 <= int(ch) <= len(files):
            return files[int(ch) - 1]
        else:
            print(f"  {clr('Invalid.', R)}")


def pick_weights(checkpoint_dir):
    section("Starting Weights  (optional)")
    print(f"  {clr('Load existing weights to continue training from.', DIM)}")
    print(f"  {clr('Skip to start fresh from the base model.', DIM)}\n")

    files = [
        f for f in find_all_weights(checkpoint_dir)
        if not re.search(r"step_\d+\.(pt|safetensors)$", f)
    ]

    if not files:
        print(f"  {clr('No extra weight files found.', DIM)}")
        print(f"  {clr('→ Starting from base model.', Y)}\n")
        return None

    print(f"  {clr('0', C)}.  None — start fresh")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f) / 1024**2
        print(f"  {clr(str(i), C)}.  {Path(f).name}  {clr(f'({size:.0f} MB)', DIM)}")
    print()

    while True:
        ch = input(f"  {clr('Choice', C)} [{clr('0', W)}]: ").strip() or "0"
        if ch == "0":
            return None
        if ch.isdigit() and 1 <= int(ch) <= len(files):
            return files[int(ch) - 1]
        print(f"  {clr('Invalid.', R)}")


def pick_training_settings(checkpoint_dir):
    section("Training Settings")
    ckpt, step = find_latest_checkpoint(checkpoint_dir)
    if ckpt:
        print(f"  {clr('✓', G)} Checkpoint found: step_{step} — will auto-resume\n")

    resolution = ask("Resolution (512 / 768 / 1024)", 512, int)
    batch_size = ask("Batch size", 4, int)
    max_steps  = ask("Max steps", 16000, int)
    save_every = ask("Save every N steps", 4000, int)
    lr         = ask("Learning rate", 3e-5, float)
    rank       = ask("LoRA rank", 128, int)
    alpha      = ask("LoRA alpha", 256, int)

    if resolution >= 1024 and batch_size > 1:
        print(f"\n  {clr('! 1024px — auto-setting batch to 1 for VRAM safety.', Y)}")
        batch_size = 1

    return dict(
        resolution=resolution, batch_size=batch_size,
        max_steps=max_steps, save_every=save_every,
        lr=lr, rank=rank, alpha=alpha
    )


def pick_finetune_settings():
    section("Fine-Tune Settings")
    print(f"  {clr('Advanced LoRA fine-tuning with separate UNet + Text Encoder rates.', DIM)}\n")

    resolution = ask("Resolution (512 / 768 / 1024)", 512, int)
    batch_size = ask("Batch size", 2, int)
    grad_accum = ask("Gradient accumulation steps", 8, int)
    max_steps  = ask("Max steps", 4000, int)
    save_every = ask("Save every N steps", 4000, int)
    unet_lr    = ask("UNet learning rate", 3e-5, float)
    text_lr    = ask("Text encoder learning rate", 1e-5, float)
    unet_rank  = ask("UNet LoRA rank", 128, int)
    unet_alpha = ask("UNet LoRA alpha", 80, int)
    te_rank    = ask("Text encoder LoRA rank", 32, int)
    te_alpha   = ask("Text encoder LoRA alpha", 24, int)
    dropout    = ask("LoRA dropout", 0.03, float)
    grad_ckpt  = ask("Gradient checkpointing (y/n)", "y", str)

    if resolution >= 1024 and batch_size > 1:
        print(f"\n  {clr('! 1024px — auto-setting batch to 1.', Y)}")
        batch_size = 1

    return dict(
        resolution=resolution, batch_size=batch_size,
        grad_accum=grad_accum, max_steps=max_steps,
        save_every=save_every, unet_lr=unet_lr,
        text_lr=text_lr, unet_rank=unet_rank,
        unet_alpha=unet_alpha, te_rank=te_rank,
        te_alpha=te_alpha, dropout=dropout,
        gradient_checkpointing=(str(grad_ckpt).strip().lower() != "n")
    )


def pick_existing_lora(checkpoint_dir):
    section("Existing LoRA Weights  (optional)")
    print(f"  {clr('Load a previously trained LoRA to continue fine-tuning from it.', DIM)}\n")

    files = find_all_weights(checkpoint_dir)
    if not files:
        print(f"  {clr('No weight files found. Starting fresh.', DIM)}\n")
        return None

    print(f"  {clr('0', C)}.  None — start from base model only")
    for i, f in enumerate(files, 1):
        size = os.path.getsize(f) / 1024**2
        print(f"  {clr(str(i), C)}.  {Path(f).name}  {clr(f'({size:.0f} MB)', DIM)}")
    print()

    while True:
        ch = input(f"  {clr('Choice', C)} [{clr('0', W)}]: ").strip() or "0"
        if ch == "0":
            return None
        if ch.isdigit() and 1 <= int(ch) <= len(files):
            return files[int(ch) - 1]
        print(f"  {clr('Invalid.', R)}")


def pick_improvements(cfg):
    section("Quality Improvements")
    rank_val = cfg.get("rank", cfg.get("unet_rank", "—"))
    print(f"  {clr('Current:', DIM)} res={cfg['resolution']}  rank={rank_val}  "
          f"steps={cfg['max_steps']}  batch={cfg['batch_size']}\n")

    for k, (name, _) in PRESETS.items():
        print(f"  {clr(k, C)}.  {name}")
    print(f"  {clr('0', C)}.  Keep current settings\n")

    ch = input(f"  {clr('Choose (combine e.g. 3,4)', C)}: ").strip()
    if not ch or ch == "0":
        return cfg

    for c in re.split(r"[,\s]+", ch):
        if c in PRESETS:
            cfg.update(PRESETS[c][1])
            print(f"  {clr('✓', G)} Applied: {PRESETS[c][0]}")

    if cfg["resolution"] >= 1024:
        cfg["batch_size"] = 1
    return cfg


last_data_dir   = ""
last_output_dir = ""


def flow_train(with_improvements=False):
    global last_data_dir, last_output_dir

    data_dir        = pick_data_dir(last_data_dir)
    output_dir      = pick_output_dir(last_output_dir)
    last_data_dir   = data_dir
    last_output_dir = output_dir

    model_id = pick_model()
    weights  = pick_weights(output_dir)
    settings = pick_training_settings(output_dir)
    cfg      = dict(model_id=model_id, weights_path=weights,
                    data_dir=data_dir, checkpoint_dir=output_dir, **settings)

    if with_improvements:
        cfg = pick_improvements(cfg)

    _confirm_and_run(cfg, mode="train")


def flow_finetune():
    global last_data_dir, last_output_dir

    data_dir        = pick_data_dir(last_data_dir)
    output_dir      = pick_output_dir(last_output_dir)
    last_data_dir   = data_dir
    last_output_dir = output_dir

    model_id      = pick_model()
    existing_lora = pick_existing_lora(output_dir)
    settings      = pick_finetune_settings()
    cfg           = dict(model_id=model_id, existing_lora=existing_lora,
                         data_dir=data_dir, checkpoint_dir=output_dir, **settings)

    _confirm_and_run(cfg, mode="finetune")


def flow_generate():
    global last_output_dir

    model_id   = pick_model()
    lora_path  = pick_lora_for_generation([last_output_dir])
    output_dir = pick_output_dir(
        os.path.join(last_output_dir, "generated") if last_output_dir else "",
        label="Image Output Folder"
    )

    section("Generation Settings")
    prompt = ""
    while not prompt:
        prompt = input(f"  {clr('Prompt', C)}: ").strip()
        if not prompt:
            print(f"  {clr('Prompt cannot be empty.', R)}")

    num_images = ask("Number of images", 4, int)
    steps      = ask("Inference steps", 30, int)
    guidance   = ask("Guidance scale (CFG)", 7.5, float)

    cfg = dict(
        model_id=model_id, lora_path=lora_path,
        output_dir=output_dir, prompt=prompt,
        num_images=num_images, steps=steps, guidance=guidance
    )

    _confirm_and_run(cfg, mode="generate")


def _confirm_and_run(cfg, mode):
    section("Summary")
    for k, v in cfg.items():
        if v is not None and v != "":
            print(f"  {clr(k, DIM)} = {clr(str(v), W)}")

    input(f"\n  {clr('Press Enter to start...', Y)}")

    try:
        if mode == "finetune":
            run_finetune(cfg)
        elif mode == "generate":
            run_generate(cfg)
        else:
            run_training(cfg)
    except KeyboardInterrupt:
        print(f"\n\n  {clr('Stopped. Latest checkpoint preserved.', Y)}\n")

    input(f"\n  {clr('Press Enter to return to menu...', DIM)}")


def main_menu():
    global last_data_dir, last_output_dir

    while True:
        print_header()

        img_count = sum(
            len(glob.glob(os.path.join(last_data_dir, f"*.{ext}")))
            for ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
        ) if last_data_dir else 0

        _, step  = find_latest_checkpoint(last_output_dir) if last_output_dir else (None, 0)
        ckpt_str = clr(f"step_{step}", G) if step else clr("none", DIM)
        data_str = clr(last_data_dir, W) if last_data_dir else clr("not set", DIM)
        out_str  = clr(last_output_dir, W) if last_output_dir else clr("not set", DIM)

        print(f"  {clr('Data dir   ', DIM)} {data_str}")
        if last_data_dir:
            print(f"  {clr('Images     ', DIM)} {clr(str(img_count), W)}")
        print(f"  {clr('Output dir ', DIM)} {out_str}")
        if last_output_dir:
            print(f"  {clr('Checkpoint ', DIM)} {ckpt_str}")

        print(f"\n  {clr('─'*50, DIM)}")
        print(f"  {clr('1', C)}.  Train          — standard LoRA training, auto-resume")
        print(f"  {clr('2', C)}.  Train + Boost  — pick quality improvements first")
        print(f"  {clr('3', C)}.  Fine-Tune      — dual LoRA, VAE latent cache, video support")
        print(f"  {clr('4', C)}.  Generate       — run inference with any LoRA")
        print(f"  {clr('5', C)}.  VRAM info")
        print(f"  {clr('6', C)}.  Exit")
        print()

        ch = input(f"  {clr('›', C)} ").strip()

        if ch == "1":
            try:
                flow_train()
            except Exception as e:
                print(f"\n  {clr(f'Error: {e}', R)}\n")
                input(f"  {clr('Press Enter...', DIM)}")

        elif ch == "2":
            try:
                flow_train(with_improvements=True)
            except Exception as e:
                print(f"\n  {clr(f'Error: {e}', R)}\n")
                input(f"  {clr('Press Enter...', DIM)}")

        elif ch == "3":
            try:
                flow_finetune()
            except Exception as e:
                print(f"\n  {clr(f'Error: {e}', R)}\n")
                input(f"  {clr('Press Enter...', DIM)}")

        elif ch == "4":
            try:
                flow_generate()
            except Exception as e:
                print(f"\n  {clr(f'Error: {e}', R)}\n")
                input(f"  {clr('Press Enter...', DIM)}")

        elif ch == "5":
            section("VRAM Info")
            if torch.cuda.is_available():
                p     = torch.cuda.get_device_properties(0)
                total = p.total_memory / 1024**3
                free  = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"  {clr('GPU          ', DIM)} {p.name}")
                print(f"  {clr('VRAM Total   ', DIM)} {total:.1f} GB")
                print(f"  {clr('VRAM Used    ', DIM)} {total - free:.1f} GB")
                print(f"  {clr('VRAM Free    ', DIM)} {free:.1f} GB")
                print(f"  {clr('CUDA         ', DIM)} {torch.version.cuda}")
                print(f"  {clr('PyTorch      ', DIM)} {torch.__version__}")
                print(f"\n  {vram_str()}")
            else:
                print(f"  {clr('No CUDA GPU detected.', R)}")
            input("\n  Press Enter...")

        elif ch == "6":
            print(f"\n  {clr('Goodbye!', C)}\n")
            sys.exit(0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n  {clr('Interrupted.', Y)}\n")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n  {clr(f'Fatal error: {e}', R)}\n")
        traceback.print_exc()
        input("\n  Press Enter to exit...")
        sys.exit(1)
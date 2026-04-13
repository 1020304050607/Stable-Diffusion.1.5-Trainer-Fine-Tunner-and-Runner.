# utils.py
import os
import re
import torch
import struct
import json
import glob
import warnings
import logging
import shutil
import cv2
from pathlib import Path
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

from boot import clr, sec, clear, R, G, Y, C, W, DIM, RST, BOLD, MAG

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

os.environ.update({
    "TRANSFORMERS_VERBOSITY": "error",
    "DIFFUSERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
})

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

IEXT = ["png","jpg","jpeg","webp","bmp","tiff"]
VEXT = ["mp4","mov","avi","mkv","webm"]
MAX_VFRAMES = 10

try:
    import cv2
    CV2 = True
except:
    CV2 = False

def gpu_gb():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3

def gpu_bar():
    if not torch.cuda.is_available(): return clr("no GPU", DIM)
    free = torch.cuda.mem_get_info()[0] / 1024**3
    tot = gpu_gb()
    used = tot - free
    n = 20
    bar = "#" * int(n * used/tot) + "." * (n - int(n * used/tot))
    col = G if used/tot < .7 else Y if used/tot < .9 else R
    return clr(bar, col) + " " + clr(str(round(used,1)), col) + "/" + str(round(tot)) + "GB"

def gpu_used():
    if not torch.cuda.is_available(): return 0.0
    return (gpu_gb() - torch.cuda.mem_get_info()[0]/1024**3)

# Model profiles
_PROFILES = {
    "sd1": dict(label="SD 1.x (512px native)", native=512, min_res=512, max_res=768,
                safe_rank=128, min_rank=4, safe_bs=2, min_bs=1, te_min_rank=4, flux=False,
                note="was trained at 512px. going above 768 wastes VRAM without much benefit."),
    "sd2": dict(label="SD 2.x (768px native)", native=768, min_res=768, max_res=1024,
                safe_rank=128, min_rank=4, safe_bs=2, min_bs=1, te_min_rank=4, flux=False,
                note="was trained at 768px. do not go below 768 or output gets noisy and broken."),
    "sdxl": dict(label="SDXL (1024px native)", native=1024, min_res=1024, max_res=1024,
                safe_rank=64, min_rank=4, safe_bs=1, min_bs=1, te_min_rank=4, flux=False,
                note="was trained at 1024px. going below 1024 breaks generation noticeably."),
    "flux": dict(label="Flux.1 (1024px native)", native=1024, min_res=1024, max_res=1024,
                safe_rank=32, min_rank=4, safe_bs=1, min_bs=1, te_min_rank=16, flux=True,
                note="1024px minimum. text encoder rank below 16 breaks generation on Flux."),
}
_DEFAULT = _PROFILES["sd1"]

def detect_local(path):
    if not path or not os.path.exists(path):
        return _DEFAULT.copy()
    try:
        ext = Path(path).suffix.lower()
        if ext == ".safetensors":
            with open(path, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8: return _DEFAULT.copy()
                hlen = struct.unpack("<Q", raw)[0]
                if hlen > 50_000_000: return _DEFAULT.copy()
                hdr = json.loads(f.read(hlen).decode("utf-8", errors="replace"))
            keys = set(hdr.keys())
        elif ext in (".pt",".pth",".ckpt"):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            keys = set((sd.get("state_dict", sd) if isinstance(sd, dict) else {}).keys())
        else:
            return _DEFAULT.copy()
        if any("double_stream" in k or "single_stream" in k or "img_attn" in k for k in keys):
            return _PROFILES["flux"].copy()
        if any("label_emb" in k or "add_embedding" in k for k in keys):
            return _PROFILES["sdxl"].copy()
        if any("cond_stage_model.model.transformer" in k or "open_clip" in k.lower() for k in keys):
            return _PROFILES["sd2"].copy()
        return _PROFILES["sd1"].copy()
    except Exception:
        return _DEFAULT.copy()

def detect_from_id(model_id):
    mid = model_id.lower()
    if "flux" in mid: return _PROFILES["flux"].copy()
    if "xl" in mid or "sdxl" in mid: return _PROFILES["sdxl"].copy()
    if "2-1" in mid or "2.1" in mid or "v2" in mid: return _PROFILES["sd2"].copy()
    return _PROFILES["sd1"].copy()

def vram_preset(prof=None):
    gb = gpu_gb()
    p = prof or _DEFAULT
    nr = p["native"]
    mr = p["min_res"]
    def cr(r): return max(r, mr)
    if gb >= 20:
        return dict(res=cr(nr), bs=4, ga=4, rank=p["safe_rank"], alpha=p["safe_rank"]*2,
                    nw=8, steps=16000, save=2000, lr=2e-5,
                    ur=p["safe_rank"], ua=p["safe_rank"]*2,
                    tr=min(64,p["safe_rank"]), ta=min(64,p["safe_rank"])*2,
                    drop=.03, gc=False, label="20GB+ RTX 3090/4090/A5000")
    elif gb >= 16:
        return dict(res=cr(min(nr,768)), bs=3, ga=4, rank=min(128,p["safe_rank"]), alpha=min(256,p["safe_rank"]*2),
                    nw=6, steps=16000, save=2000, lr=3e-5,
                    ur=min(128,p["safe_rank"]), ua=min(128,p["safe_rank"])*2,
                    tr=32, ta=64, drop=.03, gc=False, label="16GB RTX 3080Ti/4080/A4000")
    elif gb >= 12:
        return dict(res=cr(min(nr,768)), bs=2, ga=8, rank=min(64,p["safe_rank"]), alpha=min(128,p["safe_rank"]*2),
                    nw=4, steps=16000, save=2000, lr=3e-5,
                    ur=min(64,p["safe_rank"]), ua=min(128,p["safe_rank"])*2,
                    tr=32, ta=64, drop=.05, gc=True, label="12GB RTX 3060Ti/3080/4070")
    elif gb >= 8:
        return dict(res=cr(min(nr,512)), bs=1, ga=16, rank=min(32,p["safe_rank"]), alpha=min(64,p["safe_rank"]*2),
                    nw=2, steps=16000, save=2000, lr=3e-5,
                    ur=min(32,p["safe_rank"]), ua=min(64,p["safe_rank"])*2,
                    tr=16, ta=32, drop=.05, gc=True, label="8GB RTX 2080/3060/4060")
    else:
        return dict(res=cr(min(nr,512)), bs=1, ga=32, rank=min(16,p["safe_rank"]), alpha=min(32,p["safe_rank"])*2,
                    nw=2, steps=16000, save=2000, lr=3e-5,
                    ur=min(16,p["safe_rank"]), ua=min(32,p["safe_rank"])*2,
                    tr=8, ta=16, drop=.05, gc=True, label="6GB survival mode")

def show_vram_table():
    sec("GPU Size Recommendations")
    print(" " + clr("press enter after reading", DIM) + "\n")
    rows = [
        ("6GB", 512, 1, 32, 16, "gradient checkpointing on. slow but it works."),
        ("8GB", 512, 1, 16, 32, "512px only. ~1.5 steps/sec."),
        ("12GB", 768, 2, 8, 64, "768px comfortable. rank 128 possible."),
        ("16GB", 768, 3, 4,128, "768px batch 3 is the sweet spot."),
        ("20GB+",1024, 4, 4,128, "full quality. 1024px batch 4 rank 128."),
    ]
    for lbl, res, bs, ga, rk, note in rows:
        print(" " + clr(lbl.ljust(7), C+BOLD)
              + clr("res=" + str(res) + " bs=" + str(bs)
                    + " accum=" + str(ga) + " rank=" + str(rk), W))
        print(" " + clr(" " + note, DIM) + "\n")
    gb = gpu_gb()
    if gb > 0:
        p = vram_preset()
        print(" " + clr("your card: " + str(round(gb,1)) + "GB -> auto-filling for: " + p["label"], Y))

def print_header():
    clear()
    print(clr("=" * 60, C))
    print(clr(" SD LoRA Trainer | Full Pipeline | Image Edit | Face", BOLD))
    print(clr("=" * 60, C))
    if torch.cuda.is_available():
        print(" " + clr("GPU ", DIM) + torch.cuda.get_device_properties(0).name)
        print(" " + clr("VRAM ", DIM) + gpu_bar())
    print()

def run_resize(cfg):
    sec("Resize Images and Videos")
    td = cfg["data_dir"]
    sd = cfg.get("before_folder", os.path.join(os.path.dirname(td), "before"))
    w = int(cfg.get("resize_w", 512))
    h = int(cfg.get("resize_h", 512))
    sz = (w, h)
    ie = (".png",".jpg",".jpeg",".webp",".bmp")
    ve = (".mp4",".avi",".mov",".mkv")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(td, exist_ok=True)
    print(" " + clr("step 1/3 moving to staging...", C))
    moved = 0
    for f in os.listdir(td):
        try: shutil.move(os.path.join(td, f), os.path.join(sd, f)); moved += 1
        except Exception as ex: print(" " + clr("skip " + f + " " + str(ex), Y))
    print(" " + clr("moved " + str(moved) + " files", G) + "\n")
    print(" " + clr("step 2/3 resizing to " + str(w) + "x" + str(h) + "...", C) + "\n")
    staged = os.listdir(sd)
    img_files = [f for f in staged if f.lower().endswith(ie)]
    vid_files = [f for f in staged if f.lower().endswith(ve)]
    other = [f for f in staged if not f.lower().endswith(ie + ve)]
    for f in other:
        try: shutil.copy2(os.path.join(sd, f), os.path.join(td, f))
        except: pass
    iok = ifail = 0
    for f in tqdm(img_files, desc=" images", unit="img", dynamic_ncols=True, colour="cyan"):
        src = os.path.join(sd, f); dst = os.path.join(td, f)
        try:
            with Image.open(src) as img:
                img.resize(sz, Image.LANCZOS).save(dst, quality=100, subsampling=0)
            iok += 1
        except Exception as ex:
            print(" " + clr("problem: " + f + " " + str(ex), Y)); ifail += 1
    vok = vfail = 0
    if CV2 and vid_files:
        for f in tqdm(vid_files, desc=" videos", unit="vid", dynamic_ncols=True, colour="cyan"):
            src = os.path.join(sd, f); dst = os.path.join(td, f)
            try:
                cap = cv2.VideoCapture(src)
                if not cap.isOpened(): raise RuntimeError("cant open")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, sz)
                while True:
                    ret, fr = cap.read()
                    if not ret: break
                    out.write(cv2.resize(fr, sz, interpolation=cv2.INTER_AREA))
                cap.release(); out.release(); vok += 1
            except Exception as ex:
                print(" " + clr("video problem: " + f + " " + str(ex), Y)); vfail += 1
    elif vid_files:
        print(" " + clr("opencv missing, " + str(len(vid_files)) + " videos skipped", Y))
    print("\n " + clr("step 3/3 cleaning staging...", C))
    for f in os.listdir(sd):
        try: os.remove(os.path.join(sd, f))
        except: pass
    print(" " + clr("done. images " + str(iok) + " ok " + str(ifail) + " failed "
                      "videos " + str(vok) + " ok " + str(vfail) + " failed", G) + "\n")

def run_caption(cfg):
    sec("Auto Caption")
    from transformers import (BlipProcessor, BlipForConditionalGeneration,
                              AutoImageProcessor, SiglipForImageClassification)
    folder = cfg["data_dir"]
    bsz = int(cfg.get("caption_batch", 4))
    bid = cfg.get("blip_model", "Salesforce/blip-image-captioning-base")
    sid = cfg.get("style_model", "strangerguardhf/nsfw_image_detection")
    if not os.path.isdir(folder):
        print(" " + clr("folder not found: " + folder, R)); return
    old = list(Path(folder).glob("*.txt"))
    for f in old:
        try: f.unlink()
        except: pass
    if old: print(" " + clr("cleared " + str(len(old)) + " old captions", G))
    print(" " + clr("loading BLIP...", C))
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    bp = BlipProcessor.from_pretrained(bid)
    bm = BlipForConditionalGeneration.from_pretrained(bid, torch_dtype=dt).to(device)
    print(" " + clr("loading style classifier...", C))
    sp = AutoImageProcessor.from_pretrained(sid)
    sm = SiglipForImageClassification.from_pretrained(sid).to(device)
    bm.eval(); sm.eval()
    print(" " + clr("loaded on " + ("GPU" if device=="cuda" else "CPU"), G) + "\n")
    SL = {"0":"Anime","1":"Explicit","2":"Normal","3":"Explicit Photo","4":"Suggestive"}
    def get_styles(paths):
        inp = sp(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt").to(device)
        with torch.no_grad(): logits = sm(**inp).logits
        return [SL.get(str(i),"Unknown") for i in torch.softmax(logits,1).argmax(1).tolist()]
    def get_captions(paths):
        inp = bp(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt").to(device)
        with torch.no_grad(): out = bm.generate(**inp, max_new_tokens=120, min_length=20, do_sample=False)
        return [bp.decode(o, skip_special_tokens=True).strip() for o in out]
    media = sorted([p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".png",".jpg",".jpeg",".webp",".gif"}])
    if not media: print(" " + clr("no images found", Y)); return
    print(" " + clr("captioning " + str(len(media)) + " images, batch " + str(bsz) + "...", C) + "\n")
    ok = fail = 0; t0 = time.time()
    for i in tqdm(range(0, len(media), bsz), desc=" batches", unit="batch",
                  dynamic_ncols=True, colour="cyan"):
        batch = media[i:i+bsz]
        try: sl = get_styles(batch); cl_ = get_captions(batch)
        except Exception as ex:
            print(" " + clr("batch error: " + str(ex), Y))
            sl = ["Unknown"] * len(batch); cl_ = [""] * len(batch)
        for path, s, cap in zip(batch, sl, cl_):
            if cap:
                path.with_suffix(".txt").write_text("Style: " + s + "\nCaption: " + cap, encoding="utf-8")
                ok += 1
            else: fail += 1
    elapsed = time.time() - t0
    print("\n " + clr("done. " + str(ok) + " captioned " + str(fail) + " failed"
                        " avg " + str(round(elapsed/max(len(media),1),2)) + "s each", G) + "\n")

def pause(m=" press enter..."):
    input(m)

def show_tutorial():
    clear()
    print(clr("=" * 60, C))
    print(clr(" Tutorial | How to use this trainer", BOLD))
    print(clr("=" * 60, C) + "\n")
    sections = [
        ("QUICK START", [
            "1. Resize -> menu 2",
            "2. Caption -> menu 3",
            "3. Train -> menu 4 or 5",
            "4. Generate -> menu 7",
        ]),
        ("IMAGE EDITING (menu 9)", [
            "Face swap, face restore, img2img, filters, batch restore",
        ]),
    ]
    for title, lines in sections:
        print(" " + clr(title, Y))
        for l in lines: print(" " + clr(" " + l, DIM))
        print()
    print(" " + clr("=" * 58, DIM))
    pause("")

# All your picker and helper functions (added at the end)
def hf_search(query):
    try:
        import urllib.request, urllib.parse
        url = "https://huggingface.co/api/models?search=" + urllib.parse.quote(query)
        url += "&filter=diffusers&limit=10&sort=downloads"
        req = urllib.request.Request(url, headers={"User-Agent": "trainer/4.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        return [(m.get("id",""), m.get("downloads",0)) for m in data if m.get("id")]
    except: return []

HF_PRESETS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "dreamlike-art/dreamlike-photoreal-2.0",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "Lykon/dreamshaper-8",
]

def _pick_model_search():
    sec("Pick a Base Model")
    print(" " + clr("type a number, s to search HuggingFace, or u to paste a URL\n", DIM))
    for i, m in enumerate(HF_PRESETS, 1):
        prof = detect_from_id(m)
        badge = clr("[" + prof["label"].split("(")[0].strip() + "]", DIM)
        print(" " + clr(str(i), C) + " " + m + " " + badge)
    print()
    print(" " + clr("s", C) + " search HuggingFace by name")
    print(" " + clr("u", C) + " paste a URL or model ID")
    print()
    while True:
        ch = input(" " + clr("choice [1]: ", C)).strip() or "1"
        if ch == "u":
            raw = input(" " + clr("paste model ID or HF URL: ", C)).strip()
            if "huggingface.co/" in raw:
                raw = raw.rstrip("/").split("huggingface.co/")[-1].strip("/")
            if raw:
                prof = detect_from_id(raw)
                print(" " + clr("detected: " + prof["label"], G))
                return raw, prof
            print(" " + clr("nothing entered", R))
        elif ch == "s":
            q = input(" " + clr("search: ", C)).strip()
            if not q: continue
            print(" " + clr("searching...", C))
            results = hf_search(q)
            if not results:
                print(" " + clr("no results or no internet. try option u to paste an ID", Y)); continue
            print()
            for i, (mid, dl) in enumerate(results, 1):
                prof = detect_from_id(mid)
                badge = clr("[" + prof["label"].split("(")[0].strip() + "]", DIM)
                print(" " + clr(str(i), C) + " " + mid + " " + badge + " " + clr(str(dl) + " dl", DIM))
            print(" " + clr("0", C) + " search again")
            print()
            sc = input(" " + clr("pick one [1]: ", C)).strip() or "1"
            if sc == "0": continue
            if sc.isdigit() and 1 <= int(sc) <= len(results):
                mid = results[int(sc)-1][0]
                prof = detect_from_id(mid)
                print(" " + clr("selected: " + mid, G))
                print(" " + clr("detected: " + prof["label"], G))
                return mid, prof
            print(" " + clr("not valid", R))
        elif ch.isdigit() and 1 <= int(ch) <= len(HF_PRESETS):
            mid = HF_PRESETS[int(ch)-1]
            prof = detect_from_id(mid)
            print(" " + clr("detected: " + prof["label"], G))
            return mid, prof
        else:
            print(" " + clr("not valid", R))

def _pick_data_dir(last=""):
    sec("Training Data Folder")
    print(" " + clr("path to folder with your images (or videos)", DIM))
    if last: print(" " + clr("last: " + last, DIM) + "\n")
    while True:
        raw = input(" " + clr("path [" + (last or "required") + "]: ", C)).strip()
        if not raw and last: raw = last
        if not raw: print(" " + clr("required", R)); continue
        path = raw.strip('"').strip("'")
        if not os.path.isdir(path): print(" " + clr("folder not found", R)); continue
        ni = sum(len(glob.glob(os.path.join(path,"*."+e))) for e in IEXT)
        nv = sum(len(glob.glob(os.path.join(path,"*."+e))) for e in VEXT)
        print(" " + clr("found " + str(ni) + " images and " + str(nv) + " videos", G))
        return path

def _pick_out_dir(last="", label="Output Folder"):
    sec(label)
    raw = input(" " + clr("path [" + (last or "required") + "]: ", C)).strip()
    if not raw and last: return last
    path = raw.strip('"').strip("'")
    if path: os.makedirs(path, exist_ok=True); return path
    return last

def _pick_weights(ckd):
    sec("Starting Weights (optional, enter to skip)")
    files = [f for f in all_weights(ckd) if not re.search(r"step_\d+\.(pt|safetensors)$",f)]
    if not files: print(" " + clr("nothing found, starting fresh", DIM)); return None
    print(" " + clr("0", C) + " fresh start")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print(" " + clr(str(i), C) + " " + Path(f).name + " " + clr(str(round(mb))+"MB",DIM))
    print()
    while True:
        ch = input(" " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        print(" " + clr("not valid", R))

def _pick_lora(dirs):
    sec("LoRA File (optional, enter to skip)")
    files = []
    for d in list(dirs) + [os.getcwd()]:
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d,"*.safetensors")):
                if f not in files: files.append(f)
    if not files:
        raw = input(" " + clr("no files found paste path or enter to skip: ", C)).strip().strip('"').strip("'")
        return raw if raw and os.path.exists(raw) else None
    print(" " + clr("0", C) + " none")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print(" " + clr(str(i), C) + " " + Path(f).name + " " + clr(str(round(mb))+"MB",DIM))
    print(" " + clr("m", C) + " type a path")
    print()
    while True:
        ch = input(" " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch == "m":
            raw = input(" " + clr("path: ", C)).strip().strip('"').strip("'")
            if raw and os.path.exists(raw): return raw
            print(" " + clr("not found", R))
        elif ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        else: print(" " + clr("not valid", R))

def _pick_existing_lora(ckd):
    sec("Existing LoRA (optional)")
    files = all_weights(ckd)
    if not files: print(" " + clr("nothing found, starting fresh", DIM)); return None
    print(" " + clr("0", C) + " fresh start")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print(" " + clr(str(i), C) + " " + Path(f).name + " " + clr(str(round(mb))+"MB",DIM))
    print()
    while True:
        ch = input(" " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        print(" " + clr("not valid", R))

def _get_train_settings(ckd, prof=None):
    sec("Training Settings")
    p = vram_preset(prof)
    gb = gpu_gb()
    if gb > 0:
        print(" " + clr("GPU: " + str(round(gb,1)) + "GB -> " + p["label"], Y))
        if prof: print(" " + clr("model: " + prof["label"], C))
        print(" " + clr("press enter to accept each default\n", DIM))
    latest, step = find_latest(ckd)
    if latest: print(" " + clr("checkpoint at step " + str(step) + ", will auto-resume\n", G))
    res = _ask("resolution (512 / 768 / 1024)", p["res"], int)
    bs = _ask("batch size", p["bs"], int)
    ms = _ask("total steps", p["steps"], int)
    sv = _ask("save every N steps", p["save"], int)
    lr = _ask("learning rate", p["lr"], float)
    rank = _ask("LoRA rank", p["rank"], int)
    alp = _ask("LoRA alpha", p["alpha"], int)
    nw = _ask("dataloader workers", p["nw"], int)
    if res >= 1024 and bs > 1: print(" " + clr("1024px: batch set to 1",Y)); bs = 1
    return dict(resolution=res, batch_size=bs, max_steps=ms, save_every=sv,
                lr=lr, rank=rank, alpha=alp, num_workers=nw)

def _get_ft_settings(ckd, prof=None):
    sec("Fine Tune Settings")
    p = vram_preset(prof)
    pr = prof or _DEFAULT
    gb = gpu_gb()
    if gb > 0: print(" " + clr("GPU: " + str(round(gb,1)) + "GB -> " + p["label"], Y))
    if prof:
        print(" " + clr("model: " + prof["label"], C))
        print(" " + clr("note: " + prof["note"], Y))
        if prof["min_res"] > 512:
            print(" " + clr("minimum resolution for this model: " + str(prof["min_res"]) + "px", R))
    print(" " + clr("\npress enter to accept each default\n", DIM))
    res = _validated_ask("resolution", p["res"], int, pr, "res")
    bs = _ask("batch size", p["bs"], int)
    ga = _ask("gradient accumulation", p["ga"], int)
    ms = _ask("total steps", p["steps"], int)
    sv = _ask("save every N steps", p["save"], int)
    ulr = _ask("UNet learning rate", 3e-5, float)
    tlr = _ask("text encoder learning rate", 1e-5, float)
    ur = _validated_ask("UNet LoRA rank", p["ur"], int, pr, "rank")
    ua = _ask("UNet LoRA alpha", p["ua"], int)
    tr = _validated_ask("text encoder LoRA rank", p["tr"], int, pr, "te_rank")
    ta = _ask("text encoder LoRA alpha", p["ta"], int)
    drop = _ask("LoRA dropout", p["drop"], float)
    nw = _ask("dataloader workers", p["nw"], int)
    gc = _ask("gradient checkpointing y/n", "y" if p["gc"] else "n", str)
    if res >= 1024 and bs > 1: print(" " + clr("1024px: batch set to 1",Y)); bs = 1
    return dict(resolution=res, batch_size=bs, grad_accum=ga, max_steps=ms,
                save_every=sv, unet_lr=ulr, text_lr=tlr, unet_rank=ur,
                unet_alpha=ua, te_rank=tr, te_alpha=ta, dropout=drop,
                num_workers=nw, gradient_checkpointing=(str(gc).strip().lower()!="n"))

BOOSTS = {
    "1": ("resolution -> 768", dict(resolution=768, batch_size=2)),
    "2": ("resolution -> 1024", dict(resolution=1024, batch_size=1)),
    "3": ("rank -> 192", dict(rank=192, alpha=384)),
    "4": ("+8k steps", dict(max_steps=24000)),
    "5": ("fast mode (rank 64 bs 8)", dict(rank=64, alpha=128, batch_size=8)),
    "6": ("full quality pack", dict(resolution=768, rank=192, alpha=384, max_steps=20000, batch_size=2)),
}

def _apply_boosts(cfg):
    sec("Quality Boosts (standard train only)")
    print(" " + clr("note: these are for standard train. fine tune uses model-native settings.\n", DIM))
    for k,(nm,_) in BOOSTS.items(): print(" " + clr(k,C) + " " + nm)
    print(" " + clr("0",C) + " keep as is")
    print()
    ch = input(" " + clr("pick (or combine like 1,3): ", C)).strip()
    if not ch or ch == "0": return cfg
    for part in re.split(r"[,\s]+", ch):
        if part in BOOSTS:
            cfg.update(BOOSTS[part][1])
            print(" " + clr("applied: " + BOOSTS[part][0], G))
    if cfg.get("resolution",0) >= 1024: cfg["batch_size"] = 1
    return cfg

def find_latest(folder):
    hits = []
    for pat in ["step_*.pt","step_*.safetensors"]:
        for f in glob.glob(os.path.join(folder, pat)):
            m = re.search(r"step_(\d+)\.(pt|safetensors)$", os.path.basename(f))
            if m: hits.append((f, int(m.group(1))))
    if not hits: return None, 0
    return sorted(hits, key=lambda x: x[1])[-1]

def all_weights(folder):
    out = []
    for pat in ["*.pt","*.safetensors"]:
        out += glob.glob(os.path.join(folder, pat))
    return sorted(out)

def load_into(model, path):
    ext = Path(path).suffix.lower()
    state = load_file(path) if ext == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    res = model.load_state_dict(state, strict=False)
    n = len(state) - len(res.missing_keys)
    print(" " + clr("loaded " + str(n) + "/" + str(len(state)) + " tensors from " + Path(path).name, G))

def save_st(unet, te, folder, step):
    os.makedirs(folder, exist_ok=True)
    st = {k:v for k,v in unet.state_dict().items() if "lora_" in k}
    st.update({k:v for k,v in te.state_dict().items() if "lora_" in k})
    save_file(st, os.path.join(folder, "step_" + str(step) + ".safetensors"))
    print(" " + clr("checkpoint saved at step " + str(step), Y))

def save_pt(unet, folder, step):
    os.makedirs(folder, exist_ok=True)
    st = unet.state_dict()
    torch.save(st, os.path.join(folder, "step_" + str(step) + ".pt"))
    lo = {k:v for k,v in st.items() if "lora_" in k}
    if lo: save_file(lo, os.path.join(folder, "step_" + str(step) + ".safetensors"))
    print(" " + clr("checkpoint saved at step " + str(step), Y))

def _validated_ask(label, default, cast, prof, field):
    while True:
        raw = input(" " + clr(label, C) + " [" + clr(str(default), W) + "]: ").strip()
        if not raw: return default
        try: val = cast(raw)
        except: print(" " + clr("not a valid number", R)); continue
        if field == "res" and prof:
            mn = prof.get("min_res", 512)
            if val < mn:
                print(" " + clr("WARNING: this model needs at least " + str(mn) + "px", R))
                print(" " + clr(" going below " + str(mn) + "px will break generation", Y))
                c = input(" " + clr(" use minimum " + str(mn) + " instead? y/n [y]: ", C)).strip().lower()
                if c != "n":
                    print(" " + clr(" capped to " + str(mn), G))
                    return mn
                print(" " + clr(" keeping " + str(val) + "px (expect broken outputs)", Y))
                return val
        if field == "rank" and prof:
            mn = prof.get("min_rank", 4)
            if val < mn:
                print(" " + clr("WARNING: rank below " + str(mn) + " will likely fail", R))
                c = input(" " + clr(" use minimum " + str(mn) + " instead? y/n [y]: ", C)).strip().lower()
                if c != "n": return mn
                return val
        if field == "te_rank" and prof and prof.get("flux"):
            if val < 16:
                print(" " + clr("WARNING: Flux text encoder rank below 16 breaks generation", R))
                print(" " + clr(" the encoder is tightly coupled with the diffusion model here", Y))
                c = input(" " + clr(" use 16 instead? y/n [y]: ", C)).strip().lower()
                if c != "n": return 16
                return val
        return val

def _ask(label, default, cast=str):
    raw = input(" " + clr(label, C) + " [" + clr(str(default), W) + "]: ").strip()
    if not raw: return default
    try: return cast(raw)
    except: return default
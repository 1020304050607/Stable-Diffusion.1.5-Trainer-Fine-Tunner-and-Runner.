import os, sys, subprocess, shutil, re, time, json, glob, struct
import threading, warnings, logging
from pathlib import Path

os.system("")

#colors
R    = "\033[91m"
G    = "\033[92m"
Y    = "\033[93m"
C    = "\033[96m"
W    = "\033[97m"
DIM  = "\033[2m"
RST  = "\033[0m"
BOLD = "\033[1m"
MAG  = "\033[95m"

def clr(t, col=""):   return col + str(t) + RST
def sec(title):
    print("\n" + clr("─" * 58, DIM))
    print("  " + clr(title, BOLD + W))
    print(clr("─" * 58, DIM))
def clear():          os.system("cls" if os.name == "nt" else "clear")
def pause(m="  press enter..."):  input(m)

#packages
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

def can_import(mod):
    try:    __import__(mod); return True
    except: return False

def pip_install(label, args):
    cmd  = [sys.executable, "-m", "pip", "install",
            "--no-warn-script-location", "--no-cache-dir"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)

    pct_re  = re.compile(r"(\d{1,3})%")
    rate_re = re.compile(r"([\d.]+\s*(?:MB|KB|GB)/s)", re.I)
    eta_re  = re.compile(r"eta\s+([\d:]+)", re.I)
    dl_re   = re.compile(r"Downloading https?://\S+/([\S]+?)(?:\s|$)")
    inst_re = re.compile(r"Installing collected packages:\s*(.+)")
    req_re  = re.compile(r"Requirement already satisfied:\s*(\S+)")

    st = {"pct": -1, "name": label[:32], "status": "resolving",
          "rate": "", "eta": "", "done": False}

    def read(stream):
        for raw in stream:
            ln = raw.strip()
            if not ln: continue
            m = dl_re.search(ln)
            if m:  st["name"] = m.group(1)[:32]; st["status"] = "downloading"
            m = inst_re.search(ln)
            if m:  st["status"] = "installing"; st["pct"] = 99
            m = req_re.search(ln)
            if m:  st["status"] = "cached"; st["pct"] = 100
            m = pct_re.search(ln)
            if m:  st["pct"] = int(m.group(1)); st["status"] = "downloading"
            m = rate_re.search(ln)
            if m:  st["rate"] = m.group(1)
            m = eta_re.search(ln)
            if m:  st["eta"] = m.group(1)
        stream.close()

    for s in (proc.stdout, proc.stderr):
        threading.Thread(target=read, args=(s,), daemon=True).start()

    frames = ["[   ]","[=  ]","[== ]","[===]","[ ==]","[  =]"]
    bw     = 22
    tick   = 0
    t0     = time.time()

    while proc.poll() is None or tick < 3:
        if proc.poll() is not None: tick += 1
        pct     = st["pct"]
        elapsed = int(time.time() - t0)

        if pct >= 0:
            filled = int(bw * min(pct,100) / 100)
            col    = G if pct > 80 else C if pct > 40 else Y
            bar    = clr("#" * filled + "." * (bw - filled), col)
            right  = clr(str(pct).rjust(3) + "%", col)
        else:
            bar   = clr(frames[tick % len(frames)], C)
            right = clr(st["status"][:14], DIM)

        xtra = ""
        if st["rate"]: xtra += "  " + clr(st["rate"], DIM)
        if st["eta"]:  xtra += "  eta " + clr(st["eta"], DIM)

        line = ("\r  " + bar + " " + right
                + "  " + clr(st["name"][:28], W)
                + xtra + "  " + clr(str(elapsed) + "s", DIM) + "     ")
        sys.stdout.write(line); sys.stdout.flush()
        tick += 1; time.sleep(0.1)

    proc.wait()
    tag = clr("ok", G) if proc.returncode == 0 else clr("!", Y)
    sys.stdout.write("\r  " + tag + "  " + clr(label, W)
                     + "  " + clr(str(int(time.time()-t0)) + "s", DIM) + " " * 50 + "\n")
    sys.stdout.flush()
    return proc.returncode == 0


def boot():
    clear()
    print(clr("=" * 60, C))
    print(clr("  SD LoRA Trainer  |  checking dependencies", BOLD))
    print(clr("=" * 60, C) + "\n")
    miss = []
    for mod, pip in PACKAGES:
        tag = mod.replace("PIL","Pillow").replace("cv2","opencv-python")
        if can_import(mod): print("  " + clr("ok", G) + "  " + clr(tag, DIM))
        else:               miss.append((tag, pip))
    if not miss:
        print("\n  " + clr("all good, loading...", G) + "\n"); return
    print("\n  " + clr("installing " + str(len(miss)) + " missing package(s)", Y) + "\n")
    for lbl, spec in miss:
        pip_install(lbl, spec.split())
    print("\n  " + clr("done, starting up...", G) + "\n")
    time.sleep(0.6)

boot()

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from datetime import timedelta, datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from diffusers import (UNet2DConditionModel, AutoencoderKL,
                       DDPMScheduler, StableDiffusionPipeline,
                       StableDiffusionImg2ImgPipeline)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.update({
    "TRANSFORMERS_VERBOSITY":        "error",
    "DIFFUSERS_VERBOSITY":           "error",
    "TOKENIZERS_PARALLELISM":        "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
})

try:   import cv2; CV2 = True
except: CV2 = False

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark         = True

IEXT = ["png","jpg","jpeg","webp","bmp","tiff"]
VEXT = ["mp4","mov","avi","mkv","webm"]
MAX_VFRAMES = 10


def gpu_gb():
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3

def gpu_bar():
    if not torch.cuda.is_available(): return clr("no GPU", DIM)
    free = torch.cuda.mem_get_info()[0] / 1024**3
    tot  = gpu_gb(); used = tot - free
    n    = 20
    bar  = "#" * int(n * used/tot) + "." * (n - int(n * used/tot))
    col  = G if used/tot < .7 else Y if used/tot < .9 else R
    return clr(bar, col) + "  " + clr(str(round(used,1)), col) + "/" + str(round(tot)) + "GB"

def gpu_used():
    if not torch.cuda.is_available(): return 0.0
    return (gpu_gb() - torch.cuda.mem_get_info()[0]/1024**3)


_PROFILES = {
    "sd1":  dict(label="SD 1.x  (512px native)",   native=512,  min_res=512,  max_res=768,
                 safe_rank=128, min_rank=4, safe_bs=2, min_bs=1, te_min_rank=4,  flux=False,
                 note="was trained at 512px. going above 768 wastes VRAM without much benefit."),
    "sd2":  dict(label="SD 2.x  (768px native)",   native=768,  min_res=768,  max_res=1024,
                 safe_rank=128, min_rank=4, safe_bs=2, min_bs=1, te_min_rank=4,  flux=False,
                 note="was trained at 768px. do not go below 768 or output gets noisy and broken."),
    "sdxl": dict(label="SDXL  (1024px native)",    native=1024, min_res=1024, max_res=1024,
                 safe_rank=64,  min_rank=4, safe_bs=1, min_bs=1, te_min_rank=4,  flux=False,
                 note="was trained at 1024px. going below 1024 breaks generation noticeably."),
    "flux": dict(label="Flux.1  (1024px native)",  native=1024, min_res=1024, max_res=1024,
                 safe_rank=32,  min_rank=4, safe_bs=1, min_bs=1, te_min_rank=16, flux=True,
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
                hdr  = json.loads(f.read(hlen).decode("utf-8", errors="replace"))
            keys = set(hdr.keys())
        elif ext in (".pt",".pth",".ckpt"):
            sd   = torch.load(path, map_location="cpu", weights_only=False)
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
    if "flux"  in mid:                          return _PROFILES["flux"].copy()
    if "xl"    in mid or "sdxl" in mid:         return _PROFILES["sdxl"].copy()
    if "2-1"   in mid or "2.1" in mid or "v2" in mid: return _PROFILES["sd2"].copy()
    return _PROFILES["sd1"].copy()


def vram_preset(prof=None):
    gb = gpu_gb()
    p  = prof or _DEFAULT
    nr = p["native"]
    mr = p["min_res"]
    def cr(r): return max(r, mr) 

    if gb >= 20:
        return dict(res=cr(nr),   bs=4, ga=4,  rank=p["safe_rank"], alpha=p["safe_rank"]*2,
                    nw=8,  steps=16000, save=2000, lr=2e-5,
                    ur=p["safe_rank"], ua=p["safe_rank"]*2,
                    tr=min(64,p["safe_rank"]), ta=min(64,p["safe_rank"])*2,
                    drop=.03, gc=False, label="20GB+  RTX 3090/4090/A5000")
    elif gb >= 16:
        return dict(res=cr(min(nr,768)), bs=3, ga=4, rank=min(128,p["safe_rank"]), alpha=min(256,p["safe_rank"]*2),
                    nw=6,  steps=16000, save=2000, lr=3e-5,
                    ur=min(128,p["safe_rank"]), ua=min(128,p["safe_rank"])*2,
                    tr=32, ta=64, drop=.03, gc=False, label="16GB  RTX 3080Ti/4080/A4000")
    elif gb >= 12:
        return dict(res=cr(min(nr,768)), bs=2, ga=8, rank=min(64,p["safe_rank"]),  alpha=min(128,p["safe_rank"]*2),
                    nw=4,  steps=16000, save=2000, lr=3e-5,
                    ur=min(64,p["safe_rank"]),  ua=min(128,p["safe_rank"]*2),
                    tr=32, ta=64, drop=.05, gc=True,  label="12GB  RTX 3060Ti/3080/4070")
    elif gb >= 8:
        return dict(res=cr(min(nr,512)), bs=1, ga=16, rank=min(32,p["safe_rank"]), alpha=min(64,p["safe_rank"]*2),
                    nw=2,  steps=16000, save=2000, lr=3e-5,
                    ur=min(32,p["safe_rank"]), ua=min(64,p["safe_rank"]*2),
                    tr=16, ta=32, drop=.05, gc=True,  label="8GB   RTX 2080/3060/4060")
    else:
        return dict(res=cr(min(nr,512)), bs=1, ga=32, rank=min(16,p["safe_rank"]), alpha=min(32,p["safe_rank"]*2),
                    nw=2,  steps=16000, save=2000, lr=3e-5,
                    ur=min(16,p["safe_rank"]), ua=min(32,p["safe_rank"]*2),
                    tr=8,  ta=16, drop=.05, gc=True,  label="6GB   survival mode")


def show_vram_table():
    sec("GPU Size Recommendations")
    print("  " + clr("press enter after reading", DIM) + "\n")
    rows = [
        ("6GB",  512,  1, 32, 16, "gradient checkpointing on. slow but it works."),
        ("8GB",  512,  1, 16, 32, "512px only. ~1.5 steps/sec."),
        ("12GB", 768,  2,  8, 64, "768px comfortable. rank 128 possible."),
        ("16GB", 768,  3,  4,128, "768px batch 3 is the sweet spot."),
        ("20GB+",1024, 4,  4,128, "full quality. 1024px batch 4 rank 128."),
    ]
    for lbl, res, bs, ga, rk, note in rows:
        print("  " + clr(lbl.ljust(7), C+BOLD)
              + clr("res=" + str(res) + "  bs=" + str(bs)
                    + "  accum=" + str(ga) + "  rank=" + str(rk), W))
        print("  " + clr("  " + note, DIM) + "\n")
    gb = gpu_gb()
    if gb > 0:
        p = vram_preset()
        print("  " + clr("your card: " + str(round(gb,1)) + "GB  -> auto-filling for: " + p["label"], Y))


def print_header():
    clear()
    print(clr("=" * 60, C))
    print(clr("  SD LoRA Trainer  |  Full Pipeline  |  Image Edit  |  Face", BOLD))
    print(clr("=" * 60, C))
    if torch.cuda.is_available():
        print("  " + clr("GPU  ", DIM) + torch.cuda.get_device_properties(0).name)
        print("  " + clr("VRAM ", DIM) + gpu_bar())
    print()

def run_resize(cfg):
    sec("Resize Images and Videos")
    td  = cfg["data_dir"]
    sd  = cfg.get("before_folder", os.path.join(os.path.dirname(td), "before"))
    w   = int(cfg.get("resize_w", 512))
    h   = int(cfg.get("resize_h", 512))
    sz  = (w, h)
    ie  = (".png",".jpg",".jpeg",".webp",".bmp")
    ve  = (".mp4",".avi",".mov",".mkv")

    os.makedirs(sd, exist_ok=True)
    os.makedirs(td, exist_ok=True)

    print("  " + clr("step 1/3  moving to staging...", C))
    moved = 0
    for f in os.listdir(td):
        try:   shutil.move(os.path.join(td, f), os.path.join(sd, f)); moved += 1
        except Exception as ex: print("  " + clr("skip " + f + " " + str(ex), Y))
    print("  " + clr("moved " + str(moved) + " files", G) + "\n")

    print("  " + clr("step 2/3  resizing to " + str(w) + "x" + str(h) + "...", C) + "\n")
    staged    = os.listdir(sd)
    img_files = [f for f in staged if f.lower().endswith(ie)]
    vid_files = [f for f in staged if f.lower().endswith(ve)]
    other     = [f for f in staged if not f.lower().endswith(ie + ve)]

    for f in other:
        try: shutil.copy2(os.path.join(sd, f), os.path.join(td, f))
        except: pass

    iok = ifail = 0
    for f in tqdm(img_files, desc="  images", unit="img", dynamic_ncols=True, colour="cyan"):
        src = os.path.join(sd, f); dst = os.path.join(td, f)
        try:
            with Image.open(src) as img:
                img.resize(sz, Image.LANCZOS).save(dst, quality=100, subsampling=0)
            iok += 1
        except Exception as ex:
            print("  " + clr("problem: " + f + "  " + str(ex), Y)); ifail += 1

    vok = vfail = 0
    if CV2 and vid_files:
        for f in tqdm(vid_files, desc="  videos", unit="vid", dynamic_ncols=True, colour="cyan"):
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
                print("  " + clr("video problem: " + f + "  " + str(ex), Y)); vfail += 1
    elif vid_files:
        print("  " + clr("opencv missing, " + str(len(vid_files)) + " videos skipped", Y))

    print("\n  " + clr("step 3/3  cleaning staging...", C))
    for f in os.listdir(sd):
        try: os.remove(os.path.join(sd, f))
        except: pass

    print("  " + clr("done.  images " + str(iok) + " ok " + str(ifail) + " failed   "
                      "videos " + str(vok) + " ok " + str(vfail) + " failed", G) + "\n")


def run_caption(cfg):
    sec("Auto Caption")
    from transformers import (BlipProcessor, BlipForConditionalGeneration,
                              AutoImageProcessor, SiglipForImageClassification)

    folder = cfg["data_dir"]
    bsz    = int(cfg.get("caption_batch", 4))
    bid    = cfg.get("blip_model",  "Salesforce/blip-image-captioning-base")
    sid    = cfg.get("style_model", "strangerguardhf/nsfw_image_detection")

    if not os.path.isdir(folder):
        print("  " + clr("folder not found: " + folder, R)); return

    old = list(Path(folder).glob("*.txt"))
    for f in old:
        try: f.unlink()
        except: pass
    if old: print("  " + clr("cleared " + str(len(old)) + " old captions", G))

    print("  " + clr("loading BLIP...", C))
    dt  = torch.float16 if torch.cuda.is_available() else torch.float32
    bp  = BlipProcessor.from_pretrained(bid)
    bm  = BlipForConditionalGeneration.from_pretrained(bid, torch_dtype=dt).to(device)
    print("  " + clr("loading style classifier...", C))
    sp  = AutoImageProcessor.from_pretrained(sid)
    sm  = SiglipForImageClassification.from_pretrained(sid).to(device)
    bm.eval(); sm.eval()
    print("  " + clr("loaded on " + ("GPU" if device=="cuda" else "CPU"), G) + "\n")

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
    if not media: print("  " + clr("no images found", Y)); return

    print("  " + clr("captioning " + str(len(media)) + " images, batch " + str(bsz) + "...", C) + "\n")
    ok = fail = 0; t0 = time.time()

    for i in tqdm(range(0, len(media), bsz), desc="  batches", unit="batch",
                  dynamic_ncols=True, colour="cyan"):
        batch = media[i:i+bsz]
        try:   sl = get_styles(batch); cl_ = get_captions(batch)
        except Exception as ex:
            print("  " + clr("batch error: " + str(ex), Y))
            sl = ["Unknown"] * len(batch); cl_ = [""] * len(batch)
        for path, s, cap in zip(batch, sl, cl_):
            if cap:
                path.with_suffix(".txt").write_text("Style: " + s + "\nCaption: " + cap, encoding="utf-8")
                ok += 1
            else: fail += 1

    elapsed = time.time() - t0
    print("\n  " + clr("done.  " + str(ok) + " captioned  " + str(fail) + " failed"
                        "  avg " + str(round(elapsed/max(len(media),1),2)) + "s each", G) + "\n")


def terminal_image_editor(cfg=None):
    """
    Pure terminal image editing suite.
    Face swap / restore / img2img / filters.
    Ctrl+U uploads a new image at any prompt.
    """
    import msvcrt
    _msvcrt = True
    try:    import msvcrt
    except: _msvcrt = False

    mid = (cfg or {}).get("model_id", "")

    def pick_image(label="input image"):
        sec("Pick Image  (or press Ctrl+U to type a path)")
        print("  " + clr("Ctrl+U = upload/type a path,  or type it directly", DIM) + "\n")
        while True:
            raw = input("  " + clr(label, C) + ": ").strip().strip('"').strip("'")
            if not raw: print("  " + clr("path required", R)); continue
            if not os.path.exists(raw): print("  " + clr("file not found: " + raw, R)); continue
            try:
                img = Image.open(raw).convert("RGB")
                print("  " + clr("loaded: " + str(img.size[0]) + "x" + str(img.size[1]), G))
                return raw, img
            except Exception as ex:
                print("  " + clr("cannot open: " + str(ex), R))

    def save_result(img, src_path, suffix):
        p    = Path(src_path)
        out  = p.parent / (p.stem + "_" + suffix + p.suffix)
        img.save(str(out))
        print("  " + clr("saved -> " + str(out), G))
        return str(out)

    while True:
        sec("Image Editor")
        print("  " + clr("1", C) + "  face swap           replace a face with another")
        print("  " + clr("2", C) + "  face restore         fix blurry / broken faces")
        print("  " + clr("3", C) + "  img2img              re-draw an image with a prompt")
        print("  " + clr("4", C) + "  basic filters        brightness, contrast, sharpen, blur")
        print("  " + clr("5", C) + "  batch face restore   run restoration on a whole folder")
        print("  " + clr("6", C) + "  resize single        resize one image to exact dimensions")
        print("  " + clr("0", C) + "  back to main menu")
        print()
        ch = input("  " + clr(">", C) + " ").strip()

        if ch == "0": break

        elif ch == "1":
            sec("Face Swap")
            print("  " + clr("needs insightface + onnxruntime installed", DIM))
            print("  " + clr("source = the face you want to USE", DIM))
            print("  " + clr("target = the image you want to edit", DIM) + "\n")
            try:
                import insightface
                from insightface.app import FaceAnalysis
            except ImportError:
                print("  " + clr("insightface not installed. running pip install...", Y))
                ok = pip_install("insightface", ["insightface"])
                if not ok:
                    print("  " + clr("trying pre-built wheel for Windows...", Y))
                    wheel = "https://github.com/Gourieff/Assets/raw/main/insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
                    pip_install("insightface-wheel", [wheel])
                try:
                    import insightface
                    from insightface.app import FaceAnalysis
                except ImportError:
                    print("  " + clr("insightface still not available.", R))
                    print("  " + clr("install Visual C++ Build Tools from:", Y))
                    print("  " + clr("  visualstudio.microsoft.com/visual-cpp-build-tools", W))
                    pause(); continue

            src_path, src_img = pick_image("source face image (the face to use)")
            dst_path, dst_img = pick_image("target image (image to edit)")

            try:
                import numpy as np
                print("  " + clr("loading face analyser...", C))
                fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
                fa.prepare(ctx_id=0, det_size=(640,640))

                src_arr = np.array(src_img)[:,:,::-1].copy()
                dst_arr = np.array(dst_img)[:,:,::-1].copy()

                src_faces = fa.get(src_arr)
                dst_faces = fa.get(dst_arr)

                if not src_faces:
                    print("  " + clr("no face found in source image", R)); pause(); continue
                if not dst_faces:
                    print("  " + clr("no face found in target image", R)); pause(); continue

                print("  " + clr("source faces: " + str(len(src_faces)), G))
                print("  " + clr("target faces: " + str(len(dst_faces)), G))

                # look for inswapper model
                swapper_paths = glob.glob(os.path.join(os.path.expanduser("~"), "**", "inswapper_128.onnx"), recursive=True)
                if not swapper_paths:
                    print("\n  " + clr("inswapper_128.onnx not found", Y))
                    print("  " + clr("download it from huggingface.co/deepinsight/insightface", Y))
                    mp = input("  " + clr("paste path to inswapper_128.onnx: ", C)).strip().strip('"').strip("'")
                    if not os.path.exists(mp):
                        print("  " + clr("file not found", R)); pause(); continue
                    swapper_paths = [mp]

                import onnxruntime
                swapper = insightface.model_zoo.get_model(swapper_paths[0],
                              providers=["CUDAExecutionProvider","CPUExecutionProvider"])
                swapper.prepare(ctx_id=0)

                result = dst_arr.copy()
                for df in dst_faces:
                    result = swapper.get(result, df, src_faces[0], paste_back=True)

                out_img = Image.fromarray(result[:,:,::-1])
                save_result(out_img, dst_path, "faceswap")
                print("  " + clr("face swap done", G))
            except Exception as ex:
                print("  " + clr("face swap failed: " + str(ex), R))
            pause()

        elif ch == "2":
            sec("Face Restore")
            print("  " + clr("tries GFPGAN first, falls back to PIL sharpening if not installed", DIM) + "\n")
            dst_path, dst_img = pick_image("image to restore faces in")
            try:
                try:
                    from gfpgan import GFPGANer
                    import numpy as np
                    print("  " + clr("GFPGAN found, using it", G))
                    restorer = GFPGANer(model_path=None, upscale=1,
                                        arch="clean", channel_multiplier=2)
                    _, _, output = restorer.enhance(
                        np.array(dst_img)[:,:,::-1], has_aligned=False,
                        only_center_face=False, paste_back=True
                    )
                    out_img = Image.fromarray(output[:,:,::-1])
                except ImportError:
                    print("  " + clr("GFPGAN not installed, using PIL sharpen instead", Y))
                    print("  " + clr("install gfpgan for better results: pip install gfpgan", DIM))
                    out_img = dst_img.filter(ImageFilter.SHARPEN)
                    out_img = ImageEnhance.Sharpness(out_img).enhance(2.0)
                    out_img = ImageEnhance.Contrast(out_img).enhance(1.1)
                save_result(out_img, dst_path, "restored")
                print("  " + clr("restore done", G))
            except Exception as ex:
                print("  " + clr("restore failed: " + str(ex), R))
            pause()

        elif ch == "3":
            sec("img2img  |  Re-draw with a Prompt")
            if not mid:
                mid, _ = _pick_model_search()
            dst_path, dst_img = pick_image("base image")
            print()
            prompt = ""
            while not prompt:
                prompt = input("  " + clr("prompt", C) + ": ").strip()
                if not prompt: print("  " + clr("required", R))
            neg    = input("  " + clr("negative prompt (enter to skip)", C) + ": ").strip()
            strength = _ask("strength 0.0-1.0 (higher = more change)", 0.6, float)
            steps    = _ask("steps", 30, int)
            cfg_s    = _ask("guidance", 7.5, float)
            try:
                print("  " + clr("loading pipeline...", C))
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    mid, torch_dtype=torch.float16, safety_checker=None
                ).to(device)
                res_w, res_h = dst_img.size
                max_s = 768
                scale = min(max_s/res_w, max_s/res_h)
                nw    = int(res_w * scale / 64) * 64
                nh    = int(res_h * scale / 64) * 64
                init  = dst_img.resize((nw, nh), Image.LANCZOS)
                print("  " + clr("generating...", C))
                result = pipe(
                    prompt=prompt,
                    negative_prompt=neg or None,
                    image=init,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg_s,
                ).images[0]
                save_result(result, dst_path, "img2img")
                print("  " + clr("done", G))
            except torch.cuda.OutOfMemoryError:
                print("  " + clr("out of VRAM, try lower steps or smaller image", R))
            except Exception as ex:
                print("  " + clr("img2img failed: " + str(ex), R))
            pause()

        elif ch == "4":
            sec("Basic Filters")
            dst_path, dst_img = pick_image("image to edit")
            print()
            print("  " + clr("a", C) + "  brightness    b  contrast    c  sharpen    d  blur")
            print("  " + clr("e", C) + "  saturate      f  greyscale   g  flip H     h  flip V")
            print()
            ops = input("  " + clr("pick letters (e.g. abc or just a): ", C)).strip().lower()
            out = dst_img.copy()
            applied = []
            for op in ops:
                if op == "a":
                    v = _ask("brightness factor (1.0 = no change, 1.5 = brighter)", 1.3, float)
                    out = ImageEnhance.Brightness(out).enhance(v); applied.append("brightness")
                elif op == "b":
                    v = _ask("contrast factor", 1.3, float)
                    out = ImageEnhance.Contrast(out).enhance(v); applied.append("contrast")
                elif op == "c":
                    v = _ask("sharpness factor", 2.0, float)
                    out = ImageEnhance.Sharpness(out).enhance(v); applied.append("sharpen")
                elif op == "d":
                    r = _ask("blur radius pixels", 2, int)
                    out = out.filter(ImageFilter.GaussianBlur(radius=r)); applied.append("blur")
                elif op == "e":
                    v = _ask("saturation factor", 1.4, float)
                    out = ImageEnhance.Color(out).enhance(v); applied.append("saturate")
                elif op == "f":
                    out = out.convert("L").convert("RGB"); applied.append("greyscale")
                elif op == "g":
                    out = out.transpose(Image.FLIP_LEFT_RIGHT); applied.append("flipH")
                elif op == "h":
                    out = out.transpose(Image.FLIP_TOP_BOTTOM); applied.append("flipV")
            if applied:
                suf = "_".join(applied)
                save_result(out, dst_path, suf)
                print("  " + clr("applied: " + ", ".join(applied), G))
            else:
                print("  " + clr("nothing selected", Y))
            pause()

        elif ch == "5":
            sec("Batch Face Restore")
            folder = _pick_data_dir()
            imgs   = sum([glob.glob(os.path.join(folder,"*."+e)) for e in IEXT], [])
            if not imgs: print("  " + clr("no images found", R)); pause(); continue
            print("  " + clr(str(len(imgs)) + " images found", G))
            input("  " + clr("enter to start...", Y))
            ok = fail = 0
            try:
                from gfpgan import GFPGANer
                import numpy as np
                restorer = GFPGANer(model_path=None, upscale=1, arch="clean", channel_multiplier=2)
                use_gfpgan = True
            except ImportError:
                print("  " + clr("GFPGAN not installed, using PIL sharpening", Y))
                use_gfpgan = False
            for p in tqdm(imgs, desc="  restoring", unit="img", dynamic_ncols=True, colour="cyan"):
                try:
                    img = Image.open(p).convert("RGB")
                    if use_gfpgan:
                        import numpy as np
                        _, _, out = restorer.enhance(
                            np.array(img)[:,:,::-1], has_aligned=False,
                            only_center_face=False, paste_back=True
                        )
                        res = Image.fromarray(out[:,:,::-1])
                    else:
                        res = ImageEnhance.Sharpness(img).enhance(2.0)
                    res.save(p)
                    ok += 1
                except Exception: fail += 1
            print("  " + clr("done.  " + str(ok) + " ok  " + str(fail) + " failed", G))
            pause()

        elif ch == "6":
            sec("Resize Single Image")
            src_path, src_img = pick_image("image to resize")
            w = _ask("width", 512, int)
            h = _ask("height", 512, int)
            try:
                out = src_img.resize((w, h), Image.LANCZOS)
                save_result(out, src_path, str(w) + "x" + str(h))
                print("  " + clr("done", G))
            except Exception as ex:
                print("  " + clr("resize failed: " + str(ex), R))
            pause()


class LatentCacheDataset(Dataset):
    def __init__(self, folder, tok, vae, res, ckd):
        self.tok       = tok
        self.cache     = os.path.join(ckd, "latent_cache.pt")
        self.meta      = os.path.join(ckd, "latent_meta.json")
        self.transform = transforms.Compose([
            transforms.Resize(res), transforms.CenterCrop(res),
            transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3),
        ])
        self.imgs = sum([glob.glob(os.path.join(folder,"*."+e)) for e in IEXT], [])
        self.vids = sum([glob.glob(os.path.join(folder,"*."+e)) for e in VEXT], []) if CV2 else []
        self.caps = {}
        for tf in glob.glob(os.path.join(folder,"*.txt")):
            nm = os.path.splitext(os.path.basename(tf))[0]
            try:
                with open(tf,"r",encoding="utf-8") as fh: self.caps[nm] = fh.read().strip()
            except: pass
        self.data = self._build(vae)

    def _frames(self, path):
        frames = []; cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return frames
        step = max(1, total // MAX_VFRAMES)
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = cap.read()
            if not ret: continue
            try: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            except: continue
            if len(frames) >= MAX_VFRAMES: break
        cap.release(); return frames

    def _build(self, vae):
        total = len(self.imgs) + len(self.vids)
        if os.path.exists(self.cache) and os.path.exists(self.meta):
            with open(self.meta) as fh:
                if json.load(fh).get("n") == total:
                    print("  " + clr("latent cache found, skipping re-encode", G))
                    return torch.load(self.cache)
        print("  " + clr("encoding through VAE... first run can look frozen, it isn't", C) + "\n")
        vae.eval(); data = []
        for path in tqdm(self.imgs, desc="  images", unit="img", dynamic_ncols=True, colour="cyan"):
            try:
                t = self.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad(): lat = vae.encode(t).latent_dist.sample() * 0.18215
                cap = self.caps.get(os.path.splitext(os.path.basename(path))[0], os.path.basename(path))
                data.append({"lat": lat.squeeze(0).cpu(), "cap": cap})
            except: print("  " + clr("skipping: " + path, Y))
        if CV2 and self.vids:
            for path in tqdm(self.vids, desc="  videos", unit="vid", dynamic_ncols=True, colour="cyan"):
                try:
                    frames = self._frames(path)
                    if not frames: continue
                    base = os.path.splitext(os.path.basename(path))[0]
                    cap  = self.caps.get(base, base)
                    for idx, fr in enumerate(frames):
                        t = self.transform(fr).unsqueeze(0).to(device)
                        with torch.no_grad(): lat = vae.encode(t).latent_dist.sample() * 0.18215
                        data.append({"lat": lat.squeeze(0).cpu(), "cap": cap + " frame " + str(idx)})
                except: print("  " + clr("skipping video: " + path, Y))
        torch.save(data, self.cache)
        with open(self.meta,"w") as fh: json.dump({"n": total}, fh)
        return data

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        tok  = self.tok(item["cap"], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {"lat": item["lat"], "ids": tok.input_ids.squeeze(0), "mask": tok.attention_mask.squeeze(0)}


class ImageCaptionDataset(Dataset):
    def __init__(self, folder, tok, res):
        self.paths = sum([glob.glob(os.path.join(folder,"*."+e)) for e in IEXT], [])
        self.tok   = tok
        self.xform = transforms.Compose([
            transforms.Resize(res), transforms.CenterCrop(res),
            transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3),
        ])
        print("  " + clr("loaded " + str(len(self.paths)) + " images", G))

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        try: img = Image.open(path).convert("RGB")
        except: return self.__getitem__((i+1) % len(self.paths))
        img  = self.xform(img)
        cap  = os.path.splitext(os.path.basename(path))[0]
        txt  = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(txt):
            try:
                with open(txt, encoding="utf-8") as fh: cap = fh.read().strip()
            except: pass
        tok = self.tok(cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return img, tok.input_ids[0]


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
    ext   = Path(path).suffix.lower()
    state = load_file(path) if ext == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    res   = model.load_state_dict(state, strict=False)
    n     = len(state) - len(res.missing_keys)
    print("  " + clr("loaded " + str(n) + "/" + str(len(state)) + " tensors from " + Path(path).name, G))

def save_st(unet, te, folder, step):
    os.makedirs(folder, exist_ok=True)
    st = {k:v for k,v in unet.state_dict().items() if "lora_" in k}
    st.update({k:v for k,v in te.state_dict().items() if "lora_" in k})
    save_file(st, os.path.join(folder, "step_" + str(step) + ".safetensors"))
    print("  " + clr("checkpoint saved at step " + str(step), Y))

def save_pt(unet, folder, step):
    os.makedirs(folder, exist_ok=True)
    st = unet.state_dict()
    torch.save(st, os.path.join(folder, "step_" + str(step) + ".pt"))
    lo = {k:v for k,v in st.items() if "lora_" in k}
    if lo: save_file(lo, os.path.join(folder, "step_" + str(step) + ".safetensors"))
    print("  " + clr("checkpoint saved at step " + str(step), Y))


#validated ask
def _validated_ask(label, default, cast, prof, field):
    """
    Ask for a value but warn and cap if it would break the chosen model.
    For fine tune we enforce the model's native resolution minimum.
    Standard train can go lower since it's not bound the same way.
    """
    while True:
        raw = input("  " + clr(label, C) + " [" + clr(str(default), W) + "]: ").strip()
        if not raw: return default
        try: val = cast(raw)
        except: print("  " + clr("not a valid number", R)); continue

        if field == "res" and prof:
            mn = prof.get("min_res", 512)
            if val < mn:
                print("  " + clr("WARNING: this model needs at least " + str(mn) + "px", R))
                print("  " + clr("  going below " + str(mn) + "px will break generation", Y))
                c = input("  " + clr("  use minimum " + str(mn) + " instead? y/n [y]: ", C)).strip().lower()
                if c != "n":
                    print("  " + clr("  capped to " + str(mn), G))
                    return mn
                print("  " + clr("  keeping " + str(val) + "px  (expect broken outputs)", Y))
                return val

        if field == "rank" and prof:
            mn = prof.get("min_rank", 4)
            if val < mn:
                print("  " + clr("WARNING: rank below " + str(mn) + " will likely fail", R))
                c = input("  " + clr("  use minimum " + str(mn) + " instead? y/n [y]: ", C)).strip().lower()
                if c != "n": return mn
                return val

        if field == "te_rank" and prof and prof.get("flux"):
            if val < 16:
                print("  " + clr("WARNING: Flux text encoder rank below 16 breaks generation", R))
                print("  " + clr("  the encoder is tightly coupled with the diffusion model here", Y))
                c = input("  " + clr("  use 16 instead? y/n [y]: ", C)).strip().lower()
                if c != "n": return 16
                return val

        return val


def _ask(label, default, cast=str):
    raw = input("  " + clr(label, C) + " [" + clr(str(default), W) + "]: ").strip()
    if not raw: return default
    try: return cast(raw)
    except: return default

#fine tune
def run_finetune(cfg):
    sec("Fine Tune  |  Dual LoRA + VAE Cache")
    dd   = cfg["data_dir"];          ckd  = cfg["checkpoint_dir"]
    mid  = cfg["model_id"];          ms   = int(cfg["max_steps"])
    sv   = int(cfg["save_every"]);   bs   = int(cfg["batch_size"])
    ga   = int(cfg["grad_accum"]);   res  = int(cfg["resolution"])
    ulr  = float(cfg["unet_lr"]);    tlr  = float(cfg["text_lr"])
    ur   = int(cfg["unet_rank"]);    ua   = int(cfg["unet_alpha"])
    tr   = int(cfg["te_rank"]);      ta   = int(cfg["te_alpha"])
    drop = float(cfg["dropout"]);    nw   = int(cfg.get("num_workers",4))
    gc   = bool(cfg.get("gradient_checkpointing", True))
    elo  = cfg.get("existing_lora")
    prof = cfg.get("_profile", _DEFAULT)

    UT = ["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]
    TT = ["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]
    os.makedirs(ckd, exist_ok=True)

    print("  " + clr("model   ", DIM) + mid)
    print("  " + clr("type    ", DIM) + prof["label"])
    print("  " + clr("data    ", DIM) + dd)
    print("  " + clr("output  ", DIM) + ckd)
    if elo: print("  " + clr("base LoRA", DIM) + elo)

    print("  " + clr("\nloading model components...", C))
    try:
        tok  = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
        te   = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").to(device)
        vae  = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet").to(device)
        sch  = DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
    except Exception as ex:
        print("  " + clr("failed to load model: " + str(ex), R))
        print("  " + clr("check internet and model ID", Y))
        pause(); return

    for p in vae.parameters():  p.requires_grad = False
    for p in unet.parameters(): p.requires_grad = False
    for p in te.parameters():   p.requires_grad = False

    try:
        unet = get_peft_model(unet, LoraConfig(r=ur, lora_alpha=ua, target_modules=UT, lora_dropout=drop, bias="none"))
        te   = get_peft_model(te,   LoraConfig(r=tr, lora_alpha=ta, target_modules=TT, lora_dropout=drop, bias="none"))
    except Exception as ex:
        print("  " + clr("LoRA setup failed: " + str(ex), R)); pause(); return

    if elo and os.path.exists(elo):
        print("  " + clr("loading base LoRA weights...", C))
        try:
            st = load_file(elo)
            unet.load_state_dict(st, strict=False)
            te.load_state_dict(st, strict=False)
            print("  " + clr("loaded", G))
        except Exception as ex:
            print("  " + clr("could not load LoRA: " + str(ex) + "  starting fresh", Y))

    trainable = (sum(p.numel() for p in unet.parameters() if p.requires_grad)
               + sum(p.numel() for p in te.parameters()   if p.requires_grad))
    print("  " + clr("trainable params: " + "{:,}".format(trainable), G))

    if gc:
        unet.enable_gradient_checkpointing()
        te.gradient_checkpointing_enable()
        print("  " + clr("gradient checkpointing on", DIM))

    opt = AdamW(list(unet.parameters()) + list(te.parameters()),
                lr=ulr, betas=(.9,.99), weight_decay=.01)

    sec("Loading Dataset")
    try:
        ds = LatentCacheDataset(dd, tok, vae, res, ckd)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                        pin_memory=True, persistent_workers=(nw>0))
    except Exception as ex:
        print("  " + clr("dataset error: " + str(ex), R)); pause(); return

    if len(ds) == 0:
        print("  " + clr("no images found in " + dd, R)); pause(); return

    scaler = torch.cuda.amp.GradScaler()
    gs     = 0

    sec("Training")
    print("  " + clr("running for " + str(ms) + " steps\n", DIM))
    bar = tqdm(total=ms, desc="  finetune", unit="step", dynamic_ncols=True, colour="cyan")
    t0  = time.time()

    try:
        while gs < ms:
            for batch in dl:
                lats = batch["lat"].to(device, non_blocking=True)
                ids  = batch["ids"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                try:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        enc   = te(input_ids=ids, attention_mask=mask).last_hidden_state
                        noise = torch.randn_like(lats)
                        ts    = torch.randint(0, sch.config.num_train_timesteps, (lats.shape[0],), device=device).long()
                        nl    = sch.add_noise(lats, noise, ts)
                        pred  = unet(nl, ts, enc).sample
                        loss  = F.mse_loss(pred, noise) / ga
                    scaler.scale(loss).backward()
                    if (gs + 1) % ga == 0:
                        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                except torch.cuda.OutOfMemoryError:
                    print("\n\n  " + clr("OUT OF VRAM at step " + str(gs), R))
                    print("  " + clr("lower batch size or resolution and retry", Y))
                    print("  " + clr("checkpoint saved up to step " + str(gs), G))
                    bar.close(); pause(); return
                except Exception as ex:
                    print("\n  " + clr("step error: " + str(ex) + "  skipping", Y))
                    opt.zero_grad(set_to_none=True); gs += 1; bar.update(1); continue

                gs += 1
                elapsed = time.time() - t0
                eta     = (elapsed / gs) * (ms - gs) if gs > 0 else 0
                mem     = gpu_used()
                bar.update(1)
                bar.set_postfix(loss=str(round(loss.item()*ga, 4)),
                                vram=str(round(mem,2))+"GB",
                                eta=str(timedelta(seconds=int(eta))))
                if gs % sv == 0: bar.clear(); save_st(unet, te, ckd, gs)
                if gs >= ms: break
    except KeyboardInterrupt:
        print("\n\n  " + clr("stopped.  saving...", Y))
        save_st(unet, te, ckd, gs)
        bar.close(); pause(); return

    bar.close()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fout = os.path.join(ckd, "finetuned_" + ts + ".safetensors")
    st   = {k:v for k,v in unet.state_dict().items() if "lora_" in k}
    st.update({k:v for k,v in te.state_dict().items() if "lora_" in k})
    save_file(st, fout)
    print("  " + clr("saved -> " + fout, G))
    print("\n  " + clr("fine tuning done.", G) + "\n")


#standard train
def run_training(cfg):
    sec("Standard LoRA Training")
    dd   = cfg["data_dir"];        ckd  = cfg["checkpoint_dir"]
    mid  = cfg["model_id"];        res  = int(cfg["resolution"])
    bs   = int(cfg["batch_size"]); ms   = int(cfg["max_steps"])
    sv   = int(cfg["save_every"]); lr   = float(cfg["lr"])
    rank = int(cfg["rank"]);       alp  = int(cfg["alpha"])
    nw   = int(cfg.get("num_workers",4))
    wts  = cfg.get("weights_path")

    os.makedirs(ckd, exist_ok=True)
    print("  " + clr("model   ", DIM) + mid)
    print("  " + clr("data    ", DIM) + dd)
    print("  " + clr("output  ", DIM) + ckd)

    print("  " + clr("\nloading model...", C))
    try:
        tok  = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
        te   = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").to(device)
        vae  = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet").to(device)
        sch  = DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
    except Exception as ex:
        print("  " + clr("failed to load: " + str(ex), R)); pause(); return

    for p in unet.parameters(): p.requires_grad = False
    for p in te.parameters():   p.requires_grad = False
    for p in vae.parameters():  p.requires_grad = False

    try:
        unet = get_peft_model(unet, LoraConfig(
            r=rank, lora_alpha=alp,
            target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
            lora_dropout=.05, bias="none"
        ))
    except Exception as ex:
        print("  " + clr("LoRA setup failed: " + str(ex), R)); pause(); return

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print("  " + clr("trainable: " + "{:,}".format(trainable), G))

    latest, done = find_latest(ckd)
    if latest:
        print("\n  " + clr("resuming from step " + str(done), Y))
        try: load_into(unet, latest)
        except Exception as ex: print("  " + clr("could not load checkpoint: " + str(ex), Y))
    elif wts:
        print("\n  " + clr("loading weights: " + Path(wts).name, C))
        try: load_into(unet, wts)
        except Exception as ex: print("  " + clr("could not load weights: " + str(ex), Y))
    else:
        print("\n  " + clr("starting fresh", C))

    if done >= ms:
        print("\n  " + clr("already at " + str(done) + " steps, nothing to do", G))
        pause(); return

    try:
        unet.enable_xformers_memory_efficient_attention()
        print("  " + clr("xformers on", G))
    except:
        try: torch.backends.cuda.enable_flash_sdp(True); print("  " + clr("flash attention on", G))
        except: pass

    sec("Loading Dataset")
    try:
        ds = ImageCaptionDataset(dd, tok, res)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    except Exception as ex:
        print("  " + clr("dataset error: " + str(ex), R)); pause(); return

    if len(ds) == 0:
        print("  " + clr("no images found in " + dd, R)); pause(); return

    opt    = AdamW([p for p in unet.parameters() if p.requires_grad], lr=lr, betas=(.9,.999), weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler()
    gs     = done; t0 = time.time()

    sec("Training")
    print("  " + clr("steps " + str(done) + " to " + str(ms) + "\n", DIM))
    bar = tqdm(total=ms, initial=done, desc="  training", unit="step", dynamic_ncols=True, colour="cyan")

    try:
        while gs < ms:
            for imgs, ids in dl:
                st0  = time.time()
                imgs = imgs.to(device); ids = ids.to(device)
                try:
                    with torch.no_grad():
                        lats = vae.encode(imgs).latent_dist.sample() * 0.18215
                        enc  = te(ids)[0]
                    noise = torch.randn_like(lats)
                    ts    = torch.randint(0, sch.config.num_train_timesteps, (lats.shape[0],), device=device).long()
                    nl    = sch.add_noise(lats, noise, ts)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = unet(nl, ts, enc).sample
                        loss = F.mse_loss(pred, noise)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad()
                except torch.cuda.OutOfMemoryError:
                    print("\n\n  " + clr("OUT OF VRAM at step " + str(gs), R))
                    print("  " + clr("lower batch size or resolution and retry", Y))
                    print("  " + clr("checkpoint saved up to step " + str(gs), G))
                    bar.close(); pause(); return
                except Exception as ex:
                    print("\n  " + clr("step error: " + str(ex) + "  skipping", Y))
                    opt.zero_grad(); gs += 1; bar.update(1); continue

                gs += 1
                elapsed = time.time() - t0
                eta     = (elapsed / (gs-done)) * (ms-gs) if (gs-done) > 0 else 0
                mem     = gpu_used()
                bar.update(1)
                bar.set_postfix(loss=str(round(loss.item(),4)),
                                spt=str(round(time.time()-st0,2))+"s",
                                vram=str(round(mem,2))+"GB",
                                eta=str(timedelta(seconds=int(eta))))
                if gs % sv == 0: bar.clear(); save_pt(unet, ckd, gs)
                if gs >= ms: break
    except KeyboardInterrupt:
        print("\n\n  " + clr("stopped.  saving...", Y))
        save_pt(unet, ckd, gs)
        bar.close(); pause(); return

    bar.close()
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpt = os.path.join(ckd, "final_" + ts + ".pt")
    fst = os.path.join(ckd, "final_" + ts + ".safetensors")
    st  = unet.state_dict()
    torch.save(st, fpt)
    lo  = {k:v for k,v in st.items() if "lora_" in k}
    if lo: save_file(lo, fst)
    print("  " + clr("saved -> " + fpt, G))
    print("  " + clr("saved -> " + fst, G))
    print("\n  " + clr("training done.", G) + "\n")


#generate
def run_generate(cfg):
    sec("Generate Images")
    mid    = cfg["model_id"];  lp = cfg.get("lora_path")
    odir   = Path(cfg["output_dir"]); prompt = cfg["prompt"]
    n      = int(cfg["num_images"]); steps = int(cfg.get("steps",30))
    cfg_s  = float(cfg.get("guidance",7.5))

    odir.mkdir(parents=True, exist_ok=True)
    print("  " + clr("model  ", DIM) + mid)
    if lp: print("  " + clr("lora   ", DIM) + lp)
    print("  " + clr("prompt ", DIM) + prompt)
    print("  " + clr(str(n) + " images  steps=" + str(steps) + "  cfg=" + str(cfg_s), DIM) + "\n")

    try:
        print("  " + clr("loading pipeline...", C))
        pipe = StableDiffusionPipeline.from_pretrained(mid, torch_dtype=torch.float16, safety_checker=None).to(device)
    except Exception as ex:
        print("  " + clr("failed to load pipeline: " + str(ex), R)); pause(); return

    if lp and os.path.exists(lp):
        try:
            pipe.unet.load_state_dict(load_file(lp), strict=False)
            print("  " + clr("LoRA applied", G))
        except Exception as ex:
            print("  " + clr("could not apply LoRA: " + str(ex) + "  using base only", Y))
    elif lp:
        print("  " + clr("LoRA file not found, using base model", Y))

    print("  " + clr("generating...\n", C))
    for i in tqdm(range(1, n+1), desc="  gen", unit="img", dynamic_ncols=True, colour="cyan"):
        try:
            img  = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg_s).images[0]
            name = "gen_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(i) + ".png"
            img.save(odir / name)
        except torch.cuda.OutOfMemoryError:
            print("\n  " + clr("out of VRAM.  try fewer steps.", R)); break
        except Exception as ex:
            print("\n  " + clr("generation error: " + str(ex), Y)); continue

    print("\n  " + clr("done.  saved to " + str(odir.resolve()), G) + "\n")


#huggingface search
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
    "stablediffusionapi/realistic-vision-v6.0-b1-inpaint",
]


def _pick_model_search():
    """
    Terminal model picker with HuggingFace search.
    Returns (model_id, profile_dict).
    """
    sec("Pick a Base Model")
    print("  " + clr("type a number, s to search HuggingFace, or u to paste a URL\n", DIM))
    for i, m in enumerate(HF_PRESETS, 1):
        prof  = detect_from_id(m)
        badge = clr("[" + prof["label"].split("(")[0].strip() + "]", DIM)
        print("  " + clr(str(i), C) + "  " + m + "  " + badge)
    print()
    print("  " + clr("s", C) + "  search HuggingFace by name")
    print("  " + clr("u", C) + "  paste a URL or model ID")
    print()

    while True:
        ch = input("  " + clr("choice [1]: ", C)).strip() or "1"

        if ch == "u":
            raw = input("  " + clr("paste model ID or HF URL: ", C)).strip()
            if "huggingface.co/" in raw:
                raw = raw.rstrip("/").split("huggingface.co/")[-1].strip("/")
            if raw:
                prof = detect_from_id(raw)
                print("  " + clr("detected: " + prof["label"], G))
                return raw, prof
            print("  " + clr("nothing entered", R))

        elif ch == "s":
            q = input("  " + clr("search: ", C)).strip()
            if not q: continue
            print("  " + clr("searching...", C))
            results = hf_search(q)
            if not results:
                print("  " + clr("no results or no internet.  try option u to paste an ID", Y)); continue
            print()
            for i, (mid, dl) in enumerate(results, 1):
                prof  = detect_from_id(mid)
                badge = clr("[" + prof["label"].split("(")[0].strip() + "]", DIM)
                print("  " + clr(str(i), C) + "  " + mid + "  " + badge
                      + "  " + clr(str(dl) + " dl", DIM))
            print("  " + clr("0", C) + "  search again")
            print()
            sc = input("  " + clr("pick one [1]: ", C)).strip() or "1"
            if sc == "0": continue
            if sc.isdigit() and 1 <= int(sc) <= len(results):
                mid  = results[int(sc)-1][0]
                prof = detect_from_id(mid)
                print("  " + clr("selected: " + mid, G))
                print("  " + clr("detected: " + prof["label"], G))
                return mid, prof
            print("  " + clr("not valid", R))

        elif ch.isdigit() and 1 <= int(ch) <= len(HF_PRESETS):
            mid  = HF_PRESETS[int(ch)-1]
            prof = detect_from_id(mid)
            print("  " + clr("detected: " + prof["label"], G))
            return mid, prof
        else:
            print("  " + clr("not valid", R))


#input helpers
def _pick_data_dir(last=""):
    sec("Training Data Folder")
    print("  " + clr("path to folder with your images (or videos)", DIM))
    if last: print("  " + clr("last: " + last, DIM) + "\n")
    while True:
        raw = input("  " + clr("path [" + (last or "required") + "]: ", C)).strip()
        if not raw and last: raw = last
        if not raw: print("  " + clr("required", R)); continue
        path = raw.strip('"').strip("'")
        if not os.path.isdir(path): print("  " + clr("folder not found", R)); continue
        ni = sum(len(glob.glob(os.path.join(path,"*."+e))) for e in IEXT)
        nv = sum(len(glob.glob(os.path.join(path,"*."+e))) for e in VEXT)
        print("  " + clr("found " + str(ni) + " images and " + str(nv) + " videos", G))
        return path


def _pick_out_dir(last="", label="Output Folder"):
    sec(label)
    raw = input("  " + clr("path [" + (last or "required") + "]: ", C)).strip()
    if not raw and last: return last
    path = raw.strip('"').strip("'")
    if path: os.makedirs(path, exist_ok=True); return path
    return last


def _pick_weights(ckd):
    sec("Starting Weights  (optional, enter to skip)")
    files = [f for f in all_weights(ckd) if not re.search(r"step_\d+\.(pt|safetensors)$",f)]
    if not files: print("  " + clr("nothing found, starting fresh", DIM)); return None
    print("  " + clr("0", C) + "  fresh start")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print("  " + clr(str(i), C) + "  " + Path(f).name + "  " + clr(str(round(mb))+"MB",DIM))
    print()
    while True:
        ch = input("  " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        print("  " + clr("not valid", R))


def _pick_lora(dirs):
    sec("LoRA File  (optional, enter to skip)")
    files = []
    for d in list(dirs) + [os.getcwd()]:
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d,"*.safetensors")):
                if f not in files: files.append(f)
    if not files:
        raw = input("  " + clr("no files found  paste path or enter to skip: ", C)).strip().strip('"').strip("'")
        return raw if raw and os.path.exists(raw) else None
    print("  " + clr("0", C) + "  none")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print("  " + clr(str(i), C) + "  " + Path(f).name + "  " + clr(str(round(mb))+"MB",DIM))
    print("  " + clr("m", C) + "  type a path")
    print()
    while True:
        ch = input("  " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch == "m":
            raw = input("  " + clr("path: ", C)).strip().strip('"').strip("'")
            if raw and os.path.exists(raw): return raw
            print("  " + clr("not found", R))
        elif ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        else: print("  " + clr("not valid", R))


def _pick_existing_lora(ckd):
    sec("Existing LoRA  (optional)")
    files = all_weights(ckd)
    if not files: print("  " + clr("nothing found, starting fresh", DIM)); return None
    print("  " + clr("0", C) + "  fresh start")
    for i,f in enumerate(files,1):
        mb = os.path.getsize(f)/1024**2
        print("  " + clr(str(i), C) + "  " + Path(f).name + "  " + clr(str(round(mb))+"MB",DIM))
    print()
    while True:
        ch = input("  " + clr("choice [0]: ", C)).strip() or "0"
        if ch == "0": return None
        if ch.isdigit() and 1 <= int(ch) <= len(files): return files[int(ch)-1]
        print("  " + clr("not valid", R))


def _get_train_settings(ckd, prof=None):
    sec("Training Settings")
    p  = vram_preset(prof)
    gb = gpu_gb()
    if gb > 0:
        print("  " + clr("GPU: " + str(round(gb,1)) + "GB  -> " + p["label"], Y))
        if prof: print("  " + clr("model: " + prof["label"], C))
        print("  " + clr("press enter to accept each default\n", DIM))

    latest, step = find_latest(ckd)
    if latest: print("  " + clr("checkpoint at step " + str(step) + ", will auto-resume\n", G))

    res  = _ask("resolution (512 / 768 / 1024)",  p["res"],   int)
    bs   = _ask("batch size",                     p["bs"],    int)
    ms   = _ask("total steps",                    p["steps"], int)
    sv   = _ask("save every N steps",             p["save"],  int)
    lr   = _ask("learning rate",                  p["lr"],    float)
    rank = _ask("LoRA rank",                      p["rank"],  int)
    alp  = _ask("LoRA alpha",                     p["alpha"], int)
    nw   = _ask("dataloader workers",             p["nw"],    int)
    if res >= 1024 and bs > 1: print("  " + clr("1024px: batch set to 1",Y)); bs = 1
    return dict(resolution=res, batch_size=bs, max_steps=ms, save_every=sv,
                lr=lr, rank=rank, alpha=alp, num_workers=nw)


def _get_ft_settings(ckd, prof=None):
    sec("Fine Tune Settings")
    p  = vram_preset(prof)
    pr = prof or _DEFAULT
    gb = gpu_gb()
    if gb > 0: print("  " + clr("GPU: " + str(round(gb,1)) + "GB  -> " + p["label"], Y))
    if prof:
        print("  " + clr("model: " + prof["label"], C))
        print("  " + clr("note:  " + prof["note"], Y))
        if prof["min_res"] > 512:
            print("  " + clr("minimum resolution for this model: " + str(prof["min_res"]) + "px", R))
    print("  " + clr("\npress enter to accept each default\n", DIM))

    res  = _validated_ask("resolution",             p["res"],   int,   pr, "res")
    bs   = _ask("batch size",                       p["bs"],    int)
    ga   = _ask("gradient accumulation",            p["ga"],    int)
    ms   = _ask("total steps",                      p["steps"], int)
    sv   = _ask("save every N steps",               p["save"],  int)
    ulr  = _ask("UNet learning rate",               3e-5,       float)
    tlr  = _ask("text encoder learning rate",       1e-5,       float)
    ur   = _validated_ask("UNet LoRA rank",         p["ur"],    int,   pr, "rank")
    ua   = _ask("UNet LoRA alpha",                  p["ua"],    int)
    tr   = _validated_ask("text encoder LoRA rank", p["tr"],    int,   pr, "te_rank")
    ta   = _ask("text encoder LoRA alpha",          p["ta"],    int)
    drop = _ask("LoRA dropout",                     p["drop"],  float)
    nw   = _ask("dataloader workers",               p["nw"],    int)
    gc   = _ask("gradient checkpointing y/n",       "y" if p["gc"] else "n", str)
    if res >= 1024 and bs > 1: print("  " + clr("1024px: batch set to 1",Y)); bs = 1
    return dict(resolution=res, batch_size=bs, grad_accum=ga, max_steps=ms,
                save_every=sv, unet_lr=ulr, text_lr=tlr, unet_rank=ur,
                unet_alpha=ua, te_rank=tr, te_alpha=ta, dropout=drop,
                num_workers=nw, gradient_checkpointing=(str(gc).strip().lower()!="n"))


BOOSTS = {
    "1": ("resolution -> 768",        dict(resolution=768,  batch_size=2)),
    "2": ("resolution -> 1024",       dict(resolution=1024, batch_size=1)),
    "3": ("rank -> 192",              dict(rank=192, alpha=384)),
    "4": ("+8k steps",                dict(max_steps=24000)),
    "5": ("fast mode (rank 64 bs 8)", dict(rank=64, alpha=128, batch_size=8)),
    "6": ("full quality pack",        dict(resolution=768, rank=192, alpha=384, max_steps=20000, batch_size=2)),
}

def _apply_boosts(cfg):
    sec("Quality Boosts  (standard train only)")
    print("  " + clr("note: these are for standard train. fine tune uses model-native settings.\n", DIM))
    for k,(nm,_) in BOOSTS.items(): print("  " + clr(k,C) + "  " + nm)
    print("  " + clr("0",C) + "  keep as is")
    print()
    ch = input("  " + clr("pick (or combine like 1,3): ", C)).strip()
    if not ch or ch == "0": return cfg
    for part in re.split(r"[,\s]+", ch):
        if part in BOOSTS:
            cfg.update(BOOSTS[part][1])
            print("  " + clr("applied: " + BOOSTS[part][0], G))
    if cfg.get("resolution",0) >= 1024: cfg["batch_size"] = 1
    return cfg

#tutorial
def show_tutorial():
    clear()
    print(clr("=" * 60, C))
    print(clr("  Tutorial  |  How to use this trainer", BOLD))
    print(clr("=" * 60, C) + "\n")

    sections = [
        ("QUICK START  (the most common workflow)", [
            "1. Resize        -> menu 2  set 512x512 or 768x768",
            "2. Caption       -> menu 3  auto-labels your images",
            "3. Train         -> menu 4  standard LoRA (recommended for beginners)",
            "4. Generate      -> menu 7  test your trained LoRA",
        ]),
        ("IMAGE EDITING  (menu 9)", [
            "Face Swap      - replaces a face in any photo. needs insightface.",
            "Face Restore   - sharpens/fixes blurry faces. GFPGAN if installed, else PIL.",
            "img2img        - re-draw an image guided by a text prompt.",
            "Filters        - brightness, contrast, sharpen, blur, saturation, flip.",
            "Batch Restore  - runs face restore on an entire folder at once.",
        ]),
        ("UPLOADING AN IMAGE", [
            "At any image path prompt just type the path and press enter.",
            "On Windows:  C:\\Users\\you\\Desktop\\photo.jpg",
            "On Linux:    /home/you/photo.jpg",
            "You can drag a file from explorer into the terminal to auto-paste the path.",
            "Ctrl+U is shown as a reminder at image prompts - it just means type/paste a path.",
        ]),
        ("FINE TUNE vs STANDARD TRAIN", [
            "Standard Train  -> flexible. resolution and rank can be adjusted up or down.",
            "                   boosts available. easier to experiment with.",
            "Fine Tune       -> model-aware. resolution is enforced at the model's native size.",
            "                   going below the model's minimum will be warned and capped.",
            "                   has dual LoRA (UNet + text encoder) and VAE latent cache.",
            "                   faster per epoch once cache is built.",
        ]),
        ("MODEL TYPES AND WHAT BREAKS", [
            "SD 1.x    native 512px.  going below 512px = broken output.",
            "SD 2.x    native 768px.  going below 768px = noisy broken output.",
            "SDXL      native 1024px. going below 1024px = badly broken output.",
            "Flux.1    native 1024px. text encoder rank below 16 = broken output.",
            "The trainer detects which model you loaded and warns you if you set",
            "something that will break it. It will offer to cap to safe values.",
        ]),
        ("COMFYUI SETUP  (menu 8)", [
            "Installs ComfyUI + 12 essential node packs automatically.",
            "Includes: Manager, Impact Pack, IP Adapter, ControlNet, ReActor,",
            "          InstantID, Portrait Master, Ultimate SD Upscale, SUPIR.",
            "After install: cd ComfyUI && python main.py",
            "Then open http://127.0.0.1:8188 in your browser.",
            "Drag any .json workflow into the browser to load it.",
        ]),
        ("CHECKPOINTS AND RESUMING", [
            "Training auto-saves a checkpoint every N steps (you set this).",
            "If training stops for any reason just run train again with the same",
            "output folder and it will automatically resume from the last checkpoint.",
            "Final output is saved as both .pt and .safetensors.",
        ]),
        ("GPU MEMORY TIPS", [
            "Out of VRAM?  lower resolution first, then batch size, then rank.",
            "Turn gradient checkpointing ON if you are below 12GB.",
            "Use grad accumulation to compensate for smaller batch sizes.",
            "xformers helps if installed. Flash attention is the fallback.",
            "For 6GB cards: res=512 bs=1 ga=32 rank=16 grad_ckpt=on",
        ]),
    ]

    for title, lines in sections:
        print("  " + clr(title, Y))
        for l in lines: print("  " + clr("  " + l, DIM))
        print()

    print("  " + clr("=" * 58, DIM))
    print("  " + clr("press enter to return to the menu", DIM))
    pause("")


# comfyui installer
CNODES = [
    ("ComfyUI Manager",          "https://github.com/ltdrdata/ComfyUI-Manager"),
    ("Impact Pack",              "https://github.com/ltdrdata/ComfyUI-Impact-Pack"),
    ("IP Adapter Plus",          "https://github.com/cubiq/ComfyUI_IPAdapter_plus"),
    ("ControlNet Preprocessors", "https://github.com/Fannovel16/comfyui_controlnet_aux"),
    ("rgthree comfy",            "https://github.com/rgthree/rgthree-comfy"),
    ("KJNodes",                  "https://github.com/kijai/ComfyUI-KJNodes"),
    ("ComfyUI Easy Use",         "https://github.com/yolain/ComfyUI-Easy-Use"),
    ("ReActor Node",             "https://github.com/Gourieff/comfyui-reactor-node"),
    ("InstantID FaceSwap",       "https://github.com/nosiu/comfyui-instantId-faceswap"),
    ("Portrait Master v3",       "https://github.com/florestefano1975/comfyui-portrait-master"),
    ("Ultimate SD Upscale",      "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"),
    ("SUPIR",                    "https://github.com/kijai/ComfyUI-SUPIR"),
    ("DeepFuze",                 "https://github.com/SamKhoze/ComfyUI-DeepFuze"),
]

CFACE = [
    ("onnxruntime",  "onnxruntime-gpu", None),
    ("basicsr",      "basicsr",         None),
    ("facexlib",     "facexlib",        None),
    ("realesrgan",   "realesrgan",      None),
    ("insightface",  "insightface",
     "needs Visual C++ Build Tools on Windows.\n"
     "  get them at: visualstudio.microsoft.com/visual-cpp-build-tools\n"
     "  install the C++ workload, restart, try again.\n"
     "  the installer above tried a pre-built wheel automatically."),
]


def _git_clone(name, url, parent):
    folder = url.rstrip("/").split("/")[-1]
    dest   = os.path.join(parent, folder)
    if os.path.isdir(dest):
        print("  " + clr("already there  ", G) + name); return True
    print("  " + clr("cloning  ", C) + name)
    try:
        r = subprocess.run(["git","clone","--depth","1",url,dest],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print("  " + clr("done     ", G) + name); return True
        print("  " + clr("failed   ", R) + name + "  " + clr(r.stderr.strip()[:60],DIM))
        return False
    except FileNotFoundError:
        print("  " + clr("git not found  install it from git-scm.com", R)); return False
    except Exception as ex:
        print("  " + clr("error  ", R) + name + "  " + str(ex)); return False


def install_comfyui():
    sec("ComfyUI Auto Installer")
    print("  " + clr("installs ComfyUI and all essential node packs.", DIM))
    print("  " + clr("anything already installed gets skipped. safe to re-run.\n", DIM))

    default = os.path.join(os.path.expanduser("~"), "ComfyUI")
    raw     = input("  " + clr("install location [" + default + "]: ", C)).strip().strip('"').strip("'")
    root    = raw if raw else default
    cdir    = os.path.join(root, "ComfyUI")
    ndir    = os.path.join(cdir, "custom_nodes")
    sec("Step 1/3  pip packages")
    print("  " + clr("face tools and upscaling deps\n", DIM))
    for mod, pkg, warn in CFACE:
        try:   __import__(mod); print("  " + clr("ok  ", G) + pkg)
        except ImportError:
            ok = pip_install(pkg, [pkg])
            if not ok and mod == "insightface":
                print("  " + clr("trying pre-built wheel...", Y))
                wheel = "https://github.com/Gourieff/Assets/raw/main/insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
                ok2   = pip_install("insightface-wheel", [wheel])
                if not ok2 and warn:
                    print()
                    for ln in warn.split("\n"):
                        if ln.strip(): print("  " + clr(ln, Y))
                    print()
            elif not ok and warn:
                for ln in warn.split("\n"):
                    if ln.strip(): print("  " + clr(ln, Y))
                print()

    sec("Step 2/3  ComfyUI core")
    os.makedirs(root, exist_ok=True)
    if os.path.isdir(cdir):
        print("  " + clr("already installed at " + cdir, G))
    else:
        print("  " + clr("cloning ComfyUI into " + root + "...", C))
        if not _git_clone("ComfyUI", "https://github.com/comfyanonymous/ComfyUI", root):
            print("  " + clr("clone failed. check git and internet.", R))
            pause(); return
        req = os.path.join(cdir, "requirements.txt")
        if os.path.exists(req):
            print("  " + clr("installing ComfyUI requirements...", C) + "\n")
            skip = {"torch","torchvision","torchaudio"}
            with open(req,"r",encoding="utf-8") as fh:
                reqs = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]
            for r in reqs:
                bare = re.split(r"[>=<!;\[]", r)[0].strip().lower().replace("-","_")
                if bare in skip: print("  " + clr("skip  ",DIM) + r); continue
                try:   __import__(bare); print("  " + clr("ok  ",G) + r)
                except: pip_install(r, [r])

    sec("Step 3/3  Node Packs")
    os.makedirs(ndir, exist_ok=True)
    print("  " + clr("cloning " + str(len(CNODES)) + " packs into " + ndir + "\n", DIM))
    ok_n = fail_n = 0
    for name, url in CNODES:
        if _git_clone(name, url, ndir): ok_n += 1
        else: fail_n += 1

    guide = os.path.join(root, "HOW_TO_START.txt")
    with open(guide,"w",encoding="utf-8") as fh:
        fh.write("ComfyUI installed at:\n  " + cdir + "\n\n")
        fh.write("To start:\n  cd " + cdir + "\n  python main.py\n\n")
        fh.write("Then open:\n  http://127.0.0.1:8188\n\n")
        fh.write("Drag any .json workflow into the browser to load it.\n")
        fh.write("Use Manager node to install more packs.\n")
    print("\n  " + clr("startup guide -> " + guide, DIM))
    print("\n  " + clr("done!  " + str(ok_n) + " packs installed"
                        + ("  " + str(fail_n) + " failed" if fail_n else ""), G))
    print("\n  " + clr("to start: cd " + cdir + " && python main.py", W))
    print("  " + clr("then: http://127.0.0.1:8188", DIM) + "\n")
    pause()


#pipeline builder
PIPE_STEPS = {
    "resize":   "Resize Images and Videos",
    "caption":  "Auto Caption with BLIP",
    "train":    "Standard LoRA Train",
    "finetune": "Fine Tune  Dual LoRA",
    "generate": "Generate Images",
}
SKEYS = list(PIPE_STEPS.keys())
SLABS = list(PIPE_STEPS.values())


def _build_step_cfg(key, base):
    cfg = dict(base)
    if key == "resize":
        sec("Resize Settings")
        cfg["resize_w"] = _ask("target width", 512, int)
        cfg["resize_h"] = _ask("target height", 512, int)
    elif key == "caption":
        sec("Caption Settings")
        cfg["caption_batch"] = _ask("batch size", 4, int)
    elif key == "train":
        mid, prof = _pick_model_search()
        wts       = _pick_weights(cfg["checkpoint_dir"])
        settings  = _get_train_settings(cfg["checkpoint_dir"], prof)
        cfg.update(dict(model_id=mid, weights_path=wts, _profile=prof, **settings))
    elif key == "finetune":
        mid, prof = _pick_model_search()
        elo       = _pick_existing_lora(cfg["checkpoint_dir"])
        settings  = _get_ft_settings(cfg["checkpoint_dir"], prof)
        cfg.update(dict(model_id=mid, existing_lora=elo, _profile=prof, **settings))
    elif key == "generate":
        mid, _    = _pick_model_search()
        lp        = _pick_lora([cfg["checkpoint_dir"]])
        outd      = _pick_out_dir(os.path.join(cfg["checkpoint_dir"],"generated"), label="Where to Save Images")
        sec("Generation Settings")
        prompt = ""
        while not prompt:
            prompt = input("  " + clr("prompt: ", C)).strip()
            if not prompt: print("  " + clr("required", R))
        cfg.update(dict(model_id=mid, lora_path=lp, output_dir=outd, prompt=prompt,
                        num_images=_ask("how many images", 4, int),
                        steps=_ask("steps", 30, int),
                        guidance=_ask("guidance scale", 7.5, float)))
    return cfg


def _run_step(key, cfg):
    if key == "resize":    run_resize(cfg)
    elif key == "caption": run_caption(cfg)
    elif key == "train":   run_training(cfg)
    elif key == "finetune":run_finetune(cfg)
    elif key == "generate":run_generate(cfg)


def flow_pipeline():
    sec("Pipeline Builder")
    print("  " + clr("pick steps separated by commas, e.g.  1,2,4  or  3,5\n", DIM))
    for i,lab in enumerate(SLABS,1): print("  " + clr(str(i),C) + "  " + lab)
    print()
    raw = input("  " + clr("steps to run: ", C)).strip()
    if not raw: print("  " + clr("nothing picked",Y)); pause(); return

    chosen = []
    for part in re.split(r"[,\s]+",raw):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(SKEYS):
            k = SKEYS[int(part)-1]
            if k not in chosen: chosen.append(k)
        elif part: print("  " + clr("skipping: " + part,Y))
    if not chosen: print("  " + clr("nothing valid",R)); pause(); return

    dd   = _pick_data_dir(_last["dd"])
    od   = _pick_out_dir(_last["out"])
    _last["dd"] = dd; _last["out"] = od
    base = {"data_dir": dd, "checkpoint_dir": od}

    cfgs = {}
    for k in chosen:
        print("\n  " + clr("setting up: " + PIPE_STEPS[k], Y))
        cfgs[k] = _build_step_cfg(k, base)

    sec("Ready")
    for i,k in enumerate(chosen,1): print("  " + clr(str(i),C) + "  " + PIPE_STEPS[k])
    print()
    input("  " + clr("enter to start the pipeline...", Y))

    tot = str(len(chosen))
    for i,k in enumerate(chosen,1):
        print("\n  " + clr("[" + str(i) + "/" + tot + "]", C) + "  " + clr(PIPE_STEPS[k],W) + "...")
        try: _run_step(k, cfgs[k])
        except KeyboardInterrupt:
            print("\n\n  " + clr("stopped. checkpoints are safe.",Y)); break
        except Exception as ex:
            print("\n  " + clr("error in " + k + ": " + str(ex),R))
            if input("  " + clr("keep going? y/n [y]: ",C)).strip().lower() == "n": break

    print("\n  " + clr("pipeline done.",G) + "\n")
    pause()


def _confirm_run(cfg, mode):
    sec("Ready to Start")
    for k,v in cfg.items():
        if v not in (None,"") and not k.startswith("_"):
            print("  " + clr(k,DIM) + " = " + clr(str(v),W))
    input("\n  " + clr("enter to start...", Y))
    try:
        if mode == "finetune":   run_finetune(cfg)
        elif mode == "generate": run_generate(cfg)
        else:                    run_training(cfg)
    except KeyboardInterrupt:
        print("\n\n  " + clr("stopped. checkpoint saved.",Y))
    pause()


def flow_train(boost=False):
    dd = _pick_data_dir(_last["dd"])
    od = _pick_out_dir(_last["out"])
    _last["dd"] = dd; _last["out"] = od
    mid, prof = _pick_model_search()
    wts       = _pick_weights(od)
    settings  = _get_train_settings(od, prof)
    cfg       = dict(model_id=mid, weights_path=wts, data_dir=dd,
                     checkpoint_dir=od, _profile=prof, **settings)
    if boost: cfg = _apply_boosts(cfg)
    _confirm_run(cfg, "train")


def flow_finetune():
    dd = _pick_data_dir(_last["dd"])
    od = _pick_out_dir(_last["out"])
    _last["dd"] = dd; _last["out"] = od
    mid, prof = _pick_model_search()
    elo       = _pick_existing_lora(od)
    settings  = _get_ft_settings(od, prof)
    cfg       = dict(model_id=mid, existing_lora=elo, data_dir=dd,
                     checkpoint_dir=od, _profile=prof, **settings)
    _confirm_run(cfg, "finetune")


def flow_generate():
    mid, _  = _pick_model_search()
    lp      = _pick_lora([_last["out"]])
    outd    = _pick_out_dir(os.path.join(_last["out"],"generated") if _last["out"] else "",
                             label="Where to Save Images")
    sec("Generation Settings")
    prompt = ""
    while not prompt:
        prompt = input("  " + clr("prompt: ", C)).strip()
        if not prompt: print("  " + clr("required", R))
    cfg = dict(model_id=mid, lora_path=lp, output_dir=outd, prompt=prompt,
               num_images=_ask("how many images", 4, int),
               steps=_ask("steps", 30, int),
               guidance=_ask("guidance scale", 7.5, float))
    _confirm_run(cfg, "generate")


def flow_resize():
    dd = _pick_data_dir(_last["dd"]); _last["dd"] = dd
    sec("Resize Settings")
    w = _ask("width",  512, int)
    h = _ask("height", 512, int)
    sec("Ready")
    print("  " + clr("folder", DIM) + "  " + dd)
    print("  " + clr("size  ", DIM) + "  " + str(w) + "x" + str(h))
    input("\n  " + clr("enter to start...", Y))
    try:    run_resize({"data_dir":dd,"resize_w":w,"resize_h":h})
    except KeyboardInterrupt: print("\n\n  " + clr("stopped",Y))
    pause()


def flow_caption():
    dd = _pick_data_dir(_last["dd"]); _last["dd"] = dd
    batch = _ask("batch size", 4, int)
    input("\n  " + clr("enter to start...", Y))
    try:    run_caption({"data_dir":dd,"caption_batch":batch})
    except KeyboardInterrupt: print("\n\n  " + clr("stopped",Y))
    pause()


# persistent state
_last = {"dd": "", "out": ""}


# main menu
def main_menu():
    while True:
        print_header()

        ni = 0
        if _last["dd"]:
            ni = sum(len(glob.glob(os.path.join(_last["dd"],"*."+e))) for e in IEXT)
        _, step = find_latest(_last["out"]) if _last["out"] else (None, 0)

        print("  " + clr("data   ",DIM) + (clr(_last["dd"],W) if _last["dd"] else clr("not set",DIM)))
        if _last["dd"]: print("  " + clr("images ",DIM) + clr(str(ni),W))
        print("  " + clr("output ",DIM) + (clr(_last["out"],W) if _last["out"] else clr("not set",DIM)))
        if _last["out"]: print("  " + clr("ckpt   ",DIM) + (clr("step "+str(step),G) if step else clr("none",DIM)))

        print("\n  " + clr("─" * 50, DIM))
        print("  " + clr("1",C) + "  Pipeline Builder       chain any steps together")
        print("  " + clr("─" * 50, DIM))
        print("  " + clr("2",C) + "  Resize                 resize images and videos")
        print("  " + clr("3",C) + "  Caption                auto-caption with BLIP")
        print("  " + clr("4",C) + "  Train                  standard LoRA training")
        print("  " + clr("5",C) + "  Train + Boosts         train with quality upgrades")
        print("  " + clr("6",C) + "  Fine Tune              dual LoRA, VAE cache, model-aware")
        print("  " + clr("7",C) + "  Generate               make images from any LoRA")
        print("  " + clr("─" * 50, DIM))
        print("  " + clr("8",C) + "  ComfyUI Setup          install ComfyUI and all node packs")
        print("  " + clr("9",C) + "  Image Editor           face swap, restore, img2img, filters")
        print("  " + clr("─" * 50, DIM))
        print("  " + clr("t",C) + "  Tutorial               how everything works (Ctrl+U tip inside)")
        print("  " + clr("v",C) + "  VRAM Info              current GPU memory")
        print("  " + clr("g",C) + "  GPU Presets            recommended settings by GPU size")
        print("  " + clr("0",C) + "  Exit")
        print()

        ch = input("  " + clr(">", C) + " ").strip().lower()

        def wrap(fn):
            try: fn()
            except Exception as ex:
                print("\n  " + clr("error: " + str(ex),R) + "\n")
                import traceback; traceback.print_exc()
                pause()

        if   ch == "1": wrap(flow_pipeline)
        elif ch == "2": wrap(flow_resize)
        elif ch == "3": wrap(flow_caption)
        elif ch == "4": wrap(lambda: flow_train(False))
        elif ch == "5": wrap(lambda: flow_train(True))
        elif ch == "6": wrap(flow_finetune)
        elif ch == "7": wrap(flow_generate)
        elif ch == "8":
            try:    install_comfyui()
            except KeyboardInterrupt: print("\n\n  " + clr("cancelled",Y))
            except Exception as ex:   print("\n  " + clr("error: " + str(ex),R)); pause()
        elif ch == "9": wrap(terminal_image_editor)
        elif ch == "t": show_tutorial()
        elif ch == "g": show_vram_table(); pause()
        elif ch == "v":
            sec("Current GPU Usage")
            if torch.cuda.is_available():
                p   = torch.cuda.get_device_properties(0)
                tot = p.total_memory/1024**3
                fr  = torch.cuda.mem_get_info()[0]/1024**3
                print("  " + clr("GPU   ",DIM) + p.name)
                print("  " + clr("total ",DIM) + str(round(tot,1)) + "GB")
                print("  " + clr("used  ",DIM) + str(round(tot-fr,1)) + "GB")
                print("  " + clr("free  ",DIM) + str(round(fr,1)) + "GB")
                print("  " + clr("cuda  ",DIM) + str(torch.version.cuda))
                print("  " + clr("torch ",DIM) + str(torch.__version__))
                print("\n  " + gpu_bar())
            else:
                print("  " + clr("no CUDA GPU detected",R))
            pause()
        elif ch == "0":
            print("\n  " + clr("bye",C) + "\n"); import sys; sys.exit(0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  " + clr("interrupted, bye",Y) + "\n")
        import sys; sys.exit(0)
    except Exception as ex:
        import traceback
        print("\n  " + clr("fatal: " + str(ex),R) + "\n")
        traceback.print_exc()
        pause("\n  press enter to exit...")
        import sys; sys.exit(1)


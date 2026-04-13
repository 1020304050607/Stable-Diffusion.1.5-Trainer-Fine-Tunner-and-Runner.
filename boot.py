# boot.py
"""Boot and dependency installer"""
import os
import sys
import subprocess
import threading
import time
import re
import warnings
import logging
from pathlib import Path

os.system("")

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
        if can_import(mod): 
            print("  " + clr("ok", G) + "  " + clr(tag, DIM))
        else:               
            miss.append((tag, pip))

    if not miss:
        print("\n  " + clr("all good, loading...", G) + "\n")
        return

    print("\n  " + clr("installing " + str(len(miss)) + " missing package(s)", Y) + "\n")
    for lbl, spec in miss:
        pip_install(lbl, spec.split())

    print("\n  " + clr("done, starting up...", G) + "\n")
    time.sleep(0.6)

boot()

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
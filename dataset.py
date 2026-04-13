# dataset.py
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from boot import device, clr
from utils import IEXT, VEXT, MAX_VFRAMES, CV2

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


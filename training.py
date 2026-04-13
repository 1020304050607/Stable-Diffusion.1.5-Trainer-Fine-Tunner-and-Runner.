import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from boot import sec, clr
from utils import device
from utils import gpu_used, find_latest, load_into, save_st, save_pt
from dataset import LatentCacheDataset, ImageCaptionDataset

# ── fine tune ─────────────────────────────────────────────────────
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


# ── standard train ────────────────────────────────────────────────
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

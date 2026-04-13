from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from boot import device, sec, clr

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


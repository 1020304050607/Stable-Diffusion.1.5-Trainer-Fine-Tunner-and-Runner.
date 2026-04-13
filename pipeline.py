# pipeline.py
import re

from boot import sec, clr, pause
from utils import (
    _pick_data_dir, _pick_out_dir, _pick_model_search,
    _pick_weights, _pick_lora, _pick_existing_lora,
    _get_train_settings, _get_ft_settings, _apply_boosts,
    _ask, _validated_ask
)
from training import run_training, run_finetune
from inference import run_generate

# Shared state
_last = {"dd": "", "out": ""}

# Pipeline constants
PIPE_STEPS = {
    "resize": "Resize Images and Videos",
    "caption": "Auto Caption with BLIP",
    "train": "Standard LoRA Train",
    "finetune": "Fine Tune Dual LoRA",
    "generate": "Generate Images",
}
SKEYS = list(PIPE_STEPS.keys())
SLABS = list(PIPE_STEPS.values())

def flow_pipeline():
    sec("Pipeline Builder")
    print(" " + clr("pick steps separated by commas, e.g. 1,2,4 or 3,5\n", "\033[2m"))
    for i, lab in enumerate(SLABS, 1):
        print(" " + clr(str(i), "\033[96m") + " " + lab)
    print()
    raw = input(" " + clr("steps to run: ", "\033[96m")).strip()
    if not raw:
        print(" " + clr("nothing picked", "\033[93m"))
        pause()
        return

    chosen = []
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(SKEYS):
            k = SKEYS[int(part) - 1]
            if k not in chosen:
                chosen.append(k)
        elif part:
            print(" " + clr("skipping: " + part, "\033[93m"))

    if not chosen:
        print(" " + clr("nothing valid", "\033[91m"))
        pause()
        return

    dd = _pick_data_dir(_last["dd"])
    od = _pick_out_dir(_last["out"])
    _last["dd"] = dd
    _last["out"] = od

    base = {"data_dir": dd, "checkpoint_dir": od}
    cfgs = {}
    for k in chosen:
        print("\n " + clr("setting up: " + PIPE_STEPS[k], "\033[93m"))
        cfgs[k] = _build_step_cfg(k, base)

    sec("Ready")
    for i, k in enumerate(chosen, 1):
        print(" " + clr(str(i), "\033[96m") + " " + PIPE_STEPS[k])
    print()
    input(" " + clr("enter to start the pipeline...", "\033[93m"))

    tot = str(len(chosen))
    for i, k in enumerate(chosen, 1):
        print("\n " + clr("[" + str(i) + "/" + tot + "]", "\033[96m") + " " + clr(PIPE_STEPS[k], "\033[97m") + "...")
        try:
            _run_step(k, cfgs[k])
        except KeyboardInterrupt:
            print("\n\n " + clr("stopped. checkpoints are safe.", "\033[93m"))
            break
        except Exception as ex:
            print("\n " + clr("error in " + k + ": " + str(ex), "\033[91m"))
            if input(" " + clr("keep going? y/n [y]: ", "\033[96m")).strip().lower() == "n":
                break

    print("\n " + clr("pipeline done.", "\033[92m") + "\n")
    pause()


def flow_train(boost=False):
    dd = _pick_data_dir(_last["dd"])
    od = _pick_out_dir(_last["out"])
    _last["dd"] = dd
    _last["out"] = od
    mid, prof = _pick_model_search()
    wts = _pick_weights(od)
    settings = _get_train_settings(od, prof)
    cfg = dict(model_id=mid, weights_path=wts, data_dir=dd,
               checkpoint_dir=od, _profile=prof, **settings)
    if boost:
        cfg = _apply_boosts(cfg)
    _confirm_run(cfg, "train")


def flow_finetune():
    dd = _pick_data_dir(_last["dd"])
    od = _pick_out_dir(_last["out"])
    _last["dd"] = dd
    _last["out"] = od
    mid, prof = _pick_model_search()
    elo = _pick_existing_lora(od)
    settings = _get_ft_settings(od, prof)
    cfg = dict(model_id=mid, existing_lora=elo, data_dir=dd,
               checkpoint_dir=od, _profile=prof, **settings)
    _confirm_run(cfg, "finetune")


def flow_generate():
    mid, _ = _pick_model_search()
    lp = _pick_lora([_last["out"]])
    outd = _pick_out_dir(os.path.join(_last["out"], "generated") if _last["out"] else "",
                         label="Where to Save Images")
    sec("Generation Settings")
    prompt = ""
    while not prompt:
        prompt = input(" " + clr("prompt: ", "\033[96m")).strip()
        if not prompt:
            print(" " + clr("required", "\033[91m"))
    cfg = dict(model_id=mid, lora_path=lp, output_dir=outd, prompt=prompt,
               num_images=_ask("how many images", 4, int),
               steps=_ask("steps", 30, int),
               guidance=_ask("guidance scale", 7.5, float))
    _confirm_run(cfg, "generate")


def flow_resize():
    dd = _pick_data_dir(_last["dd"])
    _last["dd"] = dd
    sec("Resize Settings")
    w = _ask("width", 512, int)
    h = _ask("height", 512, int)
    sec("Ready")
    print(" " + clr("folder", "\033[2m") + " " + dd)
    print(" " + clr("size ", "\033[2m") + str(w) + "x" + str(h))
    input("\n " + clr("enter to start...", "\033[93m"))
    try:
        run_resize({"data_dir": dd, "resize_w": w, "resize_h": h})
    except KeyboardInterrupt:
        print("\n\n " + clr("stopped", "\033[93m"))
    pause()


def flow_caption():
    dd = _pick_data_dir(_last["dd"])
    _last["dd"] = dd
    batch = _ask("batch size", 4, int)
    input("\n " + clr("enter to start...", "\033[93m"))
    try:
        run_caption({"data_dir": dd, "caption_batch": batch})
    except KeyboardInterrupt:
        print("\n\n " + clr("stopped", "\033[93m"))
    pause()


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
        wts = _pick_weights(cfg["checkpoint_dir"])
        settings = _get_train_settings(cfg["checkpoint_dir"], prof)
        cfg.update(dict(model_id=mid, weights_path=wts, _profile=prof, **settings))
    elif key == "finetune":
        mid, prof = _pick_model_search()
        elo = _pick_existing_lora(cfg["checkpoint_dir"])
        settings = _get_ft_settings(cfg["checkpoint_dir"], prof)
        cfg.update(dict(model_id=mid, existing_lora=elo, _profile=prof, **settings))
    elif key == "generate":
        mid, _ = _pick_model_search()
        lp = _pick_lora([cfg["checkpoint_dir"]])
        outd = _pick_out_dir(os.path.join(cfg["checkpoint_dir"], "generated"), label="Where to Save Images")
        sec("Generation Settings")
        prompt = ""
        while not prompt:
            prompt = input(" " + clr("prompt: ", "\033[96m")).strip()
            if not prompt:
                print(" " + clr("required", "\033[91m"))
        cfg.update(dict(model_id=mid, lora_path=lp, output_dir=outd, prompt=prompt,
                        num_images=_ask("how many images", 4, int),
                        steps=_ask("steps", 30, int),
                        guidance=_ask("guidance scale", 7.5, float)))
    return cfg


def _run_step(key, cfg):
    if key == "resize":
        run_resize(cfg)
    elif key == "caption":
        run_caption(cfg)
    elif key == "train":
        run_training(cfg)
    elif key == "finetune":
        run_finetune(cfg)
    elif key == "generate":
        run_generate(cfg)


def _confirm_run(cfg, mode):
    sec("Ready to Start")
    for k, v in cfg.items():
        if v not in (None, "") and not k.startswith("_"):
            print(" " + clr(k, "\033[2m") + " = " + clr(str(v), "\033[97m"))
    input("\n " + clr("enter to start...", "\033[93m"))
    try:
        if mode == "finetune":
            run_finetune(cfg)
        elif mode == "generate":
            run_generate(cfg)
        else:
            run_training(cfg)
    except KeyboardInterrupt:
        print("\n\n " + clr("stopped. checkpoint saved.", "\033[93m"))
    pause()
# main.py
import multiprocessing
import sys
import traceback
import glob
import torch

from boot import boot, clr, sec, device
from utils import (
    print_header, show_tutorial, show_vram_table, 
    gpu_bar, find_latest, IEXT
)
from editor import terminal_image_editor
from pipeline import flow_pipeline, flow_train, flow_finetune, flow_generate, flow_resize, flow_caption
from comfy import install_comfyui

_last = {"dd": "", "out": ""}

def wrap(fn):
    try:
        fn()
    except Exception as ex:
        print("\n " + clr("error: " + str(ex), "\033[91m"))
        traceback.print_exc()
        input("\n press enter to continue...")

def main_menu():
    while True:
        print_header()

        ni = 0
        if _last["dd"]:
            ni = sum(len(glob.glob(os.path.join(_last["dd"], "*." + e))) for e in IEXT)
        _, step = find_latest(_last["out"]) if _last["out"] else (None, 0)

        print(" " + clr("data ", "\033[2m") + (clr(_last["dd"], "\033[97m") if _last["dd"] else clr("not set", "\033[2m")))
        if _last["dd"]:
            print(" " + clr("images ", "\033[2m") + clr(str(ni), "\033[97m"))
        print(" " + clr("output ", "\033[2m") + (clr(_last["out"], "\033[97m") if _last["out"] else clr("not set", "\033[2m")))
        if _last["out"]:
            print(" " + clr("ckpt ", "\033[2m") + (clr("step " + str(step), "\033[92m") if step else clr("none", "\033[2m")))

        print("\n " + clr("─" * 50, "\033[2m"))
        print(" " + clr("1", "\033[96m") + " Pipeline Builder")
        print(" " + clr("─" * 50, "\033[2m"))
        print(" " + clr("2", "\033[96m") + " Resize")
        print(" " + clr("3", "\033[96m") + " Caption")
        print(" " + clr("4", "\033[96m") + " Train standard LoRA")
        print(" " + clr("5", "\033[96m") + " Train + Boosts")
        print(" " + clr("6", "\033[96m") + " Fine Tune")
        print(" " + clr("7", "\033[96m") + " Generate")
        print(" " + clr("─" * 50, "\033[2m"))
        print(" " + clr("8", "\033[96m") + " ComfyUI Setup")
        print(" " + clr("9", "\033[96m") + " Image Editor")
        print(" " + clr("─" * 50, "\033[2m"))
        print(" " + clr("t", "\033[96m") + " Tutorial")
        print(" " + clr("v", "\033[96m") + " VRAM Info")
        print(" " + clr("g", "\033[96m") + " GPU Presets")
        print(" " + clr("0", "\033[96m") + " Exit")
        print()

        ch = input(" " + clr(">", "\033[96m") + " ").strip().lower()

        if ch == "1":   wrap(flow_pipeline)
        elif ch == "2": wrap(flow_resize)
        elif ch == "3": wrap(flow_caption)
        elif ch == "4": wrap(lambda: flow_train(False))
        elif ch == "5": wrap(lambda: flow_train(True))
        elif ch == "6": wrap(flow_finetune)
        elif ch == "7": wrap(flow_generate)
        elif ch == "8":
            try: install_comfyui()
            except: pass
        elif ch == "9": wrap(terminal_image_editor)
        elif ch == "t": show_tutorial()
        elif ch == "g":
            show_vram_table()
            input("\n press enter...")
        elif ch == "v":
            sec("Current GPU Usage")
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                tot = p.total_memory / 1024**3
                fr = torch.cuda.mem_get_info()[0] / 1024**3
                print(" " + clr("GPU ", "\033[2m") + p.name)
                print(" " + clr("total ", "\033[2m") + str(round(tot,1)) + "GB")
                print(" " + clr("used ", "\033[2m") + str(round(tot-fr,1)) + "GB")
                print(" " + clr("free ", "\033[2m") + str(round(fr,1)) + "GB")
                print("\n " + gpu_bar())
            else:
                print(" " + clr("no CUDA GPU detected", "\033[91m"))
            input("\n press enter...")
        elif ch == "0":
            print("\n " + clr("bye", "\033[96m") + "\n")
            sys.exit(0)
        else:
            print(" " + clr("invalid choice", "\033[93m"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    boot()
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n " + clr("interrupted, bye", "\033[93m"))
        sys.exit(0)
    except Exception as ex:
        print("\n " + clr("fatal: " + str(ex), "\033[91m"))
        traceback.print_exc()
        input("\n press enter to exit...")
        sys.exit(1)
# comfy.py
import subprocess
import os
import re

from boot import sec, clr, pip_install, pause

CNODES = [
    ("ComfyUI Manager", "https://github.com/ltdrdata/ComfyUI-Manager"),
    ("Impact Pack", "https://github.com/ltdrdata/ComfyUI-Impact-Pack"),
    ("IP Adapter Plus", "https://github.com/cubiq/ComfyUI_IPAdapter_plus"),
    ("ControlNet Preprocessors", "https://github.com/Fannovel16/comfyui_controlnet_aux"),
    ("rgthree comfy", "https://github.com/rgthree/rgthree-comfy"),
    ("KJNodes", "https://github.com/kijai/ComfyUI-KJNodes"),
    ("ComfyUI Easy Use", "https://github.com/yolain/ComfyUI-Easy-Use"),
    ("ReActor Node", "https://github.com/Gourieff/comfyui-reactor-node"),
    ("InstantID FaceSwap", "https://github.com/nosiu/comfyui-instantId-faceswap"),
    ("Portrait Master v3", "https://github.com/florestefano1975/comfyui-portrait-master"),
    ("Ultimate SD Upscale", "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"),
    ("SUPIR", "https://github.com/kijai/ComfyUI-SUPIR"),
    ("DeepFuze", "https://github.com/SamKhoze/ComfyUI-DeepFuze"),
]

CFACE = [
    ("onnxruntime", "onnxruntime-gpu", None),
    ("basicsr", "basicsr", None),
    ("facexlib", "facexlib", None),
    ("realesrgan", "realesrgan", None),
    ("insightface", "insightface",
     "needs Visual C++ Build Tools on Windows.\n"
     " get them at: visualstudio.microsoft.com/visual-cpp-build-tools\n"
     " install the C++ workload, restart, try again."),
]

def _git_clone(name, url, parent):
    folder = url.rstrip("/").split("/")[-1]
    dest = os.path.join(parent, folder)
    if os.path.isdir(dest):
        print(" " + clr("already there ", G) + name)
        return True
    print(" " + clr("cloning ", C) + name)
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", url, dest],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(" " + clr("done ", G) + name)
            return True
        print(" " + clr("failed ", R) + name + " " + clr(r.stderr.strip()[:60], DIM))
        return False
    except FileNotFoundError:
        print(" " + clr("git not found. install it from git-scm.com", R))
        return False
    except Exception as ex:
        print(" " + clr("error ", R) + name + " " + str(ex))
        return False

def install_comfyui():
    sec("ComfyUI Auto Installer")
    print(" " + clr("installs ComfyUI and all essential node packs.", DIM))
    print(" " + clr("anything already installed gets skipped. safe to re-run.\n", DIM))

    default = os.path.join(os.path.expanduser("~"), "ComfyUI")
    raw = input(" " + clr("install location [" + default + "]: ", C)).strip().strip('"').strip("'")
    root = raw if raw else default
    cdir = os.path.join(root, "ComfyUI")
    ndir = os.path.join(cdir, "custom_nodes")

    # Step 1: pip packages
    sec("Step 1/3 pip packages")
    print(" " + clr("face tools and upscaling deps\n", DIM))
    for mod, pkg, warn in CFACE:
        try:
            __import__(mod)
            print(" " + clr("ok ", G) + pkg)
        except ImportError:
            ok = pip_install(pkg, [pkg])
            if not ok and mod == "insightface":
                print(" " + clr("trying pre-built wheel...", Y))
                wheel = "https://github.com/Gourieff/Assets/raw/main/insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
                ok2 = pip_install("insightface-wheel", [wheel])
                if not ok2 and warn:
                    for ln in warn.split("\n"):
                        if ln.strip(): print(" " + clr(ln, Y))
            elif not ok and warn:
                for ln in warn.split("\n"):
                    if ln.strip(): print(" " + clr(ln, Y))

    # Step 2: ComfyUI core
    sec("Step 2/3 ComfyUI core")
    os.makedirs(root, exist_ok=True)
    if os.path.isdir(cdir):
        print(" " + clr("already installed at " + cdir, G))
    else:
        print(" " + clr("cloning ComfyUI into " + root + "...", C))
        if not _git_clone("ComfyUI", "https://github.com/comfyanonymous/ComfyUI", root):
            print(" " + clr("clone failed. check git and internet.", R))
            pause()
            return
        req = os.path.join(cdir, "requirements.txt")
        if os.path.exists(req):
            print(" " + clr("installing ComfyUI requirements...", C) + "\n")
            skip = {"torch", "torchvision", "torchaudio"}
            with open(req, "r", encoding="utf-8") as fh:
                reqs = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]
            for r in reqs:
                bare = re.split(r"[>=<!;\[]", r)[0].strip().lower().replace("-", "_")
                if bare in skip:
                    print(" " + clr("skip ", DIM) + r)
                    continue
                try:
                    __import__(bare)
                    print(" " + clr("ok ", G) + r)
                except:
                    pip_install(r, [r])

    # Step 3: Node packs
    sec("Step 3/3 Node Packs")
    os.makedirs(ndir, exist_ok=True)
    print(" " + clr("cloning " + str(len(CNODES)) + " packs into " + ndir + "\n", DIM))
    ok_n = fail_n = 0
    for name, url in CNODES:
        if _git_clone(name, url, ndir):
            ok_n += 1
        else:
            fail_n += 1

    guide = os.path.join(root, "HOW_TO_START.txt")
    with open(guide, "w", encoding="utf-8") as fh:
        fh.write("ComfyUI installed at:\n " + cdir + "\n\n")
        fh.write("To start:\n cd " + cdir + "\n python main.py\n\n")
        fh.write("Then open:\n http://127.0.0.1:8188\n\n")
        fh.write("Drag any .json workflow into the browser to load it.\n")
        fh.write("Use Manager node to install more packs.\n")

    print("\n " + clr("startup guide -> " + guide, DIM))
    print("\n " + clr("done! " + str(ok_n) + " packs installed" + 
                      (" " + str(fail_n) + " failed" if fail_n else ""), G))
    print("\n " + clr("to start: cd " + cdir + " && python main.py", W))
    print(" " + clr("then: http://127.0.0.1:8188", DIM) + "\n")
    pause()
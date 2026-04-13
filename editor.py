from PIL import Image, ImageFilter, ImageEnhance
import glob
import os
from pathlib import Path

from boot import sec, clr, pause
from utils import IEXT

def terminal_image_editor(cfg=None):
    """
    Pure terminal image editing suite.
    Face swap / restore / img2img / filters.
    Ctrl+U uploads a new image at any prompt.
    """
    import msvcrt  # will fail on linux - we handle that
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
            # face swap using insightface + inswapper
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
            # face restore using GFPGAN or CodeFormer
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
                    # basic PIL face-area sharpen
                    out_img = dst_img.filter(ImageFilter.SHARPEN)
                    out_img = ImageEnhance.Sharpness(out_img).enhance(2.0)
                    out_img = ImageEnhance.Contrast(out_img).enhance(1.1)
                save_result(out_img, dst_path, "restored")
                print("  " + clr("restore done", G))
            except Exception as ex:
                print("  " + clr("restore failed: " + str(ex), R))
            pause()

        elif ch == "3":
            # img2img in terminal
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
                # keep aspect ratio, round to 64
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
            # basic PIL filters
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
            # batch face restore
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
            # resize single image
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


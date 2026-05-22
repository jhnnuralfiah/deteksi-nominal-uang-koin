import cv2
import numpy as np
from pathlib import Path

CONFIG = {
    "input_dir":       "Dataset",
    "output_dir":      "output_preprocessing",
    "target_size":     (512, 512),
    "blur_ksize":      (5, 5),
    "blur_sigma":      0,
    "supported_ext":   {".jpg", ".jpeg", ".png", ".bmp"},
    "save_comparison": True,
}

def pad_to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    max_side = max(h, w)
    if len(image.shape) == 3:
        canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((max_side, max_side), dtype=np.uint8)
    top  = (max_side - h) // 2
    left = (max_side - w) // 2
    canvas[top:top+h, left:left+w] = image
    return canvas

def resize_image(image, size):
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (w > size[0] or h > size[1]) else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interp)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def gaussian_blur(image, ksize, sigma):
    return cv2.GaussianBlur(image, ksize, sigma)

def preprocess(image_path, cfg):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa baca: {image_path}")
    padded  = pad_to_square(img)
    resized = resize_image(padded, cfg["target_size"])
    gray    = to_grayscale(resized)
    blurred = gaussian_blur(gray, cfg["blur_ksize"], cfg["blur_sigma"])
    return {
        "original": img,
        "padded":   padded,
        "resized":  resized,
        "gray":     gray,
        "final":    blurred,
    }

def save_comparison(stages, out_path):
    size = (256, 256)
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    panels = [
        cv2.resize(to_bgr(stages["original"]), size),
        cv2.resize(to_bgr(stages["padded"]),   size),
        cv2.resize(to_bgr(stages["resized"]),  size),
        cv2.resize(to_bgr(stages["gray"]),     size),
        cv2.resize(to_bgr(stages["final"]),    size),
    ]
    labels = ["Original", "Padding", "Resize", "Grayscale", "GaussBlur(Final)"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(out_path, np.hstack(panels))

def run(cfg):
    input_root  = Path(cfg["input_dir"])
    output_root = Path(cfg["output_dir"])

    if not input_root.exists():
        print(f"[ERROR] Folder '{input_root}' tidak ditemukan!")
        return

    all_images = []
    for ext in cfg["supported_ext"]:
        all_images += list(input_root.rglob(f"*{ext}"))
        all_images += list(input_root.rglob(f"*{ext.upper()}"))
    all_images = list(set(all_images))   # ← fix duplikat

    if not all_images:
        print("[WARNING] Tidak ada gambar ditemukan!")
        return

    if output_root.exists():
        for old_file in output_root.rglob("*"):
            if old_file.is_file():
                old_file.unlink()
        print("  [INFO] File lama dihapus.\n")

    print(f"{'='*50}")
    print(f"  PREPROCESSING KOIN RUPIAH")
    print(f"  Alur : Padding → Resize → Grayscale → GaussBlur")
    print(f"  Total: {len(all_images)} gambar")
    print(f"{'='*50}\n")

    ok, fail = 0, 0

    for img_path in sorted(all_images):
        relative = img_path.relative_to(input_root)
        out_dir  = output_root / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            stages = preprocess(str(img_path), cfg)
            cv2.imwrite(str(out_dir / img_path.name), stages["final"])
            if cfg["save_comparison"]:
                save_comparison(stages, str(out_dir / f"_cmp_{img_path.stem}.jpg"))
            print(f"  [OK] {relative}")
            ok += 1
        except Exception as e:
            print(f"  [GAGAL] {relative} — {e}")
            fail += 1

    print(f"\n  Selesai : {ok} berhasil, {fail} gagal")
    print(f"  Output  : {output_root.resolve()}\n")

if __name__ == "__main__":
    run(CONFIG)
import cv2
import numpy as np
from pathlib import Path

CONFIG = {
    # Nama folder sesuai dengan screenshot milikmu
    "input_dir":     "Dataset",             
    "output_dir":    "output_preprocessing", 
    "supported_ext": {".jpg", ".jpeg", ".png", ".bmp"},
    
    # KUNCI UTAMA: Diset 'None' agar ukuran gambar TIDAK BERUBAH. 
    # Koin dijamin akan tetap terdeteksi oleh file deteksimu.
    "target_width":  None, 
    "blur_kernel":   (7, 7)                 
}

def preprocess_image(img_path, cfg):
    # Membaca gambar asli
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError("Gambar tidak dapat dibaca atau rusak.")
        
    # 1. Konversi ke Grayscale (Hitam Putih)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Skip Resize (Karena target_width = None, gambar dibiarkan ukuran asli)
    if cfg["target_width"] is not None:
        h, w = gray.shape[:2]
        if w > cfg["target_width"]: 
            scale = cfg["target_width"] / w
            target_height = int(h * scale)
            gray = cv2.resize(gray, (cfg["target_width"], target_height), interpolation=cv2.INTER_AREA)
    
    # 3. Smoothing / Blurring (Untuk membuang noise/bintik pada foto)
    blurred = cv2.GaussianBlur(gray, cfg["blur_kernel"], 1.5)
    
    return blurred

def run(cfg):
    input_root  = Path(cfg["input_dir"])
    output_root = Path(cfg["output_dir"])
    
    if not input_root.exists():
        print(f"[ERROR] Folder input '{cfg['input_dir']}' tidak ditemukan!")
        return

    all_images = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in cfg["supported_ext"]
    ]
    
    if not all_images:
        print(f"[WARNING] Tidak ada gambar di dalam folder '{cfg['input_dir']}'")
        return
        
    print(f"\n{'='*55}")
    print(f"  PROSES PREPROCESSING GAMBAR (UKURAN ASLI)")
    print(f"  Total Gambar Ditemukan: {len(all_images)}")
    print(f"{'='*55}\n")
    
    success_count = 0
    fail_count = 0
    
    for img_path in sorted(all_images):
        relative_path = img_path.relative_to(input_root)
        out_file_path = output_root / relative_path
        
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            processed_img = preprocess_image(img_path, cfg)
            cv2.imwrite(str(out_file_path), processed_img)
            print(f"  [OK] {relative_path}")
            success_count += 1
        except Exception as e:
            print(f"  [GAGAL] {relative_path} — Alasan: {e}")
            fail_count += 1
            
    print(f"\n{'='*55}")
    print(f"  Selesai! Berhasil: {success_count} gambar, Gagal: {fail_count} gambar.")
    print(f"  Hasil disimpan di folder: {output_root.resolve()}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    run(CONFIG)
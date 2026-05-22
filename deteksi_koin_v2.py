import cv2
import numpy as np
import csv
import sys
from pathlib import Path

CONFIG = {
    "input_dir":     "output_preprocessing", 
    "output_dir":    "output_deteksi",
    "csv_output":    "output_deteksi/data_radius_koin.csv",
    "supported_ext": {".jpg", ".jpeg", ".png", ".bmp"},
    
    # PARAMETER SUPER CEPAT (Sudah disesuaikan untuk gambar yang di-resize)
    "target_width":  800,  # Gambar otomatis dikecilkan ke 800px agar super cepat
    "dp":            1,    
    "min_dist":      100,  
    "param1":        50,   
    "param2":        30,   # Diturunkan sedikit agar lebih sensitif mendeteksi
    "min_radius":    20,   # Radius kecil karena gambar sudah di-resize
    "max_radius":    250   
}

def detect_fast(cfg):
    input_root = Path(cfg["input_dir"])
    output_root = Path(cfg["output_dir"])
    
    if not input_root.exists():
        print(f"[ERROR] Folder '{cfg['input_dir']}' tidak ditemukan.")
        return
        
    output_root.mkdir(parents=True, exist_ok=True)
    
    all_images = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in cfg["supported_ext"]]
    total_files = len(all_images)
    
    print(f"\n{'='*60}")
    print(f"  PROSES DETEKSI KOIN SUPER CEPAT (RESIZE OTOMATIS)")
    print(f"  Total Gambar: {total_files}")
    print(f"{'='*60}\n")
    
    with open(cfg["csv_output"], mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Nama_File", "Label", "Radius_Piksel"])
        
        success_count = 0
        
        for idx, img_path in enumerate(sorted(all_images), 1):
            label = img_path.parent.name.replace("koin_", "") 
            print(f"  [{idx}/{total_files}] {img_path.name[:15]:<15} ... ", end="")
            sys.stdout.flush() 
            
            # 1. Baca gambar hasil preprocessing
            gray_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            # 2. Baca gambar asli untuk visualisasi (atau pakai gray jika gagal)
            original_path = Path("Dataset") / img_path.relative_to(input_root)
            img_visual = cv2.imread(str(original_path))
            if img_visual is None:
                img_visual = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            # 3. RESIZE ON THE FLY (Ini kunci utamanya biar ngebut!)
            h, w = gray_img.shape
            scale = cfg["target_width"] / w
            target_h = int(h * scale)
            
            gray_resized = cv2.resize(gray_img, (cfg["target_width"], target_h), interpolation=cv2.INTER_AREA)
            visual_resized = cv2.resize(img_visual, (cfg["target_width"], target_h), interpolation=cv2.INTER_AREA)
            
            # 4. Deteksi lingkaran pada gambar yang sudah ringan
            circles = cv2.HoughCircles(
                gray_resized, 
                cv2.HOUGH_GRADIENT, 
                dp=cfg["dp"], 
                minDist=cfg["min_dist"],
                param1=cfg["param1"], 
                param2=cfg["param2"], 
                minRadius=cfg["min_radius"], 
                maxRadius=cfg["max_radius"]
            )
            
            radius_terdeteksi = "-"
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :1]:  
                    x, y, r = i[0], i[1], i[2]
                    radius_terdeteksi = r
                    
                    # Gambar lingkaran di atas gambar yang sudah di-resize
                    cv2.circle(visual_resized, (x, y), r, (0, 255, 0), 3)
                    cv2.circle(visual_resized, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(visual_resized, f"R: {r}px", (x - 40, y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                success_count += 1
                print(f"TERDETEKSI ({radius_terdeteksi}px)")
            else:
                print(f"LEWAT (Tidak ketemu)")
            
            # Simpan hasil
            csv_writer.writerow([img_path.name, label, radius_terdeteksi])
            out_file_path = output_root / img_path.relative_to(input_root)
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_file_path), visual_resized)
            
    print(f"\n{'='*60}")
    print(f"  SELESAI NGEBUT! Berhasil: {success_count}/{total_files} gambar.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    detect_fast(CONFIG)
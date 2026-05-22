import cv2
import numpy as np
import csv
from pathlib import Path

CONFIG = {
    "input_dir":     "output_preprocessing",
    "output_dir":    "output_deteksi",
    "supported_ext": {".jpg", ".jpeg", ".png", ".bmp"},
    "margin":        60,

    "cht": {
        "dp":        1,
        "minDist":   100,
        "param1":    80,
        "param2":    32,
        "minRadius": 55,
        "maxRadius": 80,
    }
}

def detect_circles(image_gray, cfg_cht):
    circles = cv2.HoughCircles(
        image_gray,
        cv2.HOUGH_GRADIENT,
        dp        = cfg_cht["dp"],
        minDist   = cfg_cht["minDist"],
        param1    = cfg_cht["param1"],
        param2    = cfg_cht["param2"],
        minRadius = cfg_cht["minRadius"],
        maxRadius = cfg_cht["maxRadius"],
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
    return circles

def draw_circles(image_gray, circles):
    output = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    if circles is None:
        return output
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(output, f"#{i+1} r={r}px", (x - r, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return output

def extract_features(circles, image_name, label):
    features = []
    if circles is None:
        return features
    for i, (x, y, r) in enumerate(circles):
        features.append({
            "file":        image_name,
            "label":       label,
            "koin_ke":     i + 1,
            "center_x":    x,
            "center_y":    y,
            "radius_px":   r,
            "diameter_px": r * 2,
        })
    return features

def run(cfg):
    input_root  = Path(cfg["input_dir"])
    output_root = Path(cfg["output_dir"])

    if not input_root.exists():
        print(f"[ERROR] Folder '{input_root}' tidak ditemukan!")
        return

    # Kumpulkan semua gambar + deduplikasi
    all_images = []
    for ext in cfg["supported_ext"]:
        all_images += [p for p in input_root.rglob(f"*{ext}")
                       if not p.name.startswith("_cmp_")]
        all_images += [p for p in input_root.rglob(f"*{ext.upper()}")
                       if not p.name.startswith("_cmp_")]
    all_images = list(set(all_images))  # ← deduplikasi path

    if not all_images:
        print("[WARNING] Tidak ada gambar ditemukan!")
        return

    # Bersihkan output lama
    if output_root.exists():
        for old_file in output_root.rglob("*"):
            if old_file.is_file():
                old_file.unlink()
        print("  [INFO] Output lama dihapus.\n")

    print(f"\n{'='*55}")
    print(f"  DETEKSI LINGKARAN - Circular Hough Transform")
    print(f"  Total  : {len(all_images)} gambar")
    print(f"{'='*55}\n")

    all_features           = []
    ok, fail, not_detected = 0, 0, 0

    for img_path in sorted(all_images):
        relative = img_path.relative_to(input_root)
        label    = img_path.parent.name
        out_dir  = output_root / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise FileNotFoundError("Tidak bisa dibaca")

            h, w     = img_gray.shape[:2]
            m        = cfg["margin"]
            img_gray = img_gray[m:h-m, m:w-m]

            circles    = detect_circles(img_gray, cfg["cht"])
            result_img = draw_circles(img_gray, circles)
            cv2.imwrite(str(out_dir / img_path.name), result_img)

            features = extract_features(circles, img_path.name, label)
            all_features.extend(features)

            jumlah = len(circles) if circles is not None else 0
            if jumlah == 0:
                print(f"  [!] {relative} — tidak terdeteksi")
                not_detected += 1
            else:
                print(f"  [OK] {relative} — {jumlah} koin terdeteksi")
            ok += 1

        except Exception as e:
            print(f"  [GAGAL] {relative} — {e}")
            fail += 1

    # ── Hapus duplikat sebelum simpan CSV ──────────────
    seen = set()
    unique_features = []
    for row in all_features:
        key = (row["file"], row["label"], row["koin_ke"],
               row["center_x"], row["center_y"])
        if key not in seen:
            seen.add(key)
            unique_features.append(row)
    # ───────────────────────────────────────────────────

    # Simpan CSV bersih
    csv_path = output_root / "hasil_fitur.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["file", "label", "koin_ke", "center_x", "center_y",
                      "radius_px", "diameter_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_features)

    print(f"\n{'='*55}")
    print(f"  Selesai        : {ok} diproses, {fail} gagal")
    print(f"  Tdk terdeteksi : {not_detected} gambar")
    print(f"  Total koin     : {len(unique_features)} baris di CSV")
    print(f"  CSV            : {csv_path.resolve()}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    run(CONFIG)
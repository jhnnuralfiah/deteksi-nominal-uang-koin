import cv2
import numpy as np
import csv
from pathlib import Path

CONFIG = {
    "csv_input":     "output_deteksi/hasil_fitur.csv",
    "dataset_dir":   "Dataset",        # foto ASLI
    "output_dir":    "output_ekstraksi",
    "target_size":   512,              # harus sama dengan preprocessing
    "margin":        60,               # harus sama dengan deteksi_lingkaran
}

# ─────────────────────────────────────────────
#  HITUNG FITUR GEOMETRI
# ─────────────────────────────────────────────

def hitung_fitur(radius_px):
    r        = radius_px
    diameter = r * 2
    luas     = round(np.pi * r ** 2, 2)
    keliling = round(2 * np.pi * r, 2)
    return diameter, luas, keliling


# ─────────────────────────────────────────────
#  KONVERSI KOORDINAT CSV → FOTO ASLI
# ─────────────────────────────────────────────

def konversi_ke_asli(cx_csv, cy_csv, r_csv, h_asli, w_asli, target_size, margin):
    """
    Koordinat di CSV berasal dari gambar yang sudah melalui:
      1. Padding hitam → jadi kotak max_side x max_side
      2. Resize → jadi target_size x target_size (512x512)
      3. Crop margin → jadi (512-2*margin) x (512-2*margin)

    Fungsi ini membalik semua transformasi itu untuk mendapatkan
    koordinat yang tepat di foto asli.
    """
    max_side   = max(h_asli, w_asli)

    # Offset padding (foto asli ditempel di tengah kanvas kotak)
    off_x = (max_side - w_asli) / 2
    off_y = (max_side - h_asli) / 2

    # Skala dari target_size ke max_side
    scale = max_side / target_size

    # Balik: tambah margin dulu (crop margin dibalik)
    # lalu skala ke max_side, lalu kurangi offset padding
    cx_asli = int((cx_csv + margin) * scale - off_x)
    cy_asli = int((cy_csv + margin) * scale - off_y)
    r_asli  = int(r_csv * scale)

    return cx_asli, cy_asli, r_asli


# ─────────────────────────────────────────────
#  GAMBAR ANOTASI DI FOTO ASLI
# ─────────────────────────────────────────────

def gambar_anotasi(img_asli, circles_data, target_size, margin):
    output = img_asli.copy()
    h_asli, w_asli = img_asli.shape[:2]

    for row in circles_data:
        cx_csv = int(row["center_x"])
        cy_csv = int(row["center_y"])
        r_csv  = int(row["radius_px"])

        # Konversi koordinat ke foto asli
        cx, cy, r = konversi_ke_asli(
            cx_csv, cy_csv, r_csv,
            h_asli, w_asli, target_size, margin
        )

        diameter, luas, keliling = hitung_fitur(r_csv)

        # Lingkaran hijau
        cv2.circle(output, (cx, cy), r, (0, 255, 0), 3)

        # Titik tengah merah
        cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

        # Label diameter di atas lingkaran
        cv2.putText(output,
                    f"#{row['koin_ke']} d={diameter}px",
                    (cx - r, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Label luas + keliling di bawah lingkaran
        cv2.putText(output,
                    f"L={luas} K={keliling}",
                    (cx - r, cy + r + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    return output


# ─────────────────────────────────────────────
#  BATCH PROCESSING
# ─────────────────────────────────────────────

def run(cfg):
    csv_path    = Path(cfg["csv_input"])
    dataset_dir = Path(cfg["dataset_dir"])
    output_root = Path(cfg["output_dir"])

    if not csv_path.exists():
        print(f"[ERROR] CSV tidak ditemukan: {csv_path}")
        print(f"        Jalankan deteksi_lingkaran.py terlebih dahulu.")
        return

    # Baca CSV hasil deteksi
    data_per_file = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["file"], row["label"])
            if key not in data_per_file:
                data_per_file[key] = []
            data_per_file[key].append(row)

    if not data_per_file:
        print("[WARNING] CSV kosong!")
        return

    # Bersihkan output lama
    if output_root.exists():
        for old_file in output_root.rglob("*"):
            if old_file.is_file():
                old_file.unlink()
        print("  [INFO] Output lama dihapus.\n")

    print(f"\n{'='*55}")
    print(f"  EKSTRAKSI FITUR - Bounding Circle di Foto Asli")
    print(f"  Input CSV : {csv_path.resolve()}")
    print(f"  Dataset   : {dataset_dir.resolve()}")
    print(f"  Output    : {output_root.resolve()}")
    print(f"  Total file: {len(data_per_file)}")
    print(f"{'='*55}\n")

    all_results = []
    ok, fail    = 0, 0

    for (fname, label), circles_data in sorted(data_per_file.items()):
        # Cari foto asli
        img_path = dataset_dir / label / fname
        if not img_path.exists():
            print(f"  [!] Foto asli tidak ditemukan: {img_path}")
            fail += 1
            continue

        try:
            img_asli = cv2.imread(str(img_path))
            if img_asli is None:
                raise FileNotFoundError("Tidak bisa dibaca")

            # Gambar anotasi di foto asli
            result_img = gambar_anotasi(
                img_asli, circles_data,
                cfg["target_size"], cfg["margin"]
            )

            # Simpan hasil visual
            out_dir = output_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / fname), result_img)

            # Kumpulkan fitur untuk CSV
            for row in circles_data:
                r = int(row["radius_px"])
                diameter, luas, keliling = hitung_fitur(r)
                all_results.append({
                    "file":        fname,
                    "label":       label,
                    "koin_ke":     row["koin_ke"],
                    "center_x":    row["center_x"],
                    "center_y":    row["center_y"],
                    "radius_px":   r,
                    "diameter_px": diameter,
                    "luas_px2":    luas,
                    "keliling_px": keliling,
                })

            print(f"  [OK] {label}/{fname} — {len(circles_data)} koin")
            ok += 1

        except Exception as e:
            print(f"  [GAGAL] {label}/{fname} — {e}")
            fail += 1

    # Simpan CSV lengkap
    csv_out = output_root / "hasil_ekstraksi_fitur.csv"
    with open(csv_out, "w", newline="") as f:
        fieldnames = ["file", "label", "koin_ke", "center_x", "center_y",
                      "radius_px", "diameter_px", "luas_px2", "keliling_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*55}")
    print(f"  Selesai  : {ok} berhasil, {fail} gagal")
    print(f"  Total    : {len(all_results)} koin terekstrak")
    print(f"  CSV fitur: {csv_out.resolve()}")
    print(f"  Visual   : {output_root.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run(CONFIG)
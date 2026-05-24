import cv2
import numpy as np
import csv
from pathlib import Path

CONFIG = {
    "csv_input":   "output_ekstraksi/hasil_ekstraksi_fitur.csv",
    "dataset_dir": "Dataset",
    "output_dir":  "output_klasifikasi",
    "target_size": 512,
    "margin":      60,
}

# ─────────────────────────────────────────────
#  RULE-BASED CLASSIFICATION
# ─────────────────────────────────────────────

def klasifikasi_koin(diameter_px):
    """
    Threshold berdasarkan analisis distribusi diameter aktual:

    koin_100  → 92-110px,   mayoritas 104-106px
    koin_1000 → 110-118px,  mayoritas 112-114px
    koin_200  → 110-122px,  mayoritas 114-120px
    koin_500  → 116-132px,  mayoritas 124-130px

    Batas optimal:
      >= 122px  → Rp500   (500 mulai dari 122px ke atas)
      >= 115px  → Rp200   (mayoritas 200 ada di sini)
      >= 110px  → Rp1000  (koin 1000 dimulai dari rentang ini)
      < 110px   → Rp100   (hampir seluruh koin 100 berada di bawah 110px)
    """
    if diameter_px >= 122:
        return "Rp500"
    elif diameter_px >= 115:
        return "Rp200"
    elif diameter_px >= 110:
        return "Rp1000"
    else:
        return "Rp100"


# ─────────────────────────────────────────────
#  KONVERSI KOORDINAT CSV → FOTO ASLI
# ─────────────────────────────────────────────

def konversi_ke_asli(cx_csv, cy_csv, r_csv, h_asli, w_asli, target_size, margin):
    max_side = max(h_asli, w_asli)
    off_x    = (max_side - w_asli) / 2
    off_y    = (max_side - h_asli) / 2
    scale    = max_side / target_size

    cx_asli = int((cx_csv + margin) * scale - off_x)
    cy_asli = int((cy_csv + margin) * scale - off_y)
    r_asli  = int(r_csv * scale)

    return cx_asli, cy_asli, r_asli


# ─────────────────────────────────────────────
#  WARNA PER NOMINAL
# ─────────────────────────────────────────────

WARNA = {
    "Rp100":  (255, 0, 255),    # magenta
    "Rp200":  (0, 255, 0),      # hijau
    "Rp500":  (255, 165, 0),    # oranye
    "Rp1000": (0, 0, 255),      # merah
}


# ─────────────────────────────────────────────
#  GAMBAR ANOTASI + LABEL KLASIFIKASI
# ─────────────────────────────────────────────

def gambar_klasifikasi(img_asli, circles_data, target_size, margin):
    output = img_asli.copy()
    h_asli, w_asli = img_asli.shape[:2]

    for row in circles_data:
        cx_csv = int(row["center_x"])
        cy_csv = int(row["center_y"])
        r_csv  = int(row["radius_px"])
        d_px   = int(row["diameter_px"])

        cx, cy, r = konversi_ke_asli(
            cx_csv, cy_csv, r_csv,
            h_asli, w_asli, target_size, margin
        )

        nominal = klasifikasi_koin(d_px)
        warna   = WARNA.get(nominal, (255, 255, 255))

        # Lingkaran berwarna sesuai nominal
        cv2.circle(output, (cx, cy), r, warna, 3)

        # Titik tengah
        cv2.circle(output, (cx, cy), 6, warna, -1)

        # Label nominal di atas lingkaran
        cv2.putText(output, nominal,
                    (cx - r, cy - r - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna, 3)

        # Label diameter
        cv2.putText(output, f"d={d_px}px",
                    (cx - r, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        return

    # Baca CSV
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
    print(f"  KLASIFIKASI RULE-BASED - Nominal Koin Rupiah")
    print(f"  Threshold:")
    print(f"    diameter >= 122px  → Rp500")
    print(f"    diameter 115-121px → Rp200")
    print(f"    diameter 110-114px → Rp1000")
    print(f"    diameter < 110px   → Rp100")
    print(f"  Total file: {len(data_per_file)}")
    print(f"{'='*55}\n")

    all_results  = []
    ok, fail     = 0, 0
    benar, salah = 0, 0

    # Hitung per kelas untuk analisis detail
    detail = {
        "Rp100":  {"benar": 0, "salah": 0},
        "Rp200":  {"benar": 0, "salah": 0},
        "Rp500":  {"benar": 0, "salah": 0},
        "Rp1000": {"benar": 0, "salah": 0},
    }

    for (fname, label), circles_data in sorted(data_per_file.items()):
        img_path = dataset_dir / label / fname
        if not img_path.exists():
            print(f"  [!] Foto tidak ditemukan: {img_path}")
            fail += 1
            continue

        try:
            img_asli = cv2.imread(str(img_path))
            if img_asli is None:
                raise FileNotFoundError("Tidak bisa dibaca")

            result_img = gambar_klasifikasi(
                img_asli, circles_data,
                cfg["target_size"], cfg["margin"]
            )

            out_dir = output_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / fname), result_img)

            # Label asli dari nama folder
            label_asli = "Rp" + label.split("_")[1]

            for row in circles_data:
                d_px     = int(row["diameter_px"])
                prediksi = klasifikasi_koin(d_px)
                status   = "BENAR" if prediksi == label_asli else "SALAH"

                if prediksi == label_asli:
                    benar += 1
                    detail[label_asli]["benar"] += 1
                else:
                    salah += 1
                    detail[label_asli]["salah"] += 1

                all_results.append({
                    "file":        fname,
                    "label_asli":  label_asli,
                    "koin_ke":     row["koin_ke"],
                    "diameter_px": d_px,
                    "prediksi":    prediksi,
                    "status":      status,
                })

            print(f"  [OK] {label}/{fname} — {len(circles_data)} koin")
            ok += 1

        except Exception as e:
            print(f"  [GAGAL] {label}/{fname} — {e}")
            fail += 1

    # Simpan CSV hasil klasifikasi
    csv_out = output_root / "hasil_klasifikasi.csv"
    with open(csv_out, "w", newline="") as f:
        fieldnames = ["file", "label_asli", "koin_ke",
                      "diameter_px", "prediksi", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Hitung akurasi
    total   = benar + salah
    akurasi = round((benar / total) * 100, 2) if total > 0 else 0

    print(f"\n{'='*55}")
    print(f"  HASIL KLASIFIKASI")
    print(f"{'='*55}")
    print(f"  Total koin : {total}")
    print(f"  Benar      : {benar}")
    print(f"  Salah      : {salah}")
    print(f"  Akurasi    : {akurasi}%")
    print(f"\n  Detail per kelas:")
    for kelas, hasil in detail.items():
        total_kelas = hasil["benar"] + hasil["salah"]
        if total_kelas > 0:
            acc = round(hasil["benar"] / total_kelas * 100, 2)
            print(f"    {kelas:8s} → Benar: {hasil['benar']:3d} | "
                  f"Salah: {hasil['salah']:3d} | Akurasi: {acc}%")
    print(f"\n  CSV hasil  : {csv_out.resolve()}")
    print(f"  Visual     : {output_root.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run(CONFIG)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

def jalankan_evaluasi(csv_path):
    print(f"Membaca data dari: {csv_path}...\n")

    # 1. Load Dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} tidak ditemukan!")
        return

    kolom_asli     = 'label_asli'
    kolom_prediksi = 'prediksi'

    y_true = df[kolom_asli].astype(str)
    y_pred = df[kolom_prediksi].astype(str)
    labels = ['Rp100', 'Rp200', 'Rp500', 'Rp1000']

    # 2. Buat folder output_evaluasi
    output_dir = "output_evaluasi"
    os.makedirs(output_dir, exist_ok=True)

    # 3. OUTPUT TERMINAL
    print("=" * 60)
    print("  HASIL EVALUASI SISTEM KLASIFIKASI KOIN RUPIAH")
    print("=" * 60)

    # Akurasi keseluruhan
    akurasi = accuracy_score(y_true, y_pred)
    print(f"\n  Akurasi Keseluruhan : {akurasi * 100:.2f}%")
    print(f"  Total Data          : {len(y_true)} koin")
    print(f"  Benar               : {(y_true == y_pred).sum()}")
    print(f"  Salah               : {(y_true != y_pred).sum()}")

    # Akurasi per kelas
    print("\n  Akurasi per kelas:")
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            acc  = (y_true[mask] == y_pred[mask]).mean() * 100
            total = mask.sum()
            benar = (y_true[mask] == y_pred[mask]).sum()
            print(f"    {label:8s} → {acc:6.2f}%  "
                  f"({benar}/{total} benar)")

    # Classification report
    print("\n" + "-" * 60)
    print("  CLASSIFICATION REPORT")
    print("-" * 60)
    report_teks = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    print(report_teks)

    # ─────────────────────────────────────────────
    #  WINDOW 1: Confusion Matrix
    # ─────────────────────────────────────────────
    plt.figure("Window 1 - Confusion Matrix", figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix Klasifikasi Koin Rupiah', pad=15, fontsize=13)
    plt.xlabel('\nPrediksi Sistem', fontsize=11)
    plt.ylabel('Label Asli (Ground Truth)\n', fontsize=11)
    plt.tight_layout()

    path_cm = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path_cm, dpi=300)
    print(f"[OK] Confusion matrix disimpan: {path_cm}")

    # ─────────────────────────────────────────────
    #  WINDOW 2: Metrik Evaluasi (Precision, Recall, F1)
    # ─────────────────────────────────────────────
    report_dict = classification_report(
        y_true, y_pred, labels=labels,
        output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).iloc[:-1, :len(labels)].T
    df_report = df_report[['precision', 'recall', 'f1-score']]

    plt.figure("Window 2 - Metrik Evaluasi", figsize=(8, 5))
    sns.heatmap(
        df_report,
        annot=True,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        fmt='.2f',
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Metrik Evaluasi per Kelas', pad=15, fontsize=13)
    plt.xlabel('\nMetrik', fontsize=11)
    plt.ylabel('Nominal Koin\n', fontsize=11)
    plt.tight_layout()

    path_metrics = os.path.join(output_dir, 'classification_metrics.png')
    plt.savefig(path_metrics, dpi=300)
    print(f"[OK] Metrik evaluasi disimpan: {path_metrics}")

    # ─────────────────────────────────────────────
    #  WINDOW 3: Bar Chart Akurasi per Kelas
    # ─────────────────────────────────────────────
    akurasi_per_kelas = []
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).mean() * 100
        else:
            acc = 0
        akurasi_per_kelas.append(acc)

    plt.figure("Window 3 - Akurasi per Kelas", figsize=(8, 5))
    bars = plt.bar(labels, akurasi_per_kelas,
                   color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'],
                   edgecolor='black', width=0.5)

    # Tambahkan label nilai di atas bar
    for bar, acc in zip(bars, akurasi_per_kelas):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f'{acc:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Garis akurasi keseluruhan
    plt.axhline(y=akurasi * 100, color='red', linestyle='--', linewidth=1.5,
                label=f'Akurasi Keseluruhan: {akurasi*100:.2f}%')
    plt.legend(fontsize=10)

    plt.title('Akurasi Klasifikasi per Nominal Koin', pad=15, fontsize=13)
    plt.xlabel('\nNominal Koin', fontsize=11)
    plt.ylabel('Akurasi (%)\n', fontsize=11)
    plt.ylim(0, 115)
    plt.tight_layout()

    path_bar = os.path.join(output_dir, 'akurasi_per_kelas.png')
    plt.savefig(path_bar, dpi=300)
    print(f"[OK] Bar chart akurasi disimpan: {path_bar}")

    print("\n" + "=" * 60)
    print(f"  Semua output tersimpan di folder: {output_dir}/")
    print("=" * 60)

    # Tampilkan semua window
    plt.show()


if __name__ == "__main__":
    PATH_CSV = "output_klasifikasi/hasil_klasifikasi.csv"
    jalankan_evaluasi(PATH_CSV)
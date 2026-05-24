import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────
#  CSS CUSTOM — tampilan lebih rapi & modern
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background utama */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8e8f0;
}

/* Judul utama */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #f7c948, #f4a261);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}

/* Subjudul */
h3 {
    color: #f7c948 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-top: 1.5rem !important;
}

/* Card container */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(247,201,72,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Metric boxes */
[data-testid="metric-container"] {
    background: rgba(247,201,72,0.08) !important;
    border: 1px solid rgba(247,201,72,0.3) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #a0a0c0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f7c948 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #7ee8a2 !important;
    font-size: 0.9rem !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(247,201,72,0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* Tombol proses */
.stButton > button {
    background: linear-gradient(90deg, #f7c948, #f4a261) !important;
    color: #0f0f1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(247,201,72,0.3) !important;
}

/* Info box nominal */
.koin-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
}
.koin-nominal { font-weight: 700; font-size: 1rem; color: #f7c948; }
.koin-jumlah  { color: #a0a0c0; font-size: 0.9rem; }
.koin-subtotal{ font-weight: 700; font-size: 1rem; color: #7ee8a2; }

/* Badge nominal */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.3rem;
}
.badge-100  { background: rgba(255,0,255,0.15); color: #ff77ff; border: 1px solid rgba(255,0,255,0.3); }
.badge-200  { background: rgba(0,255,0,0.1);    color: #7ee8a2; border: 1px solid rgba(0,255,0,0.25); }
.badge-500  { background: rgba(255,165,0,0.15); color: #f4a261; border: 1px solid rgba(255,165,0,0.3); }
.badge-1000 { background: rgba(80,120,255,0.15);color: #8ab4f8; border: 1px solid rgba(80,120,255,0.3); }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.5rem 0 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #f7c948 !important; }

/* Success/Error */
.stSuccess { background: rgba(126,232,162,0.1) !important; border-left: 4px solid #7ee8a2 !important; border-radius: 10px !important; }
.stError   { background: rgba(255,100,100,0.1) !important; border-left: 4px solid #ff6464 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
CONFIG_CHT = {
    "dp": 1, "minDist": 100, "param1": 80,
    "param2": 32, "minRadius": 55, "maxRadius": 80,
}
CONFIG_CHT_100 = {
    "dp": 1, "minDist": 85, "param1": 50,
    "param2": 40, "minRadius": 40, "maxRadius": 65,
}

WARNA = {
    "Rp100":  (255, 0,   255),
    "Rp200":  (0,   255, 0  ),
    "Rp500":  (255, 165, 0  ),
    "Rp1000": (0,   0,   255),
}

BADGE_CLASS = {
    "Rp100": "badge-100", "Rp200": "badge-200",
    "Rp500": "badge-500", "Rp1000": "badge-1000",
}


# ─────────────────────────────────────────────
#  TAHAP 1: PREPROCESSING
# ─────────────────────────────────────────────
def pad_to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    max_side = max(h, w)
    canvas = (np.zeros((max_side, max_side, 3), dtype=np.uint8)
              if len(image.shape) == 3
              else np.zeros((max_side, max_side), dtype=np.uint8))
    top  = (max_side - h) // 2
    left = (max_side - w) // 2
    canvas[top:top+h, left:left+w] = image
    return canvas

def resize_image(image, size):
    h, w   = image.shape[:2]
    interp = cv2.INTER_AREA if (w > size[0] or h > size[1]) else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interp)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def gaussian_blur_fn(image, ksize, sigma):
    return cv2.GaussianBlur(image, ksize, sigma)


# ─────────────────────────────────────────────
#  TAHAP 2: DETEKSI LINGKARAN
# ─────────────────────────────────────────────
def detect_circles(image_gray, cfg):
    circles = cv2.HoughCircles(
        image_gray, cv2.HOUGH_GRADIENT,
        dp=cfg["dp"], minDist=cfg["minDist"],
        param1=cfg["param1"], param2=cfg["param2"],
        minRadius=cfg["minRadius"], maxRadius=cfg["maxRadius"],
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
    return circles


# ─────────────────────────────────────────────
#  TAHAP 3: EKSTRAKSI FITUR
# ─────────────────────────────────────────────
def hitung_fitur(radius_px):
    r        = radius_px
    diameter = r * 2
    luas     = round(np.pi * r ** 2, 2)
    keliling = round(2 * np.pi * r, 2)
    return diameter, luas, keliling

def konversi_ke_asli(cx, cy, r, h_asli, w_asli, target_size, margin):
    max_side = max(h_asli, w_asli)
    off_x    = (max_side - w_asli) / 2
    off_y    = (max_side - h_asli) / 2
    scale    = max_side / target_size
    return (int((cx + margin) * scale - off_x),
            int((cy + margin) * scale - off_y),
            int(r * scale))


# ─────────────────────────────────────────────
#  TAHAP 4: KLASIFIKASI
# ─────────────────────────────────────────────
def klasifikasi_koin(diameter_px):
    if diameter_px >= 122:
        return "Rp500",  500,  WARNA["Rp500"]
    elif diameter_px >= 115:
        return "Rp200",  200,  WARNA["Rp200"]
    elif diameter_px >= 110:
        return "Rp1000", 1000, WARNA["Rp1000"]
    else:
        return "Rp100",  100,  WARNA["Rp100"]


# ─────────────────────────────────────────────
#  PROSES UTAMA
# ─────────────────────────────────────────────
def proses_gambar_koin(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_asli   = cv2.imdecode(file_bytes, 1)

    h_asli, w_asli = img_asli.shape[:2]
    img_output     = img_asli.copy()

    target_size = (512, 512)
    margin      = 60

    # Preprocessing
    padded      = pad_to_square(img_asli)
    resized     = resize_image(padded, target_size)
    gray        = to_grayscale(resized)
    blurred     = gaussian_blur_fn(gray, (5, 5), 0)
    _, enc      = cv2.imencode('.jpg', blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    preprocessed = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    # Crop margin
    h_p, w_p    = preprocessed.shape
    img_cropped = preprocessed[margin:h_p-margin, margin:w_p-margin]

    # Deteksi
    c_kecil = detect_circles(img_cropped, CONFIG_CHT_100)
    c_besar = detect_circles(img_cropped, CONFIG_CHT)

    kandidat = []
    if c_kecil is not None:
        kandidat += [(int(c[0]), int(c[1]), int(c[2])) for c in c_kecil]
    if c_besar is not None:
        kandidat += [(int(c[0]), int(c[1]), int(c[2])) for c in c_besar]

    final = []
    for c in kandidat:
        x, y, r = c
        if not any(np.sqrt((x-fx)**2 + (y-fy)**2) < 50
                   for fx, fy, fr in final):
            final.append((x, y, r))

    rekap       = {'Rp100': 0, 'Rp200': 0, 'Rp500': 0, 'Rp1000': 0}
    grand_total = 0
    data_tabel  = []

    for i, (cx, cy, r) in enumerate(final):
        d_px, luas, keliling = hitung_fitur(r)
        nominal, nilai, warna = klasifikasi_koin(d_px)

        rekap[nominal] += 1
        grand_total    += nilai

        data_tabel.append({
            "Koin ke": i+1, "Nominal": nominal,
            "Radius (px)": r, "Diameter (px)": d_px,
            "Luas (px²)": luas, "Keliling (px)": keliling,
        })

        cx_a, cy_a, r_a = konversi_ke_asli(cx, cy, r, h_asli, w_asli, target_size[0], margin)

        font_scale = max(h_asli, w_asli) / 1000.0
        cv2.circle(img_output, (cx_a, cy_a), r_a, warna, 3)
        cv2.circle(img_output, (cx_a, cy_a), 6, warna, -1)
        cv2.putText(img_output, nominal,
                    (cx_a - r_a, cy_a - r_a - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.9, font_scale), warna, max(3, int(font_scale*3)))
        cv2.putText(img_output, f"d={d_px}px",
                    (cx_a - r_a, cy_a - r_a - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, font_scale*0.7), (255,255,255), max(2, int(font_scale*2)))

    total_keping = sum(rekap.values())
    img_rgb      = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    return img_rgb, rekap, total_keping, grand_total, pd.DataFrame(data_tabel)


# ─────────────────────────────────────────────
#  UI STREAMLIT
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem'>
    <span style='font-size:2.4rem'>🪙</span>
    <h1 style='margin:0'>Kalkulator Koin Rupiah</h1>
</div>
<p style='color:#a0a0c0; font-size:0.95rem; margin-top:0.2rem; margin-bottom:1.5rem'>
    Deteksi & klasifikasi nominal koin menggunakan
    <b style='color:#f7c948'>Circular Hough Transform</b> +
    <b style='color:#f7c948'>Rule-Based Classification</b>
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload ──
uploaded_file = st.file_uploader(
    "📁 Upload foto koin (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col_prev, col_btn = st.columns([3, 1])
    with col_prev:
        with st.expander("🖼️ Preview gambar yang diupload"):
            st.image(Image.open(uploaded_file), use_container_width=True)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        proses = st.button("🔍 Proses", type="primary", use_container_width=True)

    if proses:
        with st.spinner("Menjalankan pipeline deteksi..."):
            img_hasil, rekap, jml_keping, grand_total, df = proses_gambar_koin(uploaded_file)

        if jml_keping > 0:
            # ── Gambar hasil ──
            st.markdown("### 📷 Hasil Deteksi")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<p style='color:#a0a0c0; font-size:0.85rem; margin-bottom:0.4rem'>GAMBAR ASLI</p>", unsafe_allow_html=True)
                uploaded_file.seek(0)
                st.image(Image.open(uploaded_file), use_container_width=True)
            with col2:
                st.markdown("<p style='color:#a0a0c0; font-size:0.85rem; margin-bottom:0.4rem'>HASIL KLASIFIKASI</p>", unsafe_allow_html=True)
                st.image(img_hasil, use_container_width=True)

            st.markdown("---")

            # ── Rekap nominal ──
            st.markdown("### 📋 Rincian per Nominal")
            for nominal, jumlah in rekap.items():
                if jumlah > 0:
                    nilai    = int(nominal.replace("Rp", ""))
                    subtotal = nilai * jumlah
                    badge    = BADGE_CLASS[nominal]
                    st.markdown(f"""
                    <div class='koin-card'>
                        <span>
                            <span class='badge {badge}'>{nominal}</span>
                            <span class='koin-nominal'></span>
                        </span>
                        <span class='koin-jumlah'>{jumlah} keping</span>
                        <span class='koin-subtotal'>Rp {subtotal:,}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Total ──
            st.markdown("### 💰 Total Keseluruhan")
            c1, c2 = st.columns(2)
            c1.metric("Total Keping", f"{jml_keping} Koin")
            c2.metric("Total Nominal", f"Rp {grand_total:,}".replace(",", "."))

            st.markdown("---")

            # ── Legenda warna ──
            st.markdown("### 🎨 Legenda Warna")
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.markdown("<span class='badge badge-100'>🟣 Rp100</span>",  unsafe_allow_html=True)
            lc2.markdown("<span class='badge badge-200'>🟢 Rp200</span>",  unsafe_allow_html=True)
            lc3.markdown("<span class='badge badge-500'>🟠 Rp500</span>",  unsafe_allow_html=True)
            lc4.markdown("<span class='badge badge-1000'>🔵 Rp1000</span>", unsafe_allow_html=True)

            st.markdown("---")

            # ── Tabel ekstraksi ──
            with st.expander("📊 Detail Ekstraksi Fitur"):
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.error("❌ Tidak ada koin terdeteksi. Coba foto dengan pencahayaan lebih baik.")

else:
    st.markdown("""
    <div style='text-align:center; padding:3rem 1rem; color:#505070;'>
        <div style='font-size:3.5rem; margin-bottom:0.8rem'>📷</div>
        <p style='font-size:1rem; margin:0'>Upload foto koin untuk memulai deteksi</p>
        <p style='font-size:0.85rem; margin-top:0.4rem'>Mendukung format JPG dan PNG</p>
    </div>
    """, unsafe_allow_html=True)
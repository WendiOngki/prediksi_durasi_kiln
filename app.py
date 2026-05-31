import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model

# ── Konfigurasi halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Durasi Pengeringan Kayu",
    page_icon="🪵",
    layout="centered"
)

# ── Load model (cache agar tidak reload setiap interaksi) ──────────────────
@st.cache_resource
def load():
    return load_model('model_kiln_v1')

model = load()

# ── Judul ──────────────────────────────────────────────────────────────────
st.title("🪵 Prediksi Durasi Pengeringan Kayu Kiln")
st.markdown("Isi form di bawah, lalu klik **Hitung Prediksi**.")

# ── Konstanta ──────────────────────────────────────────────────────────────
TEBAL_BINS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

# ═══════════════════════════════════════════════════════════════════════════
# FORM INPUT
# ═══════════════════════════════════════════════════════════════════════════
with st.form("input_form"):

    st.subheader("📋 Informasi Batch")
    col1, col2 = st.columns(2)

    with col1:
        jenis_kayu = st.selectbox(
            "Jenis Kayu",
            ["MAHONI", "JATI", "SENGON", "MERANTI"],
            help="Pilih jenis kayu yang akan dikeringkan"
        )
        no_kiln = st.selectbox(
            "Nomor Kiln",
            [1, 2, 3, 4, 5, 6],
            help="Pilih kiln yang digunakan"
        )
        bulan_in = st.selectbox(
            "Bulan Masuk",
            list(range(1, 13)),
            format_func=lambda x: [
                "Januari","Februari","Maret","April","Mei","Juni",
                "Juli","Agustus","September","Oktober","November","Desember"
            ][x-1],
            help="Bulan batch masuk kiln"
        )

    with col2:
        vol_total_m3 = st.number_input(
            "Volume Total (m³)", 
            min_value=0.1, max_value=100.0, value=10.0, step=0.5,
            help="Total volume kayu dalam satu batch"
        )
        total_lembar = st.number_input(
            "Total Lembar", 
            min_value=1, max_value=5000, value=500, step=10,
            help="Total jumlah lembar papan dalam batch"
        )
        jumlah_asal = st.number_input(
            "Jumlah Asal Kayu",
            min_value=1, max_value=10, value=1, step=1,
            help="Dari berapa lokasi asal kayu ini berasal"
        )

    st.subheader("🌤️ Kondisi Cuaca")
    col3, col4 = st.columns(2)

    with col3:
        kelembaban_pct = st.slider(
            "Kelembaban (%)", 
            min_value=40, max_value=100, value=75,
            help="Kelembaban udara rata-rata selama pengeringan"
        )
        curah_hujan_mm = st.number_input(
            "Curah Hujan (mm)", 
            min_value=0.0, max_value=50.0, value=3.0, step=0.5
        )

    with col4:
        suhu_maks_c = st.number_input(
            "Suhu Maksimum (°C)",
            min_value=20.0, max_value=45.0, value=32.0, step=0.5
        )
        suhu_min_c = st.number_input(
            "Suhu Minimum (°C)",
            min_value=15.0, max_value=40.0, value=24.0, step=0.5
        )

    st.subheader("📏 Komposisi Ketebalan Papan")
    st.caption("Isi jumlah lembar untuk setiap ketebalan. Kosongkan (isi 0) jika tidak ada.")

    cols_tebal = st.columns(len(TEBAL_BINS))
    komposisi = {}
    for col, t in zip(cols_tebal, TEBAL_BINS):
        with col:
            n = st.number_input(
                f"{t} cm", 
                min_value=0, max_value=3000, value=0, step=10,
                key=f"tebal_{t}"
            )
            if n > 0:
                komposisi[t] = n

    submitted = st.form_submit_button(
        "🔍 Hitung Prediksi", 
        use_container_width=True,
        type="primary"
    )

# ═══════════════════════════════════════════════════════════════════════════
# PROSES DAN TAMPILKAN HASIL
# ═══════════════════════════════════════════════════════════════════════════
if submitted:

    # Validasi komposisi ketebalan tidak kosong semua
    if not komposisi:
        st.error("⚠️ Isi minimal satu ketebalan papan.")
        st.stop()

    # Validasi suhu
    if suhu_maks_c <= suhu_min_c:
        st.error("⚠️ Suhu maksimum harus lebih besar dari suhu minimum.")
        st.stop()

    # ── Bangun input row ───────────────────────────────────────────────────
    total_lem   = sum(komposisi.values())
    props       = {f'prop_{t}': komposisi.get(t, 0) / total_lem for t in TEBAL_BINS}
    tebal_vals  = np.repeat(
        list(komposisi.keys()), 
        list(komposisi.values())
    ).astype(float)

    ket_mean    = float(np.mean(tebal_vals))
    ket_max     = float(np.max(tebal_vals))
    ket_min     = float(np.min(tebal_vals))
    ket_std     = float(np.std(tebal_vals)) if len(set(komposisi.keys())) > 1 else 0.0
    n_ketebalan = len(komposisi)
    delta_suhu  = suhu_maks_c - suhu_min_c
    musim       = 1 if bulan_in in [11, 12, 1, 2, 3, 4] else 0
    prop_tipis  = props['prop_2.0'] + props['prop_2.5']
    prop_tebal_ext = props['prop_5.0'] + props['prop_8.0']

    row = {
        'jenis_kayu'    : str(jenis_kayu),
        'no_kiln'       : str(no_kiln),
        'vol_total_m3'  : vol_total_m3,
        'total_lembar'  : total_lembar,
        'kelembaban_pct': kelembaban_pct,
        'curah_hujan_mm': curah_hujan_mm,
        'suhu_maks_c'   : suhu_maks_c,
        'suhu_min_c'    : suhu_min_c,
        'bulan_in'      : str(bulan_in),
        'ket_mean'      : ket_mean,
        'ket_max'       : ket_max,
        'ket_min'       : ket_min,
        'ket_std'       : ket_std,
        'n_ketebalan'   : n_ketebalan,
        'vol_m3_total'  : vol_total_m3,
        'lembar_total'  : total_lembar,
        'jumlah_asal'   : jumlah_asal,
        'vol_per_lembar': vol_total_m3 / max(total_lembar, 1),
        'delta_suhu'    : delta_suhu,
        'musim'         : str(musim),
        'lembab_x_tebal': kelembaban_pct * ket_max,
        'hujan_x_lembab': curah_hujan_mm * kelembaban_pct,
        'tebal_x_vol'   : ket_mean * vol_total_m3,
        'prop_tipis'    : prop_tipis,
        'prop_tebal_ext': prop_tebal_ext,
        'rasio_tebal'   : prop_tebal_ext / (prop_tipis + 1e-6),
        **props,
    }

    df_input = pd.DataFrame([row])

    # ── Prediksi ───────────────────────────────────────────────────────────
    with st.spinner("Menghitung prediksi..."):
        try:
            hasil = predict_model(model, data=df_input)
            durasi = round(float(hasil['prediction_label'].values[0]), 1)
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            st.stop()

    # ── Tampilkan hasil ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    col_hasil1, col_hasil2, col_hasil3 = st.columns(3)
    with col_hasil1:
        st.metric("⏱️ Estimasi Durasi", f"{durasi} hari")
    with col_hasil2:
        tanggal_selesai = pd.Timestamp.now() + pd.Timedelta(days=durasi)
        st.metric("📅 Perkiraan Selesai", tanggal_selesai.strftime("%d %b %Y"))
    with col_hasil3:
        musim_label = "🌧️ Musim Hujan" if musim == 1 else "☀️ Musim Kemarau"
        st.metric("Musim", musim_label)

    # Ringkasan input
    with st.expander("📋 Lihat detail input yang digunakan"):
        ringkasan = {
            "Jenis Kayu": jenis_kayu,
            "Kiln": f"Kiln {no_kiln}",
            "Volume": f"{vol_total_m3} m³",
            "Total Lembar": f"{total_lembar} lbr",
            "Kelembaban": f"{kelembaban_pct}%",
            "Suhu Maks/Min": f"{suhu_maks_c}°C / {suhu_min_c}°C",
            "Ketebalan dominan": f"{ket_mean:.2f} cm (rata-rata)",
            "Komposisi": str({f"{k}cm": v for k, v in komposisi.items()}),
        }
        for k, v in ringkasan.items():
            st.write(f"**{k}:** {v}")
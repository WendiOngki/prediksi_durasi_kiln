import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Prediksi Durasi Pengeringan Kayu",
    page_icon="🪵",
    layout="centered"
)

@st.cache_resource
def load():
    with open('model_sklearn.pkl', 'rb') as f:
        return pickle.load(f)

model = load()

st.title("🪵 Prediksi Durasi Pengeringan Kayu Kiln")
st.markdown("Isi form di bawah, lalu klik **Hitung Prediksi**.")

TEBAL_BINS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

with st.form("input_form"):
    st.subheader("📋 Informasi Batch")
    col1, col2 = st.columns(2)
    with col1:
        jenis_kayu = st.selectbox("Jenis Kayu", ["MAHONI", "JATI"])
        no_kiln    = st.selectbox("Nomor Kiln", [str(i) for i in range(1, 7)])
        bulan_in   = st.selectbox("Bulan Masuk", list(range(1, 13)),
                        format_func=lambda x: ["Januari","Februari","Maret","April",
                            "Mei","Juni","Juli","Agustus","September",
                            "Oktober","November","Desember"][x-1])
    with col2:
        vol_total_m3  = st.number_input("Volume Total (m³)",  min_value=0.1,  max_value=100.0, value=10.0,  step=0.5)
        total_lembar  = st.number_input("Total Lembar",        min_value=1,    max_value=5000,  value=500,   step=10)
        jumlah_asal   = st.number_input("Jumlah Asal Kayu",   min_value=1,    max_value=10,    value=1,     step=1)

    st.subheader("🌤️ Kondisi Cuaca")
    col3, col4 = st.columns(2)
    with col3:
        kelembaban_pct = st.slider("Kelembaban (%)",      40, 100, 75)
        curah_hujan_mm = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    with col4:
        suhu_maks_c = st.number_input("Suhu Maksimum (°C)", min_value=20.0, max_value=45.0, value=32.0, step=0.5)
        suhu_min_c  = st.number_input("Suhu Minimum (°C)",  min_value=15.0, max_value=40.0, value=24.0, step=0.5)

    st.subheader("📏 Komposisi Ketebalan Papan")
    st.caption("Isi jumlah lembar tiap ketebalan. Kosongkan (isi 0) jika tidak ada.")
    cols_tebal = st.columns(len(TEBAL_BINS))
    komposisi  = {}
    for col, t in zip(cols_tebal, TEBAL_BINS):
        with col:
            n = st.number_input(f"{t} cm", min_value=0, max_value=3000, value=0, step=10, key=f"t_{t}")
            if n > 0:
                komposisi[t] = n

    submitted = st.form_submit_button("🔍 Hitung Prediksi", use_container_width=True, type="primary")

if submitted:
    if not komposisi:
        st.error("⚠️ Isi minimal satu ketebalan papan.")
        st.stop()
    if suhu_maks_c <= suhu_min_c:
        st.error("⚠️ Suhu maksimum harus lebih besar dari suhu minimum.")
        st.stop()

    total_lem = sum(komposisi.values())
    props     = {f'prop_{t}': komposisi.get(t, 0) / total_lem for t in TEBAL_BINS}
    tebal_arr = np.repeat(list(komposisi.keys()), list(komposisi.values())).astype(float)

    ket_mean = float(np.mean(tebal_arr))
    ket_max  = float(np.max(tebal_arr))
    ket_min  = float(np.min(tebal_arr))
    ket_std  = float(np.std(tebal_arr)) if len(komposisi) > 1 else 0.0
    musim    = '1' if bulan_in in [11, 12, 1, 2, 3, 4] else '0'

    row = {
        'jenis_kayu'    : jenis_kayu,
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
        'n_ketebalan'   : len(komposisi),
        'vol_m3_total'  : vol_total_m3,
        'lembar_total'  : total_lembar,
        'jumlah_asal'   : jumlah_asal,
        'vol_per_lembar': vol_total_m3 / max(total_lembar, 1),
        'delta_suhu'    : suhu_maks_c - suhu_min_c,
        'musim'         : musim,
        'lembab_x_tebal': kelembaban_pct * ket_max,
        'hujan_x_lembab': curah_hujan_mm * kelembaban_pct,
        'tebal_x_vol'   : ket_mean * vol_total_m3,
        'prop_tipis'    : props['prop_2.0'] + props['prop_2.5'],
        'prop_tebal_ext': props['prop_5.0'] + props['prop_8.0'],
        'rasio_tebal'   : (props['prop_5.0'] + props['prop_8.0']) / (props['prop_2.0'] + props['prop_2.5'] + 1e-6),
        **props,
    }

    df_input = pd.DataFrame([row])

    with st.spinner("Menghitung prediksi..."):
        try:
            durasi = round(float(model.predict(df_input)[0]), 1)
        except Exception as e:
            st.error(f"Error prediksi: {e}")
            st.stop()

    st.divider()
    st.subheader("📊 Hasil Prediksi")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("⏱️ Estimasi Durasi", f"{durasi} hari")
    with c2:
        selesai = pd.Timestamp.now() + pd.Timedelta(days=durasi)
        st.metric("📅 Perkiraan Selesai", selesai.strftime("%d %b %Y"))
    with c3:
        st.metric("Musim", "🌧️ Hujan" if musim == '1' else "☀️ Kemarau")

    with st.expander("📋 Detail input"):
        st.dataframe(df_input.T.rename(columns={0: "Nilai"}), use_container_width=True)

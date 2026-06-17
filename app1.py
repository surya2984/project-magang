import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Dashboard Presensi Cerdas", page_icon="🏫", layout="wide")
st.title("🏫 Sistem Analitik & Deteksi Anomali Presensi")
st.markdown("Dashboard cerdas berbasis Machine Learning terintegrasi AppSheet.")
# Fungsi untuk membaca file CSS eksternal
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================
# FITUR UPLOAD DATA EXCEL (.xlsx)
# ==========================================
st.sidebar.header("📁 Input Data")
st.sidebar.markdown("Unggah file **ABSEN ONLINE1.xlsx** (Berisi tab CHECK IN & kantor)")
uploaded_file = st.sidebar.file_uploader("Upload Log Presensi", type=['xlsx'])

@st.cache_data
def process_real_data(df):
    # 1. Pembersihan Data (Fokus pada kolom CHECK IN dan LOKASI)
    df = df.dropna(subset=['LOKASI', 'CHECK IN']).copy()
    df = df[df['LOKASI'].astype(str).str.contains(',', na=False)].copy()
    
    # 2. Preprocessing Spasial
    df[['latitude', 'longitude']] = df['LOKASI'].str.split(',', n=1, expand=True).astype(float)
    
   # ==========================================
    # 3. Preprocessing Waktu (Penangkap Tanda Titik & Titik Dua)
    # ==========================================
    df['TANGGAL_DT'] = pd.to_datetime(df['TANGGAL'])
    df['Hari'] = df['TANGGAL_DT'].dt.dayofweek
    
    # 1. Paksa seluruh isi kolom menjadi string
    df['CHECK_IN_STR'] = df['CHECK IN'].astype(str)
    
    # 2. REGEX PAMUNGKAS: Cari angka yang dipisahkan oleh TITIK (:) ATAU TITIK BIASA (\.)
    ekstraksi = df['CHECK_IN_STR'].str.extract(r'(\d{1,2})[:\.](\d{2})')
    
    # 3. Hitung total menit (Kolom 0 = Jam, Kolom 1 = Menit)
    df['Menit_Masuk'] = (ekstraksi[0].astype(float) * 60) + ekstraksi[1].astype(float)
    
    # 4. Jika ada baris yang kosong/gagal, ubah jadi 0, lalu pastikan format integer
    df['Menit_Masuk'] = df['Menit_Masuk'].fillna(0).astype(int)
    
    # 5. Label Keterlambatan: Melewati 450 menit (07:30 pagi)
    df['Is_Late'] = (df['Menit_Masuk'] > 450).astype(int)
    
    # 4. Encoding Kategori (Jika kolom STATUS tersedia)
    #==========================================
    if 'STATUS' in df.columns:
        le = LabelEncoder()
        df['Status_Encoded'] = le.fit_transform(df['STATUS'].astype(str))
    else:
        df['Status_Encoded'] = 0
    
    # 5. Model DBSCAN (Deteksi Anomali Spasial)
    coords = df[['latitude', 'longitude']].values
    if len(coords) > 0:
        dbscan = DBSCAN(eps=0.0001, min_samples=2) 
        df['Kluster_Spasial'] = dbscan.fit_predict(coords)
        df['Is_Anomali_Lokasi'] = (df['Kluster_Spasial'] == -1).astype(bool)
    else:
        df['Is_Anomali_Lokasi'] = False
    
    # 6. Model Prediksi Keterlambatan (Random Forest)
    X = df[['Hari', 'Status_Encoded']]
    y = df['Is_Late']
    
    if len(y.unique()) > 1:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        df['Prediksi_Risiko_Telat'] = rf_model.predict(X)
    else:
        df['Prediksi_Risiko_Telat'] = 0
        
    # Identifikasi dinamis untuk kolom nama (Mencari kolom yang mengandung kata 'NAMA', 'GURU', atau 'PEGAWAI')
    # ==========================================
    # 7. Penyatuan Identitas Nama (GURU / PEGAWAI)
    # ==========================================
    # Memastikan kolom GURU dan PEGAWAI ada di dalam data
    if 'GURU' in df.columns and 'PEGAWAI' in df.columns:
        # Menggabungkan nilai kolom, mengganti nilai kosong (NaN) dengan teks kosong
        df['Identitas'] = df['GURU'].fillna('').astype(str) + df['PEGAWAI'].fillna('').astype(str)
        # Membersihkan sisa teks 'nan' atau '.0' yang mungkin terbawa dari Excel
        df['Identitas'] = df['Identitas'].str.replace('nan', '', case=False).str.strip()
    elif 'GURU' in df.columns:
        df['Identitas'] = df['GURU'].fillna('Unknown').astype(str)
    elif 'PEGAWAI' in df.columns:
        df['Identitas'] = df['PEGAWAI'].fillna('Unknown').astype(str)
    else:
        df['Identitas'] = "Identitas Tidak Ditemukan"
        
    return df
        
# ==========================================
# LOGIKA TAMPILAN UTAMA
# ==========================================
if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        # EKSTRAKSI DATA KANTOR
        df_kantor = pd.read_excel(xls, sheet_name='kantor')
        koordinat_str = str(df_kantor['koordinat'].iloc[0])
        center_lat, center_lng = map(float, koordinat_str.split(','))
        radius_meter = float(df_kantor['radius'].iloc[0])
        nama_lokasi = df_kantor['nama kantor'].iloc[0]

        # EKSTRAKSI DATA CHECK IN
        # ==========================================
        # EKSTRAKSI DATA CHECK IN
        # ==========================================
        raw_df = pd.read_excel(xls, sheet_name='CHECK IN')
        df = process_real_data(raw_df)
        
        # ==========================================
        # FITUR FILTER RENTANG WAKTU (SIDEBAR)
        # ==========================================
        st.sidebar.divider()
        st.sidebar.subheader("📅 Filter Kalender")
        
        # Mengambil batas tanggal minimum dan maksimum dari data asli
        min_date = df['TANGGAL_DT'].min().date()
        max_date = df['TANGGAL_DT'].max().date()
        
        # Membuat widget pemilih tanggal
        rentang_tanggal = st.sidebar.date_input(
            "Pilih Rentang Analisis:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Logika pemfilteran: Hanya proses jika user sudah memilih dua tanggal (Awal & Akhir)
        if len(rentang_tanggal) == 2:
            start_date, end_date = rentang_tanggal
            # Filter data yang hanya berada di antara tanggal terpilih
            mask = (df['TANGGAL_DT'].dt.date >= start_date) & (df['TANGGAL_DT'].dt.date <= end_date)
            df = df.loc[mask].copy() # Timpa dataframe utama dengan data yang sudah difilter
        
        # --- METRIK KPI ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Data Kedatangan (Valid)", value=len(df))
        with col2:
            total_telat = df['Is_Late'].sum()
            persentase = (total_telat/len(df))*100 if len(df) > 0 else 0
            st.metric(label="Insiden Keterlambatan Pagi", value=total_telat, delta=f"{persentase:.1f}%", delta_color="inverse")
        with col3:
            total_anomali = df['Is_Anomali_Lokasi'].sum()
            st.metric(label="Peringatan Fake GPS", value=total_anomali, delta="Luar Radius", delta_color="inverse")

        st.divider()

        # --- VISUALISASI PETA DINAMIS ---
        st.subheader("🗺️ Pemetaan Geospasial Kedatangan Pagi")
        st.markdown(f"Titik acuan: **{nama_lokasi}** (Radius Aman: {radius_meter} meter)")
        
        peta_presensi = folium.Map(
            location=[center_lat, center_lng], 
            zoom_start=19, 
            tiles='http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}',
            attr='Google Satellite Hybrid'
        )

        folium.Circle(
            location=[center_lat, center_lng],
            radius=radius_meter, 
            color='#0078FF',
            fill=True,
            fill_color='#0078FF',
            fill_opacity=0.2,
            tooltip=f"Zona Aman {nama_lokasi} ({radius_meter}m)"
        ).add_to(peta_presensi)

        for idx, baris in df.iterrows():
            if baris['Is_Anomali_Lokasi']:
                folium.CircleMarker(
                    location=[baris['latitude'], baris['longitude']],
                    radius=8, color='red', fill=True, fill_color='red', fill_opacity=0.8,
                    tooltip=f"⚠️ {baris['Identitas']} | Masuk: {baris['CHECK IN']}"
                ).add_to(peta_presensi)
            else:
                folium.CircleMarker(
                    location=[baris['latitude'], baris['longitude']],
                    radius=4, color='green', fill=True, fill_color='green', fill_opacity=0.6,
                    tooltip=f"✅ {baris['Identitas']} | Masuk: {baris['CHECK IN']}"
                ).add_to(peta_presensi)

        st_folium(peta_presensi, width=1200, height=500)

        # --- ANALITIK VISUAL (GRAFIK TREN) ---
        st.divider()
        st.subheader("📊 Analitik Tren Kedatangan")
        
        # Membagi layar menjadi dua kolom untuk grafik
        col_vis1, col_vis2 = st.columns(2)

        with col_vis1:
            st.markdown("**Distribusi Keterlambatan (Guru vs Pegawai)**")
            if 'STATUS' in df.columns:
                # Menghitung jumlah keterlambatan per status
                telat_per_status = df[df['Is_Late'] == 1].groupby('STATUS').size()
                if not telat_per_status.empty:
                    st.bar_chart(telat_per_status, color="#ffaa00")
                else:
                    st.success("Tingkat kedatangan tepat waktu 100%!")

        with col_vis2:
            st.markdown("**Frekuensi Keterlambatan Berdasarkan Hari**")
            # Memetakan angka hari menjadi nama hari yang mudah dibaca
            hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
            df['Nama_Hari'] = df['Hari'].map(hari_map)
            
            # Menghitung jumlah keterlambatan per hari
            telat_per_hari = df[df['Is_Late'] == 1].groupby('Nama_Hari').size()
            
            if not telat_per_hari.empty:
                # Mengurutkan urutan hari di grafik dari Senin - Minggu
                urutan_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                telat_per_hari = telat_per_hari.reindex(urutan_hari).dropna()
                st.bar_chart(telat_per_hari, color="#ff4b4b")
            else:
                st.success("Tingkat kedatangan tepat waktu 100%!")
        
      # --- TABEL LOG ---
        st.divider()
        st.subheader("📋 Log Kedatangan & Deteksi AI")
        
        kolom_tampil = ['TANGGAL', 'Identitas', 'STATUS', 'CHECK IN', 'Is_Late', 'Is_Anomali_Lokasi']
        kolom_tersedia = [col for col in kolom_tampil if col in df.columns]
        
        df_display = df[kolom_tersedia].copy()
        
        # 1. Bersihkan Format Teks Dasar
        if 'TANGGAL' in df_display.columns:
            df_display['TANGGAL'] = pd.to_datetime(df_display['TANGGAL'], errors='coerce').dt.strftime('%Y-%m-%d')
        if 'CHECK IN' in df_display.columns:
            df_display['CHECK IN'] = df_display['CHECK IN'].astype(str).str.replace('.', ':', regex=False)
            
        # ==========================================
        # 2. UI/UX TRANSFORMATION (Pemolesan Label)
        # ==========================================
        # Ubah angka 1/0 menjadi teks yang elegan
        if 'Is_Late' in df_display.columns:
            df_display['Status Kehadiran'] = df_display['Is_Late'].apply(lambda x: "🔴 Terlambat" if x == 1 else "✅ Tepat Waktu")
            df_display = df_display.drop(columns=['Is_Late']) # Hapus kolom angka mentah
            
        # Ubah True/False menjadi teks yang elegan
        if 'Is_Anomali_Lokasi' in df_display.columns:
            df_display['Validasi GPS'] = df_display['Is_Anomali_Lokasi'].apply(lambda x: "⚠️ Luar Radius" if x is True else "📍 Lokasi Sesuai")
            df_display = df_display.drop(columns=['Is_Anomali_Lokasi'])

        # ==========================================
        # 3. PANDAS STYLER (Pewarnaan Teks Dinamis)
        # ==========================================
        def highlight_status(val):
            if val == "🔴 Terlambat" or val == "⚠️ Luar Radius":
                # Teks merah neon dengan latar belakang merah transparan
                return 'color: #ff4b4b; font-weight: bold; background-color: rgba(255, 75, 75, 0.1);'
            elif val == "✅ Tepat Waktu" or val == "📍 Lokasi Sesuai":
                # Teks hijau neon
                return 'color: #00E676; font-weight: bold;'
            return ''

        # Terapkan gaya ke kolom yang baru dibuat menggunakan try-except untuk kompatibilitas versi Pandas
        kolom_gaya = [col for col in ['Status Kehadiran', 'Validasi GPS'] if col in df_display.columns]
        try:
            styled_df = df_display.style.map(highlight_status, subset=kolom_gaya)
        except AttributeError:
            styled_df = df_display.style.applymap(highlight_status, subset=kolom_gaya)
        
        # Tampilkan ke dashboard
        st.dataframe(styled_df, use_container_width=True)
    # ==========================================
    # PENUTUP BLOK TRY (JANGAN DIHAPUS)
    # ==========================================
    except Exception as e:
        # Menampilkan detail error yang lebih informatif jika gagal
        st.error(f"❌ Terjadi kesalahan pada arsitektur data: {e}")
        st.info("Pastikan file Excel memiliki tab bernama 'CHECK IN' dan 'kantor'.")

# PENUTUP IF UPLOADED_FILE
else:
    st.info("👈 Silakan unggah file **ABSEN ONLINE1.xlsx** untuk memulai analisis Kedatangan Pagi.")
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Custom CSS for background
st.markdown("""
<style>
    .main {
        background-color: #e6f3ff;  /* Light blue background */
    }
    .title-bg {
        background-color: #007bff;  /* Blue background for title */
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load dan preprocess data
@st.cache_data
def load_data():
    # Asumsikan file CSV ada di directory yang sama
    df = pd.read_csv('dataset_kinerja_karyawan.csv')
    # Drop kolom ID karena tidak diperlukan
    df = df.drop('ID', axis=1)
    # Encode label Kinerja Akhir
    le = LabelEncoder()
    df['Kinerja Akhir'] = le.fit_transform(df['Kinerja Akhir'])  # Rendah:0, Sedang:1, Tinggi:2
    return df, le

# Fungsi untuk train model
@st.cache_resource
def train_model(df):
    X = df.drop('Kinerja Akhir', axis=1)
    y = df['Kinerja Akhir']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluasi akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Fungsi untuk prediksi
def predict_performance(model, le, disiplin, produktivitas, kerja_sama, inisiatif):
    input_data = pd.DataFrame({
        'Disiplin': [disiplin],
        'Produktivitas': [produktivitas],
        'Kerja Sama': [kerja_sama],
        'Inisiatif': [inisiatif]
    })
    pred = model.predict(input_data)[0]
    return le.inverse_transform([pred])[0]

# Fungsi untuk saran masukan
def get_suggestions(disiplin, produktivitas, kerja_sama, inisiatif, prediksi):
    suggestions = []
    if disiplin < 5:
        suggestions.append("Tingkatkan disiplin dengan datang tepat waktu dan mematuhi aturan perusahaan.")
    if produktivitas < 5:
        suggestions.append("Tingkatkan produktivitas dengan mengatur prioritas tugas dan menggunakan tools bantu.")
    if kerja_sama < 5:
        suggestions.append("Tingkatkan kerja sama dengan berkomunikasi lebih baik dalam tim dan mendukung rekan kerja.")
    if inisiatif < 5:
        suggestions.append("Tingkatkan inisiatif dengan mencari peluang baru dan mengusulkan ide-ide inovatif.")
    
    if not suggestions:
        suggestions.append("Kinerja sudah baik, pertahankan dan terus tingkatkan!")
    
    return suggestions

# Main app
st.markdown("<div class='title-bg'><h1>🔮 Aplikasi Prediksi Kinerja Karyawan</h1></div>", unsafe_allow_html=True)

# Load data
df, le = load_data()

# Train model
model, accuracy = train_model(df)

# Sidebar menu
menu = st.sidebar.radio("📋 Menu", ["📊 Dashboard", "🔮 Prediksi Kinerja"])

if menu == "📊 Dashboard":
    st.title("📊 Dashboard Kinerja Karyawan")
    st.markdown("Pantau data dan performa model untuk analisis kinerja karyawan.")
    st.divider()
    
    st.subheader("📋 Dataset Kinerja Karyawan")
    st.dataframe(df.head(), use_container_width=True)
    
    st.divider()
    st.subheader("🎯 Akurasi Model")
    st.metric("Random Forest", f"{accuracy * 100:.2f}%")
    
    st.divider()
    st.subheader("📈 Distribusi Kinerja")
    st.bar_chart(df['Kinerja Akhir'].value_counts())

elif menu == "🔮 Prediksi Kinerja":
    st.title("🔮 Prediksi Kinerja Karyawan")
    st.markdown("Masukkan nilai kriteria kinerja (skala 1-10) untuk mendapatkan prediksi dan saran perbaikan.")
    st.divider()
    
    st.subheader("⚙️ Parameter Kinerja")
    col1, col2 = st.columns(2)
    
    with col1:
        disiplin = st.slider("Disiplin", 1, 10, 5, help="Tingkat kedisiplinan karyawan")
        produktivitas = st.slider("Produktivitas", 1, 10, 5, help="Tingkat produktivitas kerja")
    
    with col2:
        kerja_sama = st.slider("Kerja Sama", 1, 10, 5, help="Kemampuan bekerja sama dalam tim")
        inisiatif = st.slider("Inisiatif", 1, 10, 5, help="Tingkat inisiatif dan kreativitas")
    
    st.divider()
    if st.button("🔍 Prediksi Kinerja", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            prediksi = predict_performance(model, le, disiplin, produktivitas, kerja_sama, inisiatif)
        st.success(f"🎉 Prediksi Kinerja Akhir: **{prediksi}**")
        
        # Saran Masukan
        suggestions = get_suggestions(disiplin, produktivitas, kerja_sama, inisiatif, prediksi)
        st.divider()
        st.subheader("💡 Saran Masukan")
        if suggestions:
            for sug in suggestions:
                st.info(f"• {sug}")
        else:
            st.success("Kinerja sudah optimal! Terus pertahankan.")
            
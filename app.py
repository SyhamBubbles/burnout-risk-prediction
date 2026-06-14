import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Burnout Risk Checker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    margin: 0;
    font-size: 2.2rem;
}
.main-header p {
    margin-top: 0.4rem;
    opacity: 0.92;
    font-size: 1.02rem;
}
.metric-card {
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    color: white;
    text-align: center;
}
.metric-card h2 {
    margin: 0.2rem 0 0 0;
    font-size: 2rem;
}
.metric-card p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
}
.zone-card {
    background: #f8fafc;
    border-left: 6px solid #6366f1;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-top: 0.6rem;
}
.rekom-item {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.feature-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-right: 0.3rem;
    color: white;
}
.badge-aman { background-color: #22c55e; }
.badge-waspada { background-color: #eab308; }
.badge-berisiko { background-color: #f97316; }
.badge-sangat { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD MODEL DAN RESOURCE (di-cache supaya hanya load sekali)
# =====================================================================
@st.cache_resource
def load_resources():
    best_xgb = joblib.load('model_xgb.pkl')
    oe = joblib.load('ordinal_encoder.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    kmeans_final = joblib.load('kmeans_model.pkl')
    cluster_scaler = joblib.load('cluster_scaler.pkl')

    with open('recommendations.json') as f:
        recommendations = json.load(f)
        recommendations = {int(k): v for k, v in recommendations.items()}

    explainer = shap.TreeExplainer(best_xgb)

    return best_xgb, oe, scaler, kmeans_final, cluster_scaler, recommendations, explainer


best_xgb, oe, scaler, kmeans_final, cluster_scaler, recommendations, explainer = load_resources()

# =====================================================================
# KONSTANTA STRUKTUR FITUR
# =====================================================================
# Urutan kolom HARUS sama dengan urutan saat training (X_train.columns)
FEATURE_ORDER = [
    'age', 'occupation', 'work_mode', 'device_usage_type',
    'daily_screen_time', 'social_media_hours', 'doomscrolling_duration',
    'app_switch_frequency', 'notification_count', 'smartphone_unlocks',
    'late_night_device_usage', 'focus_sessions', 'deep_work_hours',
    'distraction_frequency', 'task_completion_rate', 'concentration_score',
    'sleep_hours', 'sleep_quality', 'caffeine_intake', 'physical_activity',
    'stress_level', 'workspace_quality', 'meeting_hours', 'internet_stability',
    'remote_work_days', 'motivation_level', 'mental_fatigue',
    'emotional_exhaustion', 'work_satisfaction', 'mental_state'
]

CAT_COLS = ['occupation', 'work_mode', 'device_usage_type', 'mental_state']
NUM_COLS = [c for c in FEATURE_ORDER if c not in CAT_COLS]

CLUSTER_FEATURES = [
    'emotional_exhaustion', 'stress_level', 'daily_screen_time',
    'doomscrolling_duration', 'late_night_device_usage', 'distraction_frequency',
    'work_satisfaction', 'sleep_hours', 'physical_activity',
    'deep_work_hours', 'motivation_level', 'mental_state'
]

LABEL_MAP = {0: 'Aman', 1: 'Waspada', 2: 'Berisiko', 3: 'Sangat Berisiko'}

BADGE_CLASS = {
    'Aman': 'badge-aman',
    'Waspada': 'badge-waspada',
    'Berisiko': 'badge-berisiko',
    'Sangat Berisiko': 'badge-sangat'
}

CARD_COLOR = {
    'Aman': '#22c55e',
    'Waspada': '#eab308',
    'Berisiko': '#f97316',
    'Sangat Berisiko': '#ef4444'
}

# Kategori dinamis diambil dari encoder yang sudah dilatih, supaya konsisten
OCCUPATION_OPTIONS = list(oe.categories_[0])
WORK_MODE_OPTIONS = list(oe.categories_[1])
DEVICE_USAGE_OPTIONS = list(oe.categories_[2])
MENTAL_STATE_OPTIONS = list(oe.categories_[3])

# Label singkat untuk grafik SHAP & radar (Bahasa Indonesia)
SHORT_LABELS = {
    'age': 'Usia', 'occupation': 'Pekerjaan', 'work_mode': 'Mode Kerja',
    'device_usage_type': 'Tipe Penggunaan Gadget',
    'daily_screen_time': 'Screen Time Harian', 'social_media_hours': 'Medsos',
    'doomscrolling_duration': 'Doomscrolling',
    'app_switch_frequency': 'Pindah Aplikasi', 'notification_count': 'Notifikasi',
    'smartphone_unlocks': 'Buka HP', 'late_night_device_usage': 'Gadget Malam Hari',
    'focus_sessions': 'Sesi Fokus', 'deep_work_hours': 'Deep Work',
    'distraction_frequency': 'Frekuensi Distraksi',
    'task_completion_rate': 'Penyelesaian Tugas', 'concentration_score': 'Skor Konsentrasi',
    'sleep_hours': 'Jam Tidur', 'sleep_quality': 'Kualitas Tidur',
    'caffeine_intake': 'Kafein', 'physical_activity': 'Aktivitas Fisik',
    'stress_level': 'Tingkat Stres', 'workspace_quality': 'Kualitas Ruang Kerja',
    'meeting_hours': 'Jam Meeting', 'internet_stability': 'Stabilitas Internet',
    'remote_work_days': 'Hari Kerja Remote', 'motivation_level': 'Motivasi',
    'mental_fatigue': 'Kelelahan Mental', 'emotional_exhaustion': 'Kelelahan Emosional',
    'work_satisfaction': 'Kepuasan Kerja', 'mental_state': 'Kondisi Mental'
}

# Profil rata rata (z-score) setiap zona, dipakai untuk radar chart pembanding
# Diambil dari hasil analisis cluster pada notebook (461,580 data pengguna)
CLUSTER_PROFILE = {
    0: {'emotional_exhaustion': 0.22, 'stress_level': -0.49, 'daily_screen_time': -0.67,
        'doomscrolling_duration': 0.35, 'late_night_device_usage': 0.66, 'distraction_frequency': 0.37,
        'work_satisfaction': 0.27, 'sleep_hours': 0.37, 'physical_activity': -0.05,
        'deep_work_hours': 0.07, 'motivation_level': -0.46, 'mental_state': -0.18},
    1: {'emotional_exhaustion': -0.07, 'stress_level': 0.32, 'daily_screen_time': 0.32,
        'doomscrolling_duration': -0.14, 'late_night_device_usage': 0.72, 'distraction_frequency': 0.06,
        'work_satisfaction': -0.20, 'sleep_hours': -0.05, 'physical_activity': 0.07,
        'deep_work_hours': 0.08, 'motivation_level': 0.37, 'mental_state': -0.89},
    2: {'emotional_exhaustion': -0.01, 'stress_level': 0.02, 'daily_screen_time': 0.02,
        'doomscrolling_duration': -0.01, 'late_night_device_usage': -1.36, 'distraction_frequency': -0.01,
        'work_satisfaction': -0.01, 'sleep_hours': -0.01, 'physical_activity': -0.00,
        'deep_work_hours': -0.00, 'motivation_level': 0.01, 'mental_state': 0.01},
    3: {'emotional_exhaustion': -0.09, 'stress_level': 0.07, 'daily_screen_time': 0.19,
        'doomscrolling_duration': -0.12, 'late_night_device_usage': 0.73, 'distraction_frequency': -0.30,
        'work_satisfaction': -0.03, 'sleep_hours': -0.21, 'physical_activity': -0.01,
        'deep_work_hours': -0.11, 'motivation_level': 0.02, 'mental_state': 0.84},
}

# Statistik referensi dataset (461,580 baris hasil cleaning dari 5 juta sampel)
CLUSTER_SIZE_PCT = {0: 18.7, 1: 21.0, 2: 34.2, 3: 26.1}

CROSS_TAB = {
    'Aman':            {0: 0.139, 1: 0.051, 2: 0.194, 3: 0.158},
    'Waspada':         {0: 0.385, 1: 0.269, 2: 0.407, 3: 0.406},
    'Berisiko':        {0: 0.359, 1: 0.428, 2: 0.315, 3: 0.346},
    'Sangat Berisiko': {0: 0.117, 1: 0.252, 2: 0.084, 3: 0.090},
}

# =====================================================================
# KONFIGURASI FORM INPUT (label, tipe, rentang, default)
# =====================================================================
FIELD_CONFIG = {
    'age': {'label': 'Berapa usiamu? (tahun)', 'min': 17, 'max': 65, 'default': 22, 'step': 1},
    'occupation': {'label': 'Apa pekerjaan / statusmu saat ini?', 'options': OCCUPATION_OPTIONS},
    'work_mode': {'label': 'Bagaimana mode kerja atau kuliahmu?', 'options': WORK_MODE_OPTIONS},
    'remote_work_days': {'label': 'Berapa hari per minggu kamu kerja atau kuliah dari rumah?', 'min': 0, 'max': 7, 'default': 3, 'step': 1},
    'meeting_hours': {'label': 'Berapa jam rata rata kamu menghabiskan waktu untuk meeting atau kelas per hari?', 'min': 0.0, 'max': 10.0, 'default': 2.0, 'step': 0.5},
    'workspace_quality': {'label': 'Seberapa nyaman tempat kerja atau belajarmu?', 'help': '1 = sangat tidak nyaman, 10 = sangat nyaman', 'min': 0, 'max': 10, 'default': 6, 'step': 1},
    'internet_stability': {'label': 'Seberapa stabil koneksi internetmu?', 'help': '1 = sering putus putus, 10 = sangat stabil', 'min': 0, 'max': 10, 'default': 7, 'step': 1},

    'device_usage_type': {'label': 'Bagaimana cara kamu paling sering menggunakan gadget sehari hari?', 'options': DEVICE_USAGE_OPTIONS},
    'daily_screen_time': {'label': 'Berapa jam total screen time harianmu (HP, laptop, dll)?', 'help': 'Bisa dicek di Setelan > Digital Wellbeing (Android) atau Screen Time (iPhone)', 'min': 0.0, 'max': 16.0, 'default': 6.0, 'step': 0.5},
    'social_media_hours': {'label': 'Berapa jam kamu menghabiskan waktu di media sosial per hari?', 'help': 'Bisa dicek di Digital Wellbeing / Screen Time, biasanya ada rincian per aplikasi', 'min': 0.0, 'max': 10.0, 'default': 2.0, 'step': 0.5},

    'doomscrolling_duration': {
        'label': 'Seberapa sering kamu scroll tanpa tujuan (doomscrolling)?',
        'qual_options': [
            ('Hampir tidak pernah', 0.2),
            ('Sesekali, beberapa menit', 1.0),
            ('Cukup sering, hampir 1 jam', 2.0),
            ('Sering banget, bisa berjam jam', 4.0),
        ]
    },
    'late_night_device_usage': {
        'label': 'Seberapa sering kamu pakai gadget setelah jam 10 malam?',
        'qual_options': [
            ('Tidak pernah, HP sudah disimpan sebelum jam 10 malam', 0.0),
            ('Kadang kadang, cuma sebentar', 0.5),
            ('Sering, sekitar 1-2 jam', 1.5),
            ('Hampir tiap malam, lebih dari 2 jam', 3.0),
        ]
    },

    'deep_work_hours': {
        'label': 'Berapa banyak waktu fokus penuh (deep work) yang kamu punya per hari?',
        'qual_options': [
            ('Hampir tidak punya waktu fokus penuh', 0.5),
            ('Sekitar 1-2 jam sehari', 1.5),
            ('Sekitar 3-4 jam sehari', 3.5),
            ('Lebih dari 5 jam sehari', 6.0),
        ]
    },
    'distraction_frequency': {
        'label': 'Seberapa sering kamu teralihkan saat sedang bekerja atau belajar?',
        'qual_options': [
            ('Jarang, bisa fokus lama tanpa terganggu', 10),
            ('Sesekali teralihkan', 30),
            ('Cukup sering teralihkan', 60),
            ('Sangat sering, sulit fokus lama', 100),
        ]
    },
    'task_completion_rate': {'label': 'Berapa persen tugas hari ini yang berhasil kamu selesaikan?', 'min': 0, 'max': 100, 'default': 75, 'step': 5, 'format': '%d%%'},
    'concentration_score': {'label': 'Seberapa baik tingkat konsentrasimu belakangan ini?', 'help': '1 = sangat sulit fokus, 10 = sangat mudah fokus', 'min': 0, 'max': 10, 'default': 6, 'step': 1},

    'sleep_hours': {'label': 'Berapa jam kamu tidur per hari (rata rata)?', 'min': 0.0, 'max': 12.0, 'default': 7.0, 'step': 0.5},
    'sleep_quality': {'label': 'Bagaimana kualitas tidurmu belakangan ini?', 'help': '1 = sangat buruk dan tidak nyenyak, 10 = sangat nyenyak', 'min': 0, 'max': 10, 'default': 6, 'step': 1},
    'caffeine_intake': {'label': 'Berapa gelas kopi atau teh berkafein yang kamu minum per hari?', 'min': 0, 'max': 10, 'default': 1, 'step': 1},
    'physical_activity': {'label': 'Berapa jam kamu berolahraga atau beraktivitas fisik per hari?', 'min': 0.0, 'max': 5.0, 'default': 1.0, 'step': 0.5},

    'stress_level': {'label': 'Seberapa tinggi tingkat stres yang kamu rasakan belakangan ini?', 'help': '1 = sangat rendah, 10 = sangat tinggi', 'min': 0, 'max': 10, 'default': 5, 'step': 1},
    'motivation_level': {'label': 'Seberapa termotivasi kamu belakangan ini?', 'help': '1 = sangat tidak termotivasi, 10 = sangat termotivasi', 'min': 0.0, 'max': 10.0, 'default': 5.0, 'step': 0.5},
    'mental_fatigue': {'label': 'Seberapa lelah secara mental kamu belakangan ini?', 'help': '1 = sangat segar, 10 = sangat lelah', 'min': 0, 'max': 10, 'default': 5, 'step': 1},
    'emotional_exhaustion': {'label': 'Seberapa terkuras emosimu belakangan ini?', 'help': '1 = sangat rendah, 10 = sangat tinggi', 'min': 0, 'max': 10, 'default': 5, 'step': 1},
    'work_satisfaction': {'label': 'Seberapa puas kamu dengan pekerjaan atau kuliahmu saat ini?', 'help': '1 = sangat tidak puas, 10 = sangat puas', 'min': 0, 'max': 10, 'default': 5, 'step': 1},
    'mental_state': {'label': 'Secara umum, bagaimana kondisi mentalmu belakangan ini?', 'options': MENTAL_STATE_OPTIONS},
}

# Fitur yang tidak ditanyakan ke user (sulit dijawab presisi, dampak ke hasil kecil),
# diisi dengan nilai rata rata dari data training
DEFAULT_HIDDEN_VALUES = {
    'app_switch_frequency': 80,
    'notification_count': 100,
    'smartphone_unlocks': 80,
    'focus_sessions': 3,
}

FIELD_GROUPS = {
    'Profil & Aktivitas': ['age', 'occupation', 'work_mode', 'remote_work_days', 'meeting_hours', 'workspace_quality', 'internet_stability'],
    'Kebiasaan Digital': ['device_usage_type', 'daily_screen_time', 'social_media_hours', 'doomscrolling_duration', 'late_night_device_usage'],
    'Fokus & Produktivitas': ['deep_work_hours', 'distraction_frequency', 'task_completion_rate', 'concentration_score'],
    'Kesehatan & Energi': ['sleep_hours', 'sleep_quality', 'caffeine_intake', 'physical_activity'],
    'Kondisi Mental': ['stress_level', 'motivation_level', 'mental_fatigue', 'emotional_exhaustion', 'work_satisfaction', 'mental_state'],
}


# =====================================================================
# PIPELINE PREDIKSI (klasifikasi + cluster + SHAP)
# =====================================================================
def predict_burnout(user_input, top_n=5):
    full_input = {**DEFAULT_HIDDEN_VALUES, **user_input}
    input_df = pd.DataFrame([full_input])[FEATURE_ORDER]

    input_df[CAT_COLS] = oe.transform(input_df[CAT_COLS])
    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

    burnout_pred = best_xgb.predict(input_df)[0]
    burnout_proba = best_xgb.predict_proba(input_df)[0]

    input_cluster = input_df[CLUSTER_FEATURES].copy()
    input_cluster['mental_state'] = cluster_scaler.transform(input_cluster[['mental_state']])
    cluster_pred = int(kmeans_final.predict(input_cluster)[0])

    shap_result = explainer(input_df)
    shap_for_pred_class = shap_result.values[0, :, burnout_pred]

    feature_contributions = pd.Series(shap_for_pred_class, index=input_df.columns)
    feature_contributions = feature_contributions.sort_values(key=abs, ascending=False)

    top_features = []
    for feat, val in feature_contributions.head(top_n).items():
        top_features.append({
            'fitur': feat,
            'label': SHORT_LABELS.get(feat, feat),
            'kontribusi': float(val),
            'arah': 'meningkatkan risiko' if val > 0 else 'menurunkan risiko'
        })

    return {
        'burnout_label': LABEL_MAP[burnout_pred],
        'burnout_confidence': float(burnout_proba[burnout_pred]),
        'all_proba': {LABEL_MAP[i]: float(p) for i, p in enumerate(burnout_proba)},
        'cluster_id': cluster_pred,
        'cluster': recommendations[cluster_pred]['nama'],
        'cluster_deskripsi': recommendations[cluster_pred]['deskripsi'],
        'rekomendasi': recommendations[cluster_pred]['rekomendasi'],
        'top_features': top_features,
        'user_cluster_profile': input_cluster.iloc[0].to_dict()
    }


# =====================================================================
# KOMPONEN VISUAL
# =====================================================================
def plot_shap_bar(top_features):
    feats = [f['label'] for f in top_features][::-1]
    values = [f['kontribusi'] for f in top_features][::-1]
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(feats, values, color=colors)
    ax.axvline(0, color='#94a3b8', linewidth=1)
    ax.set_xlabel('Kontribusi terhadap risiko (SHAP value)')
    ax.set_title('Faktor paling berpengaruh pada hasil prediksimu')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_radar(user_profile, cluster_profile, cluster_name):
    categories = CLUSTER_FEATURES
    labels = [SHORT_LABELS[c] for c in categories]

    user_vals = [user_profile[c] for c in categories]
    cluster_vals = [cluster_profile[c] for c in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    user_vals += user_vals[:1]
    cluster_vals += cluster_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, user_vals, color='#6366f1', linewidth=2, label='Kamu')
    ax.fill(angles, user_vals, color='#6366f1', alpha=0.2)
    ax.plot(angles, cluster_vals, color='#94a3b8', linewidth=2, linestyle='--', label=f'Rata rata {cluster_name}')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Posisimu dibanding rata rata zona', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_cluster_distribution():
    names = [recommendations[i]['nama'] for i in range(4)]
    sizes = [CLUSTER_SIZE_PCT[i] for i in range(4)]
    colors = ['#6366f1', '#ef4444', '#22c55e', '#f97316']

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(sizes, labels=names, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 8})
    ax.set_title('Distribusi Zona dari 461,580 Profil Pengguna')
    fig.tight_layout()
    return fig


def plot_cross_tab():
    cluster_names = [recommendations[i]['nama'] for i in range(4)]
    bottom = np.zeros(4)
    colors = {'Aman': '#22c55e', 'Waspada': '#eab308', 'Berisiko': '#f97316', 'Sangat Berisiko': '#ef4444'}

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in colors.items():
        vals = [CROSS_TAB[label][i] for i in range(4)]
        ax.bar(cluster_names, vals, bottom=bottom, label=label, color=color)
        bottom += np.array(vals)

    ax.set_ylabel('Proporsi')
    ax.set_title('Proporsi Tingkat Burnout di Setiap Zona')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=15, ha='right', fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 Burnout Risk Checker</h1>
    <p>Analisis risiko burnout berbasis Machine Learning, dilatih dari 461,580 profil digital
    pekerja dan mahasiswa, lengkap dengan penjelasan SHAP dan rekomendasi recovery yang dipersonalisasi.</p>
</div>
""", unsafe_allow_html=True)

tab_home, tab_check, tab_about = st.tabs(["🏠 Beranda", "📝 Cek Risiko Burnout", "📊 Tentang Model & Data"])

# =====================================================================
# TAB BERANDA
# =====================================================================
with tab_home:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Prediksi Akurat")
        st.write("Model XGBoost yang dituning dengan Optuna, mencapai ROC-AUC 0.90 dalam mengklasifikasikan 4 tingkat risiko burnout.")
    with col2:
        st.markdown("### 🔍 Bisa Dijelaskan")
        st.write("Setiap hasil prediksi disertai analisis SHAP yang menunjukkan faktor spesifik mana yang paling memengaruhi kondisimu, bukan sekadar angka.")
    with col3:
        st.markdown("### 🧭 Rekomendasi Personal")
        st.write("Kamu akan dikelompokkan ke salah satu dari 4 zona perilaku, masing masing dengan rekomendasi recovery yang berbeda.")

    st.markdown("---")
    st.markdown("""
    **Cara pakai:** buka tab **Cek Risiko Burnout**, isi seluruh pertanyaan tentang kebiasaan harianmu
    (dibagi dalam beberapa bagian agar mudah diikuti), lalu klik tombol Analisis. Hasilnya akan
    langsung muncul lengkap dengan penjelasan dan rekomendasi. Seluruh proses berjalan secara lokal,
    tidak ada data yang dikirim ke server pihak ketiga.
    """)

# =====================================================================
# TAB CEK RISIKO
# =====================================================================
with tab_check:
    if 'view' not in st.session_state:
        st.session_state['view'] = 'form'
    if 'result' not in st.session_state:
        st.session_state['result'] = None

    # -----------------------------------------------------------
    # VIEW: FORM
    # -----------------------------------------------------------
    if st.session_state['view'] == 'form':
        st.markdown("Isi seluruh pertanyaan di bawah ini berdasarkan kondisimu dalam beberapa hari terakhir.")

        user_input = {}

        for group_name, fields in FIELD_GROUPS.items():
            st.markdown(f"#### {group_name}")
            cols = st.columns(2)
            for idx, field in enumerate(fields):
                cfg = FIELD_CONFIG[field]
                col = cols[idx % 2]
                with col:
                    if 'qual_options' in cfg:
                        qual_map = dict(cfg['qual_options'])
                        selected_label = st.selectbox(
                            cfg['label'], list(qual_map.keys()), key=field,
                            help=cfg.get('help')
                        )
                        user_input[field] = qual_map[selected_label]
                    elif 'options' in cfg:
                        user_input[field] = st.selectbox(
                            cfg['label'], cfg['options'], key=field,
                            help=cfg.get('help')
                        )
                    else:
                        slider_kwargs = dict(
                            min_value=cfg['min'], max_value=cfg['max'],
                            value=cfg['default'], step=cfg['step'], key=field,
                            help=cfg.get('help')
                        )
                        if 'format' in cfg:
                            slider_kwargs['format'] = cfg['format']
                        user_input[field] = st.slider(cfg['label'], **slider_kwargs)
            st.divider()

        analyze = st.button("🔍 Analisis Risiko Burnout Saya", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("Menganalisis profil kamu..."):
                result = predict_burnout(user_input)
            st.session_state['result'] = result
            st.session_state['view'] = 'hasil'
            st.rerun()

    # -----------------------------------------------------------
    # VIEW: HASIL
    # -----------------------------------------------------------
    else:
        result = st.session_state['result']

        if st.button("← Ubah Jawaban"):
            st.session_state['view'] = 'form'
            st.rerun()

        st.markdown("## Hasil Analisis")

        col_main, col_zone = st.columns([1, 1.4])

        with col_main:
            color = CARD_COLOR[result['burnout_label']]
            st.markdown(f"""
            <div class="metric-card" style="background-color: {color};">
                <p>Tingkat Risiko Burnout</p>
                <h2>{result['burnout_label']}</h2>
                <p>Tingkat keyakinan model: {result['burnout_confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Distribusi Probabilitas")
            proba_df = pd.DataFrame({
                'Kategori': list(result['all_proba'].keys()),
                'Probabilitas': list(result['all_proba'].values())
            })
            st.bar_chart(proba_df.set_index('Kategori'))

        with col_zone:
            st.markdown(f"""
            <div class="zone-card">
                <h4>📍 Kamu masuk ke <b>{result['cluster']}</b></h4>
                <p>{result['cluster_deskripsi']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 💡 Rekomendasi untukmu")
            for rec in result['rekomendasi']:
                st.markdown(f"<div class='rekom-item'>✅ {rec}</div>", unsafe_allow_html=True)

        st.markdown("---")
        col_shap, col_radar = st.columns(2)

        with col_shap:
            st.markdown("##### Mengapa hasilnya seperti ini?")
            st.pyplot(plot_shap_bar(result['top_features']))
            st.caption("Batang merah meningkatkan risiko burnout, batang biru justru menurunkannya, dibandingkan rata rata pengguna lain.")

        with col_radar:
            st.markdown("##### Profil kebiasaanmu vs rata rata zona")
            st.pyplot(plot_radar(result['user_cluster_profile'], CLUSTER_PROFILE[result['cluster_id']], result['cluster']))
            st.caption("Semakin jauh dari titik tengah, semakin tinggi nilai fitur tersebut dibanding rata rata seluruh pengguna.")

# =====================================================================
# TAB TENTANG MODEL & DATA
# =====================================================================
with tab_about:
    st.markdown("### Tentang Sistem Ini")
    st.markdown("""
    Dashboard ini dibangun di atas pipeline Machine Learning dua lapis yang dilatih dari
    **461,580 profil pengguna** (hasil cleaning dari 500,000 sampel acak dataset 5 juta baris
    Digital Burnout & Productivity Analytics).
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Klasifikasi", "XGBoost (tuned)")
    col2.metric("Akurasi", "69%")
    col3.metric("ROC-AUC", "0.90")
    col4.metric("Jumlah Fitur", "30")

    st.markdown("---")
    st.markdown("### 4 Zona Perilaku Burnout")
    st.markdown("""
    Selain klasifikasi risiko, sistem ini mengelompokkan pengguna ke dalam 4 zona perilaku
    menggunakan KMeans Clustering, untuk memberikan rekomendasi yang lebih relevan.
    """)

    col_pie, col_stack = st.columns(2)
    with col_pie:
        st.pyplot(plot_cluster_distribution())
    with col_stack:
        st.pyplot(plot_cross_tab())

    st.markdown("---")
    st.markdown("### Mengapa Tanpa LLM atau API Eksternal?")
    st.markdown("""
    Dashboard ini sepenuhnya berjalan dari model yang sudah dilatih sendiri (XGBoost + KMeans + SHAP)
    dan tersimpan secara lokal. Keuntungannya:

    - **Privasi**, data yang kamu masukkan tidak pernah dikirim ke server pihak ketiga atau API LLM manapun
    - **Konsisten**, hasil prediksi selalu sama untuk input yang sama, tidak tergantung respons model bahasa yang bisa berubah ubah
    - **Bisa dijelaskan**, setiap angka kontribusi pada grafik SHAP berasal dari struktur pohon keputusan model, bukan teks yang digenerate
    - **Cepat**, tidak ada latensi panggilan API ke server eksternal
    """)

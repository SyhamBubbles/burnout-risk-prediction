# Burnout Risk Prediction and Personalized Recovery Recommendation System

Sistem prediksi risiko burnout berbasis Machine Learning dengan pendekatan dua lapis (klasifikasi dan clustering), dilengkapi penjelasan SHAP dan rekomendasi recovery yang dipersonalisasi. Proyek ini dikembangkan sebagai capstone project pada program Tempa Dicoding (Artificial Intelligence).

## Tentang Proyek

Dashboard ini menganalisis kebiasaan digital, pola kerja, dan kondisi mental seseorang untuk:

1. Memprediksi tingkat risiko burnout (Aman, Waspada, Berisiko, Sangat Berisiko)
2. Menjelaskan faktor faktor utama yang memengaruhi hasil prediksi menggunakan SHAP
3. Mengelompokkan pengguna ke dalam salah satu dari 4 zona perilaku (Zona Distraksi, Zona Rentan, Zona Seimbang, Zona Kurang Istirahat)
4. Memberikan rekomendasi recovery yang sesuai dengan zona masing masing

## Dataset

Model dilatih menggunakan 461.580 profil pengguna, hasil cleaning dari 500.000 sampel acak dari dataset publik Digital Burnout & Productivity Analytics (5 juta baris) di Kaggle.

https://www.kaggle.com/datasets/aiexplorer77/digital-burnout-and-productivity-analytics

Dataset asli tidak disertakan dalam repository ini karena ukurannya melebihi batas GitHub. Model yang sudah dilatih (file .pkl) dan dashboard Streamlit (app.py) tidak membutuhkan dataset ini, jadi langsung bisa dijalankan.

Dataset hanya diperlukan jika ingin menjalankan ulang notebook Burnout_PredictionNew.ipynb dari awal (training ulang). Caranya:

1. Download dataset dari link Kaggle di atas
2. Simpan file CSV di folder yang sama dengan notebook
3. Jalankan notebook dengan Run All

## Model

- **Klasifikasi**: XGBoost yang dituning menggunakan Optuna, ROC-AUC 0.90, akurasi 69%
- **Clustering**: MiniBatchKMeans dengan k=4, divalidasi menggunakan silhouette score dan distribusi label burnout per cluster
- **Explainability**: SHAP TreeExplainer untuk penjelasan per prediksi

## Struktur Proyek

```
.
├── Burnout_PredictionNew.ipynb   # Notebook lengkap (EDA, modeling, clustering, SHAP)
├── app.py                        # Dashboard Streamlit
├── requirements.txt              # Daftar dependency
├── model_xgb.pkl                 # Model klasifikasi terlatih
├── ordinal_encoder.pkl           # Encoder fitur kategorik
├── standard_scaler.pkl           # Scaler fitur numerik
├── kmeans_model.pkl              # Model clustering terlatih
├── cluster_scaler.pkl            # Scaler khusus fitur clustering
└── recommendations.json          # Mapping zona ke rekomendasi
```

## Cara Menjalankan

1. Clone repository ini
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan dashboard:

```bash
streamlit run app.py
```

## Tentang Dashboard

Dashboard berjalan sepenuhnya secara lokal tanpa API eksternal atau LLM. Seluruh prediksi, penjelasan SHAP, dan rekomendasi dihasilkan dari model Machine Learning yang sudah dilatih dan disimpan, sehingga cepat, konsisten, dan menjaga privasi data pengguna.

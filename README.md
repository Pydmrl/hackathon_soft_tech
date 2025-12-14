# 🏠 Emlak Değerlendirme Asistanı - AI_Spark_Team

Machine Learning tabanlı emlak fiyat tahmin ve yatırım danışmanlığı sistemi.

## 📊 Model Performansı

- **Test MAPE**: 18.26% (Mükemmel!)
- **Test R²**: 0.8201
- **Model**: XGBoost + OOF Correction
- **Features**: 72 özellik

### Segment Bazlı Performans:
- **<300K**: 17.6% MAPE ✅
- **300K-500K**: 15.0% MAPE ✅ (En iyi!)
- **500K-1M**: 19.7% MAPE ✅
- **1M-2M**: 25.3% MAPE ⚠️
- **>2M**: 27.6% MAPE ⚠️

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlat

```bash
python app.py
```

### 3. Tarayıcıda Aç

```
http://localhost:5000
```

## 📁 Dosya Yapısı (Profesyonel Mimari)

```
Soft_tech_ml/
├── app.py                              # Flask application factory
├── app_old.py                           # Eski monolitik versiyon (yedek)
├── requirements.txt                     # Python bağımlılıkları
├── README.md                            # Dokümantasyon
│
├── config/                              # ⚙️ Konfigürasyon modülü
│   ├── __init__.py
│   └── config.py                        # Development/Production config
│
├── services/                            # 🔧 Business Logic servisleri
│   ├── __init__.py
│   ├── model_service.py                 # Model yükleme ve yönetimi
│   ├── feature_engineering.py           # Feature hesaplamaları
│   └── prediction_service.py            # Tahmin ve correction logic
│
├── api/                                 # 🌐 REST API endpoints
│   ├── __init__.py
│   └── routes.py                        # Flask Blueprint routes
│
├── models/                              # 🤖 Eğitilmiş ML modelleri
│   ├── investment_advisor_model_v7.pkl  # XGBoost model + metadata
│   └── location_data_v7.pkl             # İlçe/mahalle verileri
│
├── templates/                           # 🎨 HTML templates
│   └── index.html                       # Web arayüzü
│
├── static/                              # 📂 Statik dosyalar
│   ├── css/
│   │   └── style.css                    # Glassmorphism tasarım
│   └── js/
│       └── script.js                    # Frontend JavaScript
│
└── logs/                                # 📝 Log dosyaları
    └── app.log
```

## 🎯 Özellikler

✅ **Gerçek Zamanlı Tahmin**: XGBoost modeli ile anlık fiyat tahmini
✅ **OOF-Based Correction**: Overfitting önleme ve doğruluk artırma
✅ **Segment-Aware**: Fiyat segmentine göre özel tahmin
✅ **Güven Aralığı**: %70 güven aralığı ile risk değerlendirmesi
✅ **Akıllı Tavsiye**: FIRSAT/NORMAL/PAHALI analizi
✅ **Responsive Tasarım**: Mobil uyumlu modern arayüz
✅ **Glassmorphism UI**: Profesyonel ve modern görünüm

## 🔧 Teknik Detaylar

### Profesyonel Mimari:
1. **Factory Pattern**: `create_app()` ile esneklik
2. **Service Layer**: Business logic ayrı servisler
3. **Blueprint Pattern**: API routes modularıte
4. **Config Management**: Environment-based konfigürasyon
5. **Dependency Injection**: Servislerin bağımsız testi

### Model Pipeline:
1. **Feature Engineering**: 72 özellik (m² segment, location encoding, interactions)
2. **Target Encoding**: K-Fold cross-validation ile leakage önleme
3. **XGBoost Regressor**: 350 estimators, max_depth=6
4. **OOF Correction**: Out-of-fold residual based correction
5. **Segment Weighting**: Büyük evlere daha fazla ağırlık

### API Endpoints:

- `GET /` - Ana sayfa
- `POST /get_neighborhoods` - İlçeye göre mahalle listesi
- `POST /predict` - Fiyat tahmini

### Tahmin İşlemi:

```python
# Input
{
  "ilce": "Kadıköy",
  "mahalle": "Fenerbahçe",
  "net_m2": 120,
  "oda": "3+1",
  "bina_yasi": "5-10 between",
  ...
}

# Output
{
  "prediction": "1,250,000 TL",
  "fair_value": "1,250,000 TL",
  "lower_bound": "1,100,000 TL",
  "upper_bound": "1,400,000 TL",
  "advice": "TAM PİYASA DEĞERİNDE - Normal fiyat",
  "status_class": "normal",
  "difference": "%+0.5",
  "reliability": "85%",
  "correction_applied": "+2.3%"
}
```

## 📈 Model Geliştirme Süreci

### v1-v3: İlk Prototipler
- Temel feature engineering
- Random Forest baseline

### v4-v6: Iyileştirmeler
- XGBoost geçişi
- Leakage düzeltmeleri
- Segment-based features

### v7: Final Version ⭐
- OOF-based correction
- K-fold target encoding
- Dengeli regularization
- Random train/test split (temporal bias fix)

## 🎨 Frontend Özellikleri

- **Glassmorphism Design**: Modern, şeffaf cam efekti
- **Responsive Layout**: Desktop ve mobil uyumlu
- **Accordion Form**: Kolay kullanım
- **Real-time Updates**: Anlık sonuç gösterimi
- **Bootstrap 5**: Modern UI components

## ⚠️ Notlar

- Model sadece "Krediye Uygun" evler için eğitildi
- En iyi performans 300K-500K segmentinde
- Yüksek fiyat segmentlerinde (%25+ MAPE) dikkatli kullanın
- Tahminler %70 güven aralığı ile verilir

**© 2025 AI_Spark_Team - SoftTech Emlak Asistanı**

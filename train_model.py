#!/usr/bin/env python3
"""
================================================================================
🏠 YATIRIM DANIŞMANI - MODEL v7 (BALANCED & CORRECTED)
================================================================================
Önceki versiyonlardaki sorunlar düzeltildi:

✅ Segment LEAKAGE düzeltildi (fiyattan türetilen segment feature YOK)
✅ m² bazlı pseudo-segment kullanılıyor (leakage yok)
✅ OOF (Out-of-Fold) residual ile correction (in-sample ezber yok)
✅ Tarih feature'ları eklendi
✅ Dengeli regularization (ne çok agresif ne çok gevşek)
✅ Segment ağırlıkları yumuşatıldı
✅ Correction ağırlıkları segmente göre dengelendi
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    USE_XGB = True
    print("✅ XGBoost kullanılacak")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_XGB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = 'C:\\Users\\ASUS\\Desktop\\hekatonDogru\\hekatonDogru\\home\\claude\\hackathon_train_set.csv'

print("\n" + "="*80)
print("🏠 YATIRIM DANIŞMANI - MODEL v7 (BALANCED & CORRECTED)")
print("="*80)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("\n📂 1. VERİ YÜKLEME...")

df = pd.read_csv(DATA_FILE, sep=';', encoding='utf-8')

def clean_price(price_str):
    if pd.isna(price_str): return np.nan
    price_str = str(price_str).replace('TL', '').replace('.', '').replace(',', '.').strip()
    try: return float(price_str)
    except: return np.nan

df['Price_Clean'] = df['Price'].apply(clean_price)
df = df[df['Price_Clean'].notna()].copy()
df_loan = df[df['Available for Loan'] == 'Yes'].copy()

# Outlier temizleme
Q1 = df_loan['Price_Clean'].quantile(0.01)
Q3 = df_loan['Price_Clean'].quantile(0.99)
df_clean = df_loan[(df_loan['Price_Clean'] >= Q1) & (df_loan['Price_Clean'] <= Q3)].copy()
print(f"   ✅ Veri: {len(df_clean):,} kayıt")

# ============================================================================
# 2. TARİH FEATURE'LARI
# ============================================================================
print("\n📅 2. TARİH FEATURE'LARI...")

# İlan tarihini parse et
df_clean['Ad_Date'] = pd.to_datetime(df_clean['Adrtisement Date'], format='%d.%m.%Y', errors='coerce')

# Referans tarih (veri setinin ortası)
ref_date = pd.Timestamp('2020-01-01')

# Tarih feature'ları
df_clean['Days_Since_Ref'] = (df_clean['Ad_Date'] - ref_date).dt.days.fillna(0)
df_clean['Ad_Month'] = df_clean['Ad_Date'].dt.month.fillna(6)
df_clean['Ad_Quarter'] = df_clean['Ad_Date'].dt.quarter.fillna(2)
df_clean['Is_YearEnd'] = df_clean['Ad_Month'].isin([11, 12, 1]).astype(int)  # Yıl sonu/başı

print(f"   ✅ Tarih feature'ları oluşturuldu")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n🔧 3. FEATURE ENGINEERING...")

def extract_age(age_str):
    if pd.isna(age_str): return 15
    age_str = str(age_str).lower().strip()
    if age_str.isdigit(): return int(age_str)
    if 'between' in age_str or '-' in age_str:
        nums = [int(s) for s in age_str.replace('-', ' ').replace('between', '').split() if s.isdigit()]
        if nums: return np.mean(nums)
    if 'more' in age_str or '31' in age_str: return 35
    return 15

def extract_rooms(room_str):
    if pd.isna(room_str): return 3
    room_str = str(room_str).strip()
    if '+' in room_str:
        parts = room_str.split('+')
        try: return float(parts[0]) + float(parts[1])
        except: pass
    return 3

def extract_floor(floor_str):
    floor_mapping = {
        'Garden Floor': 0, 'Ground floor': 0, 'Entrance floor': 1,
        'High entrance': 1, 'Mezzanine': 0.5, 'Penthouse': 20, 'Top floor': 15
    }
    if pd.isna(floor_str): return 3
    floor_str = str(floor_str).strip()
    if floor_str in floor_mapping: return floor_mapping[floor_str]
    try: return int(floor_str)
    except: return 3

# Numerik dönüşümler
for col in ['m² (Gross)', 'm² (Net)']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean['Building_Age'] = df_clean['Building Age'].apply(extract_age)
df_clean['Total_Rooms'] = df_clean['Number of rooms'].apply(extract_rooms)
df_clean['Floor_Num'] = df_clean['Floor location'].apply(extract_floor)
df_clean['Num_Bathrooms'] = pd.to_numeric(df_clean['Number of bathrooms'], errors='coerce').fillna(1)
df_clean['Num_Floors'] = pd.to_numeric(df_clean['Number of floors'], errors='coerce').fillna(10)

# m² özellikleri
df_clean['m2_Net'] = df_clean['m² (Net)'].fillna(df_clean['m² (Gross)'])
df_clean['m2_Gross'] = df_clean['m² (Gross)'].fillna(df_clean['m² (Net)'])
df_clean['m2_Log'] = np.log1p(df_clean['m2_Net'])
df_clean['m2_Sqrt'] = np.sqrt(df_clean['m2_Net'])
df_clean['m2_Ratio'] = df_clean['m2_Net'] / df_clean['m2_Gross'].replace(0, np.nan)
df_clean['m2_per_Room'] = df_clean['m2_Net'] / df_clean['Total_Rooms'].replace(0, np.nan)

# ⭐ m² BAZLI PSEUDO-SEGMENT (Fiyat değil, m² bazlı - LEAKAGE YOK)
# Bu, fiyat segmenti yerine kullanılacak
def get_m2_segment(m2):
    if m2 < 70: return 1      # Küçük
    elif m2 < 100: return 2   # Orta-küçük
    elif m2 < 130: return 3   # Orta
    elif m2 < 170: return 4   # Büyük
    else: return 5            # Çok büyük

df_clean['m2_Segment'] = df_clean['m2_Net'].apply(get_m2_segment)

# Yaş özellikleri
df_clean['Age_Decay'] = np.exp(-df_clean['Building_Age'] / 12)
df_clean['Is_New'] = (df_clean['Building_Age'] <= 5).astype(int)
df_clean['Is_Old'] = (df_clean['Building_Age'] > 20).astype(int)

# Kat özellikleri
df_clean['Floor_Relative'] = df_clean['Floor_Num'] / df_clean['Num_Floors'].replace(0, np.nan)
df_clean['Is_Ground'] = (df_clean['Floor_Num'] <= 1).astype(int)
df_clean['Is_High'] = (df_clean['Floor_Num'] >= 6).astype(int)

# Isıtma kalitesi
heating_quality = {
    'Underfloor heating': 5, 'Floor Heating': 5, 'Central system': 4,
    'Center (Share Meter)': 4, 'Natural Gas (Combi)': 3, 'Climate': 3, 'VRV': 4,
    'Heat pump': 2, 'Fireplace': 2, 'Stove': 1, 'Absent': 0, 'Solar energy': 2
}
df_clean['Heating_Quality'] = df_clean['Heating'].map(heating_quality).fillna(3)

# Kullanım durumu ve satıcı
status_map = {'Free': 3, 'Property owner': 2, 'Tenant': 1}
df_clean['Using_Status'] = df_clean['Using status'].map(status_map).fillna(2)

seller_map = {'From the real estate office': 2, 'From the construction company': 3, 'From owner': 1, 'From bank': 2}
df_clean['Seller_Type'] = df_clean['From who'].map(seller_map).fillna(2)

# Binary özellikler
binary_features = [
    'Elevator', 'Balcony', 'Parking Lot', 'Security', 'Air conditioning',
    'Swimming Pool (Open)', 'Swimming Pool (Indoor)', 'Sauna', 'Gym',
    'Jacuzzi', 'Smart House', 'Terrace', 'Garden', 'Steel door',
    'Video intercom', 'Alarm (Thief)', 'Alarm (Fire)', 'Generator',
    'Closed Garage', 'Thermal Insulation'
]

for feature in binary_features:
    if feature in df_clean.columns:
        df_clean[f'{feature}_bin'] = pd.to_numeric(df_clean[feature], errors='coerce').fillna(0)
        df_clean[f'{feature}_bin'] = (df_clean[f'{feature}_bin'] > 0).astype(int)

# Skorlar
luxury_features = ['Swimming Pool (Open)_bin', 'Swimming Pool (Indoor)_bin', 'Sauna_bin',
                   'Jacuzzi_bin', 'Smart House_bin', 'Generator_bin', 'Gym_bin']
df_clean['Luxury_Score'] = sum([df_clean.get(f, 0) for f in luxury_features])

comfort_features = ['Elevator_bin', 'Balcony_bin', 'Air conditioning_bin', 'Parking Lot_bin', 'Terrace_bin', 'Garden_bin']
df_clean['Comfort_Score'] = sum([df_clean.get(f, 0) for f in comfort_features])

security_features = ['Security_bin', 'Steel door_bin', 'Video intercom_bin', 'Alarm (Thief)_bin', 'Alarm (Fire)_bin']
df_clean['Security_Score'] = sum([df_clean.get(f, 0) for f in security_features])

df_clean['Total_Features'] = df_clean['Luxury_Score'] + df_clean['Comfort_Score'] + df_clean['Security_Score']

# Premium ilçeler
premium_districts = ['Beşiktaş', 'Sarıyer', 'Kadıköy', 'Bakırköy', 'Üsküdar', 'Şişli', 'Beyoğlu']
mid_premium_districts = ['Ataşehir', 'Maltepe', 'Kartal', 'Beykoz', 'Fatih', 'Zeytinburnu']
budget_districts = ['Sultangazi', 'Esenyurt', 'Bağcılar', 'Esenler', 'Güngören', 'Arnavutköy']

df_clean['Is_Premium_District'] = df_clean['District'].isin(premium_districts).astype(int)
df_clean['Is_Mid_Premium'] = df_clean['District'].isin(mid_premium_districts).astype(int)
df_clean['Is_Budget_District'] = df_clean['District'].isin(budget_districts).astype(int)

# ⭐ Lüks konut göstergesi (m² ve lüks skoruna göre - fiyat değil!)
df_clean['Is_Luxury_Property'] = ((df_clean['Luxury_Score'] >= 2) | (df_clean['m2_Net'] >= 150)).astype(int)

df_clean['Price_Log'] = np.log1p(df_clean['Price_Clean'])

print(f"   ✅ Feature engineering tamamlandı")

# ============================================================================
# 4. TRAIN/TEST SPLIT
# ============================================================================
print("\n📊 4. TRAIN/TEST SPLIT...")

np.random.seed(42)
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(df_clean) * 0.8)
df_train = df_clean.iloc[:split_idx].copy()
df_test = df_clean.iloc[split_idx:].copy()

print(f"   Train: {len(df_train):,} | Test: {len(df_test):,}")

# ============================================================================
# 5. TARGET ENCODING (K-Fold)
# ============================================================================
print("\n🔐 5. TARGET ENCODING...")

def target_encode_kfold(df_train, df_test, col, target_col, n_splits=5, smoothing=30):
    global_mean = df_train[target_col].mean()
    train_encoded = pd.Series(index=df_train.index, dtype=float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df_train):
        train_fold = df_train.iloc[train_idx]
        val_indices = df_train.iloc[val_idx].index
        
        agg = train_fold.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        train_encoded.loc[val_indices] = df_train.loc[val_indices, col].map(smoothed)
    
    train_encoded = train_encoded.fillna(global_mean)
    
    agg_full = df_train.groupby(col)[target_col].agg(['mean', 'count'])
    smoothed_full = (agg_full['mean'] * agg_full['count'] + global_mean * smoothing) / (agg_full['count'] + smoothing)
    test_encoded = df_test[col].map(smoothed_full).fillna(global_mean)
    
    return train_encoded, test_encoded, smoothed_full.to_dict(), global_mean

# Encoding
df_train['District_Encoded'], df_test['District_Encoded'], district_encoding, district_global = \
    target_encode_kfold(df_train, df_test, 'District', 'Price_Log', smoothing=40)

df_train['Neighborhood_Encoded'], df_test['Neighborhood_Encoded'], neighborhood_encoding, neighborhood_global = \
    target_encode_kfold(df_train, df_test, 'Neighborhood', 'Price_Log', smoothing=20)

# District varyans (heterojenlik göstergesi)
district_variance = df_train.groupby('District')['Price_Log'].std().fillna(0.3).to_dict()
df_train['District_Variance'] = df_train['District'].map(district_variance).fillna(0.3)
df_test['District_Variance'] = df_test['District'].map(district_variance).fillna(0.3)

# Neighborhood count
neighborhood_count = df_train.groupby('Neighborhood').size().to_dict()
df_train['Neighborhood_Count'] = np.log1p(df_train['Neighborhood'].map(neighborhood_count).fillna(1))
df_test['Neighborhood_Count'] = np.log1p(df_test['Neighborhood'].map(neighborhood_count).fillna(1))

print(f"   ✅ Encoding tamamlandı")

# ============================================================================
# 6. FEATURE INTERACTIONS (m² segment bazlı - fiyat segmenti YOK)
# ============================================================================
print("\n🔗 6. FEATURE INTERACTIONS...")

def create_interactions(df):
    # Lokasyon x m²
    df['m2_x_District'] = df['m2_Log'] * df['District_Encoded']
    df['m2_x_Neighborhood'] = df['m2_Log'] * df['Neighborhood_Encoded']
    
    # Yaş x lokasyon
    df['Age_x_District'] = df['Age_Decay'] * df['District_Encoded']
    
    # Kalite x lokasyon
    df['Quality_x_District'] = df['Total_Features'] * df['District_Encoded']
    df['Luxury_x_District'] = df['Luxury_Score'] * df['District_Encoded']
    
    # ⭐ m² SEGMENT bazlı interactions (fiyat segmenti YOK)
    df['m2Seg_x_District'] = df['m2_Segment'] * df['District_Encoded']
    df['m2Seg_x_Luxury'] = df['m2_Segment'] * df['Luxury_Score']
    df['m2Seg_x_Premium'] = df['m2_Segment'] * df['Is_Premium_District']
    
    # Premium kombinasyonları
    df['Premium_x_m2'] = df['Is_Premium_District'] * df['m2_Log']
    df['Premium_x_Luxury'] = df['Is_Premium_District'] * df['Luxury_Score']
    df['Luxury_Property_x_District'] = df['Is_Luxury_Property'] * df['District_Encoded']
    
    # m² x temel özellikler
    df['m2_x_Rooms'] = df['m2_Log'] * df['Total_Rooms'].fillna(3)
    df['m2_x_Age'] = df['m2_Log'] * df['Age_Decay']
    df['m2_x_Bathrooms'] = df['m2_Log'] * df['Num_Bathrooms']
    
    # Yaş x kalite
    df['New_x_Luxury'] = df['Is_New'] * df['Luxury_Score']
    df['New_x_Premium'] = df['Is_New'] * df['Is_Premium_District']
    
    return df

df_train = create_interactions(df_train)
df_test = create_interactions(df_test)

print(f"   ✅ Interaction'lar oluşturuldu")

# ============================================================================
# 7. FEATURE SET
# ============================================================================
print("\n📋 7. FEATURE SET...")

features = [
    # Temel m²
    'm2_Net', 'm2_Log', 'm2_Sqrt', 'm2_Ratio', 'm2_per_Room',
    'Total_Rooms', 'Num_Bathrooms',
    
    # m² segment (fiyat segmenti DEĞİL!)
    'm2_Segment',
    
    # Yaş
    'Building_Age', 'Age_Decay', 'Is_New', 'Is_Old',
    
    # Kat
    'Floor_Num', 'Floor_Relative', 'Num_Floors', 'Is_Ground', 'Is_High',
    
    # Lokasyon
    'District_Encoded', 'Neighborhood_Encoded', 'District_Variance', 'Neighborhood_Count',
    
    # Kategorik
    'Heating_Quality', 'Using_Status', 'Seller_Type',
    
    # Skorlar
    'Luxury_Score', 'Comfort_Score', 'Security_Score', 'Total_Features',
    
    # Premium
    'Is_Premium_District', 'Is_Mid_Premium', 'Is_Budget_District', 'Is_Luxury_Property',
    
    # Tarih
    'Days_Since_Ref', 'Ad_Month', 'Ad_Quarter', 'Is_YearEnd',
    
    # Interactionlar (m² segment bazlı)
    'm2_x_District', 'm2_x_Neighborhood', 'Age_x_District',
    'Quality_x_District', 'Luxury_x_District',
    'm2Seg_x_District', 'm2Seg_x_Luxury', 'm2Seg_x_Premium',
    'Premium_x_m2', 'Premium_x_Luxury', 'Luxury_Property_x_District',
    'm2_x_Rooms', 'm2_x_Age', 'm2_x_Bathrooms',
    'New_x_Luxury', 'New_x_Premium',
]

# Binary feature'ları ekle
for feature in binary_features:
    bin_name = f'{feature}_bin'
    if bin_name in df_train.columns:
        features.append(bin_name)

features = [f for f in features if f in df_train.columns]
n_features = len(features)
print(f"   Toplam {n_features} feature")

# X ve y hazırla
X_train = df_train[features].copy()
X_test = df_test[features].copy()
y_train = df_train['Price_Log'].copy()
y_test = df_test['Price_Log'].copy()

# Missing value handling
train_medians = {}
for col in features:
    train_medians[col] = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_medians[col])
    X_test[col] = X_test[col].fillna(train_medians[col])

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# 8. BASE MODEL EĞİTİMİ (Dengeli parametreler)
# ============================================================================
print("\n🤖 8. BASE MODEL EĞİTİMİ...")

# ⭐ Yumuşatılmış segment ağırlıkları (m² segment bazlı)
m2_segment_weights = {1: 1.0, 2: 1.05, 3: 1.1, 4: 1.2, 5: 1.3}
sample_weights = df_train['m2_Segment'].map(m2_segment_weights).fillna(1.0).values

if USE_XGB:
    model = XGBRegressor(
        n_estimators=350,
        max_depth=6,               # Dengeli derinlik
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,        # Daha dengeli
        gamma=0.08,
        reg_alpha=0.2,
        reg_lambda=1.5,            # Dengeli regularization
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
else:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=350, max_depth=6, learning_rate=0.025,
        subsample=0.8, random_state=42
    )

model.fit(X_train, y_train, sample_weight=sample_weights)

# CV
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"   CV R² Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================================
# 9. ⭐ OOF (OUT-OF-FOLD) RESIDUAL HESAPLAMA
# ============================================================================
print("\n🔄 9. OOF RESIDUAL HESAPLAMA...")

# OOF prediction (her satır kendi görmediği model tarafından tahmin edilir)
oof_predictions = np.zeros(len(X_train))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    sw_tr = sample_weights[train_idx]
    
    fold_model = XGBRegressor(
        n_estimators=350, max_depth=6, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.08, reg_alpha=0.2, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    fold_model.fit(X_tr, y_tr, sample_weight=sw_tr)
    oof_predictions[val_idx] = fold_model.predict(X_val)

# OOF residual hesapla
oof_pred_real = np.expm1(oof_predictions)
y_train_real = np.expm1(y_train)
df_train['OOF_Predicted'] = oof_pred_real
df_train['OOF_Error_TL'] = y_train_real - oof_pred_real
df_train['OOF_Error_Pct'] = (df_train['OOF_Error_TL'] / y_train_real) * 100

print(f"   ✅ OOF tahminler hesaplandı")
print(f"   OOF MAPE: {np.mean(np.abs(df_train['OOF_Error_Pct'])):.2f}%")

# ============================================================================
# 10. ⭐ OOF-BASED CORRECTION SİSTEMİ
# ============================================================================
print("\n🔧 10. OOF-BASED CORRECTION SİSTEMİ...")

# m² segment grupları
def get_m2_group(m2):
    if m2 < 80: return 'small'
    elif m2 < 120: return 'medium'
    elif m2 < 160: return 'large'
    else: return 'xlarge'

# Lüks grubu
def get_luxury_group(lux):
    if lux == 0: return 'none'
    elif lux <= 2: return 'low'
    else: return 'high'

df_train['m2_Group'] = df_train['m2_Net'].apply(get_m2_group)
df_train['Luxury_Group'] = df_train['Luxury_Score'].apply(get_luxury_group)
df_train['District_Type'] = df_train['Is_Premium_District'].map({0: 'standard', 1: 'premium'})

# ⭐ OOF residual'dan correction tablosu oluştur
correction_table = {}
for m2_seg in range(1, 6):
    for lux_grp in ['none', 'low', 'high']:
        for m2_grp in ['small', 'medium', 'large', 'xlarge']:
            for dist_type in ['standard', 'premium']:
                mask = (
                    (df_train['m2_Segment'] == m2_seg) &
                    (df_train['Luxury_Group'] == lux_grp) &
                    (df_train['m2_Group'] == m2_grp) &
                    (df_train['District_Type'] == dist_type)
                )
                
                errors = df_train.loc[mask, 'OOF_Error_Pct']
                
                if len(errors) >= 5:  # En az 5 örnek
                    key = (m2_seg, lux_grp, m2_grp, dist_type)
                    correction_table[key] = {
                        'mean_error': errors.mean(),
                        'median_error': errors.median(),
                        'std_error': errors.std(),
                        'count': len(errors)
                    }

print(f"   ✅ {len(correction_table)} kombinasyon için OOF correction hesaplandı")

# m² Segment bazlı correction (fallback)
m2_segment_corrections = {}
for m2_seg in range(1, 6):
    seg_errors = df_train[df_train['m2_Segment'] == m2_seg]['OOF_Error_Pct']
    if len(seg_errors) > 0:
        m2_segment_corrections[m2_seg] = {
            'mean': seg_errors.mean(),
            'std': seg_errors.std()
        }

# Mahalle bazlı correction (OOF)
neighborhood_corrections = {}
for neighborhood in df_train['Neighborhood'].unique():
    n_errors = df_train[df_train['Neighborhood'] == neighborhood]['OOF_Error_Pct']
    if len(n_errors) >= 10:  # En az 10 ev
        neighborhood_corrections[neighborhood] = {
            'mean': n_errors.mean(),
            'std': n_errors.std(),
            'count': len(n_errors)
        }

print(f"   ✅ {len(neighborhood_corrections)} mahalle için OOF correction hesaplandı")

# ============================================================================
# 11. CORRECTION UYGULAMA FONKSİYONU
# ============================================================================

def apply_correction(predicted_price, meta):
    """
    Hiyerarşik correction uygula (OOF-based):
    1. 4D tablo (en spesifik)
    2. Mahalle correction
    3. m² segment correction (fallback)
    
    ⭐ Correction ağırlıkları m² segmente göre dengelendi
    """
    
    m2_seg = int(meta['m2_segment'])
    lux_grp = meta['luxury_group']
    m2_grp = meta['m2_group']
    dist_type = meta['district_type']
    neighborhood = meta['neighborhood']
    
    corrections = []
    weights = []
    
    # ⭐ Segment bazlı ağırlık ayarı (küçük evlerde daha muhafazakâr)
    segment_weight_multiplier = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.2}
    base_multiplier = segment_weight_multiplier.get(m2_seg, 1.0)
    
    # 1. 4D tablo
    key_4d = (m2_seg, lux_grp, m2_grp, dist_type)
    if key_4d in correction_table:
        data = correction_table[key_4d]
        if data['count'] >= 10:
            corrections.append(data['mean_error'])
            weights.append(2.5 * base_multiplier)
        elif data['count'] >= 5:
            corrections.append(data['mean_error'])
            weights.append(1.5 * base_multiplier)
    
    # 2. Mahalle correction
    if neighborhood in neighborhood_corrections:
        n_data = neighborhood_corrections[neighborhood]
        if n_data['count'] >= 20:
            corrections.append(n_data['mean'])
            weights.append(2.0 * base_multiplier)
        elif n_data['count'] >= 10:
            corrections.append(n_data['mean'])
            weights.append(1.0 * base_multiplier)
    
    # 3. m² Segment correction (fallback)
    if m2_seg in m2_segment_corrections:
        corrections.append(m2_segment_corrections[m2_seg]['mean'])
        weights.append(0.5 * base_multiplier)
    
    # Ağırlıklı ortalama
    if corrections:
        total_weight = sum(weights)
        weighted_correction = sum(c * w for c, w in zip(corrections, weights)) / total_weight
    else:
        weighted_correction = 0
    
    # ⭐ Correction sınırı (aşırı düzeltme önleme)
    max_correction = {1: 15, 2: 18, 3: 20, 4: 25, 5: 30}  # Segment bazlı max %
    max_corr = max_correction.get(m2_seg, 20)
    weighted_correction = np.clip(weighted_correction, -max_corr, max_corr)
    
    # Correction uygula
    corrected_price = predicted_price * (1 + weighted_correction / 100)
    
    return corrected_price, weighted_correction

# ============================================================================
# 12. TEST SETİNE UYGULA
# ============================================================================
print("\n📊 12. DEĞERLENDİRME...")

# Test için grupları oluştur
df_test['m2_Group'] = df_test['m2_Net'].apply(get_m2_group)
df_test['Luxury_Group'] = df_test['Luxury_Score'].apply(get_luxury_group)
df_test['District_Type'] = df_test['Is_Premium_District'].map({0: 'standard', 1: 'premium'})

# Base tahminler
y_train_pred_log = model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_real = np.expm1(y_test)

# Correction uygula
y_test_corrected = []
for i in range(len(df_test)):
    row = df_test.iloc[i]
    meta = {
        'm2_segment': row['m2_Segment'],
        'luxury_group': row['Luxury_Group'],
        'm2_group': row['m2_Group'],
        'district_type': row['District_Type'],
        'neighborhood': row['Neighborhood']
    }
    corrected, _ = apply_correction(y_test_pred[i], meta)
    y_test_corrected.append(corrected)

y_test_corrected = np.array(y_test_corrected)

# Train için de correction
y_train_corrected = []
for i in range(len(df_train)):
    row = df_train.iloc[i]
    meta = {
        'm2_segment': row['m2_Segment'],
        'luxury_group': row['Luxury_Group'],
        'm2_group': row['m2_Group'],
        'district_type': row['District_Type'],
        'neighborhood': row['Neighborhood']
    }
    corrected, _ = apply_correction(y_train_pred[i], meta)
    y_train_corrected.append(corrected)

y_train_corrected = np.array(y_train_corrected)

# Metrikler
def calc_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

train_metrics_before = calc_metrics(y_train_real, y_train_pred)
train_metrics_after = calc_metrics(y_train_real, y_train_corrected)
test_metrics_before = calc_metrics(y_test_real, y_test_pred)
test_metrics_after = calc_metrics(y_test_real, y_test_corrected)

print(f"\n{'='*80}")
print(f"{'Metrik':<15} {'TRAIN Önce':>12} {'TRAIN Sonra':>13} {'TEST Önce':>12} {'TEST Sonra':>12}")
print(f"{'='*80}")
print(f"{'R²':<15} {train_metrics_before['r2']:>12.4f} {train_metrics_after['r2']:>13.4f} {test_metrics_before['r2']:>12.4f} {test_metrics_after['r2']:>12.4f}")
print(f"{'MAPE':<15} {train_metrics_before['mape']:>11.2f}% {train_metrics_after['mape']:>12.2f}% {test_metrics_before['mape']:>11.2f}% {test_metrics_after['mape']:>11.2f}%")
print(f"{'RMSE':<15} {train_metrics_before['rmse']:>12,.0f} {train_metrics_after['rmse']:>13,.0f} {test_metrics_before['rmse']:>12,.0f} {test_metrics_after['rmse']:>12,.0f}")
print(f"{'='*80}")

gap_r2 = train_metrics_after['r2'] - test_metrics_after['r2']
print(f"\nTrain-Test Gap (R²): {gap_r2:.4f}")

# ============================================================================
# 13. FİYAT SEGMENT ANALİZİ (Post-hoc, sadece değerlendirme için)
# ============================================================================
print("\n📊 13. FİYAT SEGMENT ANALİZİ (TEST)...")

def get_price_segment_label(price):
    if price < 300000: return '<300K'
    elif price < 500000: return '300K-500K'
    elif price < 1000000: return '500K-1M'
    elif price < 2000000: return '1M-2M'
    else: return '>2M'

def get_price_segment_num(price):
    if price < 300000: return 1
    elif price < 500000: return 2
    elif price < 1000000: return 3
    elif price < 2000000: return 4
    else: return 5

df_test['Price_Segment_Label'] = df_test['Price_Clean'].apply(get_price_segment_label)
df_test['Price_Segment_Num'] = df_test['Price_Clean'].apply(get_price_segment_num)

segment_results = []
print(f"\n{'Segment':<12} {'Önce MAPE':>12} {'Sonra MAPE':>12} {'İyileşme':>12} {'N':>8}")
print(f"{'-'*56}")

for label in ['<300K', '300K-500K', '500K-1M', '1M-2M', '>2M']:
    mask = df_test['Price_Segment_Label'] == label
    if mask.sum() > 0:
        before_mape = np.mean(np.abs((y_test_real[mask] - y_test_pred[mask]) / y_test_real[mask])) * 100
        after_mape = np.mean(np.abs((y_test_real[mask] - y_test_corrected[mask]) / y_test_real[mask])) * 100
        improvement = before_mape - after_mape
        
        segment_results.append({
            'label': label,
            'segment': get_price_segment_num(df_test.loc[mask, 'Price_Clean'].median()),
            'before_mape': float(before_mape),
            'after_mape': float(after_mape),
            'improvement': float(improvement),
            'count': int(mask.sum())
        })
        
        emoji = "🎯" if improvement > 1 else "✅" if improvement > 0 else "⚠️"
        print(f"{label:<12} {before_mape:>11.1f}% {after_mape:>11.1f}% {improvement:>+11.1f}% {mask.sum():>8} {emoji}")

# ============================================================================
# 14. GÜVEN ARALIĞI
# ============================================================================
print("\n📊 14. GÜVEN ARALIĞI...")

residuals = y_test_real - y_test_corrected
residual_lower = float(np.percentile(residuals, 15))
residual_upper = float(np.percentile(residuals, 85))

segment_ci = {}
for label in ['<300K', '300K-500K', '500K-1M', '1M-2M', '>2M']:
    mask = df_test['Price_Segment_Label'] == label
    if mask.sum() > 10:
        seg_residuals = residuals[mask]
        segment_ci[label] = {
            'lower': float(np.percentile(seg_residuals, 15)),
            'upper': float(np.percentile(seg_residuals, 85)),
        }
        print(f"   {label}: [{segment_ci[label]['lower']:+,.0f}, {segment_ci[label]['upper']:+,.0f}] TL")

# ============================================================================
# 15. FEATURE IMPORTANCE
# ============================================================================
print("\n📊 15. TOP FEATURE'LAR...")

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['importance']:.4f} - {row['feature']}")

# ============================================================================
# 16. ÖRNEK VERİLER
# ============================================================================
print("\n📋 16. ÖRNEK VERİLER...")

def create_example(row, pred_before, pred_after, actual_price, source):
    return {
        'source': source,
        'name': f"{row['District']} - {row['Neighborhood']}",
        'segment': get_price_segment_label(actual_price),
        'segment_num': get_price_segment_num(actual_price),
        'data': {
            'asking_price': float(actual_price),
            'm2_net': float(row['m2_Net']) if pd.notna(row['m2_Net']) else 100,
            'm2_gross': float(row['m2_Gross']) if pd.notna(row['m2_Gross']) else 110,
            'rooms': row['Number of rooms'] if pd.notna(row['Number of rooms']) else '3+1',
            'district': row['District'],
            'neighborhood': row['Neighborhood'],
            'building_age': row['Building Age'] if pd.notna(row['Building Age']) else '5-10 between',
            'floor': str(int(row['Floor_Num'])) if pd.notna(row['Floor_Num']) else '3',
            'num_floors': float(row['Num_Floors']) if pd.notna(row['Num_Floors']) else 10,
            'num_bathrooms': float(row['Num_Bathrooms']) if pd.notna(row['Num_Bathrooms']) else 1,
            'heating': row['Heating'] if pd.notna(row['Heating']) else 'Natural Gas (Combi)',
            'using_status': row['Using status'] if pd.notna(row['Using status']) else 'Free',
            'from_who': row['From who'] if pd.notna(row['From who']) else 'From the real estate office',
            'ad_date': row['Adrtisement Date'] if pd.notna(row['Adrtisement Date']) else '',
            'has_elevator': int(row.get('Elevator_bin', 0)),
            'has_balcony': int(row.get('Balcony_bin', 0)),
            'has_parking': int(row.get('Parking Lot_bin', 0)),
            'has_security': int(row.get('Security_bin', 0)),
            'has_air_conditioning': int(row.get('Air conditioning_bin', 0)),
            'has_pool': int(row.get('Swimming Pool (Open)_bin', 0)),
            'has_sauna': int(row.get('Sauna_bin', 0)),
            'has_gym': int(row.get('Gym_bin', 0)),
            'has_jacuzzi': int(row.get('Jacuzzi_bin', 0)),
            'has_smart_house': int(row.get('Smart House_bin', 0)),
            'has_terrace': int(row.get('Terrace_bin', 0)),
            'has_garden': int(row.get('Garden_bin', 0)),
            'has_steel_door': int(row.get('Steel door_bin', 0)),
            'has_video_intercom': int(row.get('Video intercom_bin', 0)),
            'has_alarm': int(row.get('Alarm (Thief)_bin', 0)),
            'has_generator': int(row.get('Generator_bin', 0)),
            'has_closed_garage': int(row.get('Closed Garage_bin', 0)),
            'has_thermal_insulation': int(row.get('Thermal Insulation_bin', 0)),
        },
        'actual_price': float(actual_price),
        'predicted_price_before': float(pred_before),
        'predicted_price': float(pred_after),
        'error_pct_before': float(abs((actual_price - pred_before) / actual_price) * 100),
        'error_pct': float(abs((actual_price - pred_after) / actual_price) * 100)
    }

train_examples = []
test_examples = []

np.random.seed(42)
for label in ['<300K', '300K-500K', '500K-1M', '1M-2M', '>2M']:
    mask = df_train['Price_Clean'].apply(get_price_segment_label) == label
    seg_idx = df_train[mask].index.tolist()
    if len(seg_idx) >= 2:
        samples = np.random.choice(seg_idx, 2, replace=False)
        for idx in samples:
            row = df_train.loc[idx]
            i = df_train.index.get_loc(idx)
            train_examples.append(create_example(row, y_train_pred[i], y_train_corrected[i], row['Price_Clean'], 'train'))

np.random.seed(123)
for label in ['<300K', '300K-500K', '500K-1M', '1M-2M', '>2M']:
    mask = df_test['Price_Segment_Label'] == label
    seg_idx = df_test[mask].index.tolist()
    n = min(4, len(seg_idx))
    if n > 0:
        samples = np.random.choice(seg_idx, n, replace=False)
        for idx in samples:
            row = df_test.loc[idx]
            i = df_test.index.get_loc(idx)
            test_examples.append(create_example(row, y_test_pred[i], y_test_corrected[i], row['Price_Clean'], 'test'))

all_examples = train_examples + test_examples
print(f"   Train: {len(train_examples)}, Test: {len(test_examples)}")

# ============================================================================
# 17. MODEL KAYDETME
# ============================================================================
print("\n💾 17. MODEL KAYDETME...")

save_data = {
    'model': model,
    'features': features,
    'n_features': n_features,
    
    # Correction sistemi (OOF-based)
    'correction_table': correction_table,
    'm2_segment_corrections': m2_segment_corrections,
    'neighborhood_corrections': neighborhood_corrections,
    
    # Encoding
    'district_encoding': district_encoding,
    'district_global': float(district_global),
    'neighborhood_encoding': neighborhood_encoding,
    'neighborhood_global': float(neighborhood_global),
    'district_variance': district_variance,
    'neighborhood_count': neighborhood_count,
    'heating_quality': heating_quality,
    'train_medians': train_medians,
    
    # Premium districts
    'premium_districts': premium_districts,
    'mid_premium_districts': mid_premium_districts,
    'budget_districts': budget_districts,
    
    # Güven aralığı
    'segment_ci': segment_ci,
    'residual_lower': residual_lower,
    'residual_upper': residual_upper,
    
    # Örnekler
    'train_examples': train_examples,
    'test_examples': test_examples,
    'all_examples': all_examples,
    
    # Metrikler
    'metrics_before': {k: float(v) for k, v in test_metrics_before.items()},
    'metrics_after': {k: float(v) for k, v in test_metrics_after.items()},
    'train_metrics': {k: float(v) for k, v in train_metrics_after.items()},
    'test_metrics': {k: float(v) for k, v in test_metrics_after.items()},
    'cv_scores': cv_scores.tolist(),
    'cv_mean': float(cv_scores.mean()),
    'segment_results': segment_results,
    'feature_importance': feature_importance.head(20).to_dict('records'),
}

model_path = os.path.join(SCRIPT_DIR, 'investment_advisor_model_v7.pkl')
joblib.dump(save_data, model_path)
print(f"   ✅ Model: {model_path}")

districts = sorted(df_clean['District'].unique().tolist())
neighborhoods_by_district = {d: sorted(df_clean[df_clean['District'] == d]['Neighborhood'].unique().tolist()) for d in districts}
location_data = {'districts': districts, 'neighborhoods_by_district': neighborhoods_by_district}
location_path = os.path.join(SCRIPT_DIR, 'location_data_v7.pkl')
joblib.dump(location_data, location_path)
print(f"   ✅ Lokasyon: {location_path}")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "="*80)
print("✅ MODEL v7 TAMAMLANDI!")
print("="*80)

print(f"""
📊 PERFORMANS ÖZETİ (TEST):

┌─────────────────────────────────────────────────────────────┐
│              BASE MODEL      + OOF CORRECTION    İYİLEŞME  │
├─────────────────────────────────────────────────────────────┤
│  R²:        {test_metrics_before['r2']:.4f}            {test_metrics_after['r2']:.4f}          {test_metrics_after['r2']-test_metrics_before['r2']:+.4f}    │
│  MAPE:      {test_metrics_before['mape']:.2f}%           {test_metrics_after['mape']:.2f}%          {test_metrics_before['mape']-test_metrics_after['mape']:+.2f}%   │
│  RMSE:      {test_metrics_before['rmse']:,.0f}          {test_metrics_after['rmse']:,.0f}                   │
└─────────────────────────────────────────────────────────────┘

📊 SEGMENT PERFORMANSI (Correction Sonrası):
""")

for res in segment_results:
    emoji = "🎯" if res['improvement'] > 1 else "✅" if res['improvement'] > 0 else "⚠️"
    print(f"   {emoji} {res['label']}: {res['before_mape']:.1f}% → {res['after_mape']:.1f}% ({res['improvement']:+.1f}%), N={res['count']}")

print(f"""
🔧 v7'DEKİ İYİLEŞTİRMELER:
   ✅ Fiyat segmenti LEAKAGE düzeltildi (m² segment kullanılıyor)
   ✅ OOF-based correction (in-sample ezber yok)
   ✅ Tarih feature'ları eklendi
   ✅ Segment ağırlıkları yumuşatıldı
   ✅ Correction sınırları segmente göre ayarlandı
   ✅ {n_features} feature

📋 ÖRNEKLER: {len(all_examples)} adet
""")
print("="*80)

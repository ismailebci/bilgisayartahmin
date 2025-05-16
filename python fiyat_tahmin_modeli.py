import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb
from catboost import CatBoostRegressor

# Veri setini yükleme
data = pd.read_csv('masaustu_data_kaggle.csv')

# Eksik verileri doldurma
data.ffill(inplace=True)

# Bağımsız değişkenler (X) ve bağımlı değişken (y)
X = data.drop(columns=['Fiyat'])  # 'Fiyat' sütunu hedef değişken olarak kabul edildi
y = data['Fiyat']

# Kategorik ve sayısal sütunları belirleme (X'e göre)
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri ön işleme adımları
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),  # Eksik sayısal değerleri ortalama ile doldur
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Kategorik değişkenleri one-hot encode et
    ])

# Model eğitimi ve değerlendirme için fonksiyon
def train_and_evaluate(model, params, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    grid_search = GridSearchCV(pipeline, params, cv=3, scoring='r2', n_jobs=-1, verbose=1, error_score='raise')
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2, grid_search.best_estimator_, grid_search.best_params_

# XGBoost Modeli
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_params = {
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200, 300]
}
mae_xgb, r2_xgb, xgb_best_estimator, xgb_best_params = train_and_evaluate(xgb_model, xgb_params, X_train, y_train, X_test, y_test)

# Random Forest Modeli (İyileştirilmiş)
rf_model = RandomForestRegressor(random_state=42)
rf_params = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, 15],  # max_depth sınırlandırıldı
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]  # min_samples_leaf eklendi
}
mae_rf, r2_rf, rf_best_estimator, rf_best_params = train_and_evaluate(rf_model, rf_params, X_train, y_train, X_test, y_test)

# Gradient Boosting Modeli (İyileştirilmiş)
gb_model = GradientBoostingRegressor(random_state=42)
gb_params = {
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 5, 10],  # min_samples_split eklendi
    'model__min_samples_leaf': [1, 2, 4]  # min_samples_leaf eklendi
}
mae_gb, r2_gb, gb_best_estimator, gb_best_params = train_and_evaluate(gb_model, gb_params, X_train, y_train, X_test, y_test)

# CatBoost Modeli
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_params = {
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200, 300]
}
mae_cat, r2_cat, cat_best_estimator, cat_best_params = train_and_evaluate(cat_model, cat_params, X_train, y_train, X_test, y_test)

# Model Karşılaştırması
print("\n📊 Model Karşılaştırması:")
print(f"XGBoost R²: {r2_xgb:.4f}, MAE: {mae_xgb:.2f}")
print(f"Random Forest R²: {r2_rf:.4f}, MAE: {mae_rf:.2f}")
print(f"Gradient Boosting R²: {r2_gb:.4f}, MAE: {mae_gb:.2f}")
print(f"CatBoost R²: {r2_cat:.4f}, MAE: {mae_cat:.2f}")

# Yeni bilgisayarın özelliklerini tanımlayın
new_computer = {
    'Marka': ['HP'],  # Örnek marka
    'İşlemci Tipi': ['Intel Core i5'],  # Örnek işlemci
    'SSD Kapasitesi': ['512 GB'],  # 3 TB SSD
    'Ram (Sistem Belleği)': ['8 GB'],  # 64 GB RAM
    'Ekran Kartı': ['NVIDIA GeForce RTX 3050'],  # Örnek ekran kartı
    'Kapasite': ['512 GB'],  # 3 TB depolama
    'İşletim Sistemi': ['Windows'],  # Örnek işletim sistemi
    'Ekran Kartı Bellek Tipi': ['GDDR5'],  # Örnek bellek tipi
    'Ekran Kartı Tipi': ['Dedicated'],  # Örnek ekran kartı tipi
    'Garanti Tipi': ['Resmi Garanti'],  # Örnek garanti tipi
    'Ram (Sistem Belleği) Tipi': ['DDR4'],  # Örnek RAM tipi
    'İşlemci Çekirdek Sayısı': ['16'],  # 16 çekirdek
    'İşlemci Nesli': ['11. Nesil'],  # Örnek işlemci nesli
    'İşlemci Modeli': ['i5-11500H'],  # Örnek işlemci modeli
    'Çözünürlük': ['1920x1080'],  # 4K çözünürlük
    'Power Supply': ['800W'],  # 800W güç kaynağı
    'Kullanım Amacı': ['Oyun ve İş'],  # Örnek kullanım amacı
    'Ekran Kartı Hafızası': ['4 GB'],  # 24 GB ekran kartı hafızası
    'Temel İşlemci Hızı (GHz)': ['3.5 GHz'],  # 3.5 GHz temel hız
    'Bağlantılar': ['Wi-Fi 6, Bluetooth 5.2'],  # Örnek bağlantılar
    'Cihaz Ağırlığı': ['2.5 kg'],  # 2.5 kg ağırlık
    'Ekran Boyutu': ['16 inç'],  # 16 inç ekran
    'İşlemci Frekansı': ['3.2 GHz'],  # 5.8 GHz maksimum frekans
    'Ekran Yenileme Hızı': ['144 Hz'],  # 120 Hz yenileme hızı
    'Panel Tipi': ['IPS'],  # IPS panel
    'Menşei': ['ABD'],  # Örnek menşei
    'Arttırılabilir Azami Bellek': ['128 GB']  # 128 GB'a kadar bellek
}

# Yeni bilgisayarı bir DataFrame'e dönüştürün
new_computer_df = pd.DataFrame(new_computer)

# Eksik sütunları kontrol edin ve varsayılan değerlerle doldurun
for col in X_train.columns:
    if col not in new_computer_df.columns:
        new_computer_df[col] = 'Unknown'  # Varsayılan değer

# Sütun sıralamasını eğitim verisiyle aynı yapın
new_computer_df = new_computer_df[X_train.columns]

# Tahmin yapma
y_pred_xgb_new = xgb_best_estimator.predict(new_computer_df)
y_pred_rf_new = rf_best_estimator.predict(new_computer_df)
y_pred_gb_new = gb_best_estimator.predict(new_computer_df)
y_pred_cat_new = cat_best_estimator.predict(new_computer_df)

# Sonuçları yazdırın
print("\n📊 Yeni Bilgisayarın Tahmin Fiyatları:")
print(f"XGBoost Tahmini Fiyat: {y_pred_xgb_new[0]:.2f} TL")
print(f"Random Forest Tahmini Fiyat: {y_pred_rf_new[0]:.2f} TL")
print(f"Gradient Boosting Tahmini Fiyat: {y_pred_gb_new[0]:.2f} TL")
print(f"CatBoost Tahmini Fiyat: {y_pred_cat_new[0]:.2f} TL")

# Ensemble Tahmini (Ağırlıklı Ortalama)
weights = [r2_xgb, r2_rf, r2_gb, r2_cat]  # R² skorlarına göre ağırlık
weights = np.array(weights) / np.sum(weights)  # Normalize et
y_pred_ensemble = (y_pred_xgb_new * weights[0] + y_pred_rf_new * weights[1] + y_pred_gb_new * weights[2] + y_pred_cat_new * weights[3])
print(f"\n📊 Ensemble Tahmini Fiyat: {y_pred_ensemble[0]:.2f} TL")

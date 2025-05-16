# 💻 Bilgisayar Fiyat Tahmin Modeli

Bu proje, çeşitli donanım ve yazılım özelliklerine sahip bilgisayarların fiyatlarını tahmin etmek amacıyla geliştirilmiştir. Modelleme sürecinde XGBoost, Random Forest, Gradient Boosting ve CatBoost gibi güçlü regresyon algoritmaları kullanılmıştır.

## 🔍 Proje Amacı

Makine öğrenmesi algoritmaları ile masaüstü bilgisayarların teknik özelliklerine göre fiyat tahmininde bulunmak. Özellikle e-ticaret siteleri, ikinci el satış platformları ve donanım kıyaslama uygulamaları için değerli bir içgörü sunmayı hedefler.

## 📁 Veri Kümesi
- Boyut: `N x M` (örnek sayısı x özellik sayısı)
- Hedef Değişken: `Fiyat`

### 📌 Örnek Özellikler:
- Marka, İşlemci Tipi ve Nesli
- SSD ve RAM kapasitesi
- Ekran kartı tipi ve belleği
- İşletim sistemi, bağlantı türleri, garanti tipi
- Kullanım amacı, panel tipi, ekran yenileme hızı vb.

## ⚙️ Kullanılan Kütüphaneler

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `catboost`

## 🧪 Kullanılan Modeller ve Parametre Optimizasyonu

Grid Search kullanılarak her model için hiperparametre optimizasyonu yapılmıştır. Aşağıdaki algoritmalar denenmiş ve karşılaştırılmıştır:

- **✅ XGBoostRegressor**
- **✅ RandomForestRegressor**
- **✅ GradientBoostingRegressor**
- **✅ CatBoostRegressor**

## 📊 Model Performansı (Test Verisi Üzerinde)

| Model              | R² Skoru | MAE (Ortalama Mutlak Hata) |
|--------------------|----------|-----------------------------|
| XGBoost            | 0.XXXX   | XXXX TL                     |
| Random Forest      | 0.XXXX   | XXXX TL                     |
| Gradient Boosting  | 0.XXXX   | XXXX TL                     |
| CatBoost           | 0.XXXX   | XXXX TL                     |

> Not: Yukarıdaki değerler, `GridSearchCV` ile optimize edilen modellerin test verileri üzerindeki performans sonuçlarıdır.

## 🤖 Yeni Bilgisayar Fiyat Tahmini

Aşağıdaki özelliklere sahip bir bilgisayarın fiyat tahminleri yapılmıştır:

- **İşlemci**: Intel Core i5-11500H (11. Nesil, 16 Çekirdek)
- **RAM**: 8 GB DDR4 (Azami: 128 GB)
- **Depolama**: 512 GB SSD
- **Ekran Kartı**: NVIDIA GeForce RTX 3050 (4 GB, GDDR5)
- **Ekran**: 16" IPS, 1920x1080, 144 Hz
- **Diğer**: Windows, Wi-Fi 6, Bluetooth 5.2, 800W PSU

### 🔮 Tahmin Sonuçları:

| Model              | Tahmini Fiyat |
|--------------------|----------------|
| XGBoost            | XXXX TL         |
| Random Forest      | XXXX TL         |
| Gradient Boosting  | XXXX TL         |
| CatBoost           | XXXX TL         |
| **Ensemble (Ağırlıklı Ortalama)** | **XXXX TL** |

> Ağırlıklı ortalama tahmini, modellerin `R²` skorlarına göre hesaplanmıştır.

## 📦 Proje Dosyaları

- `masaustu_data_kaggle.csv`: Veri seti
- `fiyat_tahmin_modeli.py`: Tüm eğitim ve tahmin süreçlerini içeren Python betiği
- `README.md`: Proje açıklamaları
- `requirements.txt`: Gerekli kütüphaneler (isteğe bağlı)

## 🚀 Nasıl Kullanılır?

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost
   
   fiyat_tahmin_modeli.py dosyasını çalıştırın:

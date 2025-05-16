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
| XGBoost            | 0.8556   | 2404.06 TL                  |
| Random Forest      | 0.7708   | 2971.69 TL                  |
| Gradient Boosting  | 0.8540   | 2647.94 TL                  |
| CatBoost           | 0.8771   | 2269.07 TL                  |
> CatBoost modeli en yüksek R² skoruna ve en düşük MAE değerine sahiptir.

## 🤖 Yeni Bilgisayar Fiyat Tahmini

Aşağıdaki özelliklere sahip bir bilgisayarın fiyat tahminleri yapılmıştır:

- **İşlemci**: Intel Core i5-11500H (11. Nesil, 16 Çekirdek)
- **RAM**: 8 GB DDR4 (Azami: 128 GB)
- **Depolama**: 512 GB SSD
- **Ekran Kartı**: NVIDIA GeForce RTX 3050 (4 GB, GDDR5)
- **Ekran**: 16" IPS, 1920x1080, 144 Hz
- **Diğer**: Windows, Wi-Fi 6, Bluetooth 5.2, 800W PSU

### 📈 Tahmin Sonuçları

| Model              | Tahmini Fiyat (TL) |
|--------------------|--------------------|
| XGBoost            | 27,686.59 TL       |
| Random Forest      | 49,822.92 TL       |
| Gradient Boosting  | 24,774.24 TL       |
| CatBoost           | 28,012.08 TL       |
| **Ensemble (Ağırlıklı Ortalama)** | **32,112.61 TL** |

> Ensemble tahmini, her modelin R² skoruna göre ağırlıklı ortalama alınarak hesaplanmıştır.
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
2.Çalıştırmak için:
   fiyat_tahmin_modeli.py dosyasını çalıştırın:

3. Koddaki bu kısmı ne isterseniz yazın
# Yeni bilgisayarın özelliklerini tanımlayın
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
sonra çalıştırıp sonuca bakın :D

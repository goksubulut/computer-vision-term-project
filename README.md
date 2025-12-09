# Term Project: Image Classification with Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](https://github.com/goksubulut/computer-vision-term-project)

Bu proje, transfer learning kullanarak X-ray görüntülerinde kırık (fracture) ve normal (normal) sınıflarını ayırt eden bir görüntü sınıflandırma sistemidir. MobileNetV2 mimarisi ve 10-fold Stratified Cross-Validation kullanılmıştır.

## ✨ Özellikler

- 🚀 **Transfer Learning:** ImageNet'te önceden eğitilmiş MobileNetV2 kullanımı
- 📊 **10-Fold CV:** Stratified Cross-Validation ile güvenilir model değerlendirmesi
- 🔄 **Data Augmentation:** Overfitting'i önlemek için çeşitli augmentasyon teknikleri
- ⚡ **Optimizasyon:** Early stopping, learning rate scheduling ve model checkpointing
- 🎯 **Hızlı Eğitim:** GPU ile fold başına 5 dakikanın altında eğitim süresi
- 📈 **Detaylı Raporlama:** Her fold için accuracy ve genel istatistikler

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Proje Açıklaması](#proje-açıklaması)
- [Teknolojiler](#-teknolojiler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Metodoloji](#metodoloji)
- [Sonuçlar](#-sonuçlar)
- [Akademik Referans](#akademik-referans)
- [Yazar](#-yazar)
- [Lisans](#-lisans)

## Proje Açıklaması

Bu proje, transfer learning kullanarak X-ray görüntülerinde kırık (fracture) ve normal (normal) sınıflarını ayırt eden bir görüntü sınıflandırma sistemidir. MobileNetV2 mimarisi ve 10-fold Stratified Cross-Validation kullanılmıştır.

## 🛠️ Teknolojiler

Bu proje aşağıdaki teknolojiler ve kütüphaneler kullanılarak geliştirilmiştir:

- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Model Architecture:** MobileNetV2 (Transfer Learning)
- **Data Processing:** NumPy, PIL (Pillow)
- **Machine Learning:** scikit-learn (StratifiedKFold)
- **Data Pipeline:** tf.data API
- **Platform:** Google Colab (GPU support)

## Gereksinimler

Bu proje Google Colab'da çalışacak şekilde tasarlanmıştır. Ek kütüphane kurulumu gerektirmez. Kullanılan kütüphaneler:

- TensorFlow / Keras
- NumPy
- scikit-learn
- PIL (Pillow)

Tüm bu kütüphaneler Google Colab'da varsayılan olarak yüklüdür.

## Dataset Yapısı

Proje aşağıdaki dizin yapısını beklemektedir:

```
/content/data/
    /fracture
        img1.jpg
        img2.jpg
        ...
    /normal
        imgA.jpg
        imgB.jpg
        ...
```

**Önemli:** 
- Görüntüler `.jpg`, `.jpeg` veya `.png` formatında olabilir
- `fracture` klasöründeki görüntüler kırık olarak etiketlenir (label=1)
- `normal` klasöründeki görüntüler normal olarak etiketlenir (label=0)

## 🚀 Kurulum

### GitHub'dan Klonlama

```bash
git clone https://github.com/goksubulut/computer-vision-term-project.git
cd computer-vision-term-project
```

## 📖 Kullanım

### Google Colab'da Çalıştırma

### Adım 1: GPU'yu Etkinleştirin

1. Google Colab'da yeni bir notebook oluşturun
2. `Runtime` > `Change runtime type` menüsüne gidin
3. `Hardware accelerator` olarak `GPU` seçin
4. `Save` butonuna tıklayın

### Adım 2: Dataset'i Yükleyin

Dataset'inizi Colab'a yüklemek için aşağıdaki yöntemlerden birini kullanabilirsiniz:

**Yöntem 1: Google Drive'dan Yükleme**
```python
from google.colab import drive
drive.mount('/content/drive')

# Dataset'inizi drive'a yükleyin ve aşağıdaki komutu çalıştırın
!cp -r /content/drive/MyDrive/path/to/your/data /content/data
```

**Yöntem 2: Doğrudan Yükleme**
```python
# Colab'ın dosya yükleme özelliğini kullanın
from google.colab import files
uploaded = files.upload()

# Veya zip dosyası yükleyip açın
!unzip your_dataset.zip -d /content/data
```

### Adım 3: Proje Dosyasını Çalıştırın

1. `term_project_submission.py` dosyasını Colab'a yükleyin veya içeriğini bir hücreye kopyalayın
2. Dosyayı çalıştırın:

```python
!python term_project_submission.py
```

Veya notebook içinde:

```python
exec(open('term_project_submission.py').read())
```

## Hyperparameters

Proje aşağıdaki hyperparameter'ları kullanmaktadır:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `BATCH_SIZE` | 32 | Eğitim batch boyutu |
| `IMG_SIZE` | (224, 224) | Görüntü boyutu (MobileNetV2 için standart) |
| `LEARNING_RATE` | 0.001 | Başlangıç öğrenme oranı |
| `EPOCHS` | 50 | Maksimum epoch sayısı |
| `PATIENCE` | 5 | Early stopping patience değeri |
| `NUM_CLASSES` | 2 | Sınıf sayısı (fracture, normal) |

**Not:** Early stopping ve learning rate reduction sayesinde gerçek eğitim süresi genellikle 50 epoch'tan daha kısa olacaktır.

## Runtime Constraints

- **Hedef Süre:** Her fold için eğitim 5 dakikanın altında tamamlanmalıdır
- **Toplam Süre:** 10 fold için toplam süre yaklaşık 30-50 dakika arasında olabilir
- **GPU Gereksinimi:** CUDA destekli GPU (Colab'da T4 veya daha iyi) önerilir
- **Bellek:** En az 12GB RAM önerilir

## Metodoloji

### 1. Transfer Learning

- **Base Model:** MobileNetV2 (ImageNet'te önceden eğitilmiş)
- **Fine-tuning:** Base model katmanları dondurulmuş, sadece üst katmanlar eğitiliyor
- **Özellik Çıkarımı:** Global Average Pooling kullanılıyor

### 2. 10-Fold Stratified Cross-Validation

- **Stratified:** Her fold'ta sınıf dağılımı korunur
- **Shuffle:** Veri karıştırılır (random_state=42 ile tekrarlanabilir)
- **Her Fold:** Her fold için yeni bir model eğitilir
- **Değerlendirme:** Her fold'un validation accuracy'si kaydedilir

### 3. Data Augmentation

Eğitim sırasında aşağıdaki augmentasyonlar uygulanır:
- Rastgele yatay çevirme (flip)
- Rastgele rotasyon (90° katları)
- Rastgele parlaklık ayarı
- Rastgele kontrast ayarı

### 4. Callbacks

- **Early Stopping:** Validation accuracy'de iyileşme olmazsa durdurur
- **Reduce LR on Plateau:** Validation loss düşmezse öğrenme oranını azaltır
- **Model Checkpoint:** En iyi modeli kaydeder

### 5. tf.data Pipeline

- **Prefetching:** GPU veri beklerken CPU veri hazırlar
- **Parallel Processing:** Görüntü işleme paralel yapılır
- **Memory Efficient:** Büyük dataset'ler için bellek verimli

## Çıktılar

Proje çalıştırıldığında aşağıdaki çıktılar üretilir:

1. **Console Çıktısı:** Her fold'un eğitim süreci ve sonuçları
2. **cv_results.txt:** Tüm sonuçların kaydedildiği metin dosyası
3. **Model Dosyaları:** Her fold için en iyi model (`model_fold_X.h5`)

### Örnek Çıktı Formatı

```
============================================================
10-FOLD STRATIFIED CROSS-VALIDATION
Image Classification: Fracture vs Normal
============================================================

Fold 1 Results:
  Training time: 245.32 seconds (4.09 minutes)
  Validation Accuracy: 0.8750 (87.50%)

...

FINAL RESULTS
============================================================
Mean Accuracy: 0.8625 (86.25%)
Std Deviation: 0.0234 (2.34%)
Mean ± Std: 0.8625 ± 0.0234
Mean ± Std (%): 86.25% ± 2.34%
```

## Sınırlamalar

1. **Dataset Boyutu:** Çok büyük dataset'ler için bellek sorunları yaşanabilir
2. **Eğitim Süresi:** GPU olmadan eğitim çok uzun sürebilir
3. **Model Boyutu:** Her fold için model dosyası kaydedilir (~10-15 MB)
4. **Augmentation:** Sadece temel augmentasyonlar uygulanır

## Akademik Referans

Bu proje aşağıdaki akademik makaleye referans verir:

**DOI:** 10.1002/ima.22849

**Başlık:** Image processing and machine learning-based bone fracture detection and classification using X-ray images

**Yazar:** Muhammet Emin Şahin

**Kurum:** Department of Computer Engineering, Yozgat Bozok University, Yozgat, Turkey

**Email:** emin.sahin@bozok.edu.tr

**Makale Linki:** [Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.22849)

## Sorun Giderme

### Problem: "No images found in data directory"

**Çözüm:** 
- Dataset dizin yapısını kontrol edin
- `/content/data/fracture/` ve `/content/data/normal/` klasörlerinin var olduğundan emin olun
- Görüntü dosyalarının doğru formatta olduğunu kontrol edin

### Problem: GPU kullanılmıyor

**Çözüm:**
- Runtime > Change runtime type > GPU seçildiğinden emin olun
- `tf.config.list_physical_devices('GPU')` komutu ile GPU'yu kontrol edin

### Problem: Eğitim çok uzun sürüyor

**Çözüm:**
- Batch size'ı artırın (32'den 64'e)
- Epoch sayısını azaltın
- Early stopping patience'ı azaltın
- GPU kullandığınızdan emin olun

### Problem: Düşük accuracy

**Çözüm:**
- Dataset kalitesini kontrol edin
- Daha fazla veri toplayın
- Augmentation parametrelerini ayarlayın
- Learning rate'i değiştirmeyi deneyin

## Kod Yapısı

```
term_project_submission.py
├── load_data()              # Veri yükleme
├── preprocess_image()       # Görüntü ön işleme
├── augment_image()          # Data augmentation
├── create_dataset()         # tf.data dataset oluşturma
├── create_model()           # MobileNetV2 model oluşturma
├── train_fold()             # Tek fold eğitimi
└── main()                   # Ana fonksiyon (10-fold CV)
```

## 📞 İletişim & Destek

- 💬 **Sorularınız için:** GitHub'da [Issue](https://github.com/goksubulut/computer-vision-term-project/issues) açabilirsiniz
- 🤝 **Katkıda Bulunmak:** Pull request göndermekten çekinmeyin
- 📧 **Öğretim Üyesi:** Ders ile ilgili sorular için öğretim üyesi ile iletişime geçin

## 📊 Sonuçlar

Proje çalıştırıldığında aşağıdaki metrikler hesaplanır:

- Her fold için validation accuracy
- Mean accuracy ve standard deviation
- Eğitim süreleri
- Detaylı sınıflandırma raporları

## 🤝 Katkıda Bulunma

Bu bir term projesidir. Katkılar için lütfen issue açın veya pull request gönderin.

## 📝 Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır. Kod ve dokümantasyon açık kaynak olarak paylaşılmaktadır.

**Not:** Bu bir akademik term projesidir. Eğitim ve öğrenme amaçlı kullanım için uygundur.

## 👤 Yazar

### Göksu Bulut

- 🎓 **Eğitim:** Computer Vision dersi kapsamında geliştirilmiş term projesi
- 💻 **GitHub:** [@goksubulut](https://github.com/goksubulut)
- 🔗 **Repository:** [computer-vision-term-project](https://github.com/goksubulut/computer-vision-term-project)
- 📧 **İletişim:** GitHub üzerinden issue açarak veya pull request göndererek iletişime geçebilirsiniz

### Proje Hakkında

Bu proje, Computer Vision dersi kapsamında transfer learning ve cross-validation tekniklerini uygulamalı olarak öğrenmek amacıyla geliştirilmiştir. X-ray görüntülerinde kemik kırığı tespiti gibi tıbbi görüntü analizi uygulamalarında deep learning modellerinin kullanımını göstermektedir.

### İlgi Alanları

- 🤖 Deep Learning & Neural Networks
- 👁️ Computer Vision & Image Processing
- 🏥 Medical Image Analysis
- 📊 Machine Learning & Data Science

## 🙏 Teşekkürler

- **TensorFlow/Keras** ekibine - Harika bir deep learning framework sağladıkları için
- **scikit-learn** ekibine - Cross-validation ve diğer ML araçları için
- **Google Colab** ekibine - Ücretsiz GPU erişimi sağladıkları için
- **Akademik Topluluk** - Açık kaynak araştırmalar ve paylaşımlar için
- **Öğretim Üyesi** - Proje sürecindeki rehberlik için

## 📚 İlgili Projeler

Bu projeyi beğendiyseniz, aşağıdaki konularda da projeler geliştirmeyi planlıyorum:

- 🔬 Daha gelişmiş augmentation teknikleri
- 🎯 Fine-tuning optimizasyonları
- 📱 Mobil uygulama entegrasyonu
- 🌐 Web tabanlı demo uygulaması

## ⭐ Proje İstatistikleri

- 📝 **Kod Satırı:** ~440 satır Python
- 📦 **Dosya Sayısı:** 1 ana Python dosyası
- ⏱️ **Eğitim Süresi:** Fold başına ~4-5 dakika (GPU ile)
- 🎯 **Model Mimarisi:** MobileNetV2 (Transfer Learning)
- 📊 **Validation Yöntemi:** 10-Fold Stratified Cross-Validation


# GitHub Repository Kurulum Rehberi

Bu dosya, projenizi GitHub'a yüklemek için adım adım talimatlar içerir.

## 1. GitHub'da Yeni Repository Oluşturma

1. GitHub.com'a giriş yapın
2. Sağ üst köşedeki **"+"** butonuna tıklayın
3. **"New repository"** seçeneğini seçin
4. Repository bilgilerini doldurun:
   - **Repository name:** `computer-vision-term-project` (veya istediğiniz isim)
   - **Description:** "Image Classification with Transfer Learning using MobileNetV2"
   - **Visibility:** Public veya Private (tercihinize göre)
   - **Initialize repository:** 
     - ❌ README eklemeyin (zaten var)
     - ❌ .gitignore eklemeyin (zaten var)
     - ❌ License eklemeyin (zaten var)
5. **"Create repository"** butonuna tıklayın

## 2. Yerel Git Repository Oluşturma

Proje klasörünüzde terminal/command prompt açın ve şu komutları çalıştırın:

```bash
# Git repository'yi başlat
git init

# Tüm dosyaları staging area'ya ekle
git add .

# İlk commit'i yap
git commit -m "Initial commit: Term project submission with MobileNetV2 and 10-fold CV"
```

## 3. GitHub Repository'ye Bağlama

GitHub'da oluşturduğunuz repository'nin sayfasında, **"Quick setup"** bölümünden URL'yi kopyalayın (HTTPS veya SSH).

```bash
# Remote repository'yi ekle (URL'yi kendi repository URL'inizle değiştirin)
git remote add origin https://github.com/KULLANICI-ADI/REPOSITORY-ADI.git

# Branch'i main olarak ayarla (GitHub'ın yeni default'u)
git branch -M main

# Dosyaları GitHub'a gönder
git push -u origin main
```

## 4. Alternatif: GitHub Desktop Kullanarak

Eğer GitHub Desktop kullanıyorsanız:

1. GitHub Desktop'ı açın
2. **File > Add Local Repository** seçin
3. Proje klasörünüzü seçin
4. **Publish repository** butonuna tıklayın
5. Repository adını ve açıklamasını girin
6. **Publish** butonuna tıklayın

## 5. Dosya Yapısı Kontrolü

Repository'nizde şu dosyalar olmalı:

```
computer-vision-term-project/
├── .gitignore
├── LICENSE
├── README.md
├── term_project_submission.py
└── GITHUB_SETUP.md (bu dosya)
```

## 6. Sonraki Commit'ler İçin

Kod değişikliklerinden sonra:

```bash
# Değişiklikleri kontrol et
git status

# Değişiklikleri staging area'ya ekle
git add .

# Commit yap
git commit -m "Açıklayıcı commit mesajı"

# GitHub'a gönder
git push
```

## 7. README'yi Güncelleme

README.md dosyasındaki şu kısımları kendi bilgilerinizle güncelleyin:

- Repository URL'leri
- GitHub kullanıcı adınız
- Öğrenci bilgileri (term_project_submission.py içinde de)

## 8. Repository Ayarları (Opsiyonel)

GitHub repository sayfanızda:

1. **Settings > General** bölümüne gidin
2. **Topics** ekleyin: `computer-vision`, `deep-learning`, `transfer-learning`, `tensorflow`, `image-classification`
3. **About** bölümüne kısa açıklama ekleyin
4. Website URL'i varsa ekleyin

## Sorun Giderme

### Problem: "remote origin already exists"

```bash
# Mevcut remote'u kaldır
git remote remove origin

# Yeni remote ekle
git remote add origin https://github.com/KULLANICI-ADI/REPOSITORY-ADI.git
```

### Problem: "failed to push some refs"

```bash
# Önce pull yap
git pull origin main --allow-unrelated-histories

# Sonra push yap
git push -u origin main
```

### Problem: Authentication hatası

GitHub artık password authentication'ı desteklemiyor. Personal Access Token kullanın:

1. GitHub > Settings > Developer settings > Personal access tokens
2. "Generate new token" seçin
3. Token'ı kopyalayın
4. Git push yaparken password yerine token'ı kullanın

Veya SSH key kullanın:
```bash
# SSH key oluştur (eğer yoksa)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Public key'i GitHub'a ekleyin
# Settings > SSH and GPG keys > New SSH key
```

## Faydalı Git Komutları

```bash
# Durumu kontrol et
git status

# Commit geçmişini gör
git log

# Değişiklikleri gör
git diff

# Branch oluştur
git checkout -b feature-branch

# Branch'leri listele
git branch

# Remote repository bilgilerini gör
git remote -v
```


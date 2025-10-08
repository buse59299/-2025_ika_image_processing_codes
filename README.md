# -2025_ika_image_processing_codes
Flask tabanlı gerçek zamanlı görüntü işleme sistemi. YOLOv8 modelini kullanarak kamera görüntüsünde hedef tespiti yapar.

## Özellikler
- 🎥 Gerçek zamanlı kamera görüntüsü akışı
- 🎯 YOLOv8 ile anlık nesne tespiti
- 🖥️ Web tabanlı kullanıcı arayüzü
- ⚡ Hızlı ve optimize edilmiş performans
- 📦 80 farklı COCO sınıfı tespiti

## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- Kamera (webcam veya harici)

### Adımlar

1. Repoyu klonlayın:
```bash
git clone https://github.com/buse59299/-2025_ika_image_processing_codes.git
cd -2025_ika_image_processing_codes
```

2. Sanal ortam oluşturun (önerilen):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Uygulamayı çalıştırın:
```bash
python app.py
```

5. Tarayıcınızda açın:
```
http://localhost:5000
```

## Kullanım
- Uygulama başlatıldığında, tarayıcınızda ana sayfaya gidin
- Kamera izni isteğini kabul edin (gerekirse)
- Gerçek zamanlı nesne tespiti otomatik olarak başlayacaktır
- Tespit edilen nesneler ekranda sınırlayıcı kutular ile gösterilecektir

## Proje Yapısı
```
-2025_ika_image_processing_codes/
├── app.py                 # Ana Flask uygulaması
├── templates/
│   └── index.html        # Web arayüzü şablonu
├── requirements.txt      # Python bağımlılıkları
├── .gitignore           # Git ignore dosyası
└── README.md            # Bu dosya
```

## Kullanılan Teknolojiler
- **Flask**: Web framework
- **OpenCV**: Görüntü işleme
- **YOLOv8**: Nesne tespiti modeli
- **Ultralytics**: YOLOv8 kütüphanesi
- **PyTorch**: Derin öğrenme framework

## Notlar
- İlk çalıştırmada YOLOv8n.pt model dosyası otomatik olarak indirilecektir
- Kamera cihazınızın doğru çalıştığından emin olun
- Daha iyi performans için GPU kullanımı önerilir (opsiyonel)

## Lisans
Bu proje eğitim amaçlı oluşturulmuştur.

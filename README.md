İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme
Bu proje, yazılım geliştirici pozisyonu için başvuran adayların tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmamasını tahmin eden bir makine öğrenmesi modeli içerir.

Özellikler
Rastgele veri üretimi (200+ örnek)
SVM ile sınıflandırma
Veri ölçekleme
Karar sınırı görselleştirme
FastAPI ile web servisi
Interaktif tahmin arayüzü
Gereksinimler
Python 3.11 ve üzeri gereklidir. Gerekli kütüphaneleri yüklemek için:

pip install -r requirements.txt
Kullanım
Model Eğitimi ve Test
python model.py
Bu komut:

Rastgele veri üretir
Modeli eğitir
Performans metriklerini gösterir
Karar sınırını görselleştirir
İnteraktif tahmin arayüzü sunar
Web API Servisi
python api.py
API servisi http://localhost:8000 adresinde çalışır.

API Endpoints
GET /: API bilgileri ve kullanım
POST /tahmin/: Yeni aday tahmini
Girdi formatı:
{
    "tecrube_yili": 5,
    "teknik_puan": 85
}
Veri Özellikleri
tecrube_yili: 0-10 yıl arası
teknik_puan: 0-100 arası
etiket:
1: İşe alınmadı
0: İşe alındı
Etiketleme Kriteri
Tecrübesi 2 yıldan az VE sınav puanı 60'tan düşük olanlar işe alınmıyor (etiket=1)
Diğer durumlar işe alınıyor (etiket=0)
Gelişim Alanları
Farklı kernel fonksiyonları deneme (rbf, poly)
Hiperparametre optimizasyonu (C, gamma)
Daha fazla özellik ekleme (örn. mülakat puanı)
Web arayüzü geliştirme
Model performansını artırma
Lisans

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Rastgele veri üretimi için Faker ve seed ayarı
fake = Faker('tr_TR')
np.random.seed(42)

def veri_uret(n_samples=200):
    """Rastgele başvuru verisi üretir"""
    tecrube_yili = np.random.uniform(0, 10, n_samples)
    teknik_puan = np.random.uniform(0, 100, n_samples)
    
    # Etiketleme kuralı: Tecrübesi 2 yıldan az VE sınav puanı 60'tan düşük olanlar işe alınmıyor (1)
    etiket = np.where((tecrube_yili < 2) & (teknik_puan < 60), 1, 0)
    
    return pd.DataFrame({
        'tecrube_yili': tecrube_yili,
        'teknik_puan': teknik_puan,
        'etiket': etiket
    })

def model_egit(X, y):
    """Veriyi ölçekler ve SVM modelini eğitir"""
    # Veriyi eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Veriyi ölçekle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modeli eğit
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)

def model_degerlendir(model, X_test_scaled, y_test):
    """Model performansını değerlendirir"""
    y_pred = model.predict(X_test_scaled)
    
    print("\nModel Performans Metrikleri:")
    print("-" * 30)
    print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.3f}")
    print("\nKarmaşıklık Matrisi:")
    print(confusion_matrix(y_test, y_pred))
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

def karar_siniri_goster(model, scaler, X, y):
    """Karar sınırını görselleştirir"""
    plt.figure(figsize=(10, 8))
    
    # Mesh grid oluştur
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Tüm mesh noktaları için tahmin yap
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Karar sınırını çiz
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Tecrübe Yılı (Ölçeklenmiş)')
    plt.ylabel('Teknik Puan (Ölçeklenmiş)')
    plt.title('SVM Karar Sınırı')
    plt.show()

def tahmin_yap(model, scaler, tecrube, puan):
    """Yeni bir aday için tahmin yapar"""
    X_yeni = np.array([[tecrube, puan]])
    X_yeni_scaled = scaler.transform(X_yeni)
    tahmin = model.predict(X_yeni_scaled)[0]
    
    return "İşe Alınmadı" if tahmin == 1 else "İşe Alındı"

def main():
    # Veri üret
    print("Veri üretiliyor...")
    df = veri_uret(200)
    print(f"Üretilen veri boyutu: {df.shape}")
    print("\nVeri örneği:")
    print(df.head())
    
    # Model eğitimi
    print("\nModel eğitiliyor...")
    X = df[['tecrube_yili', 'teknik_puan']].values
    y = df['etiket'].values
    
    model, scaler, (X_train_scaled, X_test_scaled, y_train, y_test) = model_egit(X, y)
    
    # Model değerlendirme
    model_degerlendir(model, X_test_scaled, y_test)
    
    # Karar sınırını görselleştir
    karar_siniri_goster(model, scaler, X_train_scaled, y_train)
    
    # Örnek tahmin
    while True:
        try:
            print("\nYeni aday değerlendirmesi için:")
            tecrube = float(input("Tecrübe yılını girin (0-10): "))
            puan = float(input("Teknik puanı girin (0-100): "))
            
            if not (0 <= tecrube <= 10 and 0 <= puan <= 100):
                print("Lütfen geçerli aralıklarda değerler girin!")
                continue
                
            sonuc = tahmin_yap(model, scaler, tecrube, puan)
            print(f"\nTahmin Sonucu: {sonuc}")
            
            devam = input("\nBaşka bir tahmin yapmak ister misiniz? (e/h): ")
            if devam.lower() != 'e':
                break
                
        except ValueError:
            print("Lütfen geçerli sayısal değerler girin!")

if __name__ == "__main__":
    main() 
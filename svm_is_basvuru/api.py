from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from model import veri_uret, model_egit

# Veri modelini tanımla
class AdayGirdi(BaseModel):
    tecrube_yili: float = Field(..., ge=0, le=10, description="Adayın tecrübe yılı (0-10 arası)")
    teknik_puan: float = Field(..., ge=0, le=100, description="Adayın teknik puanı (0-100 arası)")

class TahminSonuc(BaseModel):
    sonuc: str
    olasılık: float

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="İşe Alım Tahmin API",
    description="SVM ile aday değerlendirme API'si",
    version="1.0.0"
)

# Global değişkenler
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında modeli yükle"""
    global model, scaler
    
    # Yeni veri üret ve model eğit
    df = veri_uret(200)
    X = df[['tecrube_yili', 'teknik_puan']].values
    y = df['etiket'].values
    
    model, scaler, _ = model_egit(X, y)

@app.post("/tahmin/", response_model=TahminSonuc)
async def tahmin_yap(aday: AdayGirdi):
    """Yeni aday için tahmin yap"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi")
    
    try:
        # Girdiyi ölçekle ve tahmin yap
        X_yeni = np.array([[aday.tecrube_yili, aday.teknik_puan]])
        X_yeni_scaled = scaler.transform(X_yeni)
        
        # Tahmin yap
        tahmin = model.predict(X_yeni_scaled)[0]
        
        # Karar fonksiyonu değerini al (güven skoru olarak kullanılabilir)
        decision_value = abs(model.decision_function(X_yeni_scaled)[0])
        confidence = 1 / (1 + np.exp(-decision_value))  # Sigmoid dönüşümü
        
        sonuc = "İşe Alınmadı" if tahmin == 1 else "İşe Alındı"
        
        return TahminSonuc(sonuc=sonuc, olasılık=float(confidence))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

@app.get("/")
async def root():
    """API kök endpoint"""
    return {
        "mesaj": "İşe Alım Tahmin API'sine Hoş Geldiniz",
        "kullanım": "/tahmin/ endpoint'ini POST metodu ile kullanarak tahmin yapabilirsiniz",
        "örnek_girdi": {
            "tecrube_yili": 5,
            "teknik_puan": 85
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
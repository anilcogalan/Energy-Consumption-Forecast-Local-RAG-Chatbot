# ⚡ Enerji Tüketimi Tahminleme

Fransa (Sceaux) hane elektrik tüketimi verisi ile zaman serisi tahmini — EDA, feature engineering, LightGBM/XGBoost modelleri.

**Dataset:** [UCI Individual Household Electric Power Consumption (Kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

---

## Veri

- **Dosya:** `datas/household_power_consumption.txt`
- Kaggle’dan indirip **`datas`** klasörüne koyun (yoksa proje kökünde `datas` oluşturun). Dosya adı: `household_power_consumption.txt`.

---

## 1. Notebook’u çalıştırma

Notebook baştan sona çalışır durumdadır; veri yolu proje köküne göre ayarlıdır.

1. Sanal ortam oluşturup bağımlılıkları yükleyin:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Jupyter veya VS Code ile `Energy_Forecasting_Notebook.ipynb` dosyasını açın.
3. Kernel olarak bu projedeki `venv` (veya bağımlılıkları yüklediğiniz Python) seçin.
4. Hücreleri sırayla çalıştırın. İlk hücrelerde veri `datas/household_power_consumption.txt` dosyasından okunur.

---

## 2. Streamlit uygulamasını çalıştırma

Uygulama README’deki adımlarla çalıştırılabilir.

1. Veriyi hazırlayın: `datas/household_power_consumption.txt` mevcut olsun (yukarıdaki gibi).
2. Aynı sanal ortamı kullanın (veya yeniden oluşturup bağımlılıkları yükleyin):
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Uygulamayı başlatın:
   ```bash
   streamlit run app.py
   ```
4. Tarayıcıda açılan adrese gidin (genelde http://localhost:8501).

**Sayfalar:** Veri Görselleştirme · Tahmin Ekranı · Model Performans

**Önbelleği temizlemek:** Sidebar’da **"🗑️ Önbelleği temizle"** butonuna tıklayın; veya sağ üst menü (⋮) → **Clear cache**. Veri veya model değiştiyse bunu yapın.

---

## Gereksinimler

- Python 3.9+
- Bağımlılıklar: `requirements.txt` (Streamlit, pandas, scikit-learn, LightGBM, XGBoost, plotly, vb.)

---

## Özet

| Bileşen              | Çalışır durumda? | Nasıl çalıştırılır?                          |
|----------------------|------------------|----------------------------------------------|
| Notebook             | Evet             | Jupyter/VS Code, kernel = proje venv         |
| Streamlit uygulaması | Evet             | `streamlit run app.py` (venv aktif, veri mevcut) |

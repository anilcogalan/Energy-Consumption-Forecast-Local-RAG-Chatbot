# Projects

Bu depo, iki ayrı veri bilimi / ML projesini içerir. Her proje kendi klasöründe bağımsız çalıştırılabilir.

---

## Projeler

| Proje | Açıklama | Teknolojiler |
|-------|----------|---------------|
| **[Yerel LLM ile Hukuki Metin Analizi](Yerel%20LLM%20ile%20Hukuki%20Metin%20Analizi/)** | Hukuki belgeler üzerinde RAG (Retrieval-Augmented Generation) ile soru–cevap. Yerel LLM (LM Studio / Ollama) + ChromaDB + LexGLUE veri seti. | Streamlit, ChromaDB, Sentence Transformers, LexGLUE |
| **[Household Electric Power Consumption](Household%20Electric%20Power%20Consumption%20%20/)** | Fransa hane elektrik tüketimi verisi ile zaman serisi tahmini — EDA, feature engineering, LightGBM/XGBoost. | Streamlit, pandas, scikit-learn, LightGBM, XGBoost, Plotly |

---

## 1. Yerel LLM ile Hukuki Metin Analizi

Hukuki metinleri yerel bir LLM ile sorgulamanızı sağlayan Streamlit uygulaması. Basic / Adaptive / RAG Fusion yöntemleri, belge yükleme (PDF, DOCX, TXT), konuşma geçmişi ve PDF dışa aktarma sunar.

**Hızlı başlangıç:**

```bash
cd "Yerel LLM ile Hukuki Metin Analizi"
pip install -r requirements.txt
# ChromaDB için: LexGLUE_Legal_RAG.ipynb çalıştırın veya uygulama içinden belge yükleyin
streamlit run app.py
```

- **LLM:** LM Studio (örn. `http://localhost:1234`) veya OpenAI-uyumlu API.
- **Detaylı kurulum ve özellikler:** [Yerel LLM ile Hukuki Metin Analizi/README.md](Yerel%20LLM%20ile%20Hukuki%20Metin%20Analizi/README.md)

---

## 2. Household Electric Power Consumption

UCI hane elektrik tüketimi veri seti ile enerji tüketimi tahminleme: EDA, görselleştirme ve Streamlit ile tahmin/model performans ekranları.

**Hızlı başlangıç:**

```bash
cd "Household Electric Power Consumption  "
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Veriyi Kaggle'dan indirip datas/ klasörüne koyun (household_power_consumption.txt)
streamlit run app.py
```

- **Veri:** [UCI Electric Power Consumption (Kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) → `datas/household_power_consumption.txt`
- **Detaylı kurulum:** [Household Electric Power Consumption  /README.md](Household%20Electric%20Power%20Consumption%20%20/README.md)

---

## Depo yapısı

```
Projects/
├── README.md                    # Bu dosya
├── .gitignore
├── Yerel LLM ile Hukuki Metin Analizi/
│   ├── README.md
│   ├── app.py                   # Streamlit RAG uygulaması
│   ├── doc_upload.py            # Belge yükleme ve ChromaDB indexleme
│   ├── LexGLUE_Legal_RAG.ipynb # Veri hazırlama ve indexleme
│   ├── requirements.txt
│   └── ...
└── Household Electric Power Consumption  /
    ├── README.md
    ├── app.py                   # Streamlit tahmin uygulaması
    ├── Energy_Forecasting_Notebook.ipynb
    ├── requirements.txt
    ├── datas/                   # Veri (household_power_consumption.txt burada)
    └── ...
```

Her proje kendi `requirements.txt` ve (isteğe bağlı) `venv` ile çalıştırılabilir; ortak bir sanal ortam zorunlu değildir.

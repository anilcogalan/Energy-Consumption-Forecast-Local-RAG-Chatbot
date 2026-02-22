# ⚖️ LexGLUE Legal RAG System — Streamlit Demo

Hukuki belgelerin yerel LLM mimarisi ile işlenmesi ve sorgulanması için Streamlit demo uygulaması.

## 📋 Özellikler

- **3 RAG Yöntemi:** Basic RAG, Adaptive RAG, RAG Fusion
- **Yerel / OpenAI-uyumlu LLM:** LM Studio veya herhangi bir OpenAI-uyumlu API (Ollama vb.)
- **6 Hukuki Alan:** Case Law, Human Rights, EU Legislation, Contracts, Consumer Law, Supreme Court
- **Domain Filtreleme:** Metadata tabanlı alan filtresi
- **Kaynak Gösterimi:** Her cevapla birlikte ilgili dokümanlar, similarity skorları ve metadata
- **Konuşma Geçmişi:** Birden fazla sohbet kaydedilir; sidebar’dan yeni konuşma, geçmişe geçiş veya konuşma silme
- **PDF Dışa Aktarma:** Konuşmayı kaynaklar ve metadata ile birlikte PDF olarak indirme
- **Sohbet Arayüzü:** WhatsApp tarzı balonlar, örnek sorular ve RAG hafızası (son 5 mesaj bağlama dahil)
- **Belge Yükleme:** Sidebar’dan PDF, DOCX veya TXT yükleyebilirsiniz. Belgeler LLM ile hukuki olup olmadığı kontrol edilir; onaylananlar ChromaDB’ye chunk’lanıp indexlenir. Aynı belge tekrar yüklendiğinde (hash ile) atlanır.

## 🚀 Kurulum & Çalıştırma

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

Bağımlılıklar arasında **pypdf** ve **python-docx** belge yükleme (PDF/DOCX) için kullanılır.

### 2. LM Studio ile Yerel LLM
1. [LM Studio](https://lmstudio.ai/) indirip kurun.
2. İstediğiniz modeli arayıp indirin (örn. **meta-llama-3-8b-instruct**, **mistral-7b-instruct-v0.3**, **llama3-8b-lawyer-v2**).
3. Sol alttan **Local Server** sekmesini açıp sunucuyu başlatın (varsayılan: `http://localhost:1234`).
4. Uygulama bu adrese bağlanır; sidebar’dan farklı model seçebilirsiniz.

### 3. ChromaDB Hazırlığı
Önce Jupyter Notebook’u (`LexGLUE_Legal_RAG.ipynb`) çalıştırarak:
- LexGLUE dataset’ini indirin
- Verileri ön işleyin ve chunk’layın
- ChromaDB’ye indexleyin

Bu işlem `./chroma_db/` klasörünü oluşturur. İsterseniz sadece sidebar’daki **Belge Yükle** ile kendi PDF/DOCX/TXT belgelerinizi de indexleyebilirsiniz (önce notebook ile en az bir kez ChromaDB’nin oluşturulması gerekir).

### 4. Uygulamayı Başlat
```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## ⚙️ Yapılandırma

Sidebar’dan ayarlanabilir:

| Ayar | Açıklama | Varsayılan |
|------|----------|------------|
| Endpoint URL | LM Studio / OpenAI-uyumlu API adresi | `http://localhost:1234` |
| Model | LM Studio’daki model adı (örn. meta-llama-3-8b-instruct, mistral-7b-instruct-v0.3, llama3-8b-lawyer-v2) | `meta-llama-3-8b-instruct` |
| RAG Yöntemi | Basic / Adaptive / Fusion | Basic RAG |
| Top-K | Retrieval sonuç sayısı | 5 |
| Domain Filtresi | Hukuki alan filtresi | Tümü |
| Belge Yükle | PDF, DOCX, TXT — sadece hukuki belgeler LLM ile doğrulanıp indexlenir | — |

Embedding modeli uygulama içinde **all-MiniLM-L6-v2** (Sentence Transformers) olarak sabittir.

## 📊 RAG Yöntemleri

### Basic RAG
```
Sorgu → Embed → Top-K Retrieval → LLM → Cevap
```
Hızlı, basit sorular için ideal.

### Adaptive RAG
```
Sorgu → Karmaşıklık sınıflandırıcı → Yönlendirme:
  SIMPLE   → Minimal retrieval + LLM
  MODERATE → Standart RAG
  COMPLEX  → Parçala → Çok adımlı → Sentezle
```
Kaynak verimliliği için akıllı yönlendirme.

### RAG Fusion
```
Sorgu → N varyant üret → Her biri için retrieval → RRF → LLM → Cevap
```
En kapsamlı retrieval; birden fazla perspektifle arama.

## 📄 Belge Yükleme

Sidebar’daki **📄 Belge Yükle** bölümünden:

1. **Belge seçin:** PDF, DOCX veya TXT yükleyin.
2. **Belgeyi İşle ve İndexle:** Tıklayınca metin çıkarılır, LLM ile “hukuki belge” olup olmadığı kontrol edilir; onaylanırsa chunk’lanıp ChromaDB’ye eklenir.
3. Aynı içerik (hash ile) daha önce yüklendiyse tekrar indexlenmez.
4. **Yüklenen Belgeler** listesinden hangi dosyaların kaç chunk ile eklendiği görüntülenir.

Bu işlem `doc_upload` modülünü (metin çıkarma, hukuki doğrulama, chunking, indexleme) kullanır.

## 📁 Proje Yapısı

```
├── app.py                      # Streamlit uygulaması (RAG, sohbet, PDF export, belge yükleme)
├── doc_upload.py               # Belge yükleme: metin çıkarma (PDF/DOCX/TXT), LLM ile hukuki doğrulama, ChromaDB indexleme
├── requirements.txt            # Python bağımlılıkları (pypdf, python-docx, fpdf2, streamlit, chromadb, vb.)
├── README.md                   # Bu dosya
├── LexGLUE_Legal_RAG.ipynb     # Veri hazırlama ve ChromaDB indexleme (LexGLUE dataset)
├── chroma_db/                  # ChromaDB vektör veritabanı (notebook veya belge yükleme ile doldurulur)
├── .lexglue_conversations.json # Konuşma geçmişi (uygulama tarafından oluşturulur)
├── data/                       # LexGLUE CSV dosyaları (notebook ile indirilir)
└── example_pdfs/               # (İsteğe bağlı) Örnek PDF’ler
```

## 🔗 Kaynaklar

- **Dataset:** [LexGLUE — Kaggle](https://www.kaggle.com/datasets/thedevastator/lexglue-legal-nlp-benchmark-dataset)
- **LexGLUE Paper:** Chalkidis et al., "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English", ACL 2022
- **SaulLM-7B:** Colombo et al., "SaulLM-7B: A pioneering Large Language Model for Law", 2024

# Auto Data Bot 🧠🤖

**Auto Data Bot** is a powerful AI assistant for **data analysis, automated machine learning**, and **general question answering**.  
It can handle **large datasets with millions of rows**, analyze them in chunks, generate **visualizations, cleaned data, and trained models**, and answer questions from both uploaded datasets and a **Wikipedia-based knowledge base**.

It is fully **offline-compatible**, supports **FAISS-based vector search**, **TPOT AutoML**, and a **ChatGPT-like Flask front-end** for interactive queries.

---

## Features ✨

### 1. Dataset Analysis & AutoML
- Handles **large datasets** efficiently using chunked processing.
- **Data cleaning and preprocessing** automatically:
  - Missing value handling  
  - Categorical encoding  
  - Outlier detection  
- Generates **automated profiling reports** using **ydata-profiling**.
- Automatically selects the **best model using TPOT AutoML**.
- Saves **trained models** and **visualizations**.

### 2. General Knowledge Question Answering
- FAISS + SentenceTransformer embeddings for **vector-based retrieval**.
- Dynamically fetches **Wikipedia pages** to answer questions.
- Supports **offline caching** of previously retrieved content.
- Answers come from **uploaded datasets** and **Wikipedia**.

### 3. Front-End
- Built with **Flask**, designed like **ChatGPT interface**:
  - Single prompt bar  
  - Buttons for **Dataset Analysis** and **General Questions**  
  - Typing indicators for interactive experience  
- Returns answers and downloadable **analysis reports** in `.zip`.

### 4. Scalable & Efficient
- Works with datasets of **millions of rows** using **chunked processing**.
- FAISS vector index ensures **fast similarity search**.
- Integrates **TPOT** for automated ML pipeline generation.

---

## Installation 💻

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/auto-data-bot.git
cd auto-data-bot

### Create and activate virtual environment
python3 -m venv .bot
source .bot/bin/activate

pip install -r requirements.txt


Auto-Data-Bot/
│
├── app.py                  # Flask main app
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Dataset and FAISS indexes
│   ├── faiss_dataset/
│   └── faiss_general/
├── core/
│   ├── automl_engine.py    # TPOT AutoML & dataset processing
│   └── chatbot_engine.py   # FAISS chatbot engine + Wikipedia integration
├── models/                 # Saved trained models
└── templates/
    └── index.html          # ChatGPT-style front-end




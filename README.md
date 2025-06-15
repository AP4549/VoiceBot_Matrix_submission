# 🧠 Matrix VoiceBot

> 🔊 A multilingual, memory-augmented voice assistant powered by AWS and Supabase – for seamless Q&A from CSVs or your voice.

---

## 📌 Overview

**Matrix VoiceBot** is a voice-enabled chatbot that semantically matches user questions against a preloaded QA dataset and optionally uses AWS Bedrock for fallback LLM responses. It supports:

- ✅ **Batch CSV QA inference**
- 🎤 **Live interactive voice assistant** (Gradio-based)
- 🧠 **Memory-augmented responses with vector memory**
- 🔐 **User auth and chat logging via Supabase**

Built with AWS Transcribe (STT), Polly (TTS), Bedrock (LLM), FAISS for semantic search, and a modern Python + Gradio stack.

---

## 🔧 Features

### 🔁 Batch QA Inference (`run_inference.py`)
- Upload a CSV file with a `Questions` column
- Semantic search over `Data/qa_dataset.csv` using FAISS + Sentence Transformers
- Optional fallback to AWS Bedrock (Titan Embeddings + Claude LLM)
- Output includes:
  - Matched response
  - Confidence score
  - Source: `Dataset` / `LLM`

![image](https://github.com/user-attachments/assets/4b75106c-6c51-41df-9cb7-6cec3af158bb)


### 🎙️ Interactive Voice Assistant (`main.py`)
- Real-time mic input and audio file upload
- **Speech-to-Text**: AWS Transcribe  
- **Text-to-Speech**: AWS Polly  
- Gradio-based voice chat interface with:
  - `response_gen_rag.py`: Vanilla FAISS + polishing
  - `response_gen_ragb.py`: Hinglish + vector memory for deep context

### 🔐 Supabase Integration
- Signup/login with `create_test_user.py`
- Persistent conversation storage via `modules/supabase_client.py`
- Environment-driven user ID management

### 🧪 Testing
- `Test/test_qa.py`: End-to-end QA pipeline
- `Test/test_signup.py`: Supabase auth and storage tests

---

## ⚙️ Installation

### 📥 Prerequisites

- Python 3.8+
- AWS account with:
  - Bedrock
  - Transcribe
  - Polly
  - S3
- Supabase project with:
  - Supabase URL
  - Anon key

### 💻 Setup

```bash
git clone <repository-url>
cd MatrixVoiceBot
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file at the project root:

```ini
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token  # optional
AWS_REGION=us-west-2
AWS_S3_BUCKET=your_s3_bucket_name

SUPABASE_URL=https://your.supabase.url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_USER_ID=your_test_user_id
```

---

## 🧠 Configuration

Edit `config/config.yaml` for:

- Model IDs (Bedrock, Claude)
- FAISS confidence threshold
- Language toggle (e.g., Hinglish support)
- Memory enable/disable

---

## 📦 Data Requirements

Place the following in `Data/`:

```
qa_dataset.csv         # Base Q&A pairs
qa_embeddings.npy      # SentenceTransformer embeddings
qa_faiss_index.bin     # FAISS index
test.csv               # Input test file
userinputvoice/        # Folder for audio inputs
voicebotoutput/        # Folder for TTS outputs
```

> 🔄 To generate embeddings & FAISS index:
```bash
python modules/utils.py --build-index
```

---

## 🧪 Usage Guide

### 1️⃣ Batch QA (CSV)

```bash
python run_inference.py
```

- Reads `Data/test.csv`
- Writes to `output/output.csv` with:
  - Matched answer
  - Source (Dataset/LLM)
  - Confidence

### 2️⃣ Gradio Live Demo

```bash
python main.py
```

Opens at `http://127.0.0.1:7860` with tabs:

- **Authentication** – Sign in/out via Supabase
- **Voice Assistant** – Speak or upload audio, get response
- **Chat Inference** – Text-based fallback
- **About** – Info and system overview

### 3️⃣ User Signup (Supabase)

```bash
python create_test_user.py
```

- Creates and logs in a test user
- Stores test conversations via Supabase
- Updates `.env` with `SUPABASE_USER_ID`

---

### 🧪 Run Tests

```bash
pytest
```

- `Test/test_qa.py` – Batch QA pipeline
- `Test/test_signup.py` – Supabase auth

---

## 📁 Project Structure

```
MatrixVoiceBot/
├── .env
├── config/
│   └── config.yaml
├── create_test_user.py
├── main.py                  # Gradio UI
├── run_inference.py         # Batch mode
├── modules/
│   ├── aws_asr.py           # Transcribe wrapper
│   ├── aws_tts.py           # Polly wrapper
│   ├── response_gen.py
│   ├── response_gen_rag.py
│   ├── response_gen_ragb.py # With memory
│   ├── supabase_client.py
│   ├── vector_memory.py
│   └── utils.py
├── Data/
│   ├── qa_dataset.csv
│   ├── qa_embeddings.npy
│   ├── qa_faiss_index.bin
│   ├── test.csv
│   ├── userinputvoice/
│   └── voicebotoutput/
├── output/
│   └── output.csv
├── Test/
│   ├── test_qa.py
│   └── test_signup.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Enhancements

- 🌐 Multilingual support (Hindi, Bengali, Marathi)
- 💬 Local LLM fallback (e.g., Ollama, LM Studio)
- 🧠 Dynamic RAG with live retraining
- 📈 Analytics dashboard (Supabase/Streamlit)
- 🔗 CRM + WhatsApp Integration

---

## 💸 AWS Cost Disclaimer

> AWS Transcribe, Polly, and Bedrock services may incur charges beyond the free tier. Monitor your usage via the [AWS Console](https://console.aws.amazon.com/).

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Add your feature or fix with tests
4. Submit a pull request 🚀

---

## 📜 License

MIT License. See `LICENSE`.

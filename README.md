# 📄 Doc-Vid-Analyze: AI-Powered Document & Video Analyzer

A full-stack AI-powered tool for extracting and analyzing text from documents and videos using **OCR (Optical Character Recognition)** and **NLP (Natural Language Processing)**. The project integrates a Python backend with a JavaScript-based frontend.

## 🚀 Features

### 📄 Document Analysis
- Extracts text from PDFs and images using Tesseract OCR.
- Summarizes extracted text using NLP techniques.
- Identifies named entities (NER) and key topics.

### 🎥 Video Analysis
- Extracts frames from video files.
- Applies OCR on extracted frames.
- Generates text summaries from frames.

### 📊 Visualization & Frontend
- Web interface for user interaction.
- Word clouds for text visualization.
- Frame-by-frame analysis.

## 🛠️ Tech Stack

### Backend (Python)
- **Flask/FastAPI** - API framework
- **OpenCV** - Video processing
- **pytesseract** - OCR engine
- **spaCy / NLTK** - NLP processing
- **Streamlit** - Interactive frontend (optional)

### Frontend (JavaScript)
- **React.js / Vue.js** (based on frontend files)
- **HTML, CSS, JavaScript** for UI/UX

## 📁 Project Structure

```
doc-vid-analyze-main/
│
├── backend/                  # Python-based API & OCR processing
│   ├── run.py                # Main script
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Deployment containerization
│   ├── notebooks/            # Jupyter notebooks for analysis
│
├── frontend/                 # JavaScript-based UI
│   ├── public/               # Static assets
│   ├── src/                  # React/Vue components
│   ├── package.json          # NPM dependencies
│
└── README.md                 # You're here!
```

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/tejash-300/doc-vid-analyze-main.git
cd doc-vid-analyze-main
```

### 2️⃣ Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Install Tesseract OCR
- **Ubuntu**: `sudo apt install tesseract-ocr`
- **Windows**: [Download from Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)

### 4️⃣ Run the Backend API
```bash
python run.py
```

### 5️⃣ Install Frontend Dependencies & Run
```bash
cd frontend
npm install
npm start
```

## 📌 TODO
- [ ] Improve NLP-based summarization
- [ ] Integrate audio-to-text transcription
- [ ] Enhance frontend UI with more visualizations

## 🤝 Contributing
Pull requests are welcome! Open an issue for discussions on improvements.

## 📄 License
[MIT License](LICENSE)

## 👨‍💻 Author
**Tejash Pandey**  
Connect: [LinkedIn](https://www.linkedin.com/in/tejashpandey) | [GitHub](https://github.com/tejash-300)


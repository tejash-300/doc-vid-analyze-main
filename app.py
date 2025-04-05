import os
os.environ["TRANSFORMERS_NO_FAST"] = "1"  # Force use of slow tokenizers

import io
import torch
import uvicorn
import spacy
import pdfplumber
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pyngrok import ngrok
from threading import Thread
import time
import uuid
import subprocess  # For running ffmpeg commands
import hashlib  # For caching file results

# For asynchronous blocking calls
from starlette.concurrency import run_in_threadpool

# Import gensim for topic modeling
import gensim
from gensim import corpora, models

# Import spacy stop words
from spacy.lang.en.stop_words import STOP_WORDS

# Global cache for analysis results based on file hash
analysis_cache = {}

# Ensure compatibility with Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass  # Skip drive mount if not in Google Colab

# Ensure required directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Ensure GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize FastAPI
app = FastAPI(title="Legal Document and Video Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for document text and chat history
document_storage = {}
chat_history = []

# Function to store document context by task ID
def store_document_context(task_id, text):
    document_storage[task_id] = text
    return True

# Function to load document context by task ID
def load_document_context(task_id):
    return document_storage.get(task_id, "")

# Utility to compute MD5 hash from file content
def compute_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

#############################
#   Fine-tuning on CUAD QA   #
#############################

def fine_tune_cuad_model():
    from datasets import load_dataset
    import numpy as np
    from transformers import Trainer, TrainingArguments, AutoModelForQuestionAnswering

    print("✅ Loading CUAD dataset for fine tuning...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    if "train" in dataset:
        train_dataset = dataset["train"].select(range(50))
        if "validation" in dataset:
            val_dataset = dataset["validation"].select(range(10))
        else:
            split = train_dataset.train_test_split(test_size=0.2)
            train_dataset = split["train"]
            val_dataset = split["test"]
    else:
        raise ValueError("CUAD dataset does not have a train split")

    print("✅ Preparing training features...")
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                tokenized_start_index = 0
                while sequence_ids[tokenized_start_index] != 1:
                    tokenized_start_index += 1
                tokenized_end_index = len(input_ids) - 1
                while sequence_ids[tokenized_end_index] != 1:
                    tokenized_end_index -= 1
                if not (offsets[tokenized_start_index][0] <= start_char and offsets[tokenized_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while tokenized_start_index < len(offsets) and offsets[tokenized_start_index][0] <= start_char:
                        tokenized_start_index += 1
                    tokenized_examples["start_positions"].append(tokenized_start_index - 1)
                    while offsets[tokenized_end_index][1] >= end_char:
                        tokenized_end_index -= 1
                    tokenized_examples["end_positions"].append(tokenized_end_index + 1)
        return tokenized_examples

    print("✅ Tokenizing dataset...")
    train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(prepare_train_features, batched=True, remove_columns=val_dataset.column_names)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

    training_args = TrainingArguments(
        output_dir="./fine_tuned_legal_qa",
        max_steps=1,
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=1,
        load_best_model_at_end=False,
        report_to=[]
    )

    print("✅ Starting fine tuning on CUAD QA dataset...")
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    print("✅ Fine tuning completed. Saving model...")
    model.save_pretrained("./fine_tuned_legal_qa")
    tokenizer.save_pretrained("./fine_tuned_legal_qa")
    return tokenizer, model

#############################
#    Load NLP Models       #
#############################

try:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    print("✅ Loading NLP models...")

    # Update summarizer to use facebook/bart-large-cnn for summarization
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )
    if device == "cuda":
        try:
            summarizer.model.half()
        except Exception as e:
            print("FP16 conversion failed:", e)

    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    ner_model = pipeline("ner", model="dslim/bert-base-NER", device=0 if torch.cuda.is_available() else -1)
    speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-medium", chunk_length_s=30,
                              device_map="auto" if torch.cuda.is_available() else "cpu")
    if os.path.exists("fine_tuned_legal_qa"):
        print("✅ Loading fine-tuned CUAD QA model from fine_tuned_legal_qa...")
        cuad_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_legal_qa")
        from transformers import AutoModelForQuestionAnswering
        cuad_model = AutoModelForQuestionAnswering.from_pretrained("fine_tuned_legal_qa")
        cuad_model.to(device)
        if device == "cuda":
            cuad_model.half()
    else:
        print("⚠️ Fine-tuned QA model not found. Starting fine tuning on CUAD QA dataset. This may take a while...")
        cuad_tokenizer, cuad_model = fine_tune_cuad_model()
        cuad_model.to(device)
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading models: {str(e)}")
    raise RuntimeError(f"Error loading models: {str(e)}")

from transformers import pipeline
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

def legal_chatbot(user_input, context):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    response = qa_model(question=user_input, context=context)["answer"]
    chat_history.append({"role": "assistant", "content": response})
    return response

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip() if text else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

async def process_video_to_text(video_file_path):
    try:
        print(f"Processing video file at {video_file_path}")
        temp_audio_path = os.path.join("temp", "extracted_audio.wav")
        cmd = [
            "ffmpeg", "-i", video_file_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            temp_audio_path, "-y"
        ]
        await run_in_threadpool(subprocess.run, cmd, check=True)
        print(f"Audio extracted to {temp_audio_path}")
        result = await run_in_threadpool(speech_to_text, temp_audio_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return transcript
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Video processing failed: {str(e)}")

async def process_audio_to_text(audio_file_path):
    try:
        print(f"Processing audio file at {audio_file_path}")
        result = await run_in_threadpool(speech_to_text, audio_file_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        return transcript
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

def extract_named_entities(text):
    max_length = 10000
    entities = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        doc = nlp(chunk)
        entities.extend([{"entity": ent.text, "label": ent.label_} for ent in doc.ents])
    return entities

# -----------------------------
# Enhanced Risk Analysis Functions
# -----------------------------

def analyze_sentiment(text):
    sentences = [sent.text for sent in nlp(text).sents]
    if not sentences:
        return 0
    results = sentiment_pipeline(sentences, batch_size=16)
    scores = [res["score"] if res["label"] == "POSITIVE" else -res["score"] for res in results]
    avg_sentiment = sum(scores) / len(scores) if scores else 0
    return avg_sentiment

def analyze_topics(text, num_topics=3):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    if not tokens:
        return []
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_topics=num_topics)
    return topics

def get_enhanced_context_info(text):
    enhanced = {}
    enhanced["average_sentiment"] = analyze_sentiment(text)
    enhanced["topics"] = analyze_topics(text, num_topics=5)
    return enhanced

# New function to create a detailed, dynamic explanation for each topic
def explain_topics(topics):
    explanation = {}
    for topic_idx, topic_str in topics:
        # Split topic string into individual weighted terms
        parts = topic_str.split('+')
        terms = []
        for part in parts:
            part = part.strip()
            if '*' in part:
                weight_str, word = part.split('*', 1)
                word = word.strip().strip('\"').strip('\'')
                try:
                    weight = float(weight_str)
                except:
                    weight = 0.0
                # Filter out common stop words
                if word.lower() not in STOP_WORDS and len(word) > 1:
                    terms.append((weight, word))
        terms.sort(key=lambda x: -x[0])
        # Create a plain language label based on dominant words
        if terms:
            if any("liability" in word.lower() for weight, word in terms):
                label = "Liability & Penalty Risk"
            elif any("termination" in word.lower() for weight, word in terms):
                label = "Termination & Refund Risk"
            elif any("compliance" in word.lower() for weight, word in terms):
                label = "Compliance & Regulatory Risk"
            else:
                label = "General Risk Language"
        else:
            label = "General Risk Language"
        explanation_text = (
            f"Topic {topic_idx} ({label}) is characterized by dominant terms: " +
            ", ".join([f"'{word}' ({weight:.3f})" for weight, word in terms[:5]])
        )
        explanation[topic_idx] = {
            "label": label,
            "explanation": explanation_text,
            "terms": terms
        }
    return explanation

def analyze_risk_enhanced(text):
    enhanced = get_enhanced_context_info(text)
    avg_sentiment = enhanced["average_sentiment"]
    risk_score = abs(avg_sentiment) if avg_sentiment < 0 else 0
    topics_raw = enhanced["topics"]
    topics_explanation = explain_topics(topics_raw)
    return {
        "risk_score": risk_score,
        "average_sentiment": avg_sentiment,
        "topics": topics_raw,
        "topics_explanation": topics_explanation
    }

def analyze_contract_clauses(text):
    max_length = 512
    step = 256
    clauses_detected = []
    try:
        clause_types = list(cuad_model.config.id2label.values())
    except Exception:
        clause_types = [
            "Obligations of Seller", "Governing Law", "Termination", "Indemnification",
            "Confidentiality", "Insurance", "Non-Compete", "Change of Control",
            "Assignment", "Warranty", "Limitation of Liability", "Arbitration",
            "IP Rights", "Force Majeure", "Revenue/Profit Sharing", "Audit Rights"
        ]
    chunks = [text[i:i+max_length] for i in range(0, len(text), step) if i+step < len(text)]
    for chunk in chunks:
        inputs = cuad_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = cuad_model(**inputs)
        predictions = torch.sigmoid(outputs.start_logits).cpu().numpy()[0]
        for idx, confidence in enumerate(predictions):
            if confidence > 0.5 and idx < len(clause_types):
                clauses_detected.append({"type": clause_types[idx], "confidence": float(confidence)})
    aggregated_clauses = {}
    for clause in clauses_detected:
        clause_type = clause["type"]
        if clause_type not in aggregated_clauses or clause["confidence"] > aggregated_clauses[clause_type]["confidence"]:
            aggregated_clauses[clause_type] = clause
    return list(aggregated_clauses.values())

# -----------------------------
# Endpoints
# -----------------------------

@app.post("/analyze_legal_document")
async def analyze_legal_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_hash = compute_md5(content)
        if file_hash in analysis_cache:
            return analysis_cache[file_hash]
        text = await run_in_threadpool(extract_text_from_pdf, io.BytesIO(content))
        if not text:
            return {"status": "error", "message": "No valid text found in the document."}
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Document too short for meaningful summarization."
        entities = extract_named_entities(text)
        risk_analysis = analyze_risk_enhanced(text)
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        result = {
            "status": "success",
            "task_id": generated_task_id,
            "summary": summary,
            "named_entities": entities,
            "risk_analysis": risk_analysis,
            "clauses_detected": clauses
        }
        analysis_cache[file_hash] = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze_legal_video")
async def analyze_legal_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        content = await file.read()
        file_hash = compute_md5(content)
        if file_hash in analysis_cache:
            return analysis_cache[file_hash]
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        text = await process_video_to_text(temp_file_path)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if not text:
            return {"status": "error", "message": "No speech could be transcribed from the video."}
        transcript_path = os.path.join("static", f"transcript_{int(time.time())}.txt")
        with open(transcript_path, "w") as f:
            f.write(text)
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Transcript too short for meaningful summarization."
        entities = extract_named_entities(text)
        risk_analysis = analyze_risk_enhanced(text)
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        result = {
            "status": "success",
            "task_id": generated_task_id,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "named_entities": entities,
            "risk_analysis": risk_analysis,
            "clauses_detected": clauses
        }
        analysis_cache[file_hash] = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze_legal_audio")
async def analyze_legal_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        content = await file.read()
        file_hash = compute_md5(content)
        if file_hash in analysis_cache:
            return analysis_cache[file_hash]
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        text = await process_audio_to_text(temp_audio_path=temp_file_path)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if not text:
            return {"status": "error", "message": "No speech could be transcribed from the audio."}
        transcript_path = os.path.join("static", f"transcript_{int(time.time())}.txt")
        with open(transcript_path, "w") as f:
            f.write(text)
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Transcript too short for meaningful summarization."
        entities = extract_named_entities(text)
        risk_analysis = analyze_risk_enhanced(text)
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        result = {
            "status": "success",
            "task_id": generated_task_id,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "named_entities": entities,
            "risk_analysis": risk_analysis,
            "clauses_detected": clauses
        }
        analysis_cache[file_hash] = result
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    transcript_path = os.path.join("static", f"transcript_{transcript_id}.txt")
    if os.path.exists(transcript_path):
        return FileResponse(transcript_path)
    else:
        raise HTTPException(status_code=404, detail="Transcript not found")

@app.post("/legal_chatbot")
async def legal_chatbot_api(query: str = Form(...), task_id: str = Form(...)):
    document_context = load_document_context(task_id)
    if not document_context:
        return {"response": "⚠️ No relevant document found for this task ID."}
    response = legal_chatbot(query, document_context)
    return {"response": response, "chat_history": chat_history[-5:]}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "models_loaded": True,
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }

def setup_ngrok():
    try:
        auth_token = os.environ.get("NGROK_AUTH_TOKEN")
        if auth_token:
            ngrok.set_auth_token(auth_token)
        ngrok.kill()
        time.sleep(1)
        ngrok_tunnel = ngrok.connect(8500, "http")
        public_url = ngrok_tunnel.public_url
        print(f"✅ Ngrok Public URL: {public_url}")
        def keep_alive():
            while True:
                time.sleep(60)
                try:
                    tunnels = ngrok.get_tunnels()
                    if not tunnels:
                        print("⚠️ Ngrok tunnel closed. Reconnecting...")
                        ngrok_tunnel = ngrok.connect(8500, "http")
                        print(f"✅ Reconnected. New URL: {ngrok_tunnel.public_url}")
                except Exception as e:
                    print(f"⚠️ Ngrok error: {e}")
        Thread(target=keep_alive, daemon=True).start()
        return public_url
    except Exception as e:
        print(f"⚠️ Ngrok setup error: {e}")
        return None

# ------------------------------
# Clause Visualization Endpoints
# ------------------------------

@app.get("/download_clause_bar_chart")
async def download_clause_bar_chart(task_id: str):
    try:
        text = load_document_context(task_id)
        if not text:
            raise HTTPException(status_code=404, detail="Document context not found")
        clauses = analyze_contract_clauses(text)
        if not clauses:
            raise HTTPException(status_code=404, detail="No clauses detected.")
        clause_types = [c["type"] for c in clauses]
        confidences = [c["confidence"] for c in clauses]
        plt.figure(figsize=(10, 6))
        plt.bar(clause_types, confidences, color='blue')
        plt.xlabel("Clause Type")
        plt.ylabel("Confidence Score")
        plt.title("Extracted Legal Clause Confidence Scores")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        bar_chart_path = os.path.join("static", f"clause_bar_chart_{task_id}.png")
        plt.savefig(bar_chart_path)
        plt.close()
        return FileResponse(bar_chart_path, media_type="image/png", filename=f"clause_bar_chart_{task_id}.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating clause bar chart: {str(e)}")

@app.get("/download_clause_donut_chart")
async def download_clause_donut_chart(task_id: str):
    try:
        text = load_document_context(task_id)
        if not text:
            raise HTTPException(status_code=404, detail="Document context not found")
        clauses = analyze_contract_clauses(text)
        if not clauses:
            raise HTTPException(status_code=404, detail="No clauses detected.")
        from collections import Counter
        clause_counter = Counter([c["type"] for c in clauses])
        labels = list(clause_counter.keys())
        sizes = list(clause_counter.values())
        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title("Clause Type Distribution")
        plt.tight_layout()
        donut_chart_path = os.path.join("static", f"clause_donut_chart_{task_id}.png")
        plt.savefig(donut_chart_path)
        plt.close()
        return FileResponse(donut_chart_path, media_type="image/png", filename=f"clause_donut_chart_{task_id}.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating clause donut chart: {str(e)}")

@app.get("/download_clause_radar_chart")
async def download_clause_radar_chart(task_id: str):
    try:
        text = load_document_context(task_id)
        if not text:
            raise HTTPException(status_code=404, detail="Document context not found")
        clauses = analyze_contract_clauses(text)
        if not clauses:
            raise HTTPException(status_code=404, detail="No clauses detected.")
        labels = [c["type"] for c in clauses]
        values = [c["confidence"] for c in clauses]
        labels += labels[:1]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.set_title("Legal Clause Radar Chart", y=1.1)
        radar_chart_path = os.path.join("static", f"clause_radar_chart_{task_id}.png")
        plt.savefig(radar_chart_path)
        plt.close()
        return FileResponse(radar_chart_path, media_type="image/png", filename=f"clause_radar_chart_{task_id}.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating clause radar chart: {str(e)}")

def run():
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8500, timeout_keep_alive=600)

if __name__ == "__main__":
    public_url = setup_ngrok()
    if public_url:
        print(f"\n✅ Your API is publicly available at: {public_url}/docs\n")
    else:
        print("\n⚠️ Ngrok setup failed. API will only be available locally.\n")
    run()


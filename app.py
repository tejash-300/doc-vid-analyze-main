import os
import io
import torch
import uvicorn
import spacy
import pdfplumber
import moviepy.editor as mp
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pyngrok import ngrok
from threading import Thread
import time
import uuid

# Ensure compatibility with Google Colab (if applicable)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
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

# Initialize document storage
document_storage = {}
chat_history = []  # Global chat history

# Function to store document context by task ID
def store_document_context(task_id, text):
    """Store document text for retrieval by chatbot."""
    document_storage[task_id] = text
    return True

# Function to load document context by task ID
def load_document_context(task_id):
    """Retrieve document text for chatbot context."""
    return document_storage.get(task_id, "")

#############################
#   Fine-tuning on CUAD QA   #
#############################

def fine_tune_cuad_model():
    """
    Fine tunes a question-answering model on the CUAD (Contract Understanding Atticus Dataset)
    for detailed clause extraction. This demo function uses one epoch for demonstration;
    adjust training parameters as needed.
    """
    from datasets import load_dataset
    import numpy as np
    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForQuestionAnswering

    print("✅ Loading CUAD dataset for fine tuning...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    if "train" in dataset:
        train_dataset = dataset["train"].select(range(1000))
        if "validation" in dataset:
            val_dataset = dataset["validation"].select(range(200))
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
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=100,
        load_best_model_at_end=True,
        report_to=[]
    )

    print("✅ Starting fine tuning on CUAD QA dataset...")
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
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    print("✅ Loading NLP models...")

    summarizer = pipeline("summarization", model="nsi319/legal-pegasus",
                            device=0 if torch.cuda.is_available() else -1)
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    ner_model = pipeline("ner", model="dslim/bert-base-NER",
                         device=0 if torch.cuda.is_available() else -1)
    speech_to_text = pipeline("automatic-speech-recognition",
                              model="openai/whisper-medium",
                              chunk_length_s=30,
                              device_map="auto" if torch.cuda.is_available() else "cpu")

    # Load or Fine Tune CUAD QA Model
    if os.path.exists("fine_tuned_legal_qa"):
        print("✅ Loading fine-tuned CUAD QA model from fine_tuned_legal_qa...")
        cuad_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_legal_qa")
        from transformers import AutoModelForQuestionAnswering
        cuad_model = AutoModelForQuestionAnswering.from_pretrained("fine_tuned_legal_qa")
        cuad_model.to(device)
    else:
        print("⚠️ Fine-tuned QA model not found. Starting fine tuning on CUAD QA dataset. This may take a while...")
        cuad_tokenizer, cuad_model = fine_tune_cuad_model()
        cuad_model.to(device)

    print("✅ All models loaded successfully")

except Exception as e:
    print(f"⚠️ Error loading models: {str(e)}")
    raise RuntimeError(f"Error loading models: {str(e)}")

qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

def legal_chatbot(user_input, context):
    """Uses a real NLP model for legal Q&A."""
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    response = qa_model(question=user_input, context=context)["answer"]
    chat_history.append({"role": "assistant", "content": response})
    return response

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file using pdfplumber."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip() if text else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def process_video_to_text(video_file_path):
    """Extract audio from video and convert to text."""
    try:
        print(f"Processing video file at {video_file_path}")
        temp_audio_path = os.path.join("temp", "extracted_audio.wav")
        video = mp.VideoFileClip(video_file_path)
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        print(f"Audio extracted to {temp_audio_path}")
        result = speech_to_text(temp_audio_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return transcript
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Video processing failed: {str(e)}")

def process_audio_to_text(audio_file_path):
    """Process audio file and convert to text."""
    try:
        print(f"Processing audio file at {audio_file_path}")
        result = speech_to_text(audio_file_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        return transcript
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

def extract_named_entities(text):
    """Extracts named entities from legal text."""
    max_length = 10000
    entities = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        doc = nlp(chunk)
        entities.extend([{"entity": ent.text, "label": ent.label_} for ent in doc.ents])
    return entities

def analyze_risk(text):
    """Analyzes legal risk in the document using keyword-based analysis."""
    risk_keywords = {
        "Liability": ["liability", "responsible", "responsibility", "legal obligation"],
        "Termination": ["termination", "breach", "contract end", "default"],
        "Indemnification": ["indemnification", "indemnify", "hold harmless", "compensate", "compensation"],
        "Payment Risk": ["payment", "terms", "reimbursement", "fee", "schedule", "invoice", "money"],
        "Insurance": ["insurance", "coverage", "policy", "claims"],
    }
    risk_scores = {category: 0 for category in risk_keywords}
    lower_text = text.lower()
    for category, keywords in risk_keywords.items():
        for keyword in keywords:
            risk_scores[category] += lower_text.count(keyword.lower())
    return risk_scores

def extract_context_for_risk_terms(text, risk_keywords, window=1):
    """
    Extracts and summarizes the context around risk terms.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    risk_contexts = {category: [] for category in risk_keywords}
    for i, sent in enumerate(sentences):
        sent_text_lower = sent.text.lower()
        for category, details in risk_keywords.items():
            for keyword in details["keywords"]:
                if keyword.lower() in sent_text_lower:
                    start_idx = max(0, i - window)
                    end_idx = min(len(sentences), i + window + 1)
                    context_chunk = " ".join([s.text for s in sentences[start_idx:end_idx]])
                    risk_contexts[category].append(context_chunk)
    summarized_contexts = {}
    for category, contexts in risk_contexts.items():
        if contexts:
            combined_context = " ".join(contexts)
            try:
                summary_result = summarizer(combined_context, max_length=100, min_length=30, do_sample=False)
                summary = summary_result[0]['summary_text']
            except Exception as e:
                summary = "Context summarization failed."
            summarized_contexts[category] = summary
        else:
            summarized_contexts[category] = "No contextual details found."
    return summarized_contexts

def get_detailed_risk_info(text):
    """
    Returns detailed risk information by merging risk scores with descriptive details
    and contextual summaries from the document.
    """
    risk_details = {
        "Liability": {
            "description": "Liability refers to the legal responsibility for losses or damages.",
            "common_concerns": "Broad liability clauses may expose parties to unforeseen risks.",
            "recommendations": "Review and negotiate clear limits on liability.",
            "example": "E.g., 'The party shall be liable for direct damages due to negligence.'"
        },
        "Termination": {
            "description": "Termination involves conditions under which a contract can be ended.",
            "common_concerns": "Unilateral termination rights or ambiguous conditions can be risky.",
            "recommendations": "Ensure termination clauses are balanced and include notice periods.",
            "example": "E.g., 'Either party may terminate the agreement with 30 days notice.'"
        },
        "Indemnification": {
            "description": "Indemnification requires one party to compensate for losses incurred by the other.",
            "common_concerns": "Overly broad indemnification can shift significant risk.",
            "recommendations": "Negotiate clear limits and carve-outs where necessary.",
            "example": "E.g., 'The seller shall indemnify the buyer against claims from product defects.'"
        },
        "Payment Risk": {
            "description": "Payment risk pertains to terms regarding fees, schedules, and reimbursements.",
            "common_concerns": "Vague payment terms or hidden charges increase risk.",
            "recommendations": "Clarify payment conditions and include penalties for delays.",
            "example": "E.g., 'Payments must be made within 30 days, with a 2% late fee thereafter.'"
        },
        "Insurance": {
            "description": "Insurance risk covers the adequacy and scope of required coverage.",
            "common_concerns": "Insufficient insurance can leave parties exposed in unexpected events.",
            "recommendations": "Review insurance requirements to ensure they meet the risk profile.",
            "example": "E.g., 'The contractor must maintain liability insurance with at least $1M coverage.'"
        }
    }
    risk_scores = analyze_risk(text)
    risk_keywords_context = {
        "Liability": {"keywords": ["liability", "responsible", "responsibility", "legal obligation"]},
        "Termination": {"keywords": ["termination", "breach", "contract end", "default"]},
        "Indemnification": {"keywords": ["indemnification", "indemnify", "hold harmless", "compensate", "compensation"]},
        "Payment Risk": {"keywords": ["payment", "terms", "reimbursement", "fee", "schedule", "invoice", "money"]},
        "Insurance": {"keywords": ["insurance", "coverage", "policy", "claims"]}
    }
    risk_contexts = extract_context_for_risk_terms(text, risk_keywords_context, window=1)
    detailed_info = {}
    for risk_term, score in risk_scores.items():
        if score > 0:
            info = risk_details.get(risk_term, {"description": "No details available."})
            detailed_info[risk_term] = {
                "score": score,
                "description": info.get("description", ""),
                "common_concerns": info.get("common_concerns", ""),
                "recommendations": info.get("recommendations", ""),
                "example": info.get("example", ""),
                "context_summary": risk_contexts.get(risk_term, "No context available.")
            }
    return detailed_info

def analyze_contract_clauses(text):
    """Analyzes contract clauses using the fine-tuned CUAD QA model."""
    max_length = 512
    step = 256
    clauses_detected = []
    try:
        clause_types = list(cuad_model.config.id2label.values())
    except Exception as e:
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

@app.post("/analyze_legal_document")
async def analyze_legal_document(file: UploadFile = File(...)):
    """Analyzes a legal document for clause detection and compliance risks."""
    try:
        print(f"Processing file: {file.filename}")
        content = await file.read()
        text = extract_text_from_pdf(io.BytesIO(content))
        if not text:
            return {"status": "error", "message": "No valid text found in the document."}
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Document too short for meaningful summarization."
        print("Extracting named entities...")
        entities = extract_named_entities(text)
        print("Analyzing risk...")
        risk_scores = analyze_risk(text)
        detailed_risk = get_detailed_risk_info(text)
        print("Analyzing contract clauses...")
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        return {
            "status": "success",
            "task_id": generated_task_id,
            "summary": summary,
            "named_entities": entities,
            "risk_scores": risk_scores,
            "detailed_risk": detailed_risk,
            "clauses_detected": clauses
        }
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/analyze_legal_video")
async def analyze_legal_video(file: UploadFile = File(...)):
    """Analyzes a legal video by transcribing audio and analyzing the transcript."""
    try:
        print(f"Processing video file: {file.filename}")
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        print(f"Temporary file saved at: {temp_file_path}")
        text = process_video_to_text(temp_file_path)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if not text:
            return {"status": "error", "message": "No speech could be transcribed from the video."}
        transcript_path = os.path.join("static", f"transcript_{int(time.time())}.txt")
        with open(transcript_path, "w") as f:
            f.write(text)
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Transcript too short for meaningful summarization."
        print("Extracting named entities from transcript...")
        entities = extract_named_entities(text)
        print("Analyzing risk from transcript...")
        risk_scores = analyze_risk(text)
        detailed_risk = get_detailed_risk_info(text)
        print("Analyzing legal clauses from transcript...")
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        return {
            "status": "success",
            "task_id": generated_task_id,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "named_entities": entities,
            "risk_scores": risk_scores,
            "detailed_risk": detailed_risk,
            "clauses_detected": clauses
        }
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/analyze_legal_audio")
async def analyze_legal_audio(file: UploadFile = File(...)):
    """Analyzes legal audio by transcribing and analyzing the transcript."""
    try:
        print(f"Processing audio file: {file.filename}")
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        print(f"Temporary file saved at: {temp_file_path}")
        text = process_audio_to_text(temp_file_path)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if not text:
            return {"status": "error", "message": "No speech could be transcribed from the audio."}
        transcript_path = os.path.join("static", f"transcript_{int(time.time())}.txt")
        with open(transcript_path, "w") as f:
            f.write(text)
        summary_text = text[:4096] if len(text) > 4096 else text
        summary = summarizer(summary_text, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] if len(text) > 100 else "Transcript too short for meaningful summarization."
        print("Extracting named entities from transcript...")
        entities = extract_named_entities(text)
        print("Analyzing risk from transcript...")
        risk_scores = analyze_risk(text)
        detailed_risk = get_detailed_risk_info(text)
        print("Analyzing legal clauses from transcript...")
        clauses = analyze_contract_clauses(text)
        generated_task_id = str(uuid.uuid4())
        store_document_context(generated_task_id, text)
        return {
            "status": "success",
            "task_id": generated_task_id,
            "transcript": text,
            "transcript_path": transcript_path,
            "summary": summary,
            "named_entities": entities,
            "risk_scores": risk_scores,
            "detailed_risk": detailed_risk,
            "clauses_detected": clauses
        }
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/transcript/{transcript_id}")
async def get_transcript(transcript_id: str):
    """Retrieves a previously generated transcript."""
    transcript_path = os.path.join("static", f"transcript_{transcript_id}.txt")
    if os.path.exists(transcript_path):
        return FileResponse(transcript_path)
    else:
        raise HTTPException(status_code=404, detail="Transcript not found")

@app.post("/legal_chatbot")
async def legal_chatbot_api(query: str = Form(...), task_id: str = Form(...)):
    """Handles legal Q&A using chat history and document context."""
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
    """Sets up ngrok tunnel for Google Colab."""
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

@app.get("/download_risk_chart")
async def download_risk_chart():
    """Generate and return a risk assessment chart as an image file."""
    try:
        os.makedirs("static", exist_ok=True)
        risk_scores = {
            "Liability": 11,
            "Termination": 12,
            "Indemnification": 10,
            "Payment Risk": 41,
            "Insurance": 71
        }
        plt.figure(figsize=(8, 5))
        plt.bar(risk_scores.keys(), risk_scores.values(), color='red')
        plt.xlabel("Risk Categories")
        plt.ylabel("Risk Score")
        plt.title("Legal Risk Assessment")
        plt.xticks(rotation=30)
        risk_chart_path = "static/risk_chart.png"
        plt.savefig(risk_chart_path)
        plt.close()
        return FileResponse(risk_chart_path, media_type="image/png", filename="risk_chart.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk chart: {str(e)}")

@app.get("/download_risk_pie_chart")
async def download_risk_pie_chart():
    try:
        risk_scores = {
            "Liability": 11,
            "Termination": 12,
            "Indemnification": 10,
            "Payment Risk": 41,
            "Insurance": 71
        }
        plt.figure(figsize=(6, 6))
        plt.pie(risk_scores.values(), labels=risk_scores.keys(), autopct='%1.1f%%', startangle=90)
        plt.title("Legal Risk Distribution")
        pie_chart_path = "static/risk_pie_chart.png"
        plt.savefig(pie_chart_path)
        plt.close()
        return FileResponse(pie_chart_path, media_type="image/png", filename="risk_pie_chart.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pie chart: {str(e)}")

@app.get("/download_risk_radar_chart")
async def download_risk_radar_chart():
    try:
        risk_scores = {
            "Liability": 11,
            "Termination": 12,
            "Indemnification": 10,
            "Payment Risk": 41,
            "Insurance": 71
        }
        categories = list(risk_scores.keys())
        values = list(risk_scores.values())
        categories += categories[:1]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Legal Risk Radar Chart", y=1.1)
        radar_chart_path = "static/risk_radar_chart.png"
        plt.savefig(radar_chart_path)
        plt.close()
        return FileResponse(radar_chart_path, media_type="image/png", filename="risk_radar_chart.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating radar chart: {str(e)}")

@app.get("/download_risk_trend_chart")
async def download_risk_trend_chart():
    try:
        dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
        risk_history = {
            "Liability": [10, 12, 11, 13],
            "Termination": [12, 15, 14, 13],
            "Indemnification": [9, 10, 11, 10],
            "Payment Risk": [40, 42, 41, 43],
            "Insurance": [70, 69, 71, 72]
        }
        plt.figure(figsize=(10, 6))
        for category, scores in risk_history.items():
            plt.plot(dates, scores, marker='o', label=category)
        plt.xlabel("Date")
        plt.ylabel("Risk Score")
        plt.title("Historical Legal Risk Trends")
        plt.xticks(rotation=45)
        plt.legend()
        trend_chart_path = "static/risk_trend_chart.png"
        plt.savefig(trend_chart_path, bbox_inches="tight")
        plt.close()
        return FileResponse(trend_chart_path, media_type="image/png", filename="risk_trend_chart.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating trend chart: {str(e)}")

import pandas as pd
import plotly.express as px
from fastapi.responses import HTMLResponse

@app.get("/interactive_risk_chart", response_class=HTMLResponse)
async def interactive_risk_chart():
    try:
        risk_scores = {
            "Liability": 11,
            "Termination": 12,
            "Indemnification": 10,
            "Payment Risk": 41,
            "Insurance": 71
        }
        df = pd.DataFrame({
            "Risk Category": list(risk_scores.keys()),
            "Risk Score": list(risk_scores.values())
        })
        fig = px.bar(df, x="Risk Category", y="Risk Score", title="Interactive Legal Risk Assessment")
        return fig.to_html()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating interactive chart: {str(e)}")

def run():
    """Starts the FastAPI server."""
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8500, timeout_keep_alive=600)

if __name__ == "__main__":
    public_url = setup_ngrok()
    if public_url:
        print(f"\n✅ Your API is publicly available at: {public_url}/docs\n")
    else:
        print("\n⚠️ Ngrok setup failed. API will only be available locally.\n")
    run()

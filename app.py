from fastapi import FastAPI, HTTPException, Body, Query, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from typing import List, Optional, Dict, Any
import sqlite3
import os
import uuid
import json
import requests
from datetime import datetime
import shutil
from pathlib import Path
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Document extraction
import docx
from PyPDF2 import PdfReader

# Constants
UPLOAD_DIR = "uploads"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ensure uploads directory exists
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# Check if GROQ API key is provided
if not GROQ_API_KEY:
    print("⚠️  PERINGATAN: GROQ_API_KEY tidak ditemukan di variabel lingkungan!")
    print("   Harap buat file .env dengan kunci API GROQ Anda")
    print("   Dapatkan kunci API gratis Anda di: https://console.groq.com/")

# Initialize FastAPI
app = FastAPI(
    title="Chatbot Dinas Arpus Jateng",
    description="Sistem Analisis Dokumen dan Chat untuk Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah (Akses Publik)",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    document_ids: List[str] = []
    is_predefined: bool = False

class ChatResponse(BaseModel):
    response: str
    source_documents: List[str] = []
    predefined_questions: List[str] = []

class PredefinedQuestion(BaseModel):
    question: str
    document_id: str

class SystemHealth(BaseModel):
    status: str
    groq_api: str
    database: str
    model_info: Optional[Dict[str, Any]] = None

class AdminStats(BaseModel):
    total_documents: int
    total_chats: int
    recent_activity: List[Dict[str, Any]]

# Predefined questions for documents (disesuaikan agar lebih umum)
PREDEFINED_QUESTIONS = [
    "Apa ringkasan dari dokumen ini?",
    "Kapan dokumen ini dibuat dan oleh siapa?",
    "Apa tujuan utama dokumen ini?",
    "Bagaimana relevansi dokumen ini dengan Dinas Arpus Jateng?",
    "Informasi penting apa yang ada dalam dokumen ini?",
    "Bagian mana dari dokumen ini yang membahas tentang [topik spesifik]?",
    "Bisakah Anda menemukan data atau angka penting di dokumen ini?",
    "Apakah ada rekomendasi atau kebijakan yang disebutkan dalam dokumen ini?",
    "Apa kesimpulan dari dokumen ini?"
]

# Database connection
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# Dummy current_user for admin endpoints (for demonstration purposes only!)
def get_admin_status_dummy(is_admin_query: bool = Query(False, description="Set to true to access admin features. FOR DEMO ONLY!")):
    """
    Dummy function to simulate admin access via query parameter.
    **DO NOT USE IN PRODUCTION WITHOUT PROPER AUTHENTICATION!**
    """
    if is_admin_query: # For demonstration, allow admin access if query param is set
        return {"username": "admin_demo", "role": "admin"}
    raise HTTPException(status_code=403, detail="Admin access required (set ?is_admin_query=true for demo)")


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_file(file_path):
    file_extension = file_path.split(".")[-1].lower()
    
    if file_extension == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_extension == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def query_groq(prompt, max_tokens=2000, model="llama3-8b-8192"):
    """
    Query GROQ API for AI responses
    """
    if not GROQ_API_KEY:
        return "Error: GROQ API key not configured. Please check your .env file."
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "Anda adalah asisten analisis dokumen yang membantu staf dan publik di Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah (Dinas Arpus Jateng). Berikan jawaban yang akurat, informatif, dan relevan dalam bahasa Indonesia."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: Invalid response format from GROQ API"
        elif response.status_code == 401:
            return "Error: Invalid GROQ API key. Please check your credentials."
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please try again later."
        else:
            print(f"GROQ API error: {response.status_code} {response.text}")
            return f"Error: GROQ API returned status {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to GROQ API. Please check your internet connection."
    except requests.exceptions.Timeout:
        return "Error: GROQ API request timed out. Please try again."
    except Exception as e:
        print(f"Error querying GROQ: {e}")
        return f"Error: {str(e)}"

def test_groq_connection():
    """Test GROQ API connection and return status info"""
    if not GROQ_API_KEY:
        return {"status": "error", "message": "API key not configured"}
    
    try:
        response = query_groq("Hello, this is a connection test. Respond with 'Connection successful.'", max_tokens=50)
        if "Connection successful" in response or "successful" in response.lower():
            return {
                "status": "connected", 
                "message": "GROQ API is working properly",
                "model": "llama3-8b-8192"
            }
        else:
            return {"status": "error", "message": f"Unexpected response: {response[:100]}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- API ENDPOINTS ---

@app.get("/health", response_model=SystemHealth, tags=["System"])
def health_check():
    """Check if API and dependencies are healthy"""
    health_status = {
        "status": "healthy",
        "groq_api": "disconnected",
        "database": "disconnected",
        "model_info": None
    }
    
    # Check GROQ API connection
    groq_test = test_groq_connection()
    health_status["groq_api"] = groq_test["status"]
    if groq_test["status"] == "connected":
        health_status["model_info"] = {
            "provider": "GROQ",
            "model": "llama3-8b-8192",
            "status": "operational"
        }
    
    # Check database connection
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        health_status["database"] = "connected"
    except:
        health_status["database"] = "disconnected"
    
    # Overall status
    if health_status["groq_api"] == "connected" and health_status["database"] == "connected":
        health_status["status"] = "healthy"
    else:
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/upload", tags=["Documents"])
def upload_documents(
    files: List[UploadFile] = File(...),
):
    """Upload documents for processing"""
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maksimal 5 file diizinkan")
    
    uploaded_docs = []
    for file in files:
        doc_id = str(uuid.uuid4())
        
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "docx", "doc", "txt"]:
            continue
        
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.{file_extension}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        text = extract_text_from_file(file_path)
        
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO documents (id, filename, file_path, upload_date, text_content, file_size) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, file.filename, file_path, datetime.now().isoformat(), text[:1000], len(text))
        )
        conn.commit()
        conn.close()
        
        uploaded_docs.append({
            "document_id": doc_id,
            "filename": file.filename,
            "size": len(text),
            "predefined_questions": PREDEFINED_QUESTIONS
        })
    
    return {"uploaded_documents": uploaded_docs}

@app.get("/predefined-questions/{document_id}", tags=["Chat"])
def get_predefined_questions(
    document_id: str,
):
    """Get predefined questions for a document"""
    conn = get_db_connection()
    doc = conn.execute(
        "SELECT id FROM documents WHERE id = ?", 
        (document_id,)
    ).fetchone()
    conn.close()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan")
    
    return {"questions": PREDEFINED_QUESTIONS, "document_id": document_id}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    message: ChatMessage,
):
    """Chat with documents using GROQ AI"""
    session_id = "guest_session" 
    
    document_texts = []
    document_names = []
    
    if message.document_ids:
        conn = get_db_connection()
        for doc_id in message.document_ids:
            doc = conn.execute(
                "SELECT filename, file_path FROM documents WHERE id = ?", 
                (doc_id,)
            ).fetchone()
            
            if not doc:
                continue
                
            text = extract_text_from_file(doc["file_path"])
            if text:
                document_texts.append(text)
                document_names.append(doc["filename"])
        conn.close()
    
    strict_rules = """
Anda adalah asisten AI untuk Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah (Dinas Arpus Jateng).
Tugas utama Anda adalah menganalisis dokumen arsip, perpustakaan, dan dokumen resmi lainnya serta menjawab pertanyaan yang relevan.
**ATURAN UTAMA:**
1.  **FOKUS TOPIK:** Anda **HANYA** boleh menjawab pertanyaan yang berkaitan dengan:
    a. Isi dari dokumen yang diberikan.
    b. Informasi umum yang relevan dengan konteks Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah (Dinas Arpus Jateng).
2.  **TOLAK PERTANYAAN TIDAK RELEVAN:** Jika pertanyaan pengguna berada di luar dua topik di atas (contoh: pertanyaan tentang cuaca, resep, pemrograman, atau topik umum lainnya), Anda **HARUS** menolak dengan sopan. Gunakan jawaban ini: "Maaf, saya hanya dapat merespons pertanyaan yang berkaitan dengan analisis dokumen atau informasi seputar Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah."
3.  **BAHASA:** Selalu gunakan Bahasa Indonesia yang baik dan formal.
---
"""
    
    prompt = strict_rules
    
    if document_texts:
        prompt += "Konteks dari dokumen yang disediakan:\n"
        for i, (text, name) in enumerate(zip(document_texts, document_names)):
            prompt += f"\n--- DOKUMEN {i+1}: {name} ---\n{text[:4000]}\n" 
        
        prompt += "\nBerdasarkan aturan di atas dan konteks dari dokumen yang disediakan, jawablah pertanyaan berikut.\n"
    else:
        prompt += "\nBerdasarkan aturan di atas dan pengetahuan umum Anda tentang Dinas Kearsipan dan Perpustakaan Provinsi Jawa Tengah, jawablah pertanyaan berikut. Tidak ada dokumen yang disediakan.\n"

    prompt += f"\nPertanyaan Pengguna: \"{message.message}\""

    response = query_groq(prompt, max_tokens=1500)
    
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO chat_history (session_id, message, response, timestamp, is_predefined, document_ids) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, message.message, response, datetime.now().isoformat(), message.is_predefined, json.dumps(message.document_ids))
    )
    conn.commit()
    conn.close()
    
    predefined_questions = PREDEFINED_QUESTIONS if message.document_ids and not message.is_predefined else []
    
    return {
        "response": response, 
        "source_documents": document_names,
        "predefined_questions": predefined_questions
    }

@app.get("/documents", tags=["Documents"])
def get_documents():
    """Get list of all available documents (Public access)"""
    conn = get_db_connection()
    documents = conn.execute(
        "SELECT id, filename, upload_date, text_content, file_size FROM documents ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()
    
    return {"documents": [dict(doc) for doc in documents]}

@app.get("/history", tags=["Chat"])
def get_chat_history():
    """Get recent chat history (Public access, simplified)"""
    conn = get_db_connection()
    history = conn.execute(
        "SELECT session_id, message, response, timestamp, is_predefined, document_ids FROM chat_history ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    conn.close()
    
    return {"history": [dict(item, username=item['session_id']) for item in history]}

# Endpoint /profile dan /api-info dihapus sesuai permintaan
# @app.get("/profile", tags=["User"])
# def get_user_profile():
#     """Get dummy user profile (No user login)"""
#     return {
#         "username": "Pengguna Tamu",
#         "email": "pengguna@dinasarpusjateng.go.id",
#         "role": "public_user"
#     }

# @app.get("/api-info", tags=["System"])
# def get_api_info():
#     """Get information about the AI API being used"""
#     groq_test = test_groq_connection()
#     return {
#         "provider": "GROQ",
#         "model": "llama3-8b-8192",
#         "status": groq_test["status"],
#         "features": [
#             "Kecepatan inferensi tinggi",
#             "Respons berkualitas tinggi",
#             "Dukungan Bahasa Indonesia",
#             "Kemampuan analisis dokumen"
#         ],
#         "limits": {
#             "monthly_tokens": "1,000,000 (free tier)",
#             "max_tokens_per_request": 32768,
#             "concurrent_requests": 20
#         }
#     }

# --- Admin-only endpoints ---
@app.get("/admin/users", tags=["Admin"])
def get_all_users_admin(admin_status: dict = Depends(get_admin_status_dummy)):
    """Get list of dummy users (Admin only)"""
    return []

@app.delete("/admin/users/{username}", tags=["Admin"])
def delete_user_admin(username: str, admin_status: dict = Depends(get_admin_status_dummy)):
    """Dummy endpoint for deleting a user (Admin only)"""
    raise HTTPException(status_code=400, detail="Manajemen pengguna dinonaktifkan dalam mode akses publik.")

@app.get("/admin/stats", response_model=AdminStats, tags=["Admin"])
def get_admin_stats(admin_status: dict = Depends(get_admin_status_dummy)):
    """Get system statistics (Admin only)"""
    conn = get_db_connection()
    
    total_documents = conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()["count"]
    total_chats = conn.execute("SELECT COUNT(*) as count FROM chat_history").fetchone()["count"]
    
    recent_activity = []
    recent_chats = conn.execute(
        """
        SELECT session_id, message, timestamp 
        FROM chat_history 
        ORDER BY timestamp DESC 
        LIMIT 5
        """
    ).fetchall()
    
    for chat in recent_chats:
        recent_activity.append({
            "type": "chat",
            "username": chat["session_id"],
            "description": f"Bertanya: {chat['message'][:50]}{'...' if len(chat['message']) > 50 else ''}",
            "timestamp": chat["timestamp"]
        })
    
    recent_uploads = conn.execute(
        """
        SELECT filename, upload_date 
        FROM documents 
        ORDER BY upload_date DESC 
        LIMIT 5
        """
    ).fetchall()
    
    for upload in recent_uploads:
        recent_activity.append({
            "type": "upload",
            "username": "Pengunggah", 
            "description": f"Mengunggah: {upload['filename']}",
            "timestamp": upload["upload_date"]
        })
    
    recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_activity = recent_activity[:10]
    
    conn.close()
    
    return AdminStats(
        total_documents=total_documents,
        total_chats=total_chats,
        recent_activity=recent_activity
    )

@app.get("/admin/documents", tags=["Admin"])
def get_all_documents_admin(admin_status: dict = Depends(get_admin_status_dummy)):
    """Get all documents in the system (Admin only)"""
    conn = get_db_connection()
    documents = conn.execute(
        """
        SELECT id, filename, upload_date, text_content, file_size
        FROM documents
        ORDER BY upload_date DESC
        """
    ).fetchall()
    conn.close()
    
    return {"documents": [dict(doc, username="N/A", email="N/A") for doc in documents]}

@app.delete("/admin/documents/{document_id}", tags=["Admin"])
def delete_document_admin(document_id: str, admin_status: dict = Depends(get_admin_status_dummy)):
    """Delete a document (Admin only)"""
    conn = get_db_connection()
    
    doc = conn.execute(
        "SELECT file_path FROM documents WHERE id = ?", 
        (document_id,)
    ).fetchone()
    
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan")
    
    try:
        if os.path.exists(doc["file_path"]):
            os.remove(doc["file_path"])
        
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.execute("DELETE FROM chat_history WHERE document_ids LIKE ?", (f"%\"{document_id}\"%",))
        conn.commit()
        conn.close()
        
        return {"message": "Dokumen berhasil dihapus"}
    except Exception as e:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail="Gagal menghapus dokumen")


# --- FRONTEND SERVING ---
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_index():
    return "index.html"

app.mount("/", StaticFiles(directory="."), name="static_root")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Memulai Chatbot Dinas Arpus Jateng dengan GROQ AI (Akses Publik)")
    print("📡 Dokumentasi API: http://localhost:8000/docs")
    print("🌐 Aplikasi Frontend: http://localhost:8000")
    print("⚠️  Demo akses admin: http://localhost:8000/?is_admin_query=true (hanya untuk pengujian)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
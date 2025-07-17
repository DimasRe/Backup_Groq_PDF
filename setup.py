#!/usr/bin/env python3
"""
Database setup script for Chatbot Dinas Arpus Jateng
This script creates the necessary database tables and initial setup
"""

import sqlite3
import os
from pathlib import Path

def create_database():
    """Create database and tables."""
    print("🗄️  Menyiapkan database...")

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            text_content TEXT,
            file_size INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            is_predefined BOOLEAN DEFAULT FALSE,
            document_ids TEXT
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp)')

    conn.commit()
    conn.close()
    print("✅ Tabel database berhasil dibuat.")

def create_directories():
    """Create necessary directories."""
    print("📁 Membuat direktori...")

    directories = ['uploads']

    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Direktori berhasil dibuat: {dir_name}")

def check_env_file():
    """Check if .env file exists and has required variables."""
    print("🔧 Memeriksa konfigurasi lingkungan...")

    if not os.path.exists('.env'):
        print("⚠️  PERINGATAN: File .env tidak ditemukan!")
        print("   Harap buat file .env dengan variabel berikut:")
        print("   - GROQ_API_KEY=kunci-api-groq-anda")
        print("   Dapatkan kunci API GROQ Anda di: https://console.groq.com/")
        return False

    # Check if required variables are present
    with open('.env', 'r') as f:
        env_content = f.read()

    required_vars = ['GROQ_API_KEY']
    missing_vars = []

    for var in required_vars:
        if var not in env_content or f'{var}=kunci-api-groq-anda' in env_content:
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  PERINGATAN: Variabel lingkungan yang hilang atau tidak terkonfigurasi: {', '.join(missing_vars)}")
        return False

    print("✅ Konfigurasi lingkungan terlihat bagus.")
    return True

def install_dependencies():
    """Display dependency installation instructions."""
    print("📦 Dependensi yang diperlukan:")
    print("   pip install fastapi uvicorn python-multipart")
    print("   pip install python-docx PyPDF2 requests python-dotenv")
    print("   ")
    print("   Atau instal semua sekaligus:")
    print("   pip install -r requirements.txt")

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic[email]==2.5.0
python-docx==1.1.0
PyPDF2==3.0.1
requests==2.31.0
python-dotenv==1.0.0
"""

    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())

    print("✅ Berhasil membuat requirements.txt.")

def main():
    """Main setup function to configure the Chatbot Dinas Arpus Jateng."""
    print("🚀 Chatbot Dinas Arpus Jateng - Setup (Tanpa Login Pengguna)")
    print("=" * 50)

    create_requirements_file()
    print()

    install_dependencies()
    print()

    create_directories()
    print()

    create_database()
    print()

    env_ok = check_env_file()
    print()

    print("=" * 50)
    if env_ok:
        print("✅ Setup berhasil diselesaikan!")
        print("🚀 Anda sekarang dapat menjalankan: python app.py")
    else:
        print("⚠️  Setup selesai dengan peringatan.")
        print("📝 Harap konfigurasikan file .env Anda sebelum menjalankan aplikasi.")

    print("\n📖 Langkah selanjutnya:")
    print("1. Konfigurasikan file .env Anda dengan kunci API GROQ.")
    print("2. Instal dependensi: pip install -r requirements.txt.")
    ("3. Jalankan aplikasi: python app.py.")
    ("4. Buka browser: http://localhost:8000.")
    ("   Untuk demo akses admin: http://localhost:8000/?is_admin_query=true.")


if __name__ == "__main__":
    main()
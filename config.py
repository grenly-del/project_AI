import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    """Konfigurasi utama untuk aplikasi Flask"""

    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    GEMINI_KEY = os.getenv("GEMINI_KEY")
    if not GEMINI_KEY:
        raise ValueError("API key tidak ditemukan! Pastikan GEMINI_KEY ada di file .env")

    CORS_ORIGINS = ["http://localhost:3000"]
    DEBUG = True

    @staticmethod
    def init_app(app):
        pass

import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    """Konfigurasi utama untuk aplikasi Flask"""

    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Gemini API Key (required)
    GEMINI_KEY = os.getenv("GEMINI_KEY")
    if not GEMINI_KEY:
        raise ValueError("API key tidak ditemukan! Pastikan GEMINI_KEY ada di file .env")

    # Cloudinary configuration (optional - will use base64 fallback if not provided)
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
    
    # MongoDB configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "kidneystone_db")

    # CORS configuration
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://192.168.56.1:3000"
    ).split(",")

    # Debug mode
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Flask secret key
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    @staticmethod
    def init_app(app):
        pass

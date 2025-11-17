from flask import Flask
from flask_cors import CORS
from config import Config
from pymongo import MongoClient


def create_app():
    print('Backend Berhasil di jalankan')
    app = Flask(__name__)
    app.config.from_object(Config)

    # MongoDB setup
    try:
        mongo_client = MongoClient(Config.MONGO_URI)
        app.config['MONGO_CLIENT'] = mongo_client
        app.config['MONGO_DB'] = mongo_client[Config.MONGO_DB_NAME]
        print(f"MongoDB connected to: {Config.MONGO_URI}")
    except Exception as e:
        print(f"Warning: MongoDB connection failed: {e}")
        app.config['MONGO_CLIENT'] = None
        app.config['MONGO_DB'] = None

    # CORS
    CORS(app, origins=Config.CORS_ORIGINS)

    # Register blueprints
    from app.routes.detection_routes import detection_bp
    from app.routes.auth_routes import auth_bp
    from app.routes.patient_routes import patient_bp
    from app.routes.analytics_routes import analytics_bp
    app.register_blueprint(detection_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(analytics_bp, url_prefix="/api")

    return app

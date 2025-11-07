from flask import Flask
from flask_cors import CORS
from config import Config



def create_app():
    print('test')
    app = Flask(__name__)
    app.config.from_object(Config)

    # CORS
    CORS(app, origins=Config.CORS_ORIGINS)

    # Register blueprint
    from app.routes.detection_routes import detection_bp
    app.register_blueprint(detection_bp)



    return app

from flask import Flask
from flask_cors import CORS
import os
import logging
from app.utils.model_loader import download_model
from app.routes import create_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # Créer les dossiers de travail
    for folder in ['static/upload', 'static/processed', 'models']:
        os.makedirs(folder, exist_ok=True)

    # Vérifiez si le modèle est déjà présent avant de le télécharger
    model_path = os.path.join('models', 'u2net.pth')
    if not os.path.exists(model_path):
        try:
            download_model()
        except Exception as e:
            logger.error(f"Erreur fatale lors du téléchargement du modèle: {str(e)}")
            raise

    # CORS configuration
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,https://dashboard-podologie.netlify.app,https://backend-podologie-production.up.railway.app').split(',')
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

    create_routes(app)

    return app


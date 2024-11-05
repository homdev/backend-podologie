from flask import Flask
from flask_cors import CORS
import os
import logging
from app.utils.model_loader import download_model

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    try:
        # Création des dossiers nécessaires
        for folder in ['static/upload', 'static/processed', 'models']:
            os.makedirs(folder, exist_ok=True)
        
        # Configuration
        app.config['UPLOAD_FOLDER'] = 'static/upload/'
        app.config['PROCESSED_FOLDER'] = 'static/processed/'
        
        # CORS configuration
        CORS(app, resources={r"/*": {"origins": [
            "http://localhost:3000",
            "https://dashboard-podologie.netlify.app"
        ]}})
        
        # Téléchargement du modèle
        download_model()
        
        # Import et création des routes
        from app.routes import create_routes
        create_routes(app)
        
        return app
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de l'application: {str(e)}")
        raise

# Création de l'instance de l'application
app = create_app()
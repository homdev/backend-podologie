import os
from flask import Flask, jsonify
from flask_cors import CORS
from app.routes import create_routes
from dotenv import load_dotenv
from app.utils.model_loader import download_model
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

try:
    # Création des dossiers nécessaires
    for folder in ['static/upload', 'static/processed']:
        os.makedirs(folder, exist_ok=True)

    # Configuration
    app.config['UPLOAD_FOLDER'] = 'static/upload/'
    app.config['PROCESSED_FOLDER'] = 'static/processed/'

    # CORS configuration
    CORS(app, resources={r"/*": {"origins": [
        "http://localhost:3000",
        "https://dashboard-podologie.netlify.app"
    ]}})

    # Téléchargement du modèle au démarrage
    download_model()

    # Créer les routes de l'API
    create_routes(app)

except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de l'application: {str(e)}")
    
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        debug = os.environ.get("DEBUG", "False").lower() == "true"
        app.run(host="0.0.0.0", port=port, debug=debug)
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {str(e)}")
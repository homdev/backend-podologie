import os
from flask import Flask
from flask_cors import CORS
import logging
from app.utils.model_loader import download_model
from app.routes import create_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    flask_app = Flask(__name__)

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
    CORS(flask_app, resources={r"/*": {
        "origins": ["http://localhost:3001", "https://dashboard-podologie.netlify.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }})

    create_routes(flask_app)
    return flask_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
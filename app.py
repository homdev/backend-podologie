import os
from flask import Flask
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Créer les dossiers de travail
for folder in ['static/upload', 'static/processed', 'models']:
    os.makedirs(folder, exist_ok=True)

# CORS configuration
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://dashboard-podologie.netlify.app"]}})

try:
    from app.utils.model_loader import download_model
    download_model()
except Exception as e:
    logger.error(f"Erreur fatale lors du téléchargement du modèle: {str(e)}")
    raise

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

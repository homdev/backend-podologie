import os
from flask import Flask, request
from flask_cors import CORS
import logging
from app.utils.model_loader import download_model
from app.routes import create_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


create_routes(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

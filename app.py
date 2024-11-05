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
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://dashboard-podologie.netlify.app",
            "https://backend-podologie-production.up.railway.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": False
    }
})

# Middleware pour les en-têtes CORS
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin == "https://dashboard-podologie.netlify.app":
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

create_routes(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

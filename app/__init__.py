from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import logging
from .utils.model_loader import get_project_root

logger = logging.getLogger(__name__)

def create_app():
    project_root = get_project_root()
    static_folder = os.path.join(project_root, 'static')
    
    app = Flask(__name__, static_folder=static_folder)
    
    # Optimisation des fichiers statiques
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # Cache 1 an
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite 16MB
    
    # Configuration du serveur statique optimisé
    @app.after_request
    def add_header(response):
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response

    @app.route('/static/<path:folder>/<path:filename>')
    def serve_static(folder, filename):
        response = send_from_directory(
            os.path.join(static_folder, folder), 
            filename,
            conditional=True  # Support du cache côté client
        )
        return response

    # Créer les dossiers de travail avec chemins absolus
    for folder in ['upload', 'processed']:
        path = os.path.join(static_folder, folder)
        os.makedirs(path, exist_ok=True)
        logger.info(f"Dossier créé: {path}")

    # Créer le dossier models
    models_path = os.path.join(project_root, 'models')
    os.makedirs(models_path, exist_ok=True)

    # CORS configuration
    CORS(app, resources={r"/*": {
        "origins": ["http://localhost:3001", "https://dashboard-podologie.netlify.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }})

    # Import et enregistrement des routes
    from .routes import create_routes
    create_routes(app)

    return app
from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import logging
from .utils.model_loader import get_project_root

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    project_root = get_project_root()

    # Créer les dossiers de travail avec chemins absolus
    for folder in ['static/upload', 'static/processed', 'models']:
        path = os.path.join(project_root, folder)
        os.makedirs(path, exist_ok=True)
        logger.info(f"Dossier créé: {path}")

    # Configuration des dossiers statiques
    @app.route('/static/<path:folder>/<path:filename>')
    def serve_static(folder, filename):
        static_folder = os.path.join(project_root, 'static', folder)
        return send_from_directory(static_folder, filename)

    # CORS configuration
    CORS(app, resources={r"/*": {
        "origins": ["http://localhost:3000", "https://dashboard-podologie.netlify.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }})

    # Import et enregistrement des routes
    from .routes import create_routes
    create_routes(app)

    return app
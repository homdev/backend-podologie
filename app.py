from flask import Flask
from flask_cors import CORS
from app.routes import create_routes
import os

app = Flask(__name__)

# Configuration des dossiers d'uploads et des images traitées
app.config['UPLOAD_FOLDER'] = 'static/upload/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

# Utiliser une variable d'environnement pour les origines CORS
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,https://dashboard-podologie.netlify.app/,https://backend-podologie-production.up.railway.app').split(',')

# CORS pour permettre les requêtes provenant des origines autorisées
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# Créer les routes de l'API
create_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
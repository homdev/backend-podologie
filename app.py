from flask import Flask
from flask_cors import CORS
from app.routes import create_routes

app = Flask(__name__)

# Configuration des dossiers d'uploads et des images traitées
app.config['UPLOAD_FOLDER'] = 'static/upload/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

# CORS pour permettre les requêtes provenant de 'http://localhost:3000'
CORS(app, resources={r"/*": {"origins": "https://671be00c81a5f4a93bd4891e--dashboard-podologie.netlify.app/"}})

# Créer les routes de l'API
create_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
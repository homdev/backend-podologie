import os
import gdown
import logging
import torch

logger = logging.getLogger(__name__)

def get_project_root():
    """Retourne le chemin racine du projet"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def download_model():
    """Télécharge le modèle U2NET dans le dossier 'models'."""
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'models', 'u2net.pth')
    
    # Créer le dossier 'models' s'il n'existe pas
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        logger.info(f"Téléchargement du modèle U2NET vers {model_path}...")
        url = "https://www.dropbox.com/scl/fi/o97mt8jknqnk7umggkn1k/u2net.pth?rlkey=wsnfn5tusvxm7ec1vdjac07og&dl=1"
        try:
            gdown.download(url, model_path, quiet=False)
            logger.info("Modèle téléchargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
            raise FileNotFoundError("Le modèle n'a pas pu être téléchargé")

    # Vérification de la compatibilité du modèle
    try:
        torch.load(model_path, map_location='cpu', weights_only=True)
        logger.info("Le modèle est prêt à être utilisé")
        return model_path
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise FileNotFoundError("Le fichier modèle est corrompu")
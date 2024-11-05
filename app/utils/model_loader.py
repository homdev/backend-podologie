import os
import gdown
import logging
import torch

logger = logging.getLogger(__name__)

def download_model():
    """Télécharge le modèle U2NET dans le dossier 'models' s'il n'est pas présent."""
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    model_path = os.path.join(model_dir, 'u2net.pth')
    
    # Créer le dossier 'models' s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        logger.info("Téléchargement du modèle U2NET...")
        url = "https://www.dropbox.com/scl/fi/o97mt8jknqnk7umggkn1k/u2net.pth?rlkey=wsnfn5tusvxm7ec1vdjac07og&dl=1"  # Remplacez par l'URL directe
        try:
            gdown.download(url, model_path, quiet=False)
            logger.info("Modèle téléchargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
            raise FileNotFoundError("Le modèle n'a pas pu être téléchargé. Vérifiez les permissions.")
    else:
        logger.info("Modèle U2NET déjà présent")

    # Vérification de la compatibilité du modèle
    try:
        torch.load(model_path, map_location='cpu')
        logger.info("Le modèle est prêt à être utilisé.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        raise FileNotFoundError("Le fichier modèle est corrompu ou non compatible.")
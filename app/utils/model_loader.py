import os
import gdown
import logging
import torch

logger = logging.getLogger(__name__)

def download_model():
    """Télécharge le modèle U2NET depuis Google Drive avec gestion des erreurs"""
    model_path = os.path.join(os.path.dirname(__file__), 'models/u2net.pth')
    if not os.path.exists(model_path):
        logger.info("Téléchargement du modèle U2NET...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        file_id = "11nsdLnQIR2JxxAqnclqZJ7lqINODTANK"
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, model_path, quiet=False)
            logger.info("Modèle téléchargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
            raise
    else:
        logger.info("Modèle U2NET déjà présent")
    # Vérification de la compatibilité du modèle
    try:
        torch.load(model_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        os.remove(model_path)  # Supprime le fichier si non compatible
        raise

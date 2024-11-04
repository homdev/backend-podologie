import os
import gdown
import logging

logger = logging.getLogger(__name__)

def download_model():
    """Télécharge le modèle U2NET depuis Google Drive avec gestion des erreurs"""
    model_path = "app/models/u2net.pth"
    try:
        if not os.path.exists(model_path):
            logger.info("Téléchargement du modèle U2NET...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            file_id = "1xf2gsCU2dQxKDGqw-_Z7CFeAXLHV_NfG"
            url = f"https://drive.google.com/uc?id={file_id}"
            success = gdown.download(url, model_path, quiet=False)
            if not success:
                raise Exception("Échec du téléchargement du modèle")
            logger.info("Modèle téléchargé avec succès")
        else:
            logger.info("Modèle U2NET déjà présent")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
        raise
        
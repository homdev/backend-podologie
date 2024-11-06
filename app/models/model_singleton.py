import torch
from .u2net import U2NET
import os
import logging
from app.utils.model_loader import download_model, get_project_root

logger = logging.getLogger(__name__)

class U2NetModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(U2NetModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        try:
            # Utiliser le chemin retourné par download_model
            model_path = download_model()
            logger.info(f"Chargement du modèle depuis {model_path}")

            self._model = U2NET(3, 1)
            self._model.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            )
            self._model.eval()
            logger.info("Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    @property
    def model(self):
        return self._model

    def __call__(self, x):
        with torch.no_grad():
            return self._model(x)

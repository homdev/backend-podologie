import logging
import warnings
from app import create_app

# Ignorer les avertissements de torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    application = create_app()
    app = application
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de l'application: {str(e)}")
    raise

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

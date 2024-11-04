import os
import gdown

def download_model():
    model_path = "app/models/u2net.pth"
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Corriger l'URL de Google Drive (doit être un lien de partage direct)
        file_id = "1xf2gsCU2dQxKDGqw-_Z7CFeAXLHV_NfG"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from app.models.u2net import U2NET
import os
import logging
from typing import Tuple, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessingError(Exception):
    """Exception personnalisée pour les erreurs de traitement d'image"""
    pass

def load_model():
    """Charge le modèle U2NET avec gestion des erreurs"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models/u2net.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle U2NET n'a pas été trouvé: {model_path}")
        
        u2net = U2NET(3, 1)
        u2net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        u2net.eval()
        return u2net
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise ImageProcessingError("Impossible de charger le modèle de traitement d'image")

# Charger le modèle
try:
    u2net = load_model()
except Exception as e:
    logger.error(f"Erreur fatale lors du chargement du modèle: {str(e)}")
    raise

def validate_image(image_path: str) -> None:
    """Valide le fichier image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Le fichier image n'existe pas: {image_path}")
    
    # Vérifier l'extension du fichier
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Format de fichier non supporté. Extensions valides: {valid_extensions}")
    
    # Vérifier la taille du fichier (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if os.path.getsize(image_path) > max_size:
        raise ValueError(f"Le fichier est trop volumineux (max {max_size/1024/1024}MB)")

def remove_background(image_path: str) -> np.ndarray:
    """Utilise U^2-Net pour supprimer le fond d'une image avec gestion des erreurs."""
    try:
        validate_image(image_path)
        
        # Charge l'image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ImageProcessingError(f"Impossible d'ouvrir l'image: {str(e)}")

        original_size = image.size
        
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        try:
            input_image = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                d1, *_ = u2net(input_image)
                mask = d1.squeeze().cpu().numpy()
                mask = (mask - mask.min()) / (mask.max() - mask.min())
                mask = (mask * 255).astype(np.uint8)
        except Exception as e:
            raise ImageProcessingError(f"Erreur lors du traitement de l'image par le modèle: {str(e)}")

        # Redimensionne le masque
        try:
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = cv2.bitwise_and(image_np, mask_3channel)
            result = cv2.merge([result[:,:,0], result[:,:,1], result[:,:,2], mask])
            return result
        except Exception as e:
            raise ImageProcessingError(f"Erreur lors du post-traitement de l'image: {str(e)}")
            
    except Exception as e:
        logger.error(f"Erreur dans remove_background: {str(e)}")
        raise

def crop_to_content(image):
    """Recadre l'image en gardant les dimensions originales et en centrant le pied."""
    # Utiliser le canal alpha pour trouver la zone non transparente
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    
    if coords is None:
        return image
    
    # Obtenir les dimensions originales
    original_height, original_width = image.shape[:2]
    
    # Obtenir le rectangle englobant du pied
    x, y, w, h = cv2.boundingRect(coords)
    
    # Calculer le centre du pied
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculer les nouvelles coordonnées pour centrer le pied
    new_x = max(0, center_x - original_width // 2)
    new_y = max(0, center_y - original_height // 2)
    
    # Ajuster si on dépasse les bords
    if new_x + original_width > image.shape[1]:
        new_x = image.shape[1] - original_width
    if new_y + original_height > image.shape[0]:
        new_y = image.shape[0] - original_height
    
    # Recadrer l'image en gardant les dimensions originales
    return image[new_y:new_y+original_height, new_x:new_x+original_width]    

def split_feet(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Sépare les pieds avec gestion des erreurs."""
    try:
        if image is None:
            raise ValueError("L'image d'entrée est nulle")
            
        if len(image.shape) != 3 or image.shape[2] != 4:
            raise ValueError("Format d'image incorrect: nécessite une image BGRA")

        alpha = image[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_contour_area = 1000
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        if len(valid_contours) == 0:
            logger.warning("Aucun contour valide trouvé dans l'image")
            return None, None
        
        if len(valid_contours) > 2:
            logger.warning(f"Trop de contours trouvés ({len(valid_contours)}), utilisation des 2 plus grands")
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        
        height, width = image.shape[:2]
        left_foot = np.zeros((height, width, 4), dtype=np.uint8)
        right_foot = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Tri des contours par position x
        contour_info = [(cv2.boundingRect(cnt)[0], cnt) for cnt in valid_contours]
        contour_info.sort(key=lambda x: x[0])
        
        for i, (_, cnt) in enumerate(contour_info):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (255), -1)
            
            if i == 0:
                left_foot = cv2.bitwise_and(image, image, mask=mask)
            else:
                right_foot = cv2.bitwise_and(image, image, mask=mask)
        
        return crop_to_content(left_foot), crop_to_content(right_foot)
        
    except Exception as e:
        logger.error(f"Erreur dans split_feet: {str(e)}")
        raise ImageProcessingError(f"Erreur lors de la séparation des pieds: {str(e)}")

def save_image(image: np.ndarray, path: str) -> None:
    """Sauvegarde l'image avec gestion des erreurs."""
    try:
        if image is None:
            raise ValueError("L'image à sauvegarder est nulle")
            
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        success = cv2.imwrite(path, image)
        if not success:
            raise ImageProcessingError("Échec de la sauvegarde de l'image")
            
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'image {path}: {str(e)}")
        raise ImageProcessingError(f"Impossible de sauvegarder l'image: {str(e)}")
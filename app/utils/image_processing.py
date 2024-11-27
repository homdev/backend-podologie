import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from app.utils.model_loader import download_model, get_project_root
from app.models.model_singleton import U2NetModel
from app.models.u2net import U2NET
import os
import logging
from typing import Tuple, Optional
import time
from functools import wraps


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessingError(Exception):
    """Exception personnalisée pour les erreurs de traitement d'image"""
    pass

# Initialisation du modèle comme variable globale
u2net = U2NetModel()

def load_model():
    """Charge le modèle U2NET avec gestion des erreurs et téléchargement si nécessaire"""
    try:
        model_path = os.path.join(get_project_root(), 'models', 'u2net.pth')
        if not os.path.exists(model_path):
            download_model()
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise ImageProcessingError("Impossible de charger le modèle de traitement d'image")

try:
    u2net = load_model()
except Exception as e:
    logger.error(f"Erreur fatale lors du chargement du modèle: {str(e)}")
    raise

def validate_image(image_path: str) -> None:
    """Valide le fichier image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Le fichier image n'existe pas: {image_path}")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Format de fichier non supporté: {file_ext}")
    if os.path.getsize(image_path) > 10 * 1024 * 1024:
        raise ValueError("Le fichier est trop volumineux (max 10MB)")

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Affiner le masque pour réduire les contours noirs et les impuretés restantes."""
    # Seuil adaptatif pour capturer les zones d'ombre subtiles
    _, refined_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Appliquer un flou gaussien léger pour adoucir les contours
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)

    # Effectuer des opérations morphologiques pour combler les trous et lisser les bords
    kernel = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return refined_mask


def correct_color_cast(image: Image.Image) -> Image.Image:
    # Corrige la teinte en ajustant les couleurs pour un rendu plus naturel
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.5)  # Ajustez le niveau selon le besoin

def refine_and_clean_mask(mask: np.ndarray) -> np.ndarray:
    # Affine le masque pour réduire les contours noirs et les impuretés restantes
    _, refined_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask

def enhance_mask_edges(mask: np.ndarray) -> np.ndarray:
    """Améliore les contours du masque pour un détourage plus précis."""
    edges = cv2.Canny(mask, 100, 200)

    # Dilatation pour renforcer les bords
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Fusionner les bords avec le masque d'origine
    mask = cv2.add(mask, edges)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    return mask



def adaptive_contrast_brightness(image: Image.Image) -> Image.Image:
    gray_image = image.convert("L")
    mean_brightness = np.mean(np.array(gray_image))
    contrast_factor = 1.5 if mean_brightness < 100 else 1.2
    brightness_factor = 1.3 if mean_brightness < 100 else 1.1
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return ImageEnhance.Brightness(image).enhance(brightness_factor)


def clean_transparency(image: np.ndarray) -> np.ndarray:
    """Nettoie la transparence en éliminant les petites impuretés autour du sujet."""
    alpha = image[:, :, 3]
    
    # Utilisation du seuil d'Otsu pour distinguer le sujet du fond
    _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Filtrage morphologique pour éliminer les impuretés
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

    image[:, :, 3] = alpha
    
    return image


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} a pris {end - start:.2f} secondes")
        return result
    return wrapper

def remove_background(image_path: str) -> np.ndarray:
    logger.info("=== Début suppression du fond ===")
    try:
        # 1. Chargement de l'image
        original_image = Image.open(image_path)
        print("1. Format d'origine:", original_image.mode)
        
        # 2. Conversion pour le modèle (sans modifier l'original)
        model_input = original_image.convert('RGB')
        
        # 3. Préparation du tensor
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()
        ])
        input_tensor = transform(model_input).unsqueeze(0)
        
        # 4. Génération du masque
        with torch.no_grad():
            d1 = u2net(input_tensor)[0]
            mask = d1.squeeze().cpu().numpy()
        
        # 5. Traitement du masque
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (original_image.size[0], original_image.size[1]))
        
        # 6. Création de l'image finale en préservant les couleurs exactes
        result = np.array(original_image)
        if result.shape[2] == 3:
            result = np.dstack((result, np.ones(result.shape[:2], dtype=np.uint8) * 255))
            
        # Application du masque uniquement sur le canal alpha
        result[:, :, 3] = mask
        
        # 7. Conversion finale
        # Assurons-nous que les couleurs RGB restent intactes
        result_image = Image.fromarray(result)
        result_array = np.array(result_image)
        
        print("Vérification finale - Shape:", result_array.shape)
        print("Vérification finale - Type:", result_array.dtype)
        print("Vérification finale - Valeurs RGB min/max:", 
              result_array[:,:,:3].min(), result_array[:,:,:3].max())
        
        logger.info(f"Dimensions après suppression du fond: {result_array.shape}")
        return result_array

    except Exception as e:
        logger.error(f"Erreur dans remove_background: {str(e)}")
        raise ImageProcessingError(f"Erreur lors de la suppression du fond: {str(e)}")

def crop_to_content(image):
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return image
    original_height, original_width = image.shape[:2]
    x, y, w, h = cv2.boundingRect(coords)
    center_x = x + w // 2
    center_y = y + h // 2
    new_x = max(0, center_x - original_width // 2)
    new_y = max(0, center_y - original_height // 2)
    return image[new_y:new_y+original_height, new_x:new_x+original_width]


def clean_foot_image(foot_image: np.ndarray) -> np.ndarray:
    """Nettoie l'image du pied en supprimant le bruit et en améliorant les bords."""
    if foot_image is None:
        return None
    kernel = np.ones((3,3), np.uint8)
    alpha = foot_image[:, :, 3]
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    foot_image[:, :, 3] = alpha
    return foot_image

def find_separation_valley(alpha_channel: np.ndarray) -> int:
    """
    Trouve la vallée de séparation optimale entre les pieds.
    """
    # Création de l'histogramme vertical
    histogram = np.sum(alpha_channel, axis=0)
    
    # Lissage de l'histogramme avec cv2.GaussianBlur au lieu de filters.gaussian
    # Conversion de l'histogramme en format compatible avec cv2.GaussianBlur
    histogram_2d = histogram.reshape(-1, 1)
    # Application du flou gaussien
    histogram_smooth_2d = cv2.GaussianBlur(histogram_2d.astype(np.float32), (1, 11), 5)
    # Reconversion en 1D
    histogram_smooth = histogram_smooth_2d.reshape(-1)
    
    # Trouver tous les minimums locaux
    valleys = []
    window = len(histogram_smooth) // 4  # Fenêtre de recherche adaptative
    
    for i in range(window, len(histogram_smooth) - window):
        if histogram_smooth[i] == min(histogram_smooth[i-window:i+window]):
            valleys.append((i, histogram_smooth[i]))
    
    # Si aucune vallée n'est trouvée, retourner le centre
    if not valleys:
        return len(histogram_smooth) // 2
        
    # Trier les vallées par profondeur (valeur minimale)
    valleys.sort(key=lambda x: x[1])
    
    # Prendre la vallée la plus profonde proche du centre
    center = len(histogram_smooth) // 2
    center_valleys = [(pos, val) for pos, val in valleys 
                     if abs(pos - center) < window]
    
    if center_valleys:
        return center_valleys[0][0]
    return valleys[0][0]

def calculate_foot_masses(alpha_channel: np.ndarray, split_point: int) -> Tuple[float, float]:
    """
    Calcule les centres de masse des deux pieds.
    """
    left_mass = np.sum(alpha_channel[:, :split_point])
    right_mass = np.sum(alpha_channel[:, split_point:])
    return left_mass, right_mass

def create_adaptive_mask(height: int, width: int, split_point: int, 
                        transition_width: int = 30) -> np.ndarray:
    """
    Crée un masque progressif pour la transition entre les pieds.
    """
    mask = np.zeros((height, width))
    
    # Création d'une transition progressive
    for i in range(transition_width):
        pos = split_point - transition_width//2 + i
        if 0 <= pos < width:
            value = i / transition_width
            mask[:, pos] = value
            
    # Remplissage des zones pleines
    mask[:, :split_point-transition_width//2] = 0
    mask[:, split_point+transition_width//2:] = 1
    
    return mask

def verify_a3_dimensions(image: np.ndarray) -> Tuple[float, float]:
    """
    Vérifie et calcule les facteurs de conversion pour une image A3.
    """
    height, width = image.shape[:2]
    
    # Dimensions A3 en cm
    A3_WIDTH_CM = 29.7
    A3_HEIGHT_CM = 42.0
    
    # Calcul des facteurs de conversion
    px_to_cm_x = A3_WIDTH_CM / width
    px_to_cm_y = A3_HEIGHT_CM / height
    
    logger.info(f"Facteurs de conversion A3: {px_to_cm_x:.4f}cm/px x {px_to_cm_y:.4f}cm/px")
    
    return px_to_cm_x, px_to_cm_y

def split_feet_improved(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sépare l'image en deux parties avec une détection intelligente de la ligne de séparation.
    """
    try:
        if image is None:
            raise ValueError("L'image d'entrée est nulle")

        height, width = image.shape[:2]
        alpha_channel = image[:, :, 3]

        # 1. Amélioration de la détection de la vallée
        split_point = find_separation_valley(alpha_channel)
        
        # 2. Calcul plus précis des masses avec une marge de sécurité
        margin = width // 10  # 10% de la largeur
        left_mass = np.sum(alpha_channel[:, :split_point-margin])
        right_mass = np.sum(alpha_channel[:, split_point+margin:])
        total_mass = left_mass + right_mass
        
        if total_mass == 0:
            logger.warning("Aucun contenu détecté dans l'image")
            return None, None
            
        mass_ratio = left_mass / total_mass
        
        # Ajustement plus flexible de la séparation
        if not (0.30 < mass_ratio < 0.70):  # Seuils plus stricts
            logger.warning(f"Distribution déséquilibrée détectée ({mass_ratio:.2f}), ajustement...")
            # Recherche itérative d'un meilleur point de séparation
            best_split = width // 2
            best_ratio = abs(0.5 - mass_ratio)
            
            for test_point in range(width//3, 2*width//3, width//50):
                left = np.sum(alpha_channel[:, :test_point])
                right = np.sum(alpha_channel[:, test_point:])
                test_ratio = left / (left + right)
                if abs(0.5 - test_ratio) < best_ratio:
                    best_ratio = abs(0.5 - test_ratio)
                    best_split = test_point
            
            split_point = best_split

        # 3. Transition progressive améliorée
        transition_width = min(80, width // 8)  # Transition plus large
        mask = create_adaptive_mask(height, width, split_point, transition_width)
        
        # 4. Création des images avec transition douce
        left_foot = image.copy()
        right_foot = image.copy()
        
        # Application du masque avec lissage gaussien
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
        
        for i in range(4):
            left_foot[:, :, i] = left_foot[:, :, i] * (1 - mask_smooth)
            right_foot[:, :, i] = right_foot[:, :, i] * mask_smooth
        
        # 5. Nettoyage amélioré
        left_foot = clean_foot_image(left_foot)
        right_foot = clean_foot_image(right_foot)
        
        # 6. Recadrage intelligent avec marges
        left_foot = crop_to_content_with_margin(left_foot)
        right_foot = crop_to_content_with_margin(right_foot)
        
        # 7. Validation finale plus stricte
        min_content = 500  # Seuil minimum de contenu
        if left_foot is not None and np.sum(left_foot[:, :, 3]) < min_content:
            left_foot = None
        if right_foot is not None and np.sum(right_foot[:, :, 3]) < min_content:
            right_foot = None
            
        return left_foot, right_foot

    except Exception as e:
        logger.error(f"Erreur lors de la séparation des pieds: {str(e)}")
        return None, None

def crop_to_content_with_margin(image: np.ndarray, margin_percent: float = 0.1) -> Optional[np.ndarray]:
    """
    Recadre l'image avec une marge de sécurité.
    """
    if image is None:
        return None
        
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    
    if coords is None:
        return None
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Ajout d'une marge proportionnelle
    margin_x = int(w * margin_percent)
    margin_y = int(h * margin_percent)
    
    height, width = image.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    
    return image[y1:y2, x1:x2]

def optimize_image_for_web(image: np.ndarray) -> np.ndarray:
    """Optimise l'image pour le web tout en préservant la qualité."""
    MAX_DIM = 1500
    height, width = image.shape[:2]
    
    # Redimensionnement intelligent
    if max(height, width) > MAX_DIM:
        scale = MAX_DIM / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Optimisation de la transparence
    alpha = image[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    image[:, :, 3] = alpha
    
    return image

def save_image(image, path, dpi=None):
    """
    Sauvegarde une image avec gestion des erreurs et logging.
    :param image: L'image à sauvegarder, sous forme de tableau numpy ou objet PIL.
    :param path: Chemin de sauvegarde de l'image.
    :param dpi: Résolution de l'image en DPI.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 4:  # RGBA
                pil_image = Image.fromarray(image, 'RGBA')
            else:  # RGB
                pil_image = Image.fromarray(image, 'RGB')
        elif isinstance(image, Image.Image):  # Image PIL
            pil_image = image
        else:
            raise ValueError("Format d'image non pris en charge")
        
        pil_image.save(path, 'PNG', optimize=True, dpi=dpi)
        logger.info(f"Image sauvegardée avec succès : {path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'image {path} : {str(e)}")
        raise ImageProcessingError(f"Erreur lors de la sauvegarde de l'image : {str(e)}")
    
def isolate_foot(image: np.ndarray, side: str = 'right') -> np.ndarray:
    """
    Version améliorée avec focus sur la détection des orteils
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    # 1. Prétraitement spécifique pour les orteils
    alpha = image[:, :, 3]
    
    # Amélioration du contraste pour mieux détecter les orteils
    alpha_enhanced = cv2.equalizeHist(alpha)
    
    # 2. Double seuillage adaptatif pour les zones claires et sombres
    block_size = 11
    C = 2
    adaptive_thresh = cv2.adaptiveThreshold(
        alpha_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )
    
    # 3. Détection spécifique des orteils
    top_region = alpha_enhanced[:height//3, :]  # Focus sur la région des orteils
    _, top_thresh = cv2.threshold(
        top_region,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 4. Fusion des masques
    binary = np.zeros_like(alpha)
    binary[:height//3, :] = top_thresh
    binary[height//3:, :] = adaptive_thresh[height//3:, :]
    
    # 5. Détection et amélioration des contours
    edges = cv2.Canny(binary, 30, 150)  # Seuils plus sensibles
    kernel_edge = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel_edge, iterations=1)
    
    # 6. Analyse des composantes avec paramètres optimisés pour les orteils
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, 
        connectivity=8  # Connectivité 8 pour mieux capturer les diagonales
    )
    
    # 7. Sélection des composantes avec critères adaptés aux orteils
    min_size = int(0.001 * height * width)  # Seuil plus bas pour les petits orteils
    valid_components = []
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            center_x = centroids[i][0]
            is_right = center_x > width / 2
            
            # Analyse de la forme pour identifier les orteils
            component_mask = (labels == i).astype(np.uint8) * 255
            aspect_ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]
            
            # Les orteils ont généralement un ratio largeur/hauteur spécifique
            is_toe = aspect_ratio < 2.0 and stats[i, cv2.CC_STAT_TOP] < height // 2
            
            valid_components.append({
                'label': i,
                'is_right': is_right,
                'center_x': center_x,
                'stats': stats[i],
                'is_toe': is_toe
            })
    
    # 8. Création du masque final avec traitement spécial pour les orteils
    mask = np.zeros_like(alpha)
    for comp in valid_components:
        if (side == 'right' and comp['is_right']) or (side == 'left' and not comp['is_right']):
            comp_mask = (labels == comp['label']).astype(np.uint8) * 255
            
            # Traitement spécial pour les orteils
            if comp['is_toe']:
                # Dilatation plus importante pour les orteils
                kernel_toe = np.ones((3,3), np.uint8)
                comp_mask = cv2.dilate(comp_mask, kernel_toe, iterations=2)
            
            mask = cv2.add(mask, comp_mask)
    
    # 9. Nettoyage final avec préservation des détails
    kernel_small = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # 10. Lissage des contours
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Application du masque final
    result[:, :, 3] = mask
    
    # Recadrage optimisé
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        result = result[y:y+h, x:x+w]
    
    return result

def verify_dimensions(image: np.ndarray, original_width: float, original_height: float, dpi: int = 120):
    """Vérifie que l'image a les mêmes dimensions en centimètres que les dimensions d'origine."""
    width_pixels, height_pixels = image.shape[1], image.shape[0]
    
    # Convertir les dimensions en pixels à cm pour l'impression
    width_cm = width_pixels / dpi * 2.54
    height_cm = height_pixels / dpi * 2.54
    
    print(f"Dimensions calculées : {width_cm:.2f} cm x {height_cm:.2f} cm")
    print(f"Dimensions attendues : {original_width} cm x {original_height} cm")
    
    if abs(width_cm - original_width) < 0.1 and abs(height_cm - original_height) < 0.1:
        print("Les dimensions sont correctes.")
    else:
        print("Les dimensions ne correspondent pas aux attentes.")

def create_a4_image_with_scale(image, output_path, dpi_x, dpi_y):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Trouver la zone non transparente réelle du pied
        if image.mode == 'RGBA':
            bbox = Image.fromarray(np.array(image)[:,:,3]).getbbox()
            if bbox:
                image = image.crop(bbox)
        
        # Dimensions originales après crop
        original_width_px, original_height_px = image.size
        logger.info(f"Dimensions image entrée après crop: {original_width_px}x{original_height_px}")
        
        # Dimensions A4 en pixels
        a4_width_px = int((210 / 25.4) * dpi_x)  # 210mm en pixels
        a4_height_px = int((297 / 25.4) * dpi_y)  # 297mm en pixels
        
        # Création du canvas A4 blanc
        a4_image = Image.new("RGBA", (a4_width_px, a4_height_px), (0, 0, 0, 0))
        
        # Calcul du centrage
        x_offset = (a4_width_px - original_width_px) // 2
        y_offset = (a4_height_px - original_height_px) // 2
        
        # Ajustement vertical pour placer le pied un peu plus haut sur la page
        # y_offset = int(y_offset * 0.4)  # Décalage vers le haut de 20%
        
        # Collage avec masque alpha pour la transparence
        a4_image.paste(image, (x_offset, y_offset), image if image.mode == 'RGBA' else None)
        
        # Sauvegarde avec les DPI originaux
        a4_image.save(output_path, "PNG", dpi=(dpi_x, dpi_y))
        logger.info(f"Image A4 générée et centrée : {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du fichier A4 : {e}")
        raise ImageProcessingError(f"Erreur lors de la création du fichier A4 : {e}")

def process_and_save_image(image_path: str, output_path: str, side: str = 'right'):
    # Image originale
    original = np.array(Image.open(image_path))
    logger.info("Dimensions de l'image originale :")
    verify_dimensions(original, 21.0, 29.7)  # Dimensions A4 en cm
    
    # Après remove_background et nettoyage
    processed = remove_background(image_path)
    cleaned = clean_transparency(processed)
    
    # Isolation du pied spécifié
    isolated_foot = isolate_foot(cleaned, side)
    logger.info("Dimensions après isolation du pied :")
    verify_dimensions(isolated_foot, 21.0, 29.7)  # Dimensions A4 en cm
    
    # Utiliser create_a4_image_with_scale au lieu de place_on_a4_canvas
    pil_image = Image.fromarray(isolated_foot)
    create_a4_image_with_scale(pil_image, output_path, 120, 120)  # DPI fixé à 120

def verify_image_dimensions(image: np.ndarray, format_type: str, dpi_x: float, dpi_y: float) -> dict:
    """
    Vérifie et retourne les dimensions réelles de l'image.
    
    Args:
        image: Image numpy
        format_type: "A3" ou "A4"
        dpi_x: DPI horizontal
        dpi_y: DPI vertical
    
    Returns:
        dict: Dimensions en cm et en pixels
    """
    height, width = image.shape[:2]
    
    # Dimensions théoriques en mm
    if format_type == "A3":
        real_width_mm = 297
        real_height_mm = 420
    else:  # A4
        real_width_mm = 210
        real_height_mm = 297
    
    # Conversion pixels vers cm
    width_cm = (width / dpi_x) * 25.4 / 10
    height_cm = (height / dpi_y) * 25.4 / 10
    
    logger.info(f"=== Vérification dimensions {format_type} ===")
    logger.info(f"Dimensions théoriques: {real_width_mm/10:.1f}x{real_height_mm/10:.1f} cm")
    logger.info(f"Dimensions calculées: {width_cm:.1f}x{height_cm:.1f} cm")
    
    return {
        "format": format_type,
        "width_px": width,
        "height_px": height,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "theoretical_width_cm": real_width_mm/10,
        "theoretical_height_cm": real_height_mm/10
    }

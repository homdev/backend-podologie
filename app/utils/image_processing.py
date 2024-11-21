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
from skimage import filters
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
    
    # Lissage de l'histogramme pour réduire le bruit
    histogram_smooth = filters.gaussian(histogram, sigma=5)
    
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

def split_feet_improved(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sépare l'image en deux parties avec une détection intelligente de la ligne de séparation.
    """
    try:
        if image is None:
            raise ValueError("L'image d'entrée est nulle")

        height, width = image.shape[:2]
        alpha_channel = image[:, :, 3]

        # 1. Trouver le point de séparation optimal
        split_point = find_separation_valley(alpha_channel)
        
        # 2. Vérifier la distribution des masses
        left_mass, right_mass = calculate_foot_masses(alpha_channel, split_point)
        mass_ratio = left_mass / (left_mass + right_mass)
        
        # Ajuster le point de séparation si la distribution est trop déséquilibrée
        if not (0.3 < mass_ratio < 0.7):
            logger.warning("Distribution déséquilibrée détectée, ajustement...")
            split_point = width // 2
        
        # 3. Créer un masque de transition progressif
        transition_width = min(60, width // 10)  # Adaptatif à la largeur de l'image
        mask = create_adaptive_mask(height, width, split_point, transition_width)
        
        # 4. Créer les images des pieds avec la transition
        left_foot = image.copy()
        right_foot = image.copy()
        
        # Appliquer les masques avec transition progressive
        for i in range(4):  # Pour tous les canaux (RGBA)
            left_foot[:, :, i] = left_foot[:, :, i] * (1 - mask)
            right_foot[:, :, i] = right_foot[:, :, i] * mask
        
        # 5. Nettoyer et recadrer chaque pied
        left_foot = clean_foot_image(left_foot)
        right_foot = clean_foot_image(right_foot)
        
        # 6. Recadrage intelligent
        left_foot = crop_to_content(left_foot)
        right_foot = crop_to_content(right_foot)
        
        # 7. Validation finale
        if left_foot is not None and np.sum(left_foot[:, :, 3]) < 100:
            left_foot = None
        if right_foot is not None and np.sum(right_foot[:, :, 3]) < 100:
            right_foot = None
            
        return left_foot, right_foot

    except Exception as e:
        logger.error(f"Erreur lors de la séparation des pieds: {str(e)}")
        return None, None

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

def save_image(image: np.ndarray, path: str) -> None:
    try:
        start_time = time.time()
        
        # Debug: vérifier les valeurs avant traitement
        print("Save_image - Valeurs RGB avant:", image[:,:,:3].mean(axis=(0,1)))
        
        # Optimisation avant sauvegarde
        image = optimize_image_for_web(image)
        
        # Conversion directe en PIL sans passer par cv2
        image_pil = Image.fromarray(image)  # Suppression de cv2.cvtColor
        print("Save_image - Mode PIL:", image_pil.mode)
        
        # Debug: vérifier les valeurs après conversion
        image_array = np.array(image_pil)
        print("Save_image - Valeurs RGB après:", image_array[:,:,:3].mean(axis=(0,1)))
        
        # Paramètres optimisés pour PNG
        image_pil.save(
            path, 
            'PNG',
            optimize=True,
            quality=90,
            compress_level=3
        )
        
        logger.info(f"Image sauvegardée ({os.path.getsize(path)/1024:.1f}KB) en {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde: {str(e)}")
        raise
    
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

def place_on_a4_canvas(image: np.ndarray) -> Image.Image:
    """Place l'image sur un canevas A4 en conservant l'échelle réelle"""
    # Dimensions A4 en pixels à 120 DPI
    A4_DPI = 120
    A4_WIDTH_PX = int(210 * A4_DPI / 25.4)  # 210mm en pixels
    A4_HEIGHT_PX = int(297 * A4_DPI / 25.4)  # 297mm en pixels
    
    # Création du canevas A4
    a4_canvas = Image.new("RGBA", (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255, 0))
    
    # Conversion de l'image d'entrée en Image PIL
    foot_img = Image.fromarray(image)
    
    # Calcul de la taille cible pour un pied standard (environ 25cm de longueur)
    TARGET_FOOT_LENGTH_CM = 27.0  # Longueur standard d'un pied
    target_height_px = int(TARGET_FOOT_LENGTH_CM * A4_DPI / 2.54)
    
    # Calcul du ratio pour redimensionner l'image
    current_height = foot_img.height
    scale_factor = target_height_px / current_height
    
    # Redimensionnement de l'image en conservant les proportions
    new_width = int(foot_img.width * scale_factor)
    new_height = target_height_px
    foot_img = foot_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Centrage de l'image sur le canevas
    x_offset = (A4_WIDTH_PX - new_width) // 2
    y_offset = (A4_HEIGHT_PX - new_height) // 2
    
    # Collage de l'image sur le canevas
    a4_canvas.paste(foot_img, (x_offset, y_offset), foot_img)
    
    # Ajout des métadonnées DPI
    a4_canvas.info['dpi'] = (A4_DPI, A4_DPI)
    
    return a4_canvas

        
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
    
    # Place le pied sur le canevas A4 en conservant l'échelle 1:1
    a4_canvas = place_on_a4_canvas(isolated_foot)
    
    # Vérification finale des dimensions
    logger.info("Dimensions finales sur le canevas A4 :")
    verify_dimensions(np.array(a4_canvas), 21.0, 29.7)
    
    # Sauvegarde finale avec DPI correct
    pil_image = Image.fromarray(np.array(a4_canvas))
    pil_image.save(output_path, 'PNG', dpi=(120, 120))


import cv2
import numpy as np

def load_image(image_path):
    """Charge une image à partir d'un chemin."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    return image

def split_image(image):
    """Divise l'image en deux parties (gauche et droite)."""
    height, width = image.shape[:2]
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]
    return left_half, right_half

def preprocess_image_with_morphology(image):
    """Prétraite l'image avec un seuillage manuel et fermeture morphologique."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed

def detect_and_crop_foot(image, thresh, margin=50, side='left'):
    """Détecte et recadre le pied avec ajustement dynamique."""
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
    if len(filtered_contours) == 0:
        raise ValueError("Aucun contour pertinent détecté.")
    contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    x = max(x - margin * 1.5, 0) if side == 'left' else max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, image.shape[1] - x)
    h = min(h + 2 * margin, image.shape[0] - y)
    cropped_foot = image[y:y+h, x:x+w]
    return cropped_foot

def combine_feet_on_a4_final_optimized(left_foot, right_foot, a4_width=2480, a4_height=3508):
    """Combine les pieds sur une feuille A4."""
    left_foot_resized = cv2.resize(left_foot, (a4_width // 2, a4_height))
    right_foot_resized = cv2.resize(right_foot, (a4_width // 2, a4_height))
    a4_image = np.zeros((a4_height, a4_width, 3), dtype=np.uint8)
    a4_image[:, :a4_width // 2] = left_foot_resized
    a4_image[:, a4_width // 2:] = right_foot_resized
    return a4_image

def process_foot_scan_with_visualization(image_path, output_path):
    """Pipeline complet avec visualisation des étapes et enregistrement de l'image finale."""
    image = load_image(image_path)
    left_half, right_half = split_image(image)
    edges_left = preprocess_image_with_morphology(left_half)
    edges_right = preprocess_image_with_morphology(right_half)
    left_foot = detect_and_crop_foot(left_half, edges_left, margin=80, side='left')
    right_foot = detect_and_crop_foot(right_half, edges_right, margin=50, side='right')
    a4_image = combine_feet_on_a4_final_optimized(left_foot, right_foot)
    cv2.imwrite(output_path, a4_image)
    print(f"Image finale enregistrée à {output_path}")
    return left_foot, right_foot

def process_single_foot(image_path, side, output_path):
    """Traite un seul pied (gauche ou droit) et l'enregistre au format demi A4."""
    image = load_image(image_path)
    left_half, right_half = split_image(image)
    
    if side == 'left':
        foot_half = left_half
        margin = 80
    else:
        foot_half = right_half
        margin = 50
    
    edges = preprocess_image_with_morphology(foot_half)
    cropped_foot = detect_and_crop_foot(foot_half, edges, margin=margin, side=side)
    
    # Redimensionner au format demi A4 (1240x3508)
    resized_foot = cv2.resize(cropped_foot, (1240, 3508))
    
    cv2.imwrite(output_path, resized_foot)
    print(f"Image du pied {side} enregistrée à {output_path}")
    return resized_foot
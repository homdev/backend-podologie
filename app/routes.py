from flask import request, jsonify, current_app
from app.utils import (
    remove_background, 
    split_feet_improved, 
    save_image, 
    process_and_save_image, 
    create_a4_image_with_scale, 
    clean_transparency,
    ImageProcessingError
)
from app.utils.model_loader import get_project_root
import os
from werkzeug.utils import secure_filename
import logging
from typing import Tuple
from datetime import datetime
import time
from PIL import Image
import numpy as np
import cv2
from app.utils.visualization import visualize_processing_steps

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file(file) -> Tuple[bool, str]:
    if not file or '.' not in file.filename:
        return False, "Fichier invalide ou sans extension"
    if file.filename.rsplit('.', 1)[1].lower() not in {'png', 'jpg', 'jpeg', 'bmp'}:
        return False, "Extension non autorisée"
    if len(file.read()) > 10 * 1024 * 1024:
        file.seek(0)
        return False, "Le fichier dépasse la taille maximale"
    file.seek(0)
    return True, ""

def clean_old_files(folder):
    """Nettoie les anciens fichiers du dossier"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de {file_path}: {e}")

def measure_foot_length(foot_image: np.ndarray, dpi_x: float, dpi_y: float, format_type: str = "A3") -> dict:
    """
    Mesure la longueur réelle du pied en centimètres.
    """
    # Conversion en masque binaire pour isoler le pied
    if foot_image.shape[-1] == 4:  # Image RGBA
        mask = foot_image[:, :, 3] > 0
    else:  # Image RGB
        gray = cv2.cvtColor(foot_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    mask = mask.astype(np.uint8)
    
    # Trouver les contours du pied
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"length_cm": 0, "width_cm": 0}
    
    # Prendre le plus grand contour (le pied)
    foot_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(foot_contour)
    
    # Dimensions de l'image complète
    total_height, total_width = foot_image.shape[:2]
    
    # Calcul des facteurs de conversion selon le format
    if format_type == "A3":
        # A3: 420mm x 297mm
        px_to_cm_y = 42.0 / total_height  # hauteur A3 en cm
        px_to_cm_x = 29.7 / total_width   # largeur A3 en cm
    else:
        # A4: 297mm x 210mm
        px_to_cm_y = 29.7 / total_height
        px_to_cm_x = 21.0 / total_width
    
    # Calcul des dimensions réelles du pied
    length_cm = h * px_to_cm_y
    width_cm = w * px_to_cm_x
    
    logger.info(f"Mesures du pied - Format {format_type}:")
    logger.info(f"Longueur: {length_cm:.1f} cm, Largeur: {width_cm:.1f} cm")
    
    return {
        "length_cm": round(length_cm, 1),
        "width_cm": round(width_cm, 1),
        "length_px": h,
        "width_px": w
    }

def create_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload_file():
        total_start = time.time()
        logger.info("=== Début du traitement de la requête upload ===")
        
        try:
            # Dossiers - Correction des chemins
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            upload_folder = os.path.join(project_root, 'static', 'upload')
            processed_folder = os.path.join(project_root, 'static', 'processed')
            
            # Vérification des dossiers
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(processed_folder, exist_ok=True)
            logger.info(f"Dossiers de travail: {upload_folder}, {processed_folder}")

            # Validation du fichier
            if 'file' not in request.files:
                return jsonify({'error': 'Aucun fichier envoyé'}), 400
            file = request.files['file']
            is_valid, error_message = validate_file(file)
            if not is_valid:
                return jsonify({'error': error_message}), 400

            # Noms de fichiers standardisés
            original_path = os.path.join(upload_folder, "original_foot_scan.png")
            processed_path = os.path.join(processed_folder, "processed_foot_scan.png")

            # Nettoyage des anciens fichiers
            clean_old_files(upload_folder)
            clean_old_files(processed_folder)
            
            # Sauvegarde initiale
            file.save(original_path)
            logger.info(f"Image originale sauvegardée: {original_path}")

            # Calcul des DPI basé sur la résolution de l'image A3
            with Image.open(original_path) as img:
                width, height = img.size
                logger.info(f"=== Informations image originale ===")
                logger.info(f"Dimensions en pixels : {width}x{height}")
                
                # Dimensions A3 en mm
                A3_WIDTH_MM = 297
                A3_HEIGHT_MM = 420
                MM_TO_INCHES = 25.4
                
                # Calcul des DPI pour A3
                dpi_x = width / (A3_WIDTH_MM / MM_TO_INCHES)
                dpi_y = height / (A3_HEIGHT_MM / MM_TO_INCHES)
                
                # Facteur de conversion A3 vers A4
                A4_WIDTH_MM = 210
                A4_HEIGHT_MM = 297
                scale_factor = A4_WIDTH_MM / A3_WIDTH_MM
                
                logger.info(f"=== Calculs détaillés ===")
                logger.info(f"Format d'origine: A3 ({A3_WIDTH_MM}x{A3_HEIGHT_MM} mm)")
                logger.info(f"DPI effectifs A3: {dpi_x:.2f}x{dpi_y:.2f} px/in")
                logger.info(f"Facteur d'échelle A3->A4: {scale_factor:.3f}")

            # Traitement de l'image
            start_time = time.time()
            processed_image = remove_background(original_path)
            logger.info(f"Suppression du fond: {time.time() - start_time:.2f} secondes")

            # Nettoyage de la transparence
            cleaned_image = clean_transparency(processed_image)

            # Sauvegarde avec les DPI calculés
            save_image(processed_image, processed_path, dpi=(dpi_x, dpi_y))
            
            # Création de l'image A4 avec les DPI ajustés
            pil_image = Image.fromarray(processed_image)
            a4_path = os.path.join(processed_folder, "foot_scan_a4.png")
            create_a4_image_with_scale(pil_image, a4_path, dpi_x, dpi_y)

            # Séparation des pieds
            left_foot_path = os.path.join(processed_folder, "left_foot_scan.png")
            right_foot_path = os.path.join(processed_folder, "right_foot_scan.png")
            left_foot_a4_path = os.path.join(processed_folder, "left_foot_scan_a4.png")
            right_foot_a4_path = os.path.join(processed_folder, "right_foot_scan_a4.png")

            left_foot, right_foot = split_feet_improved(processed_image)
            
            # Sauvegarde des pieds séparés avec les DPI calculés
            save_image(left_foot, left_foot_path, dpi=(dpi_x, dpi_y))
            save_image(right_foot, right_foot_path, dpi=(dpi_x, dpi_y))

            # Création des versions A4 pour chaque pied
            left_foot_pil = Image.fromarray(left_foot)
            right_foot_pil = Image.fromarray(right_foot)
            
            create_a4_image_with_scale(left_foot_pil, left_foot_a4_path, dpi_x, dpi_y)
            create_a4_image_with_scale(right_foot_pil, right_foot_a4_path, dpi_x, dpi_y)

            # Mesure des pieds sur l'image originale A3
            original_image = np.array(Image.open(original_path))
            original_left_foot, original_right_foot = split_feet_improved(cleaned_image)

            # Calcul des mesures pour l'image originale (format A3)
            original_measures = {
                'left': measure_foot_length(original_left_foot, dpi_x, dpi_y, format_type="A3"),
                'right': measure_foot_length(original_right_foot, dpi_x, dpi_y, format_type="A3")
            }

            logger.info("=== Mesures des pieds originaux (A3) ===")
            logger.info(f"Pied gauche: {original_measures['left']['length_cm']}cm x {original_measures['left']['width_cm']}cm")
            logger.info(f"Pied droit: {original_measures['right']['length_cm']}cm x {original_measures['right']['width_cm']}cm")

            # Dictionnaire pour stocker chaque étape
            processing_steps = {}
            
            # 1. Image originale
            original_image = np.array(Image.open(original_path))
            
            # 2. Suppression du fond
            processed_image = remove_background(original_path)
            processing_steps["Suppression du fond"] = processed_image
            
            # 3. Nettoyage de la transparence
            cleaned_image = clean_transparency(processed_image)
            processing_steps["Nettoyage transparence"] = cleaned_image
            
            # 4. Séparation des pieds
            left_foot, right_foot = split_feet_improved(cleaned_image)
            processing_steps["Pied gauche"] = left_foot
            processing_steps["Pied droit"] = right_foot
            
            # 5. Mesures
            left_foot_measures = measure_foot_length(left_foot, dpi_x, dpi_y)
            right_foot_measures = measure_foot_length(right_foot, dpi_x, dpi_y)
            
            measures = {
                'left': left_foot_measures,
                'right': right_foot_measures
            }
            
            # Mesures pour les pieds séparés (format A4)
            left_foot_measures = measure_foot_length(left_foot, dpi_x, dpi_y, format_type="A4")
            right_foot_measures = measure_foot_length(right_foot, dpi_x, dpi_y, format_type="A4")
            
            # Mise à jour du dictionnaire processing_steps avec les mesures appropriées
            processing_steps = {
                "Suppression du fond": processed_image,
                "Nettoyage transparence": cleaned_image,
                "Pied gauche": left_foot,
                "Pied droit": right_foot
            }
            
            # Création de la visualisation avec les mesures
            visualization_path = os.path.join(processed_folder, "processing_steps.png")
            visualize_processing_steps(
                original_image,
                processing_steps,
                original_measures,  # Utilisation des mesures originales pour toutes les étapes
                visualization_path
            )
            
            # Réponse avec les URLs
            base_url = request.host_url.rstrip('/')
            response = {
                'success': True,
                'original_image': f"{base_url}/static/upload/original_foot_scan.png",
                'processed_image': f"{base_url}/static/processed/processed_foot_scan.png",
                'a4_image': f"{base_url}/static/processed/foot_scan_a4.png",
                'left_foot': f"{base_url}/static/processed/left_foot_scan.png",
                'right_foot': f"{base_url}/static/processed/right_foot_scan.png",
                'left_foot_a4': f"{base_url}/static/processed/left_foot_scan_a4.png",
                'right_foot_a4': f"{base_url}/static/processed/right_foot_scan_a4.png",
                'measurements': {
                    'left_foot': left_foot_measures,
                    'right_foot': right_foot_measures
                },
                'visualization': f"{base_url}/static/processed/processing_steps.png"
            }
            
            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Erreur critique dans upload_file: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/health')
    def health_check():
        model_path = "models/u2net.pth"
        return jsonify({
            'status': 'healthy',
            'model_loaded': os.path.exists(model_path),
            'version': '1.0.0'
        })

    @app.route('/test-static')
    def test_static():
        """Route de test pour vérifier l'accès aux fichiers statiques"""
        project_root = get_project_root()
        static_folders = {
            'upload': os.path.join(project_root, 'static', 'upload'),
            'processed': os.path.join(project_root, 'static', 'processed')
        }
        
        files = {
            folder: os.listdir(path) 
            for folder, path in static_folders.items()
        }
        
        return jsonify({
            'static_folders': static_folders,
            'files': files,
            'root_path': project_root
        })

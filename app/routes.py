from flask import request, jsonify, current_app
from app.utils import (
    remove_background, 
    split_feet_improved, 
    save_image, 
    process_and_save_image, 
    create_a4_image_with_scale, 
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

            # Sauvegarde initiale
            file.save(original_path)
            logger.info(f"Image originale sauvegardée: {original_path}")

            # Calcul des DPI basé sur la résolution de l'image A3
            with Image.open(original_path) as img:
                width, height = img.size
                logger.info(f"=== Informations image originale ===")
                logger.info(f"Dimensions en pixels : {width}x{height}")
                
                # Dimensions physiques en mm
                A3_WIDTH_MM = 297
                A3_HEIGHT_MM = 420
                MM_TO_INCHES = 25.4
                
                # Calcul des dimensions en pouces
                A3_WIDTH_INCHES = A3_WIDTH_MM / MM_TO_INCHES
                A3_HEIGHT_INCHES = A3_HEIGHT_MM / MM_TO_INCHES
                
                # Calcul des DPI réels pour A3
                dpi_x = width / A3_WIDTH_INCHES  # pixels par pouce
                dpi_y = height / A3_HEIGHT_INCHES
                
                # Dimensions A4 en mm et pouces
                A4_WIDTH_MM = 210
                A4_HEIGHT_MM = 297
                A4_WIDTH_INCHES = A4_WIDTH_MM / MM_TO_INCHES
                A4_HEIGHT_INCHES = A4_HEIGHT_MM / MM_TO_INCHES
                
                # Calcul des dimensions A4 en pixels pour conserver la même densité
                a4_width_px = int(A4_WIDTH_INCHES * dpi_x)
                a4_height_px = int(A4_HEIGHT_INCHES * dpi_y)
                
                logger.info(f"=== Calculs détaillés ===")
                logger.info(f"A3 dimensions physiques : {A3_WIDTH_MM:.1f}x{A3_HEIGHT_MM:.1f} mm")
                logger.info(f"A3 dimensions pouces : {A3_WIDTH_INCHES:.2f}x{A3_HEIGHT_INCHES:.2f} in")
                logger.info(f"A3 dimensions pixels : {width}x{height} px")
                logger.info(f"DPI effectifs : {dpi_x:.2f}x{dpi_y:.2f} px/in")
                logger.info(f"A4 dimensions physiques : {A4_WIDTH_MM:.1f}x{A4_HEIGHT_MM:.1f} mm")
                logger.info(f"A4 dimensions pouces : {A4_WIDTH_INCHES:.2f}x{A4_HEIGHT_INCHES:.2f} in")
                logger.info(f"A4 dimensions pixels : {a4_width_px}x{a4_height_px} px")
                logger.info(f"Ratio A4/A3 : {A4_WIDTH_MM/A3_WIDTH_MM:.3f}")
                logger.info(f"Taille physique d'un pixel : {25.4/dpi_x:.3f}x{25.4/dpi_y:.3f} mm")

            # Traitement de l'image
            start_time = time.time()
            processed_image = remove_background(original_path)
            logger.info(f"Suppression du fond: {time.time() - start_time:.2f} secondes")

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

            # Réponse avec les URLs
            base_url = request.host_url.rstrip('/')
            return jsonify({
                'success': True,
                'original_image': f"{base_url}/static/upload/original_foot_scan.png",
                'processed_image': f"{base_url}/static/processed/processed_foot_scan.png",
                'a4_image': f"{base_url}/static/processed/foot_scan_a4.png",
                'left_foot': f"{base_url}/static/processed/left_foot_scan.png",
                'right_foot': f"{base_url}/static/processed/right_foot_scan.png",
                'left_foot_a4': f"{base_url}/static/processed/left_foot_scan_a4.png",
                'right_foot_a4': f"{base_url}/static/processed/right_foot_scan_a4.png"
            }), 200

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

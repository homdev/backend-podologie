from flask import request, jsonify, current_app
from app.utils import remove_background, split_feet_improved, save_image, ImageProcessingError
from app.utils.model_loader import get_project_root
import os
from werkzeug.utils import secure_filename
import logging
from typing import Tuple
from datetime import datetime
from multiprocessing import Pool
import time

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
            # Initialisation des chemins
            start_init = time.time()
            project_root = get_project_root()
            upload_folder = os.path.join(project_root, 'static', 'upload')
            processed_folder = os.path.join(project_root, 'static', 'processed')
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(processed_folder, exist_ok=True)
            logger.info(f"Initialisation des dossiers: {time.time() - start_init:.2f} secondes")

            # Validation du fichier
            start_validation = time.time()
            if 'file' not in request.files:
                return jsonify({'error': 'Aucun fichier envoyé'}), 400
            file = request.files['file']
            is_valid, error_message = validate_file(file)
            if not is_valid:
                return jsonify({'error': error_message}), 400
            logger.info(f"Validation du fichier: {time.time() - start_validation:.2f} secondes")

            # Génération des noms de fichiers
            start_naming = time.time()
            timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            original_filename = secure_filename(file.filename)
            base_filename = f"{timestamp}_{os.path.splitext(original_filename)[0]}"
            
            original_path = os.path.join(upload_folder, f"{base_filename}.png")
            processed_path = os.path.join(processed_folder, f"processed_{base_filename}.png")
            left_foot_path = os.path.join(processed_folder, f"left_{base_filename}.png")
            right_foot_path = os.path.join(processed_folder, f"right_{base_filename}.png")
            logger.info(f"Génération des noms de fichiers: {time.time() - start_naming:.2f} secondes")

            # Sauvegarde du fichier original
            start_save = time.time()
            file.save(original_path)
            logger.info(f"Sauvegarde du fichier original: {time.time() - start_save:.2f} secondes")

            # Traitement de l'image
            start_processing = time.time()
            processed_image = remove_background(original_path)
            logger.info(f"Suppression du fond: {time.time() - start_processing:.2f} secondes")

            # Sauvegarde de l'image traitée
            start_save_processed = time.time()
            save_image(processed_image, processed_path)
            logger.info(f"Sauvegarde de l'image traitée: {time.time() - start_save_processed:.2f} secondes")

            # Séparation et sauvegarde des pieds
            start_split = time.time()
            left_foot, right_foot = split_feet_improved(processed_image)
            if left_foot is not None:
                save_image(left_foot, left_foot_path)
            if right_foot is not None:
                save_image(right_foot, right_foot_path)
            logger.info(f"Séparation et sauvegarde des pieds: {time.time() - start_split:.2f} secondes")

            # Préparation de la réponse
            start_response = time.time()
            base_url = request.host_url.rstrip('/')
            response_data = {
                'success': True,
                'original_image': f'{base_url}/static/upload/{base_filename}.png',
                'processed_image': f'{base_url}/static/processed/processed_{base_filename}.png',
                'left_foot': f'{base_url}/static/processed/left_{base_filename}.png',
                'right_foot': f'{base_url}/static/processed/right_{base_filename}.png'
            }
            logger.info(f"Préparation de la réponse: {time.time() - start_response:.2f} secondes")
            
            total_time = time.time() - total_start
            logger.info(f"=== Temps total de traitement de la requête: {total_time:.2f} secondes ===")
            
            return jsonify(response_data), 200

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
from flask import request, jsonify
from app.utils import remove_background, split_feet, save_image, ImageProcessingError
import os
from werkzeug.utils import secure_filename
import logging
from typing import Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file(file) -> Tuple[bool, str]:
    """Valide le fichier uploadé"""
    # Vérifier l'extension
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    if '.' not in file.filename:
        return False, "Le fichier n'a pas d'extension"
    
    if file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return False, f"Extension non autorisée. Extensions permises: {ALLOWED_EXTENSIONS}"
    
    # Vérifier la taille (max 10MB)
    if len(file.read()) > 10 * 1024 * 1024:
        file.seek(0)  # Réinitialiser le curseur du fichier
        return False, "Le fichier est trop volumineux (max 10MB)"
    
    file.seek(0)  # Réinitialiser le curseur du fichier
    return True, ""

def create_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            # Vérification de base du fichier
            if 'file' not in request.files:
                return jsonify({
                    'error': 'Aucun fichier n\'a été envoyé',
                    'details': 'Le champ "file" est requis dans la requête'
                }), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'error': 'Aucun fichier sélectionné',
                    'details': 'Un nom de fichier est requis'
                }), 400

            # Validation approfondie du fichier
            is_valid, error_message = validate_file(file)
            if not is_valid:
                return jsonify({
                    'error': 'Fichier invalide',
                    'details': error_message
                }), 400

            # Création des dossiers
            upload_folder = os.path.join('static', 'upload')
            processed_folder = os.path.join('static', 'processed')
            
            try:
                os.makedirs(upload_folder, exist_ok=True)
                os.makedirs(processed_folder, exist_ok=True)
            except Exception as e:
                logger.error(f"Erreur lors de la création des dossiers: {str(e)}")
                return jsonify({
                    'error': 'Erreur serveur',
                    'details': 'Impossible de créer les dossiers de stockage'
                }), 500

            # Sauvegarde sécurisée du fichier
            filename = secure_filename(file.filename)
            original_path = os.path.join(upload_folder, filename)
            
            try:
                file.save(original_path)
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du fichier original: {str(e)}")
                return jsonify({
                    'error': 'Erreur de sauvegarde',
                    'details': 'Impossible de sauvegarder le fichier original'
                }), 500

            # Traitement de l'image
            try:
                processed_image = remove_background(original_path)
                left_foot, right_foot = split_feet(processed_image)

                if left_foot is None or right_foot is None:
                    return jsonify({
                        'error': 'Erreur de traitement',
                        'details': 'Impossible de détecter les pieds dans l\'image'
                    }), 422

                # Noms des fichiers traités
                processed_filename = 'processed_' + filename
                left_foot_filename = 'left_' + filename
                right_foot_filename = 'right_' + filename

                # Chemins des fichiers traités
                processed_path = os.path.join(processed_folder, processed_filename)
                left_foot_path = os.path.join(processed_folder, left_foot_filename)
                right_foot_path = os.path.join(processed_folder, right_foot_filename)

                # Sauvegarde des images traitées
                save_image(processed_image, processed_path)
                save_image(left_foot, left_foot_path)
                save_image(right_foot, right_foot_path)

                # Construction des URLs
                base_url = request.host_url.rstrip('/')
                return jsonify({
                    'success': True,
                    'original_image': f'{base_url}/static/upload/{filename}',
                    'processed_image': f'{base_url}/static/processed/{processed_filename}',
                    'left_foot': f'{base_url}/static/processed/{left_foot_filename}',
                    'right_foot': f'{base_url}/static/processed/{right_foot_filename}'
                }), 200

            except ImageProcessingError as e:
                logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
                return jsonify({
                    'error': 'Erreur de traitement',
                    'details': str(e)
                }), 422

            except Exception as e:
                logger.error(f"Erreur inattendue: {str(e)}")
                return jsonify({
                    'error': 'Erreur serveur',
                    'details': 'Une erreur inattendue est survenue'
                }), 500

        except Exception as e:
            logger.error(f"Erreur critique: {str(e)}")
            return jsonify({
                'error': 'Erreur serveur critique',
                'details': 'Une erreur critique est survenue'
            }), 500
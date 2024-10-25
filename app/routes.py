import os
import cv2
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app.utils import process_foot_scan_with_visualization, process_single_foot
import uuid

def create_routes(app):

    @app.route('/upload', methods=['POST'])
    def upload_image():
        """Route pour uploader une image et la traiter."""
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id + "_" + filename)
        file.save(filepath)

        output_filename = file_id + "_processed.jpg"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        left_foot, right_foot = process_foot_scan_with_visualization(filepath, output_path)

        # Sauvegarder les pieds individuels
        left_output = os.path.join(app.config['PROCESSED_FOLDER'], file_id + "_left.jpg")
        right_output = os.path.join(app.config['PROCESSED_FOLDER'], file_id + "_right.jpg")
        cv2.imwrite(left_output, left_foot)
        cv2.imwrite(right_output, right_foot)

        return jsonify({
            "message": "Image traitée avec succès",
            "processed_image": f"http://192.168.1.174:5000/processed/{output_filename}",
            "left_foot": f"http://192.168.1.174:5000/processed/{file_id}_left.jpg",
            "right_foot": f"http://192.168.1.174:5000/processed/{file_id}_right.jpg"
        })

    @app.route('/process_single_foot/<side>', methods=['POST'])
    def process_single_foot_route(side):
        """Route pour traiter un seul pied (gauche ou droit)."""
        if side not in ['left', 'right']:
            return jsonify({"error": "Invalid side. Use 'left' or 'right'."}), 400

        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id + "_" + filename)
        file.save(filepath)

        output_filename = file_id + f"_{side}_processed.jpg"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        process_single_foot(filepath, side, output_path)

        return jsonify({
            "message": f"{side.capitalize()} foot processed successfully",
            "processed_image": f"http://192.168.1.174:5000/processed/{output_filename}"
        })

    @app.route('/processed/<filename>')
    def download_image(filename):
        """Route pour télécharger l'image traitée."""
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
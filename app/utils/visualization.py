import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def visualize_processing_steps(original_image, processed_steps, measures, output_path):
    """
    Visualise toutes les étapes avec mesures détaillées.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    
    # Configuration des sous-plots
    steps = len(processed_steps) + 1
    cols = 3
    rows = (steps + cols - 1) // cols
    
    def add_measurement_overlay(ax, image, title, format_type="A3", measures=None, show_combined=False):
        """Ajoute les mesures sur l'image"""
        height, width = image.shape[:2]
        
        # Dimensions réelles en cm
        if format_type == "A3":
            page_width_cm = 29.7
            page_height_cm = 42.0
            scale_text = "Format A3"
        else:  # A4
            page_width_cm = 21.0
            page_height_cm = 29.7
            scale_text = "Format A4"
        
        ax.imshow(image)
        ax.set_title(title)
        
        # Texte avec les dimensions
        text = f"{scale_text}\n{page_width_cm:.1f}x{page_height_cm:.1f} cm"
        
        if show_combined and measures:
            text += f"\nPied gauche: {measures['left']['length_cm']:.1f} x {measures['left']['width_cm']:.1f} cm"
            text += f"\nPied droit: {measures['right']['length_cm']:.1f} x {measures['right']['width_cm']:.1f} cm"
        elif measures:
            if isinstance(measures, dict) and 'length_cm' in measures:
                text += f"\nLongueur: {measures['length_cm']:.1f} cm"
                text += f"\nLargeur: {measures['width_cm']:.1f} cm"
        
        if format_type == "A4":
            text += "\n(échelle réduite)"
        
        ax.text(0.02, 0.98, text,
                transform=ax.transAxes,
                color='white',
                bbox=dict(facecolor='black', alpha=0.7),
                verticalalignment='top')
        ax.axis('off')
    
    # 1. Image originale A3 avec mesures combinées
    ax1 = plt.subplot(rows, cols, 1)
    add_measurement_overlay(ax1, original_image, "1. Image originale (A3)", 
                          format_type="A3", measures=measures, show_combined=True)
    
    # Étapes de traitement avec mesures appropriées
    for idx, (step_name, step_image) in enumerate(processed_steps.items(), 2):
        ax = plt.subplot(rows, cols, idx)
        format_type = "A4" if "pied" in step_name.lower() else "A3"
        
        # Déterminer les mesures à afficher
        step_measures = None
        if "Suppression du fond" in step_name or "Nettoyage transparence" in step_name:
            # Montrer les mesures combinées pour les étapes intermédiaires
            add_measurement_overlay(ax, step_image, f"{idx}. {step_name}", 
                                 format_type, measures, show_combined=True)
        else:
            # Pour les pieds individuels
            foot_side = 'left' if 'gauche' in step_name.lower() else 'right'
            if foot_side in measures:
                step_measures = measures[foot_side]
                add_measurement_overlay(ax, step_image, f"{idx}. {step_name}", 
                                     format_type, step_measures)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Model path
MODEL_PATH = os.path.join('app', 'model', 'best.pt')

# Global model instance (lazy loaded)
_model = None

def get_model():
    """Load YOLO model lazily"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
    return _model

def detect_image(image_path, output_dir):
    """
    Detect kidney stones in an image using YOLO model.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output image
        
    Returns:
        tuple: (result_image_path, confidence, prediction, model_version)
    """
    try:
        model = get_model()
        
        # Run inference
        results = model(image_path)
        
        # Get the first result (assuming single image)
        result = results[0]
        
        # Get predictions
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy() if len(boxes) > 0 else []
        classes = boxes.cls.cpu().numpy() if len(boxes) > 0 else []
        
        # Calculate average confidence if detections exist
        if len(confidences) > 0:
            confidence = float(np.mean(confidences))
            # Determine prediction based on detections
            if len(confidences) > 0:
                prediction = "Kidney Stone Detected"
            else:
                prediction = "No Kidney Stone Detected"
        else:
            confidence = 0.0
            prediction = "No Kidney Stone Detected"
        
        # Save annotated image
        annotated_img = result.plot()
        output_filename = f"detected_{os.path.basename(image_path)}"
        result_image_path = os.path.join(output_dir, output_filename)
        
        # Save image
        cv2.imwrite(result_image_path, annotated_img)
        
        # Get model version/info
        model_version = "YOLOv8"  # Default version
        
        return result_image_path, confidence, prediction, model_version
        
    except Exception as e:
        raise Exception(f"Error during YOLO detection: {str(e)}")

def generate_gradcam(image_path, output_dir):
    """
    Generate Grad-CAM visualization for the image.
    Note: This is a simplified version. Full Grad-CAM requires more complex implementation.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output image
        
    Returns:
        str: Path to generated Grad-CAM image, or None if failed
    """
    try:
        # For now, return None as Grad-CAM requires more complex implementation
        # This can be implemented later with proper gradient computation
        # For a basic implementation, we can create a simple heatmap
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Create a simple heatmap visualization (placeholder)
        # In a real implementation, this would compute gradients from the model
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Blend with original
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save
        output_filename = f"gradcam_{os.path.basename(image_path)}"
        gradcam_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(gradcam_path, overlay)
        
        return gradcam_path
        
    except Exception as e:
        print(f"[Grad-CAM Error] {e}")
        return None

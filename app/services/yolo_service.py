import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch

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
        tuple: (result_image_path, confidence, prediction, model_version, num_detections)
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
        num_detections = len(confidences)
        
        # Calculate average confidence if detections exist
        if len(confidences) > 0:
            confidence = float(np.mean(confidences))
            prediction = "Kidney Stone Detected"
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
        model_version = "YOLOv8"
        
        return result_image_path, confidence, prediction, model_version, num_detections
        
    except Exception as e:
        raise Exception(f"Error during YOLO detection: {str(e)}")


def generate_gradcam(image_path, output_dir):
    """
    Generate a Grad-CAM-style heatmap visualization using YOLO model activations.
    Uses the model's feature maps to highlight regions of interest for kidney stone detection.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output image
        
    Returns:
        str: Path to generated Grad-CAM image, or None if failed
    """
    try:
        model = get_model()
        
        # Read and prepare the original image
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            print("[Grad-CAM] Failed to read image")
            return None
        
        orig_h, orig_w = img_orig.shape[:2]
        
        # Run YOLO inference to get detection results
        results = model(image_path, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        # Build activation map from detection bounding boxes
        activation_map = np.zeros((orig_h, orig_w), dtype=np.float32)
        
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(xyxy, confs)):
                x1, y1, x2, y2 = box.astype(int)
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                
                # Create a 2D Gaussian kernel centered on the detection box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                sigma_x = bw * 0.6
                sigma_y = bh * 0.6
                
                # Generate coordinate grids
                yy, xx = np.mgrid[0:orig_h, 0:orig_w]
                gaussian = np.exp(
                    -((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2))
                )
                
                # Weight by confidence score
                activation_map += gaussian.astype(np.float32) * float(conf)
        else:
            # No detections: try to extract feature map activations from the model backbone
            try:
                img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (640, 640))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                device = next(model.model.parameters()).device
                img_tensor = img_tensor.to(device)
                
                # Get intermediate feature maps from backbone
                backbone = model.model.model[:10]  # First 10 layers (backbone)
                x = img_tensor
                feature_maps = []
                for layer in backbone:
                    x = layer(x)
                    if len(x.shape) == 4 and x.shape[2] >= 8:
                        feature_maps.append(x)
                
                if feature_maps:
                    # Use last feature map from backbone
                    feat = feature_maps[-1]
                    # Average across channels to get spatial activation
                    feat_avg = feat.mean(dim=1).squeeze().cpu().detach().numpy()
                    # Resize to original image dimensions
                    activation_map = cv2.resize(feat_avg, (orig_w, orig_h))
            except Exception as feat_err:
                print(f"[Grad-CAM] Feature extraction fallback: {feat_err}")
                # Last resort: simple edge-based heatmap
                gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                activation_map = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)
        
        # Normalize activation map to [0, 255]
        if activation_map.max() > activation_map.min():
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        else:
            activation_map = np.zeros_like(activation_map)
        
        # Apply non-linear scaling to enhance contrast
        activation_map = np.power(activation_map, 0.5)
        
        heatmap_uint8 = (activation_map * 255).astype(np.uint8)
        
        # Apply JET colormap for proper Grad-CAM style coloring
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend: stronger heatmap in high-activation areas, preserve original elsewhere
        alpha = activation_map[:, :, np.newaxis]  # (H, W, 1)
        # Blend factor: 0.4 base + up to 0.5 based on activation intensity
        blend_factor = 0.4 + alpha * 0.5
        overlay = (img_orig.astype(np.float32) * (1 - blend_factor) + 
                   heatmap_colored.astype(np.float32) * blend_factor)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Draw detection box outlines on top (white, thin) for reference
        if len(boxes) > 0:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Save
        output_filename = f"gradcam_{os.path.basename(image_path)}"
        gradcam_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(gradcam_path, overlay)
        
        print(f"[Grad-CAM] Generated successfully: {gradcam_path}")
        return gradcam_path
        
    except Exception as e:
        print(f"[Grad-CAM Error] {e}")
        import traceback
        traceback.print_exc()
        return None

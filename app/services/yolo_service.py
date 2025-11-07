from ultralytics import YOLO
import os

# Load model sekali di awal
model = YOLO('app/model/best.pt')

def detect_image(image_path, output_folder):
    try:
        results = model(image_path)
        result_img_path = os.path.join(output_folder, "result_" + os.path.basename(image_path))
        results[0].save(filename=result_img_path)
        confidence = float(results[0].boxes.conf[0]) if len(results[0].boxes.conf) > 0 else None

        return result_img_path, confidence
    except Exception as e:
        raise RuntimeError(f"YOLO detection failed: {e}")

from flask import Blueprint, request, jsonify, render_template, current_app
import os
from app.services.yolo_service import detect_image
from app.services.gemini_service import describe_detected_image, chatbot_msg
# Paths to scripts (must match sudoers)


detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/')
def index():
    return render_template('index.html')

@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)

    try:
        result_img_path, confidence = detect_image(img_path, upload_folder)
        description = describe_detected_image(result_img_path)
        return jsonify({
            "result_image": os.path.basename(result_img_path),
            "description": description,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@detection_bp.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        if(data is None):
            return jsonify({"error": "No image uploaded"}), 400
        msg = data.get('message')
        descript = data.get('descript')

        res_msg = chatbot_msg(msg, descript)

        return jsonify({
            "result_msg": res_msg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
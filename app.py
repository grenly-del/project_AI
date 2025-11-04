from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import torch
from PIL import Image
import os
from flask_cors import CORS
app = Flask(__name__)

# Folder upload
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLO
model_path = 'model/best.pt'  # ubah sesuai lokasi file kamu
model = YOLO(model_path)

CORS(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Simpan file yang diupload
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Deteksi dengan YOLO
    results = model(img_path)

    # Simpan hasil deteksi ke file baru
    result_img_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
    results[0].save(filename=result_img_path)

    # Kembalikan hasil ke browser
    return send_file(result_img_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv
import google.generativeai as genai
import os
from flask_cors import CORS

# === Setup dasar Flask ===
app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Load environment (.env) ===
load_dotenv()
key_gemini = os.getenv("GEMINI_KEY")
if not key_gemini:
    raise ValueError("API key tidak ditemukan! Pastikan GEMINI_KEY ada di file .env")

# === Konfigurasi Gemini ===
genai.configure(api_key=key_gemini)

# === Load YOLO model ===
model = YOLO('model/best.pt')


# === Fungsi: Deskripsi gambar menggunakan Gemini ===
def describe_detected_image(image_path):
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
        Gambar berikut adalah hasil deteksi batu ginjal menggunakan model YOLO.
        Analisis dan jelaskan gambar ini secara rinci dengan fokus pada hal-hal berikut:

        1. Gambarkan secara singkat apa yang tampak pada gambar, termasuk posisi atau area di mana batu ginjal terdeteksi (misalnya kiri, kanan, tengah, atas, bawah).
        2. Sebutkan jumlah area yang dilingkari (bounding boxes) yang menunjukkan lokasi batu ginjal.
        3. Deskripsikan ukuran atau kepadatan relatif area batu ginjal berdasarkan visualisasi hasil deteksi (misalnya kecil, sedang, besar).
        4. Berdasarkan penampilan visual (tanpa menyatakan diagnosis medis), berikan **tingkat keparahan atau risiko visual** menggunakan salah satu kategori berikut:
        - **Normal:** Tidak ada indikasi batu ginjal yang jelas.
        - **Mild:** Area batu ginjal sangat kecil, tampak ringan.
        - **Moderate:** Area cukup jelas dan tampak berpotensi menimbulkan ketidaknyamanan.
        - **Severe:** Area batu ginjal besar dan terlihat padat, berisiko tinggi.
        - **Critical:** Batu ginjal tampak dominan di sebagian besar area gambar, kemungkinan kondisi serius.

        5. Akhiri deskripsi dengan kalimat umum tentang pentingnya pemeriksaan lanjutan oleh tenaga medis profesional.

        Gunakan bahasa Indonesia yang singkat, profesional, dan mudah dipahami.
        """

        response = model_gemini.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "Tidak ada deskripsi yang dihasilkan."

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Gagal menghasilkan deskripsi dengan Gemini."


# === Route: Halaman utama ===
@app.route('/')
def index():
    return render_template('index.html')


# === Route: Proses deteksi YOLO + deskripsi Gemini ===
@app.route('/detect', methods=['POST'])
def detect():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Simpan gambar upload
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Jalankan deteksi YOLO
    try:
        results = model(img_path)
        result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "result_" + file.filename)
        results[0].save(filename=result_img_path)
    except Exception as e:
        return jsonify({"error": f"YOLO detection failed: {str(e)}"}), 500

    # Ambil deskripsi dari Gemini
    description = describe_detected_image(result_img_path)

    # Ambil nilai confidence (jika tersedia)
    try:
        confidence = float(results[0].boxes.conf[0]) if len(results[0].boxes.conf) > 0 else None
    except Exception:
        confidence = None

    # Kembalikan hasil dalam format JSON
    return jsonify({
        "result_image": "result_" + file.filename,
        "description": description,
        "confidence": confidence
    })


# === Jalankan aplikasi ===
if __name__ == '__main__':
    app.run(debug=True)

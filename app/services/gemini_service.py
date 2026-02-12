import google.generativeai as genai
from config import Config

genai.configure(api_key=Config.GEMINI_KEY)

def describe_detected_image(image_path, confidence=0.0, prediction="Unknown", num_detections=0):
    """
    Generate a detailed medical description of a kidney stone detection result using Gemini AI.
    
    Args:
        image_path: Path to the annotated detection image
        confidence: Detection confidence score (0.0 - 1.0)
        prediction: Detection prediction label (e.g., "Kidney Stone Detected")
        num_detections: Number of bounding boxes detected
    """
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Convert confidence to percentage
        confidence_pct = round(confidence * 100, 1) if confidence <= 1 else round(confidence, 1)
        
        # Determine severity based on confidence and count
        if num_detections == 0 or confidence_pct < 20:
            severity = "Normal"
        elif confidence_pct < 40:
            severity = "Ringan (Mild)"
        elif confidence_pct < 60:
            severity = "Sedang (Moderate)"
        elif confidence_pct < 80:
            severity = "Berat (Severe)"
        else:
            severity = "Kritis (Critical)"
        
        stone_detected = num_detections > 0 and "Detected" in prediction

        prompt = f"""Anda adalah dokter spesialis urologi berpengalaman yang menganalisis hasil CT Scan ginjal.

**DATA DETEKSI DARI MODEL AI (YOLOv8):**
- Hasil Prediksi: {"Batu Ginjal Terdeteksi" if stone_detected else "Tidak Ada Batu Ginjal Terdeteksi"}
- Jumlah area deteksi (bounding box): {num_detections}
- Tingkat kepercayaan model: {confidence_pct}%
- Estimasi tingkat keparahan: {severity}

**INSTRUKSI:**
Analisis gambar CT Scan berikut yang sudah dianotasi dengan bounding box dari model YOLO.
Berikan deskripsi medis yang SESUAI dengan data deteksi di atas.

{"Karena model mendeteksi " + str(num_detections) + " area batu ginjal dengan confidence " + str(confidence_pct) + "%, fokuskan analisis pada:" if stone_detected else "Karena model TIDAK mendeteksi batu ginjal, jelaskan bahwa:"}

{"1. Lokasi batu ginjal yang terdeteksi (kiri/kanan/bilateral)" if stone_detected else "1. Hasil pemeriksaan menunjukkan tidak ada indikasi batu ginjal"}
{"2. Deskripsi area yang ditandai bounding box" if stone_detected else "2. Kondisi ginjal tampak normal berdasarkan analisis AI"}
{"3. Perkiraan ukuran dan kepadatan batu" if stone_detected else "3. Tetap disarankan pemeriksaan rutin"}
{"4. Tingkat risiko berdasarkan confidence " + str(confidence_pct) + "% dan jumlah deteksi " + str(num_detections) if stone_detected else "4. Gaya hidup sehat untuk pencegahan"}
5. Rekomendasi medis yang sesuai dengan tingkat keparahan: {severity}

**FORMAT OUTPUT:**
Gunakan format Markdown dengan heading dan bullet points.
Gunakan bahasa Indonesia yang profesional, jelas, dan mudah dipahami.
PENTING: Jangan membuat informasi yang bertentangan dengan data deteksi di atas.
Jika model mendeteksi {num_detections} batu, jangan katakan jumlah yang berbeda.
Jika confidence {confidence_pct}%, sesuaikan deskripsi tingkat keparahan."""

        response = model_gemini.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return "Tidak ada deskripsi yang dihasilkan."
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Gagal menghasilkan deskripsi dengan Gemini."


def chatbot_msg(msg: str, data: str) -> str:
    try:
        # Pastikan model name sesuai dengan yang tersedia
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")

        # Gabungkan konteks dan pesan
        prompt = f"{data}\n{msg}"

        # Kirim prompt sebagai string langsung (bukan list dict)
        response = model_gemini.generate_content(prompt)

        # Akses teks respons langsung dari atribut `.text`
        if response.text:
            return response.text.strip()

        return "Tidak ada deskripsi yang dihasilkan."

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Gagal menghasilkan deskripsi dengan Gemini."
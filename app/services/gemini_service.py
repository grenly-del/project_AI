import google.generativeai as genai
from config import Config

genai.configure(api_key=Config.GEMINI_KEY)

def describe_detected_image(image_path):
    try:
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
        Gambar berikut adalah hasil deteksi batu ginjal menggunakan model YOLO.
        Analisis dan jelaskan gambar ini secara rinci dengan fokus pada hal-hal berikut:
        1. Gambarkan secara singkat area deteksi.
        2. Sebutkan jumlah bounding boxes.
        3. Deskripsikan ukuran dan kepadatan area.
        4. Beri tingkat risiko visual (Normal, Mild, Moderate, Severe, Critical).
        5. Akhiri dengan anjuran pemeriksaan medis lanjutan.
        Gunakan bahasa Indonesia profesional dan mudah dipahami.
        """

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
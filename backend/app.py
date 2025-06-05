from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Multimodal Assistant Backend Çalışıyor!"

@app.route("/extract-text", methods=["POST"])
def extract_text():
    if "pdf" not in request.files:
        return jsonify({"error": "PDF dosyası gönderilmedi!"}), 400

    pdf_file = request.files["pdf"]
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text()

    return jsonify({"text": all_text})

@app.route("/extract-images", methods=["POST"])
def extract_images():
    if "pdf" not in request.files:
        return jsonify({"error": "PDF dosyası gönderilmedi!"}), 400

    pdf_file = request.files["pdf"]
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images_data = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Görseli base64 ile encode et
            img_pil = Image.open(io.BytesIO(image_bytes))
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images_data.append({
                "page": page_number + 1,
                "image_index": img_index + 1,
                "image_base64": img_str
            })

    return jsonify({"images": images_data})

import whisper

@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası gönderilmedi!"}), 400

    audio_file = request.files["audio"]
    temp_path = "temp_audio_file.wav"
    audio_file.save(temp_path)

    # Küçük modeli kullanıyoruz, istersen "base", "small", "medium", "large" da deneyebilirsin
    model = whisper.load_model("tiny")
    result = model.transcribe(temp_path)
    os.remove(temp_path)

    return jsonify({"transcript": result["text"]})

if __name__ == "__main__":
    app.run(debug=True)
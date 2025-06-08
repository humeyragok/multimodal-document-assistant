from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import os
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
@app.route("/summarize-text", methods=["POST"])
def summarize_text():
    # Dosya veya metin var mı kontrol et
    text = ""
    if "text" in request.form and request.form["text"].strip():
        text = request.form["text"]
    elif "pdf" in request.files and request.files["pdf"]:
        pdf_file = request.files["pdf"]
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        except Exception as e:
            return jsonify({"error": f"PDF okunamadı: {str(e)}"}), 400
    else:
        return jsonify({"error": "Metin veya PDF yükleyin"}), 400

    # Çok uzun metinleri bölerek özetle
    max_chunk_len = 2000  # daha uzun chunk (karakter cinsinden)
    chunks = []
    text = text.replace("\n", " ")
    while len(text) > 0:
        if len(text) > max_chunk_len:
            cut_idx = text[:max_chunk_len].rfind('.')
            if cut_idx == -1 or cut_idx < max_chunk_len // 2:
                cut_idx = max_chunk_len  # cümle sonu yoksa düz böl
            chunks.append(text[:cut_idx].strip())
            text = text[cut_idx:].strip()
        else:
            chunks.append(text.strip())
            break

    # Her chunk için özet uzunluğunu otomatik seç
    summary = ""
    for chunk in chunks:
        if len(chunk.split()) < 20:
            summary += chunk + " "
            continue
        min_len = min(50, max(15, len(chunk.split()) // 4))
        max_len = min(180, max(60, len(chunk.split()) // 2))
        out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summary += out[0]['summary_text'] + " "

    return jsonify({"summary": summary.strip()})
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
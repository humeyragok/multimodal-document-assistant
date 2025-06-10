from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import os
from transformers import pipeline
from flask import Flask, request, jsonify, send_file
import whisper
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import simpleSplit
import re
from googletrans import Translator



app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
@app.route("/summarize-text", methods=["POST"])
def summarize_text():
    from flask import request, jsonify
    import fitz  # PyMuPDF

    # Metin veya PDF mi geldi kontrol et
    if "text" in request.form:
        text = request.form["text"]
    elif "pdf" in request.files:
        pdf_file = request.files["pdf"]
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
    else:
        return jsonify({"error": "Metin veya PDF yükleyin"}), 400

    # Çok uzun metinleri güvenle parçalara böl
    max_chunk_len = 900  # karakter cinsinden, güvenli sınır
    chunks = []
    text = text.replace("\n", " ")
    while len(text) > 0:
        if len(text) > max_chunk_len:
            cut_idx = text[:max_chunk_len].rfind('.')
            if cut_idx == -1:
                cut_idx = max_chunk_len
            chunks.append(text[:cut_idx].strip())
            text = text[cut_idx:].strip()
        else:
            chunks.append(text.strip())
            break

    # Her chunk için minimum/max özet uzunluğunu ayarla, kısa chunk'ları atla
    summary = ""
    for chunk in chunks:
        if len(chunk.split()) < 20:  # 20 kelimeden kısa parçayı özetlemeye gönderme
            summary += chunk + " "
            continue
        min_len = min(30, max(10, len(chunk.split()) // 4))
        max_len = min(130, max(30, len(chunk.split()) // 2))
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

@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası gönderilmedi!"}), 400

    audio_file = request.files["audio"]
    temp_path = "temp_audio_file.wav"
    audio_file.save(temp_path)

    # Küçük modeli kullanıyoruz, istersen "base", "small", "medium", "large" da deneyebilirsin
    model = whisper.load_model("medium")
    result = model.transcribe(temp_path)
    os.remove(temp_path)

    return jsonify({"transcript": result["text"]})

# Türkçe karakter desteği için font kaydı
FONT_PATH = "DejaVuSans.ttf"  # backend klasöründe olmalı
FONT_NAME = "DejaVu"

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError("DejaVuSans.ttf font dosyasını backend klasörüne kopyalayın!")

pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

def metni_duzenle(metin):
    # . ? ! sonrasında büyük harf başlat
    metin = re.sub(r'([.!?])\s*([a-zğüşöçıiA-ZĞÜŞÖÇİ])', lambda m: m.group(1) + " " + m.group(2).upper(), metin)
    # Satır başı büyük harf
    metin = metin[:1].upper() + metin[1:]
    # Çoklu boşlukları tek boşluğa indir
    metin = re.sub(' +', ' ', metin)
    return metin

def metni_pdf_yap(pdf_path, metin, baslik="Ses Transkripti"):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Başlık
    c.setFont(FONT_NAME, 16)
    c.drawString(50, height - 50, baslik)
    c.setFont(FONT_NAME, 12)

    y = height - 80
    max_width = width - 100  # Sağdan 50 px boşluk

    # Metni satır kaydırarak yaz
    lines = metin.split('\n')
    for line in lines:
        wrapped_lines = simpleSplit(line, FONT_NAME, 12, max_width)
        for wrap in wrapped_lines:
            c.drawString(50, y, wrap)
            y -= 18
            if y < 50:
                c.showPage()
                c.setFont(FONT_NAME, 12)
                y = height - 50
    c.save()

@app.route("/audio-to-pdf", methods=["POST"])
def audio_to_pdf():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası gönderilmedi!"}), 400

    audio_file = request.files["audio"]
    temp_audio_path = "temp_audio_file.wav"
    audio_file.save(temp_audio_path)

    # Whisper ile transkript al
    model = whisper.load_model("small")  # daha iyi sonuç için "medium" veya "large" da kullanabilirsin
    result = model.transcribe(temp_audio_path, language="tr")
    transcript = result["text"]
    os.remove(temp_audio_path)

    # Metni düzenle
    transcript = metni_duzenle(transcript)

    # PDF oluştur
    pdf_path = "audio_transcript.pdf"
    metni_pdf_yap(pdf_path, transcript, baslik="Ses Transkripti")

    return jsonify({"download_url": f"/download-pdf/{os.path.basename(pdf_path)}"})

@app.route("/download-pdf/<filename>")
def download_pdf(filename):
    return send_file(filename, as_attachment=True)

# Görselden kısa açıklama (caption) için BLIP modeli
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
translator = Translator()

def cevir_en_tr(text):
    return translator.translate(text, src="en", dest="tr").text

@app.route("/image-to-short-caption", methods=["POST"])
def image_to_short_caption():
    if "image" not in request.files:
        return jsonify({"error": "Resim dosyası gönderilmedi!"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    short_caption_en = captioner(image)[0]['generated_text']
    short_caption_tr = cevir_en_tr(short_caption_en)

    return jsonify({
        "short_caption": short_caption_tr
    })



if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import io
from PIL import Image
import base64

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


if __name__ == "__main__":
    app.run(debug=True)
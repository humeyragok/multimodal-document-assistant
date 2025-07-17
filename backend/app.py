from flask import Flask, request, render_template, send_file
import requests
import os
import io
import whisper
import base64
import templates


app = Flask(__name__)

MAX_CHARS = 1200  # Özetleme ve transcript için maksimum karakter sınırı

def ollama_summarize(text, model="llama3"):
    prompt = f"Bu metni Türkçe olarak özetle:\n\n{text}\n\nKısa özet:"
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        },
        timeout=120
    )
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

    
def create_pdf_from_transcript(transcript):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import simpleSplit
    import os
    import io

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Türkçe karakter desteği için DejaVuSans fontu ekle
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

    y = height - 50
    c.setFont("DejaVu", 12)
   
    max_width = width - 70  # Sağdan ve soldan boşluk

    # Tüm transcripti satır kaydırmalı şekilde yazdır
    lines = transcript.split("\n")
    for line in lines:
        wrapped_lines = simpleSplit(line, "DejaVu", 12, max_width)
        for wrap in wrapped_lines:
            c.drawString(35, y, wrap)
            y -= 18
            if y < 50:
                c.showPage()
                c.setFont("DejaVu", 12)
                y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

def create_pdf(transcript, summary):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(35, y, "Transkript:")
    c.setFont("Helvetica", 12)
    y -= 25
    for line in transcript.split("\n"):
        c.drawString(35, y, line)
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(35, y, "Özet:")
    y -= 25
    c.setFont("Helvetica", 12)
    for line in summary.split("\n"):
        c.drawString(35, y, line)
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)
    c.save()
    buffer.seek(0)
    return buffer
REPLACEMENTS = {
    '実施': 'uygulaması',
    # İstersen buraya başka karakterler de ekleyebilirsin
}

def temizle_ve_duzelt(metin):
    # Otomatik değiştir
    for yanlis, dogru in REPLACEMENTS.items():
        if yanlis in metin:
            print(f"Uyarı: '{yanlis}' karakteri bulundu, '{dogru}' ile değiştiriliyor.")
        metin = metin.replace(yanlis, dogru)
    # Ek olarak, sadece Türkçe harf, rakam, noktalama ve boşluk dışındaki karakterleri bul
    yabanci = re.findall(r'[^\sa-zA-Z0-9çÇğĞıİöÖşŞüÜ.,;:!?"\'()-]', metin)
    if yabanci:
        print("Uyarı: Şu karakterler yabancı veya beklenmeyen: ", set(yabanci))
    return metin

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    summary = ""
    summary_ready = False
    transcript = ""
    transcript_ready = False
    pdf_ready = False
    pdf_url = None
    active_tab = "text" 

    if request.method == "POST":
        # Metin özetleme sekmesi
        if "text" in request.form:
            active_tab = "text"
            text = request.form["text"]
            if len(text.strip()) == 0:
                error = "Lütfen özetlemek için bir metin giriniz."
            elif len(text) > MAX_CHARS:
                error = f"Metin çok uzun! Lütfen {MAX_CHARS} karakterden kısa bir metin giriniz."
            else:
                summary = ollama_summarize(text, model="llama3")
                summary_ready = True
            return render_template("index.html",
                                   summary=summary,
                                   summary_ready=summary_ready,
                                   transcript=transcript,
                                   transcript_ready=transcript_ready,
                                   pdf_ready=pdf_ready,
                                   pdf_url=pdf_url,
                                   error=error,
                                   max_chars=MAX_CHARS,
                                   active_tab=active_tab)

        # Ses yükleme sekmesi
        elif "audio" in request.files:
            active_tab = "audio"
            audio = request.files["audio"]
            if audio.filename == "":
                error = "Lütfen bir ses dosyası seçiniz."
            else:
                import tempfile, os
                # Daha iyi model ve dil belirt
                model = whisper.load_model("large")  # veya "medium"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.save(tmp)
                    tmp_path = tmp.name
                try:
                    result = model.transcribe(tmp_path, language="tr")
                    transcript = result["text"]
                    transcript_ready = True
                    if len(transcript) > MAX_CHARS:
                        error = f"Transkript çok uzun! Lütfen daha kısa bir ses dosyası yükleyin veya metni kısaltın. ({MAX_CHARS} karakter sınırı)"
                    else:
                        summary = ollama_summarize(transcript, model="llama3")
                        summary_ready = True
                        pdf_ready = True
                        pdf_url = "/download-pdf?transcript=" + transcript + "&summary=" + summary
                finally:
                    os.remove(tmp_path)
            return render_template("index.html",
                summary="",            # transcript sonucu özetleme kısmında görünmesin
                summary_ready=False,
                transcript=transcript,
                transcript_ready=transcript_ready,
                pdf_ready=pdf_ready,
                pdf_url=pdf_url,
                error=error,
                max_chars=MAX_CHARS,
                active_tab=active_tab)



    return render_template("index.html",
                           summary=summary,
                           summary_ready=summary_ready,
                           transcript=transcript,
                           transcript_ready=transcript_ready,
                           pdf_ready=pdf_ready,
                           pdf_url=pdf_url,
                           error=error,
                           max_chars=MAX_CHARS,
                           active_tab=active_tab)

@app.route("/download-pdf")
def download_pdf():
    transcript = request.args.get("transcript", "")
    buffer = create_pdf_from_transcript(transcript)
    return send_file(buffer, as_attachment=True, download_name="transkript.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

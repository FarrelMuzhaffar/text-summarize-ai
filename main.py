from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import PyPDF2
from docx import Document
import aiohttp
import os
import io
from typing import Optional

app = FastAPI()

# Tambahkan CORS untuk mengizinkan frontend mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend Anda di produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")  # Simpan di variabel lingkungan

# Endpoint untuk menyajikan halaman utama
@app.get("/")
async def serve_html():
    return FileResponse("text-summarize.html")

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Ekstrak teks dari file .txt, .docx, atau .pdf.
    """
    content = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".txt"):
            return content.decode("utf-8")
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            raise HTTPException(status_code=400, detail="Tipe file tidak didukung. Gunakan .txt, .docx, atau .pdf.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengekstrak teks dari file: {str(e)}")

async def summarize_text(input_text: str) -> str:
    """
    Kirim teks ke OpenRouter untuk ringkasan.
    """
    prompt = f"Buat ringkasan dari teks berikut: {input_text}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [
            {"role": "system", "content": "Buat ringkasan dari teks berikut."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=30) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"Gagal menghubungi API: {response.status}")
            data = await response.json()
            if "choices" not in data or not data["choices"]:
                raise HTTPException(status_code=500, detail="API tidak mengembalikan konten.")
            return data["choices"][0]["message"]["content"].strip()

@app.post("/summarize")
async def summarize(
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Endpoint untuk meringkas teks atau file.
    """
    if not prompt and not file:
        raise HTTPException(status_code=400, detail="Tidak ada teks atau file yang diunggah.")

    input_text = ""
    if file:
        input_text = await extract_text_from_file(file)
    elif prompt:
        input_text = prompt.strip()

    if not input_text:
        raise HTTPException(status_code=400, detail="Teks yang diekstrak kosong atau tidak valid.")

    try:
        summary = await summarize_text(input_text)
        return {"success": True, "data": {"text": summary}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
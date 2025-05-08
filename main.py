from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import PyPDF2
from docx import Document
import aiohttp
import os
import io
import logging
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Tambahkan CORS untuk mengizinkan frontend mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sementara izinkan semua untuk debugging
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
    logger.info("Serving text-summarize.html")
    return FileResponse("text-summarize.html")

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Ekstrak teks dari file .txt, .docx, atau .pdf (maksimum 50 halaman untuk PDF).
    """
    start_time = time.time()
    logger.info(f"Extracting text from file: {file.filename}")
    content = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".txt"):
            text = content.decode("utf-8")
            logger.info(f"Successfully extracted text from .txt in {time.time() - start_time:.2f} seconds")
            return text
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            logger.info(f"Successfully extracted text from .docx in {time.time() - start_time:.2f} seconds")
            return text
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            max_pages = 50  # Batas maksimum halaman
            for page_num, page in enumerate(pdf_reader.pages[:max_pages], 1):
                extracted = page.extract_text()
                text += extracted or ""
                logger.info(f"Extracted page {page_num}/{min(len(pdf_reader.pages), max_pages)}")
            if len(pdf_reader.pages) > max_pages:
                logger.warning(f"PDF truncated to {max_pages} pages (total pages: {len(pdf_reader.pages)})")
            logger.info(f"Successfully extracted text from .pdf in {time.time() - start_time:.2f} seconds")
            return text
        else:
            logger.error(f"Unsupported file type: {filename}")
            raise HTTPException(status_code=400, detail="Tipe file tidak didukung. Gunakan .txt, .docx, atau .pdf.")
    except Exception as e:
        logger.error(f"Failed to extract text from file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal mengekstrak teks dari file: {str(e)}")

async def summarize_text(input_text: str) -> str:
    """
    Kirim teks ke OpenRouter untuk ringkasan.
    """
    start_time = time.time()
    logger.info("Starting text summarization")
    prompt = f"Buat ringkasan dari teks berikut: {input_text}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [
            {"role": "system", "content": "Buat ringkasan dari teks berikut."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=90) as response:
                logger.info(f"OpenRouter API response status: {response.status} in {time.time() - start_time:.2f} seconds")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise HTTPException(status_code=500, detail=f"Gagal menghubungi API: {response.status}")
                data = await response.json()
                if "choices" not in data or not data["choices"]:
                    logger.error("OpenRouter API response missing choices")
                    raise HTTPException(status_code=500, detail="API tidak mengembalikan konten.")
                logger.info(f"Summarization completed in {time.time() - start_time:.2f} seconds")
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error in summarize_text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Gagal meringkas teks: {str(e)}")

@app.post("/summarize")
async def summarize(
    request: Request,
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Endpoint untuk meringkas teks atau file.
    """
    start_time = time.time()
    logger.info(f"Received summarize request: {request.method} {request.url}")
    logger.info(f"Request headers: {dict(request.headers)}")
    if not prompt and not file:
        logger.error("No text or file provided")
        raise HTTPException(status_code=400, detail="Tidak ada teks atau file yang diunggah.")

    if file and file.size > 5 * 1024 * 1024:  # 5MB limit
        logger.error(f"File size too large: {file.size} bytes")
        raise HTTPException(status_code=400, detail="Ukuran file terlalu besar. Maksimum 5MB.")

    input_text = ""
    if file:
        input_text = await extract_text_from_file(file)
    elif prompt:
        input_text = prompt.strip()

    if not input_text:
        logger.error("Extracted text is empty or invalid")
        raise HTTPException(status_code=400, detail="Teks yang diekstrak kosong atau tidak valid.")

    try:
        summary = await summarize_text(input_text)
        logger.info(f"Summarization successful in {time.time() - start_time:.2f} seconds")
        return {"success": True, "data": {"text": summary}}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
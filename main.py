from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
import pdfplumber
from reportlab.pdfgen import canvas
import uuid
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (allows your Hostinger frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later you can restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load DeepSeek API key securely from environment
api_key = os.getenv("DEEPSEEK_API_KEY")

client = None

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

# Home route for testing
@app.get("/")
def home():
    return {"status": "Resume Optimizer Backend Running"}

# Function to extract text from uploaded PDF
def extract_text_from_pdf(file):

    try:
        text = ""

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to read PDF file"
        )

# Function to optimize resume using DeepSeek
def optimize_resume(resume_text):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable"
        )

    prompt = f"""
You are a professional ATS resume optimizer.

Analyze the resume and improve it.

Return strictly in this format:

ATS SCORE: number

OPTIMIZED RESUME:

Full professional ATS-optimized resume.

Resume:
{resume_text}
"""

    try:

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert ATS resume optimizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="AI optimization failed"
        )

# Function to create PDF from optimized text
def create_pdf(content):

    try:

        filename = f"optimized_resume_{uuid.uuid4()}.pdf"

        c = canvas.Canvas(filename)

        y = 800

        for line in content.split("\n"):

            if y < 50:
                c.showPage()
                y = 800

            c.drawString(50, y, line)
            y -= 20

        c.save()

        return filename

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to create PDF"
        )

# Optimize endpoint
@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Extract text
    resume_text = extract_text_from_pdf(file.file)

    if not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF contains no readable text"
        )

    # Optimize resume
    optimized_text = optimize_resume(resume_text)

    # Create optimized PDF
    pdf_file = create_pdf(optimized_text)

    return {
        "success": True,
        "optimized_text": optimized_text,
        "download_url": f"/download/{pdf_file}"
    }

# Download endpoint
@app.get("/download/{filename}")
def download_file(filename: str):

    if not os.path.exists(filename):
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path=filename,
        filename=filename,
        media_type="application/pdf"
    )

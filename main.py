from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
import pdfplumber
from reportlab.pdfgen import canvas
import uuid
import os

# Load API key from environment variable
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder to store generated PDFs
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"status": "Resume Optimizer Backend Running"}


# -------- PDF TEXT EXTRACTION --------

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -------- AI OPTIMIZATION --------

def optimize_resume(text):

    prompt = f"""
You are an expert ATS resume optimizer.

Improve this resume professionally.

Return JSON format only:

{{
  "optimized_text": "full optimized resume text"
}}

Resume:
{text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a professional resume writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------- PDF GENERATION --------

def create_pdf(content):

    filename = f"optimized_resume_{uuid.uuid4()}.pdf"
    path = os.path.join(OUTPUT_DIR, filename)

    c = canvas.Canvas(path)

    y = 800

    for line in content.split("\n"):
        c.drawString(50, y, line)
        y -= 20

        if y < 50:
            c.showPage()
            y = 800

    c.save()

    return filename


# -------- PDF UPLOAD ENDPOINT --------

@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):

    text = extract_text(file.file)

    optimized = optimize_resume(text)

    filename = create_pdf(optimized)

    return JSONResponse({
        "success": True,
        "optimized_text": optimized,
        "download_url": f"/download/{filename}"
    })


# -------- STRUCTURED BUILDER ENDPOINT --------

@app.post("/optimize/structured")
async def optimize_structured(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(""),
    skills: str = Form(""),
    projects: str = Form(""),
    experience: str = Form(""),
    education: str = Form("")
):

    structured_text = f"""
Name: {name}
Email: {email}
Phone: {phone}

Skills:
{skills}

Projects:
{projects}

Experience:
{experience}

Education:
{education}
"""

    optimized = optimize_resume(structured_text)

    filename = create_pdf(optimized)

    return JSONResponse({
        "success": True,
        "optimized_text": optimized,
        "download_url": f"/download/{filename}"
    })


# -------- DOWNLOAD ENDPOINT --------

@app.get("/download/{filename}")
def download(filename: str):

    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename
    )

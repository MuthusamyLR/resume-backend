from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import pdfplumber
from reportlab.pdfgen import canvas
import uuid
import os

app = FastAPI()

# Allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def optimize_resume(text):

    prompt = f"""
You are an ATS resume optimizer.

Analyze and rewrite this resume professionally.

Return:

ATS SCORE: number

OPTIMIZED RESUME:

Full resume

Resume:
{text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert resume writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def create_pdf(content):

    filename = f"{uuid.uuid4()}.pdf"

    c = canvas.Canvas(filename)

    y = 800

    for line in content.split("\n"):
        c.drawString(50, y, line)
        y -= 20

    c.save()

    return filename


@app.get("/")
def home():
    return {"status": "Backend running"}


@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):

    text = extract_text(file.file)

    optimized = optimize_resume(text)

    pdf_file = create_pdf(optimized)

    return {
        "optimized_text": optimized,
        "pdf_file": pdf_file
    }

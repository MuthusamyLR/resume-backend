from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
import pdfplumber
from reportlab.pdfgen import canvas
import uuid
import os
import json
import time
import re
from typing import Optional

# -------------------------
# CONFIG
# -------------------------

OUTPUT_DIR = "generated"
MAX_RETRIES = 3
RETRY_DELAY = 2
TIMEOUT_SECONDS = 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load API key safely
# -------------------------

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=TIMEOUT_SECONDS
)

# -------------------------
# FastAPI init
# -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Render cold start indicator
# -------------------------

startup_time = time.time()


@app.get("/")
def home():
    uptime = int(time.time() - startup_time)
    return {
        "status": "Resume Optimizer Backend Running",
        "uptime_seconds": uptime
    }


# -------------------------
# PDF TEXT EXTRACTION
# -------------------------

def extract_text(file) -> str:

    try:

        text = ""

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise HTTPException(400, "No readable text in PDF")

        return text

    except Exception as e:

        raise HTTPException(500, f"PDF extraction failed: {str(e)}")


# -------------------------
# GUARANTEED JSON PARSER
# -------------------------

def extract_json_from_text(text: str) -> Optional[dict]:

    try:

        # Direct parse
        return json.loads(text)

    except:

        pass

    # Try extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:

        try:
            return json.loads(match.group())
        except:
            pass

    return None


# -------------------------
# AI CALL WITH RETRY + TIMEOUT
# -------------------------

def call_ai_with_retry(prompt: str) -> str:

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(

                model="deepseek-chat",

                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional ATS resume optimizer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,
                max_tokens=2000
            )

            if not response:
                raise Exception("Empty response object")

            content = response.choices[0].message.content

            if not content:
                raise Exception("Empty content")

            return content

        except Exception as e:

            last_error = str(e)

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"AI failed after retries: {last_error}")


# -------------------------
# OPTIMIZER
# -------------------------

def optimize_resume(text: str) -> str:

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

    result = call_ai_with_retry(prompt)

    parsed = extract_json_from_text(result)

    if parsed and "optimized_text" in parsed:

        return parsed["optimized_text"]

    # fallback if AI fails format
    return result


# -------------------------
# SAFE PDF GENERATION
# -------------------------

def create_pdf(content: str) -> str:

    try:

        filename = f"optimized_resume_{uuid.uuid4().hex}.pdf"

        path = os.path.join(OUTPUT_DIR, filename)

        c = canvas.Canvas(path)

        y = 800

        for line in content.split("\n"):

            safe_line = line[:100]  # prevent overflow

            c.drawString(50, y, safe_line)

            y -= 20

            if y < 50:

                c.showPage()
                y = 800

        c.save()

        return filename

    except Exception as e:

        raise HTTPException(500, f"PDF creation failed: {str(e)}")


# -------------------------
# MAIN ENDPOINT
# -------------------------

@app.post("/optimize")
async def optimize(file: UploadFile = File(...)):

    try:

        text = extract_text(file.file)

        optimized_text = optimize_resume(text)

        filename = create_pdf(optimized_text)

        return JSONResponse({

            "success": True,

            "optimized_text": optimized_text,

            "download_url": f"/download/{filename}"

        })

    except Exception as e:

        return JSONResponse({

            "success": False,

            "error": str(e)

        }, status_code=500)


# -------------------------
# STRUCTURED ENDPOINT
# -------------------------

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

    try:

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

        optimized_text = optimize_resume(structured_text)

        filename = create_pdf(optimized_text)

        return JSONResponse({

            "success": True,

            "optimized_text": optimized_text,

            "download_url": f"/download/{filename}"

        })

    except Exception as e:

        return JSONResponse({

            "success": False,

            "error": str(e)

        }, status_code=500)


# -------------------------
# DOWNLOAD
# -------------------------

@app.get("/download/{filename}")
def download(filename: str):

    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):

        raise HTTPException(404, "File not found")

    return FileResponse(

        path,
        media_type="application/pdf",
        filename=filename

    )

"""
pdf_to_markdown.py

This script:
1. Reads a law PDF
2. Extracts raw text
3. Uses Gemini to convert it into clean Markdown
4. Saves the Markdown file

Beginner-friendly, step-by-step.
"""

import os
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# STEP 1: Load API key
# -----------------------------
load_dotenv()  # Loads variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# STEP 2: File paths
# -----------------------------
PDF_PATH = "data/pdfs/India Penal Code.pdf"        # 👈 path to your PDF
OUTPUT_MD_PATH = "data/markdown/IPC.md"  # 👈 where markdown will be saved

# -----------------------------
# STEP 3: Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"

    return all_text


print("📄 Extracting text from PDF...")
raw_text = extract_text_from_pdf(PDF_PATH)

# -----------------------------
# STEP 4: Send text to Gemini
# -----------------------------
print("🤖 Sending text to Gemini for Markdown conversion...")

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are converting an Indian law document into clean Markdown.

Instructions:
- Use proper Markdown headings
- Convert each Section into a heading (## Section X – Title)
- Remove page numbers, footers, and repeated headers
- Do NOT summarize or change meaning
- Keep the original legal wording
- Output ONLY Markdown, nothing else

Text:
{raw_text}
"""

response = model.generate_content(prompt)

markdown_text = response.text

# -----------------------------
# STEP 5: Save Markdown file
# -----------------------------
os.makedirs("data/markdown", exist_ok=True)

with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"✅ Markdown file saved at: {OUTPUT_MD_PATH}")

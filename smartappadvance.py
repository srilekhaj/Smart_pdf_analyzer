from fastapi import FastAPI, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import shutil
from pdf2image import convert_from_path
import pytesseract
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
api_key = os.getenv('GROQ_API_KEY')
client = Groq()

# Set Poppler path in the environment variables dynamically
os.environ["PATH"] += ";C:\\Program Files\\poppler-24.08.0\\Library\\bin;C:\\poppler\\bin"

# Set the path to Tesseract executable (if not added to PATH globally)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_pypdf2(file_path):
    text=""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
    return text.strip() if text.strip() else None

def extract_text_ocr(pdf_path):
    """Extract text from scanned PDFs using OCR."""
    """Yes, Each page of the PDF is converted into an image, and the text from each page is extracted using Tesseract. Finally, the text from all pages is combined into one large string with newlines separating the text from each page."""
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img, lang='eng', config='--psm 6') for img in images]).strip()
    return text if text else None

def detect_pdf_type_and_extract(pdf_path):
    """Determine whether the PDF is text-based or scanned, and extract text accordingly."""
    text = extract_text_pypdf2(pdf_path)
    if text:
        return text, "Text-based PDF"
    
    # If no text is found, force OCR processing
    text = extract_text_ocr(pdf_path)
    if text:
        return text, "Scanned PDF"
    
    return None, "Unknown PDF type"

def summarize_text(text):
    try:
        
        chat_completion = client.chat.completions.create(
            messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": text,
            }
        ],
        
        # The language model which will generate the completion.
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        # The maximum number of tokens to generate. Requests can use up to
        # 32,768 tokens shared between prompt and completion.
        max_completion_tokens=1024,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,
        # If set, partial message deltas will be sent.
        stream=False,
        )
        summary = chat_completion.choices[0].message.content
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with AI summarization: {str(e)}")

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text, pdf_type = detect_pdf_type_and_extract(file_path)  #text, "Text-based PDF" text, "Scanned PDF"
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    
    summary = summarize_text(text)
    return {"pdf_type": pdf_type, "summary": summary}

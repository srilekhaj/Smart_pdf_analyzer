from fastapi import FastAPI, UploadFile, File
import shutil
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq()
api_key = os.getenv('GROQ_API_KEY')
app = FastAPI()
from PyPDF2 import PdfReader

def extract_text_pymupdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text = text + page.extract_text()
    return text

def summarize_text(text):
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

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_pymupdf(file_path)
    summary = summarize_text(text)
    return {"summary": summary}

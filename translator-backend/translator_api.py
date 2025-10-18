from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rene-04x.github.io",
        "https://rene-04x.github.io/English-Tagalog-Translator/"
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom dictionary
custom_dict = {
    "computer": "kompyuter",
    "internet": "internet",
    "good morning": "magandang umaga",
    "AI": "artipisyal na intelihensiya"
}

def apply_custom_dict(text):
    for k, v in custom_dict.items():
        text = text.replace(k, v)
    return text

# Request schema
class TextRequest(BaseModel):
    text: str
    direction: str  # "en-tl" or "tl-en"

# Translation endpoint (uses OpenAI GPT for translation + refinement)
@app.post("/translate")
async def translate(request: TextRequest):
    text = request.text.strip()
    if not text:
        return {"translation": ""}

    # Set source/target languages
    if request.direction == "en-tl":
        src, tgt = "English", "Tagalog"
    elif request.direction == "tl-en":
        src, tgt = "Tagalog", "English"
    else:
        return {"error": "Invalid direction"}

    # Build prompt for GPT
    prompt = f"""
Translate the following text from {src} to {tgt}. 
Make the translation accurate, natural, and grammatically correct.

Text: {text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content":prompt}],
            temperature=0
        )
        translation = response.choices[0].message.content.strip()

        # Apply custom dictionary
        translation = apply_custom_dict(translation)

        return {"translation": translation}

    except Exception as e:
        return {"translation": f"❌ Error: {str(e)}"}

# Dictionary definition endpoint
@app.get("/define")
async def define(word: str):
    try:
        res = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if res.status_code == 200:
            data = res.json()
            meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
            return {"definition": meaning}

        # fallback: GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": f"Define '{word}' in simple English, briefly."}]
        )
        return {"definition": response.choices[0].message.content.strip()}

    except Exception as e:
        return {"definition": f"❌ Error fetching definition: {str(e)}"}

# Health check endpoint
@app.get("/")
def health():
    return {"status": "✅ API is running"}

# Render entrypoint
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("translator_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

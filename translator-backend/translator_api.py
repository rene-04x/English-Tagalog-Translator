from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch, os, requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rene-04x.github.io", "https://rene-04x.github.io/English-Tagalog-Translator/"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Global variables (lazy loaded) ---
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Custom dictionary ---
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


# --- AI Refinement Function ---
def refine_translation_with_ai(original_text, raw_translation, direction):
    prompt = f"""
You are a bilingual English–Tagalog translator.
Refine the translation to make it accurate, natural, and grammatically correct.

Direction: {direction}
Original text: {original_text}
Raw translation: {raw_translation}

Return only the improved translation, no explanations.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional English–Tagalog translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ AI refinement skipped: {e}")
        return raw_translation


# --- Request Schema ---
class TextRequest(BaseModel):
    text: str
    direction: str  # "en-tl" or "tl-en"


# --- Translation Endpoint ---
@app.post("/translate")
def translate(request: TextRequest):
    global model, tokenizer

    text = request.text.strip()
    if not text:
        return {"translation": ""}

    # Lazy-load model
    if model is None or tokenizer is None:
        try:
            print("🧩 Loading translation model...")
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            model_name = "facebook/m2m100_418M"
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model could not be loaded (memory issue).")

    # Set languages
    if request.direction == "en-tl":
        src_lang, tgt_lang = "en", "tl"
    elif request.direction == "tl-en":
        src_lang, tgt_lang = "tl", "en"
    else:
        return {"error": "Invalid direction"}

    # Run translation
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=200,
        num_beams=5,
        length_penalty=1.2
    )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Apply dictionary + refinement
    translated_text = apply_custom_dict(translated_text)
    translated_text = refine_translation_with_ai(request.text, translated_text, request.direction)

    return {"translation": translated_text}


# --- Dictionary Definition Endpoint ---
@app.get("/define")
async def define(word: str):
    try:
        res = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if res.status_code == 200:
            data = res.json()
            meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
            return {"definition": meaning}

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": f"Define '{word}' in simple English, briefly."}
            ],
        )
        return {"definition": response.choices[0].message.content.strip()}

    except Exception as e:
        return {"definition": f"❌ Error fetching definition: {str(e)}"}


# --- Render entrypoint ---
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("translator_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


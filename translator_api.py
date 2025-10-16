from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load Hugging Face model
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom dictionary (optional)
custom_dict = {
    "computer": "kompyuter",
    "internet": "internet",
    "good morning": "magandang umaga",
    "AI": "artipisyal na intelihensiya"
}

def apply_custom_dict(text):
    """Replace known terms from custom dictionary."""
    for k, v in custom_dict.items():
        text = text.replace(k, v)
    return text

# --- AI Refinement Function ---
def refine_translation_with_ai(original_text, raw_translation, direction):
    """Improve translation accuracy and fluency using GPT-4o-mini."""
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
    text = request.text.strip()
    if not text:
        return {"translation": ""}

    # Define direction
    if request.direction == "en-tl":
        src_lang, tgt_lang = "en", "tl"
    elif request.direction == "tl-en":
        src_lang, tgt_lang = "tl", "en"
    else:
        return {"error": "Invalid direction"}

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(device)

    # Hugging Face translation
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        max_length=200,
        num_beams=5,        # better accuracy
        length_penalty=1.2  # prevents too short/long outputs
    )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Apply dictionary & AI refinement
    translated_text = apply_custom_dict(translated_text)
    translated_text = refine_translation_with_ai(request.text, translated_text, request.direction)

    return {"translation": translated_text}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("translator_api:app", host="127.0.0.1", port=8000, reload=True)

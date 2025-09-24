from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import zipfile
import requests
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI(title="Misinformation Detection ML Service")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    is_misinformation: bool
    confidence: float
    category: str
    explanation: str

model = None
tokenizer = None

def download_and_extract_model():
    model_dir = Path("models")
    model_zip_path = Path("model.zip")
    
    if not model_dir.exists():
        print("Model directory not found. Downloading model...")
        model_url = os.getenv("MODEL_ZIP_URL")
        if not model_url:
            print("MODEL_ZIP_URL not configured, skipping model download")
            return False
        
        try:
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_zip_path, "wb") as f:
                f.write(response.content)
            model_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
                zip_ref.extractall(model_dir)
            model_zip_path.unlink()
            print("Model extracted successfully")
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    return True

def load_model():
    global model, tokenizer
    model_dir = Path("models")
    try:
        if model_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            print("Custom model loaded successfully")
        else:
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            print("Using fallback pre-trained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        tokenizer = None
        model = None


def predict_misinformation(text: str) -> dict:
    if model is None or tokenizer is None:
        return {
            "is_misinformation": len(text.split()) > 10,
            "confidence": 0.7,
            "category": "health" if "covid" in text.lower() else "general",
            "explanation": "Dummy prediction - model not available"
        }
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = float(predictions.max())
        is_misinformation = bool(predictions.argmax() == 1)
        text_lower = text.lower()
        if any(word in text_lower for word in ["covid", "vaccine", "health", "medicine"]):
            category = "health"
        elif any(word in text_lower for word in ["election", "vote", "politics", "government"]):
            category = "politics"
        elif any(word in text_lower for word in ["climate", "environment", "global warming"]):
            category = "climate"
        else:
            category = "general"
        return {
            "is_misinformation": is_misinformation,
            "confidence": confidence,
            "category": category,
            "explanation": f"ML model prediction with {confidence:.2f} confidence"
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            "is_misinformation": False,
            "confidence": 0.5,
            "category": "unknown",
            "explanation": f"Error during prediction: {str(e)}"
        }


async def call_gemini_api(text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "is_misinformation": False,
            "confidence": 0.5,
            "category": "unknown",
            "explanation": "Gemini API key not configured"
        }
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        json_body = {
            "contents": [{
                "parts": [{
                    "text": f'Analyze this text for misinformation. Respond with JSON format: {{"is_misinformation": boolean, "confidence": number (0-1), "category": "string", "explanation": "string"}}. Text: "{text}"'
                }]
            }]
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=json_body)
            response.raise_for_status()
            content = response.json()
            text_resp = content["candidates"][0]["content"]["parts"][0]["text"]
            import re, json
            json_match = re.search(r"\{[\s\S]*\}", text_resp)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {
                    "is_misinformation": False,
                    "confidence": 0.5,
                    "category": "unknown",
                    "explanation": "Invalid response format from Gemini"
                }
    except Exception as e:
        return {
            "is_misinformation": False,
            "confidence": 0.5,
            "category": "unknown",
            "explanation": f"Gemini API error: {str(e)}"
        }

@app.on_event("startup")
async def startup_event():
    print("Starting ML service...")
    download_and_extract_model()
    load_model()
    print("ML service ready!")

@app.get("/")
async def root():
    return {"message": "Misinformation Detection ML Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = await call_gemini_api(request.text)
    return PredictionResponse(**result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

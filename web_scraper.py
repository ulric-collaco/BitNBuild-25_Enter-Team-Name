from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import re
import requests
import logging
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Review Radar - Web Scraper API",
    description="",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class UrlRequest(BaseModel):
    url: str

class ScrapedResponse(BaseModel):
    url: str
    gemini_response: Any
    status: str
    message: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
MAIN_API_URL = os.getenv("MAIN_API_URL", "http://localhost:8000")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. Gemini requests will fail until it's provided.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

model = None

def get_gemini_model() -> genai.GenerativeModel:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is missing")

    try:
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as exc:
        logger.error("Failed to initialise Gemini model: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to initialise Gemini model: {exc}")

def ensure_model() -> genai.GenerativeModel:
    global model
    if model is None:
        model = get_gemini_model()
    return model

def extract_json_block(text: str) -> str:
    cleaned = text.strip()

    # Remove markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)

    # Fix common JSON issues - trailing commas before closing brackets
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    
    # Fix unquoted property names (common Gemini issue)
    cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)
    
    # Fix single quotes to double quotes
    cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
    
    # Fix escaped quotes that might break JSON
    cleaned = re.sub(r'\\"', '"', cleaned)
    
    # Remove any trailing text after the JSON ends
    cleaned = re.sub(r'(\]|\})\s*.*$', r'\1', cleaned, flags=re.DOTALL)
    
    # Try to locate a JSON object/array within the text
    json_match = re.search(r"\[.*\]|\{.*\}", cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        return json_str

    return cleaned

def normalize_gemini_response(raw_text: str, source_url: str) -> Dict[str, Any]:
    json_payload = extract_json_block(raw_text)
    data = json.loads(json_payload)

    if isinstance(data, list):
        data = {
            "source_url": source_url,
            "total_reviews": len(data),
            "reviews": data,
        }
    elif isinstance(data, dict):
        reviews = data.get("reviews")
        if not isinstance(reviews, list):
            raise ValueError("Parsed JSON does not contain a 'reviews' list")
        data.setdefault("source_url", source_url)
        data.setdefault("total_reviews", len(reviews))
    else:
        raise ValueError("Unexpected JSON structure returned by Gemini")

    return data

@app.get("/")
async def root():
    return {
        "message": "Review Radar - Web Scraper API",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "scrape": "/scrape-url - Main endpoint for URL processing"
        }
    }

@app.get("/health")
async def health_check():
    gemini_available = bool(GEMINI_API_KEY)
    
    return {
        "status": "healthy",
        "gemini_configured": gemini_available,
        "main_api_url": MAIN_API_URL
    }

@app.post("/scrape-url", response_model=ScrapedResponse)
async def scrape_url_endpoint(request: UrlRequest):
    print("\n" + "🎯" * 40)
    print("🚀 NEW FRONTEND REQUEST RECEIVED!")
    print("🎯" * 40)
    print(f"📱 URL from Frontend: {request.url}")
    print(f"⏰ Timestamp: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    print("-" * 80)

    logger.info("Received URL for scraping: %s", request.url)

    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    if not (request.url.startswith("http://") or request.url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    gemini_prompt = f"""Generate exactly 60-75 realistic customer reviews for: {request.url}

IMPORTANT: Make reviews highly specific to this product mentioning: 

For each review, include realistic details about the product experience. Avoid generic comments.

Requirements:
- Mix of detailed and brief reviews
- Realistic names (mix of Indian and international)
- Authentic customer language and experiences
- Specific product features mentioned

Return ONLY this JSON format:
[
  {{"id": "r1", "name": "Name", "rating": 5, "comments": "Specific review about the product"}},
  {{"id": "r2", "name": "Name", "rating": 4, "comments": "Another specific review"}},
  ...
]

Generate all 60-75 reviews with unique, specific comments. make sure its a random amount between 60-75 comments in the above give format"""

    logger.info("Sending prompt to AI service...")

    try:
        gemini_client = ensure_model()
        gemini_response = gemini_client.generate_content(gemini_prompt)
        gemini_text = gemini_response.text if hasattr(gemini_response, "text") else str(gemini_response)

        print("🧠 RAW GEMINI RESPONSE:")
        print(gemini_text)
        print("-" * 80)

        normalized = normalize_gemini_response(gemini_text, request.url)

        print("✅ NORMALISED GEMINI OUTPUT:")
        print(json.dumps(normalized, indent=2)[:4000])  # avoid flooding terminal
        print("-" * 80)

    except HTTPException:
        raise
    except (json.JSONDecodeError, ValueError) as parse_error:
        logger.error("Failed to parse Gemini response: %s", parse_error)
        
        # Try alternative parsing methods
        try:
            print("🔄 ATTEMPTING ALTERNATIVE JSON PARSING...")
            
            # Method 1: Try to fix and parse again
            json_payload = extract_json_block(gemini_text)
            
            # Method 2: Try to manually extract just the array part
            array_match = re.search(r'\[[\s\S]*\]', gemini_text)
            if array_match:
                json_payload = array_match.group(0)
                json_payload = re.sub(r',(\s*[\]\}])', r'\1', json_payload)
                json_payload = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_payload)
                
            normalized = json.loads(json_payload)
            if isinstance(normalized, list):
                normalized = {
                    "source_url": request.url,
                    "total_reviews": len(normalized),
                    "reviews": normalized,
                }
            
            print("✅ ALTERNATIVE PARSING SUCCESSFUL!")
            
        except Exception as alt_error:
            logger.error("Alternative parsing also failed: %s", alt_error)
            print("❌ ALL PARSING METHODS FAILED")
            print(f"Raw Gemini response (first 1000 chars): {gemini_text[:1000]}")
            
            return ScrapedResponse(
                url=request.url,
                gemini_response=gemini_text if "gemini_text" in locals() else str(parse_error),
                status="partial_success",
                message=f"Gemini response could not be parsed: {parse_error}. Alternative parsing: {alt_error}",
            )
    except Exception as exc:
        logger.error("Gemini request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}")

    try:
        print("\n🔄 FORWARDING TO MAIN API...")
        print(f"📤 SENDING {len(normalized.get('reviews', []))} REVIEWS TO MAIN.PY")
        print(f"🎯 Target: {MAIN_API_URL.rstrip('/')}/analyze-dashboard")
        print("-" * 40)
        
        analysis_response = requests.post(
            f"{MAIN_API_URL.rstrip('/')}/analyze-dashboard",
            json=normalized,
            headers={"Content-Type": "application/json"},
            timeout=45,
        )
        analysis_response.raise_for_status()
        analysis_result = analysis_response.json()

        print("✅ ANALYSIS COMPLETE!")
        print(json.dumps(analysis_result, indent=2)[:4000])
        print("=" * 80 + "\n")

        return ScrapedResponse(
            url=request.url,
            gemini_response=analysis_result,
            status="success",
            message="URL successfully processed and analyzed",
        )
    except requests.RequestException as req_error:
        logger.error("Forwarding to main API failed: %s", req_error)
        return ScrapedResponse(
            url=request.url,
            gemini_response=normalized,
            status="partial_success",
            message=f"Gemini succeeded but analysis service failed: {req_error}",
        )

@app.post("/test-ai")
async def test_ai_connection():
    try:
        gemini_client = ensure_model()
        test_response = gemini_client.generate_content("Hello, this is a test. Please respond with 'AI service is working correctly.'")
        return {
            "status": "success",
            "message": "AI API is working",
            "response": test_response.text if hasattr(test_response, "text") else str(test_response),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("AI test failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"AI API test failed: {exc}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
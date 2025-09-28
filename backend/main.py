from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Review Radar - Sentiment Analysis API (Cloud)",
    description="Lightweight FastAPI service using Hugging Face Inference API",
    version="2.0.0-cloud"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DashboardReviewItem(BaseModel):
    id: str
    name: str
    rating: int
    comments: str

class DashboardRequest(BaseModel):
    reviews: List[DashboardReviewItem]

class SentimentChecklist(BaseModel):
    positive: int
    neutral: int
    negative: int

class ChecklistData(BaseModel):
    sentiment: SentimentChecklist

class KeywordResult(BaseModel):
    keyword: str
    score: float

class KeywordData(BaseModel):
    positive_keywords: List[KeywordResult]
    negative_keywords: List[KeywordResult]

class DashboardResponse(BaseModel):
    checklist: ChecklistData
    keywords: KeywordData
    total_reviews: int
    average_rating: float

HF_API_KEY = os.getenv("HF_API_KEY")
SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
KEYWORD_API_URL = "https://api-inference.huggingface.co/models/ml6team/keyphrase-extraction-kbir-inspec"

headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def analyze_sentiment_cloud(text: str) -> tuple:
    try:
        response = requests.post(
            SENTIMENT_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                
                sentiment_map = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral', 
                    'LABEL_2': 'positive'
                }
                
                best_prediction = max(predictions, key=lambda x: x['score'])
                sentiment = sentiment_map.get(best_prediction['label'], 'neutral')
                confidence = best_prediction['score']
                
                return sentiment, confidence
        
        return "neutral", 0.5
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return "neutral", 0.5

def extract_keywords_cloud(text: str, max_keywords: int = 10) -> List[tuple]:
    try:
        response = requests.post(
            KEYWORD_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                keywords = [(item.get('word', ''), item.get('score', 0.5)) for item in result[:max_keywords]]
                return keywords
        
        fallback_keywords = []
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:max_keywords]
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        return []

print("\n" + "="*60)
print("🎉 REVIEW RADAR - CLOUD ML API SERVER READY!")
print("="*60)
print("☁️  Using Hugging Face Inference API (Cloud-based)")
print("✅ Sentiment Analysis: CardiffNLP RoBERTa (Cloud)")
print("✅ Keyword Extraction: ML6Team KeyBERT (Cloud)")
print("🚀 Server running on: http://localhost:8000")
print("📊 Ready to process dashboard analysis requests!")
print("="*60 + "\n")

@app.get("/")
async def root():
    return {
        "message": "Review Radar - Cloud ML API",
        "status": "healthy",
        "version": "2.0.0-cloud",
        "endpoints": {
            "dashboard": "/analyze-dashboard - Main endpoint for dashboard analysis"
        }
    }

@app.get("/health")
async def health_check():
    hf_configured = bool(HF_API_KEY)
    
    return {
        "status": "healthy",
        "service_type": "cloud_ml",
        "huggingface_configured": hf_configured,
        "models": {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "keywords": "ml6team/keyphrase-extraction-kbir-inspec"
        }
    }

@app.post("/analyze-dashboard", response_model=DashboardResponse)
async def analyze_dashboard_endpoint(request: DashboardRequest):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 200:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 200 reviews per request.")
    
    try:
        start_time = time.time()
        print("\n" + "📥" * 40)
        print("📥 RECEIVED REQUEST FROM WEB_SCRAPER.PY!")
        print("📥" * 40)
        print(f"📊 Processing {len(request.reviews)} reviews using CLOUD ML APIs...")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        print("-" * 80)
        
        logger.info(f"Processing {len(request.reviews)} reviews for dashboard analysis")
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        positive_comments = []
        negative_comments = []
        total_rating = 0
        valid_reviews = 0
        
        for review_item in request.reviews:
            if not review_item.comments.strip():
                continue
            
            sentiment, confidence = analyze_sentiment_cloud(review_item.comments)
            
            sentiment_counts[sentiment] += 1
            total_rating += review_item.rating
            valid_reviews += 1
            
            if sentiment == "positive":
                positive_comments.append(review_item.comments)
            elif sentiment == "negative":
                negative_comments.append(review_item.comments)
        
        positive_keywords = []
        negative_keywords = []
        
        if positive_comments:
            pos_text = " ".join(positive_comments[:50])
            pos_kw = extract_keywords_cloud(pos_text, 10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in pos_kw if kw[0]]
        
        if negative_comments:
            neg_text = " ".join(negative_comments[:50])
            neg_kw = extract_keywords_cloud(neg_text, 10)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in neg_kw if kw[0]]
        
        average_rating = total_rating / valid_reviews if valid_reviews > 0 else 0
        
        response = DashboardResponse(
            checklist=ChecklistData(
                sentiment=SentimentChecklist(
                    positive=sentiment_counts["positive"],
                    neutral=sentiment_counts["neutral"],
                    negative=sentiment_counts["negative"]
                )
            ),
            keywords=KeywordData(
                positive_keywords=positive_keywords,
                negative_keywords=negative_keywords
            ),
            total_reviews=valid_reviews,
            average_rating=round(average_rating, 2)
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Dashboard analysis completed in {processing_time:.2f} seconds")
        
        print("\n✅ ANALYSIS COMPLETE - SENDING TO FRONTEND!")
        print("="*50)
        print(f"📈 Sentiment Analysis: {response.checklist.sentiment.positive}+ | {response.checklist.sentiment.neutral}≈ | {response.checklist.sentiment.negative}-")
        print(f"🔑 Keywords Extracted: {len(response.keywords.positive_keywords)} positive, {len(response.keywords.negative_keywords)} negative")
        print(f"⚡ Processing Time: {processing_time:.2f} seconds")
        print(f"☁️  Powered by Hugging Face Inference API")
        print(f"🎯 Frontend will redirect to /dashboard")
        print("="*50 + "\n")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in dashboard analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in dashboard analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
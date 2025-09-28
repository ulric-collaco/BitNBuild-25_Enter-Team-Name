from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import logging
import time
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Review Radar - Sentiment Analysis API",
    description="FastAPI service for sentiment analysis and keyword extraction",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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

# Global model variables
sentiment_tokenizer = None
sentiment_model = None
keyword_model = None

# Model configuration
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
KEYWORD_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

# Words to avoid (too generic for any product)
GENERIC_WORDS = {'thing', 'stuff', 'item', 'product', 'good', 'bad', 'nice', 'ok', 'okay'}

def load_models():
    """Load sentiment analysis and keyword extraction models"""
    global sentiment_tokenizer, sentiment_model, keyword_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model.to(device)
        logger.info("Sentiment model loaded successfully")
        
        logger.info(f"Loading keyword extraction model: {KEYWORD_MODEL_NAME}")
        sentence_model = SentenceTransformer(KEYWORD_MODEL_NAME)
        keyword_model = KeyBERT(model=sentence_model)
        logger.info("Keyword extraction model loaded successfully")
        
        logger.info("All models loaded successfully")
        print("\n" + "="*60)
        print("🎉 REVIEW RADAR - MAIN API SERVER READY!")
        print("="*60)
        print("✅ Sentiment Analysis Model: CardiffNLP RoBERTa (LOADED)")
        print("✅ Keyword Extraction Model: KeyBERT + MiniLM (LOADED)")
        print("🚀 Server running on: http://localhost:8000")
        print("📊 Ready to process dashboard analysis requests!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        print("❌ FAILED TO START - Models could not be loaded!")
        raise e

def preprocess_text(text: str) -> str:
    """Basic text preprocessing for sentiment analysis"""
    text = text.replace('@', '@user')
    text = text.replace('http', 'http')
    return text

def analyze_sentiment(text: str) -> tuple:
    """Analyze sentiment of a text and return sentiment label and confidence"""
    try:
        processed_text = preprocess_text(text)
        
        inputs = sentiment_tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
        
        device = next(sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            predictions = outputs.logits.cpu().numpy()[0]
            
        probabilities = softmax(predictions)
        predicted_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_idx])
        
        sentiment = SENTIMENT_LABELS[predicted_idx]
        
        return sentiment, confidence
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return "neutral", 0.0

def extract_repeated_keywords(comments: List[str], top_k: int = 10) -> List[tuple]:
    """Extract repeated keywords/phrases from a list of comments"""
    if not comments:
        return []
    
    try:
        # Combine all comments
        combined_text = " ".join(comments)
        
        # Extract keywords using KeyBERT (correct API)
        keywords = keyword_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 3),  # 1 to 3 word phrases  
            stop_words='english',
            use_mmr=True,  # Use Maximal Marginal Relevance for diversity
            diversity=0.5
        )
        
        # Filter out generic words and short phrases
        filtered_keywords = []
        for keyword, score in keywords:
            if (len(keyword) > 2 and 
                not any(generic in keyword.lower() for generic in GENERIC_WORDS) and
                not keyword.lower().isdigit()):
                filtered_keywords.append((keyword, score))
        
        # Return top results
        return filtered_keywords[:top_k]
        
    except Exception as e:
        # Fallback to simple approach if KeyBERT fails
        logger.error(f"KeyBERT failed: {str(e)}, trying simple extraction")
        try:
            combined_text = " ".join(comments)
            simple_keywords = keyword_model.extract_keywords(combined_text)
            return simple_keywords[:top_k] if simple_keywords else []
        except Exception as e2:
            logger.error(f"Simple extraction also failed: {str(e2)}")
            return []

# Load models on startup
load_models()

@app.get("/")
async def root():
    return {
        "message": "Review Radar - Sentiment Analysis API",
        "status": "healthy",
        "version": "2.0.0",
        "endpoints": {
            "dashboard": "/analyze-dashboard - Main endpoint for dashboard analysis"
        }
    }

@app.get("/health")
async def health_check():
    sentiment_loaded = sentiment_model is not None and sentiment_tokenizer is not None
    keyword_loaded = keyword_model is not None
    
    return {
        "status": "healthy" if sentiment_loaded and keyword_loaded else "unhealthy",
        "models": {
            "sentiment_model": SENTIMENT_MODEL_NAME,
            "keyword_model": KEYWORD_MODEL_NAME
        },
        "models_loaded": {
            "sentiment": sentiment_loaded,
            "keywords": keyword_loaded
        }
    }

@app.post("/analyze-dashboard", response_model=DashboardResponse)
async def analyze_dashboard_endpoint(request: DashboardRequest):
    """Main endpoint for dashboard analysis - processes reviews and returns sentiment counts and keywords"""
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not keyword_model:
        raise HTTPException(status_code=503, detail="Keyword extraction model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 200:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 200 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Processing {len(request.reviews)} reviews for dashboard analysis")
        
        # Initialize counters
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        positive_comments = []
        negative_comments = []
        total_rating = 0
        valid_reviews = 0
        
        # Process each review
        for review_item in request.reviews:
            if not review_item.comments.strip():
                continue
            
            # Analyze sentiment
            sentiment, confidence = analyze_sentiment(review_item.comments)
            
            # Update counters
            sentiment_counts[sentiment] += 1
            total_rating += review_item.rating
            valid_reviews += 1
            
            # Collect comments for keyword extraction
            if sentiment == "positive":
                positive_comments.append(review_item.comments)
            elif sentiment == "negative":
                negative_comments.append(review_item.comments)
        
        # Extract keywords from positive and negative comments
        positive_keywords = []
        negative_keywords = []
        
        if positive_comments:
            pos_kw = extract_repeated_keywords(positive_comments, top_k=10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in pos_kw]
        
        if negative_comments:
            neg_kw = extract_repeated_keywords(negative_comments, top_k=10)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in neg_kw]
        
        # Calculate average rating
        average_rating = total_rating / valid_reviews if valid_reviews > 0 else 0
        
        # Build response
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
        
        return response
        
    except Exception as e:
        logger.error(f"Error in dashboard analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in dashboard analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
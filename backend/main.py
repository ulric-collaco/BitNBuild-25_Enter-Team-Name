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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Review Radar - Sentiment & Emotion Analysis API",
    description="FastAPI service for sentiment and emotion analysis using CardiffNLP RoBERTa and J-Hartmann DistilRoBERTa models",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ReviewRequest(BaseModel):
    reviews: List[str]

class IndividualResult(BaseModel):
    review: str
    sentiment: str
    confidence: float

class SentimentCounts(BaseModel):
    positive: int
    neutral: int
    negative: int

class SentimentResponse(BaseModel):
    sentiment_counts: SentimentCounts
    individual_results: List[IndividualResult]

class EmotionIndividualResult(BaseModel):
    review: str
    emotion: str
    confidence: float
    all_scores: dict

class EmotionSummary(BaseModel):
    anger: int
    disgust: int
    fear: int
    joy: int
    neutral: int
    sadness: int
    surprise: int

class EmotionResponse(BaseModel):
    emotion_summary: EmotionSummary
    individual_results: List[EmotionIndividualResult]

class FullAnalysisResult(BaseModel):
    review: str
    sentiment: str
    sentiment_confidence: float
    emotion: str
    emotion_confidence: float

class FullAnalysisResponse(BaseModel):
    sentiment_counts: SentimentCounts
    emotion_summary: EmotionSummary
    individual_results: List[FullAnalysisResult]

# Global variables for models and tokenizers
sentiment_tokenizer = None
sentiment_model = None
emotion_tokenizer = None
emotion_model = None

# Model configurations
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

def load_models():
    """Load both sentiment and emotion analysis models and tokenizers"""
    global sentiment_tokenizer, sentiment_model, emotion_tokenizer, emotion_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load sentiment model
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model.to(device)
        logger.info("Sentiment model loaded successfully")
        
        # Load emotion model
        logger.info(f"Loading emotion model: {EMOTION_MODEL_NAME}")
        emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        emotion_model.to(device)
        logger.info("Emotion model loaded successfully")
        
        logger.info(f"Both models loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

def preprocess_text(text: str) -> str:
    """Preprocess text for the model"""
    # Basic preprocessing for Twitter RoBERTa model
    text = text.replace('@', '@user')  # Replace mentions
    text = text.replace('http', 'http')  # Keep URLs as is
    return text

def analyze_sentiment(text: str) -> tuple:
    """Analyze sentiment of a single text"""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and encode
        inputs = sentiment_tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(sentiment_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            predictions = outputs.logits.cpu().numpy()[0]
        
        # Apply softmax to get probabilities
        probabilities = softmax(predictions)
        
        # Get the predicted label and confidence
        predicted_idx = np.argmax(probabilities)
        sentiment = SENTIMENT_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return sentiment, confidence
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for text: {text[:50]}... Error: {str(e)}")
        return "neutral", 0.0

def analyze_emotion(text: str) -> tuple:
    """Analyze emotion of a single text"""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize and encode
        inputs = emotion_tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(emotion_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            predictions = outputs.logits.cpu().numpy()[0]
        
        # Apply softmax to get probabilities
        probabilities = softmax(predictions)
        
        # Get the predicted label and confidence
        predicted_idx = np.argmax(probabilities)
        emotion = EMOTION_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create all scores dictionary
        all_scores = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}
        
        return emotion, confidence, all_scores
        
    except Exception as e:
        logger.error(f"Error analyzing emotion for text: {text[:50]}... Error: {str(e)}")
        return "neutral", 0.0, {label: 0.0 for label in EMOTION_LABELS}

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up the application...")
    load_models()
    logger.info("Application startup complete!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Review Radar Sentiment Analysis API",
        "status": "healthy",
        "sentiment_model": SENTIMENT_MODEL_NAME,
        "emotion_model": EMOTION_MODEL_NAME
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    sentiment_loaded = sentiment_model is not None and sentiment_tokenizer is not None
    emotion_loaded = emotion_model is not None and emotion_tokenizer is not None
    
    return {
        "status": "healthy",
        "sentiment_model_loaded": sentiment_loaded,
        "emotion_model_loaded": emotion_loaded,
        "device": str(next(sentiment_model.parameters()).device) if sentiment_model else "not loaded"
    }

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: ReviewRequest):
    """
    Analyze sentiment for a list of reviews
    
    - **reviews**: List of review text strings to analyze
    
    Returns sentiment counts and individual results for each review
    """
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Analyzing {len(request.reviews)} reviews")
        
        # Initialize counters
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        individual_results = []
        
        # Process each review
        for review in request.reviews:
            if not review.strip():  # Skip empty reviews
                continue
                
            sentiment, confidence = analyze_sentiment(review)
            
            # Update counters
            sentiment_counts[sentiment] += 1
            
            # Add to individual results
            individual_results.append(IndividualResult(
                review=review,
                sentiment=sentiment,
                confidence=round(confidence, 4)
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        response = SentimentResponse(
            sentiment_counts=SentimentCounts(**sentiment_counts),
            individual_results=individual_results
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing reviews: {str(e)}")

# Additional endpoint for single review analysis
@app.post("/analyze-single")
async def analyze_single_review(review: dict):
    """
    Analyze sentiment for a single review
    
    - **text**: Single review text string to analyze
    """
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    text = review.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No review text provided")
    
    try:
        sentiment, confidence = analyze_sentiment(text)
        
        return {
            "review": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        logger.error(f"Error processing single review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing review: {str(e)}")

@app.post("/analyze-emotions", response_model=EmotionResponse)
async def analyze_emotions_endpoint(request: ReviewRequest):
    """
    Analyze emotions for a list of reviews
    
    - **reviews**: List of review text strings to analyze for emotions
    
    Returns emotion counts and individual emotion results for each review
    """
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Analyzing emotions for {len(request.reviews)} reviews")
        
        # Initialize counters
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        
        # Process each review
        for review in request.reviews:
            if not review.strip():  # Skip empty reviews
                continue
                
            emotion, confidence, all_scores = analyze_emotion(review)
            
            # Update counters
            emotion_counts[emotion] += 1
            
            # Add to individual results
            individual_results.append(EmotionIndividualResult(
                review=review,
                emotion=emotion,
                confidence=round(confidence, 4),
                all_scores={k: round(v, 4) for k, v in all_scores.items()}
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"Emotion analysis completed in {processing_time:.2f} seconds")
        
        response = EmotionResponse(
            emotion_summary=EmotionSummary(**emotion_counts),
            individual_results=individual_results
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing emotion analysis: {str(e)}")

@app.post("/analyze-full", response_model=FullAnalysisResponse)
async def analyze_full_endpoint(request: ReviewRequest):
    """
    Analyze both sentiment and emotions for a list of reviews
    
    - **reviews**: List of review text strings to analyze for both sentiment and emotions
    
    Returns combined sentiment and emotion analysis for each review
    """
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Analyzing sentiment and emotions for {len(request.reviews)} reviews")
        
        # Initialize counters
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        
        # Process each review
        for review in request.reviews:
            if not review.strip():  # Skip empty reviews
                continue
                
            # Analyze both sentiment and emotion
            sentiment, sentiment_confidence = analyze_sentiment(review)
            emotion, emotion_confidence, _ = analyze_emotion(review)
            
            # Update counters
            sentiment_counts[sentiment] += 1
            emotion_counts[emotion] += 1
            
            # Add to individual results
            individual_results.append(FullAnalysisResult(
                review=review,
                sentiment=sentiment,
                sentiment_confidence=round(sentiment_confidence, 4),
                emotion=emotion,
                emotion_confidence=round(emotion_confidence, 4)
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"Full analysis completed in {processing_time:.2f} seconds")
        
        response = FullAnalysisResponse(
            sentiment_counts=SentimentCounts(**sentiment_counts),
            emotion_summary=EmotionSummary(**emotion_counts),
            individual_results=individual_results
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing full analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing full analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
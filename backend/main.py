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
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Review Radar - Sentiment & Emotion Analysis API",
    description="FastAPI service for sentiment and emotion analysis using CardiffNLP RoBERTa and J-Hartmann DistilRoBERTa models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class KeywordRequest(BaseModel):
    positive_reviews: List[str]
    negative_reviews: List[str]

class KeywordResult(BaseModel):
    keyword: str
    score: float

class KeywordResponse(BaseModel):
    positive_keywords: List[KeywordResult]
    negative_keywords: List[KeywordResult]

class CombinedAnalysisRequest(BaseModel):
    reviews: List[str]

class CombinedAnalysisResponse(BaseModel):
    sentiment_counts: SentimentCounts
    emotion_summary: EmotionSummary
    individual_results: List[FullAnalysisResult]
    positive_keywords: List[KeywordResult]
    negative_keywords: List[KeywordResult]

class ReviewItem(BaseModel):
    id: int
    rating: int
    review: str

class MixedReviewRequest(BaseModel):
    reviews: List[ReviewItem]

class AutoKeywordRequest(BaseModel):
    reviews: List[str]

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

class EmotionChecklist(BaseModel):
    joy: int
    anger: int
    sadness: int
    surprise: int
    fear: int
    disgust: int
    neutral: int

class ChecklistData(BaseModel):
    sentiment: SentimentChecklist
    emotions: EmotionChecklist

class KeywordData(BaseModel):
    positive_keywords: List[KeywordResult]
    negative_keywords: List[KeywordResult]

class DashboardResponse(BaseModel):
    checklist: ChecklistData
    keywords: KeywordData
    total_reviews: int
    average_rating: float

sentiment_tokenizer = None
sentiment_model = None
emotion_tokenizer = None
emotion_model = None
keyword_model = None

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
KEYWORD_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Words to avoid (too generic for any product)
GENERIC_WORDS = {'thing', 'stuff', 'item', 'product', 'good', 'bad', 'nice', 'ok', 'okay'}

def load_models():
    global sentiment_tokenizer, sentiment_model, emotion_tokenizer, emotion_model, keyword_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model.to(device)
        logger.info("Sentiment model loaded successfully")
        
        logger.info(f"Loading emotion model: {EMOTION_MODEL_NAME}")
        emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        emotion_model.to(device)
        logger.info("Emotion model loaded successfully")
        
        logger.info(f"Loading keyword extraction model: {KEYWORD_MODEL_NAME}")
        sentence_model = SentenceTransformer(KEYWORD_MODEL_NAME)
        keyword_model = KeyBERT(model=sentence_model)
        logger.info("Keyword extraction model loaded successfully")
        
        logger.info(f"All models loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

def preprocess_text(text: str) -> str:
    text = text.replace('@', '@user')
    text = text.replace('http', 'http')
    return text

def analyze_sentiment(text: str) -> tuple:
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
        sentiment = SENTIMENT_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return sentiment, confidence
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for text: {text[:50]}... Error: {str(e)}")
        return "neutral", 0.0

def analyze_emotion(text: str) -> tuple:
    try:
        processed_text = preprocess_text(text)
        
        inputs = emotion_tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
        
        device = next(emotion_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            predictions = outputs.logits.cpu().numpy()[0]
        
        probabilities = softmax(predictions)
        
        predicted_idx = np.argmax(probabilities)
        emotion = EMOTION_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        all_scores = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}
        
        return emotion, confidence, all_scores
        
    except Exception as e:
        logger.error(f"Error analyzing emotion for text: {text[:50]}... Error: {str(e)}")
        return "neutral", 0.0, {label: 0.0 for label in EMOTION_LABELS}

def extract_intelligent_keywords(texts: List[str], top_k: int = 12) -> List[tuple]:
    """Extract meaningful keywords and phrases from reviews using intelligent filtering"""
    try:
        if not texts or not keyword_model:
            return []
        
        combined_text = " ".join(texts)
        if not combined_text.strip():
            return []
        
        # Extract keywords with KeyBERT using multiple approaches for diversity
        raw_keywords = keyword_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 4),  # Allow 1-4 word phrases
            stop_words='english',
            top_k=30,  # Get more candidates for intelligent filtering
            use_maxsum=True,
            nr_candidates=100,
            diversity=0.9  # High diversity for varied keywords
        )
        
        # Intelligent filtering without product-specific bias
        filtered_keywords = []
        
        for keyword, score in raw_keywords:
            keyword_lower = keyword.lower().strip()
            keyword_words = keyword_lower.split()
            
            # Skip if contains overly generic words
            if any(generic in keyword_words for generic in GENERIC_WORDS):
                continue
            
            # Skip single-word generic adjectives/adverbs
            if len(keyword_words) == 1 and keyword_lower in {
                'good', 'bad', 'great', 'poor', 'nice', 'awful', 'terrible', 
                'amazing', 'excellent', 'perfect', 'horrible', 'fantastic',
                'wonderful', 'disappointing', 'satisfactory', 'adequate'
            }:
                continue
            
            # Skip very short or very long keywords
            if len(keyword) < 3 or len(keyword) > 50:
                continue
            
            # Boost score for multi-word phrases (they're usually more descriptive)
            if len(keyword_words) > 1:
                score = min(score + 0.1, 1.0)
            
            # Boost score for keywords that appear to describe specific features/issues
            descriptive_patterns = [
                r'\b(?:very|really|extremely|quite|pretty)\s+\w+',  # intensifiers
                r'\w+\s+(?:quality|performance|experience|issue|problem)',  # feature descriptions
                r'(?:easy|hard|difficult)\s+to\s+\w+',  # usability
                r'\w+\s+(?:than|compared)',  # comparisons
                r'(?:works|doesn\'t work|broken|fixed)',  # functionality
            ]
            
            for pattern in descriptive_patterns:
                if re.search(pattern, keyword_lower):
                    score = min(score + 0.05, 1.0)
                    break
            
            # Only include keywords with decent relevance
            if score > 0.25:  # Lower threshold for more diverse extraction
                filtered_keywords.append((keyword, score))
        
        # Remove duplicates and similar keywords
        unique_keywords = []
        seen_keywords = set()
        
        for keyword, score in filtered_keywords:
            keyword_clean = keyword_lower.strip()
            
            # Check for similar keywords (avoid near-duplicates)
            is_duplicate = False
            for seen in seen_keywords:
                if keyword_clean in seen or seen in keyword_clean:
                    # Keep the one with higher score
                    if score > 0.5:  # Only replace if significantly better
                        seen_keywords.discard(seen)
                        unique_keywords = [(k, s) for k, s in unique_keywords if k.lower().strip() != seen]
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_keywords.add(keyword_clean)
                unique_keywords.append((keyword, score))
        
        # Sort by relevance score and return top results
        unique_keywords.sort(key=lambda x: x[1], reverse=True)
        return unique_keywords[:top_k]
        
    except Exception as e:
        logger.error(f"Error extracting intelligent keywords: {str(e)}")
        return []



# Keep backward compatibility with original function name
def extract_keywords(texts: List[str], top_k: int = 10) -> List[tuple]:
    """Wrapper for backward compatibility"""
    return extract_intelligent_keywords(texts, top_k)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    load_models()
    logger.info("Application startup complete!")

@app.get("/")
async def root():
    return {
        "message": "Review Radar - Universal Product Review Analysis API",
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "sentiment_analysis": "3-class sentiment classification",
            "emotion_detection": "7-emotion recognition",
            "keyword_extraction": "Universal product keyword extraction",
            "dashboard_api": "Frontend-ready structured analysis"
        },
        "models": {
            "sentiment_model": SENTIMENT_MODEL_NAME,
            "emotion_model": EMOTION_MODEL_NAME,
            "keyword_model": KEYWORD_MODEL_NAME
        },
        "main_endpoints": {
            "dashboard": "/analyze-dashboard - Main endpoint for frontend integration",
            "sentiment": "/analyze-sentiment - Basic sentiment analysis",
            "emotions": "/analyze-emotions - Emotion detection",
            "keywords": "/extract-keywords - Keyword extraction",
            "combined": "/analyze-reviews - Full analysis pipeline"
        }
    }

@app.get("/health")
async def health_check():
    sentiment_loaded = sentiment_model is not None and sentiment_tokenizer is not None
    emotion_loaded = emotion_model is not None and emotion_tokenizer is not None
    keyword_loaded = keyword_model is not None
    
    return {
        "status": "healthy",
        "sentiment_model_loaded": sentiment_loaded,
        "emotion_model_loaded": emotion_loaded,
        "keyword_model_loaded": keyword_loaded,
        "device": str(next(sentiment_model.parameters()).device) if sentiment_model else "not loaded"
    }

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(request: ReviewRequest):
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")

    try:
        start_time = time.time()
        logger.info(f"Analyzing {len(request.reviews)} reviews")
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        individual_results = []
        
        for review in request.reviews:
            if not review.strip():
                continue
                
            sentiment, confidence = analyze_sentiment(review)
            
            sentiment_counts[sentiment] += 1
            
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

@app.post("/analyze-single")
async def analyze_single_review(review: dict):
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
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Analyzing emotions for {len(request.reviews)} reviews")
        
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        
        for review in request.reviews:
            if not review.strip():
                continue
                
            emotion, confidence, all_scores = analyze_emotion(review)
            
            emotion_counts[emotion] += 1
            
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
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Analyzing sentiment and emotions for {len(request.reviews)} reviews")
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        
        for review in request.reviews:
            if not review.strip():
                continue
                
            sentiment, sentiment_confidence = analyze_sentiment(review)
            emotion, emotion_confidence, _ = analyze_emotion(review)
            
            sentiment_counts[sentiment] += 1
            emotion_counts[emotion] += 1
            
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

@app.post("/extract-keywords", response_model=KeywordResponse)
async def extract_keywords_endpoint(request: KeywordRequest):
    if not keyword_model:
        raise HTTPException(status_code=503, detail="Keyword extraction model not loaded. Please try again later.")
    
    if not request.positive_reviews and not request.negative_reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    try:
        start_time = time.time()
        logger.info(f"Extracting keywords from {len(request.positive_reviews)} positive and {len(request.negative_reviews)} negative reviews")
        
        positive_keywords = []
        negative_keywords = []
        
        if request.positive_reviews:
            pos_kw = extract_intelligent_keywords(request.positive_reviews, top_k=12)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in pos_kw]
        
        if request.negative_reviews:
            neg_kw = extract_intelligent_keywords(request.negative_reviews, top_k=12)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in neg_kw]
        
        processing_time = time.time() - start_time
        logger.info(f"Keyword extraction completed in {processing_time:.2f} seconds")
        
        return KeywordResponse(
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting keywords: {str(e)}")

@app.post("/analyze-reviews", response_model=CombinedAnalysisResponse)
async def analyze_reviews_endpoint(request: CombinedAnalysisRequest):
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not keyword_model:
        raise HTTPException(status_code=503, detail="Keyword extraction model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Performing combined analysis on {len(request.reviews)} reviews")
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        positive_reviews = []
        negative_reviews = []
        
        for review in request.reviews:
            if not review.strip():
                continue
                
            sentiment, sentiment_confidence = analyze_sentiment(review)
            emotion, emotion_confidence, _ = analyze_emotion(review)
            
            sentiment_counts[sentiment] += 1
            emotion_counts[emotion] += 1
            
            individual_results.append(FullAnalysisResult(
                review=review,
                sentiment=sentiment,
                sentiment_confidence=round(sentiment_confidence, 4),
                emotion=emotion,
                emotion_confidence=round(emotion_confidence, 4)
            ))
            
            if sentiment == "positive":
                positive_reviews.append(review)
            elif sentiment == "negative":
                negative_reviews.append(review)
        
        positive_keywords = []
        negative_keywords = []
        
        if positive_reviews:
            pos_kw = extract_keywords(positive_reviews, top_k=10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in pos_kw]
        
        if negative_reviews:
            neg_kw = extract_keywords(negative_reviews, top_k=10)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in neg_kw]
        
        processing_time = time.time() - start_time
        logger.info(f"Combined analysis completed in {processing_time:.2f} seconds")
        
        return CombinedAnalysisResponse(
            sentiment_counts=SentimentCounts(**sentiment_counts),
            emotion_summary=EmotionSummary(**emotion_counts),
            individual_results=individual_results,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in combined analysis: {str(e)}")

@app.post("/analyze-mixed-reviews", response_model=CombinedAnalysisResponse)
async def analyze_mixed_reviews_endpoint(request: MixedReviewRequest):
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
    if not keyword_model:
        raise HTTPException(status_code=503, detail="Keyword extraction model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    if len(request.reviews) > 100:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 100 reviews per request.")
    
    try:
        start_time = time.time()
        logger.info(f"Performing mixed review analysis on {len(request.reviews)} reviews")
        
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        individual_results = []
        positive_reviews = []
        negative_reviews = []
        
        for review_item in request.reviews:
            review_text = review_item.review
            if not review_text.strip():
                continue
                
            sentiment, sentiment_confidence = analyze_sentiment(review_text)
            emotion, emotion_confidence, _ = analyze_emotion(review_text)
            
            sentiment_counts[sentiment] += 1
            emotion_counts[emotion] += 1
            
            individual_results.append(FullAnalysisResult(
                review=review_text,
                sentiment=sentiment,
                sentiment_confidence=round(sentiment_confidence, 4),
                emotion=emotion,
                emotion_confidence=round(emotion_confidence, 4)
            ))
            
            if sentiment == "positive":
                positive_reviews.append(review_text)
            elif sentiment == "negative":
                negative_reviews.append(review_text)
        
        positive_keywords = []
        negative_keywords = []
        
        if positive_reviews:
            pos_kw = extract_keywords(positive_reviews, top_k=10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in pos_kw]
        
        if negative_reviews:
            neg_kw = extract_keywords(negative_reviews, top_k=10)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in neg_kw]
        
        processing_time = time.time() - start_time
        logger.info(f"Mixed review analysis completed in {processing_time:.2f} seconds")
        
        return CombinedAnalysisResponse(
            sentiment_counts=SentimentCounts(**sentiment_counts),
            emotion_summary=EmotionSummary(**emotion_counts),
            individual_results=individual_results,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )
        
    except Exception as e:
        logger.error(f"Error in mixed review analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in mixed review analysis: {str(e)}")

@app.post("/extract-keywords-auto", response_model=KeywordResponse)
async def extract_keywords_auto_endpoint(request: AutoKeywordRequest):
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not keyword_model:
        raise HTTPException(status_code=503, detail="Keyword extraction model not loaded. Please try again later.")
    
    if not request.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")
    
    try:
        start_time = time.time()
        logger.info(f"Auto-extracting keywords from {len(request.reviews)} reviews")
        
        positive_reviews = []
        negative_reviews = []
        
        for review_text in request.reviews:
            if not review_text.strip():
                continue
                
            sentiment, _ = analyze_sentiment(review_text)
            
            if sentiment == "positive":
                positive_reviews.append(review_text)
            elif sentiment == "negative":
                negative_reviews.append(review_text)
        
        positive_keywords = []
        negative_keywords = []
        
        if positive_reviews:
            pos_kw = extract_keywords(positive_reviews, top_k=10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in pos_kw]
        
        if negative_reviews:
            neg_kw = extract_keywords(negative_reviews, top_k=10)
            negative_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 4)) for kw in neg_kw]
        
        processing_time = time.time() - start_time
        logger.info(f"Auto keyword extraction completed in {processing_time:.2f} seconds")
        
        return KeywordResponse(
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )
        
    except Exception as e:
        logger.error(f"Error in auto keyword extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in auto keyword extraction: {str(e)}")

@app.post("/analyze-dashboard", response_model=DashboardResponse)
async def analyze_dashboard_endpoint(request: DashboardRequest):
    """Main endpoint for dashboard visualization - processes reviews with id, name, rating, comments"""
    if not sentiment_model or not sentiment_tokenizer:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded. Please try again later.")
    
    if not emotion_model or not emotion_tokenizer:
        raise HTTPException(status_code=503, detail="Emotion model not loaded. Please try again later.")
    
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
        emotion_counts = {label: 0 for label in EMOTION_LABELS}
        positive_comments = []
        negative_comments = []
        total_rating = 0
        valid_reviews = 0
        
        # Process each review
        for review_item in request.reviews:
            if not review_item.comments.strip():
                continue
            
            # Analyze sentiment and emotion
            sentiment, sentiment_confidence = analyze_sentiment(review_item.comments)
            emotion, emotion_confidence, _ = analyze_emotion(review_item.comments)
            
            # Update counters
            sentiment_counts[sentiment] += 1
            emotion_counts[emotion] += 1
            total_rating += review_item.rating
            valid_reviews += 1
            
            # Collect comments for keyword extraction
            if sentiment == "positive":
                positive_comments.append(review_item.comments)
            elif sentiment == "negative":
                negative_comments.append(review_item.comments)
        
        # Extract keywords
        positive_keywords = []
        negative_keywords = []
        
        if positive_comments:
            pos_kw = extract_intelligent_keywords(positive_comments, top_k=10)
            positive_keywords = [KeywordResult(keyword=kw[0], score=round(kw[1], 3)) for kw in pos_kw]
        
        if negative_comments:
            neg_kw = extract_intelligent_keywords(negative_comments, top_k=10)
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
                ),
                emotions=EmotionChecklist(
                    joy=emotion_counts["joy"],
                    anger=emotion_counts["anger"],
                    sadness=emotion_counts["sadness"],
                    surprise=emotion_counts["surprise"],
                    fear=emotion_counts["fear"],
                    disgust=emotion_counts["disgust"],
                    neutral=emotion_counts["neutral"]
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
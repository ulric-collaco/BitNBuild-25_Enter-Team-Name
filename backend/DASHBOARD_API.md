# Review Radar Dashboard API Documentation

## 🎯 Main Dashboard Endpoint

### **POST** `/analyze-dashboard`

The primary endpoint designed for frontend dashboard integration. Processes customer reviews and returns structured data optimized for meters, charts, and visualization components.

## 📥 Input Format

```json
{
  "reviews": [
    {
      "id": "r1",
      "name": "Alice", 
      "rating": 5,
      "comments": "Amazing camera and great battery life."
    },
    {
      "id": "r2",
      "name": "Bob",
      "rating": 2, 
      "comments": "Cheap quality, broke quickly."
    }
  ]
}
```

### Input Fields:
- **`id`**: Unique identifier for the review (string)
- **`name`**: Reviewer's name or anonymized ID (string)  
- **`rating`**: Numerical rating 1-5 stars (integer)
- **`comments`**: Textual review content (string)

## 📤 Output Format

```json
{
  "checklist": {
    "sentiment": {
      "positive": 60,
      "neutral": 25, 
      "negative": 15
    },
    "emotions": {
      "joy": 45,
      "anger": 10,
      "sadness": 5,
      "surprise": 20,
      "fear": 3,
      "disgust": 2,
      "neutral": 15
    }
  },
  "keywords": {
    "positive_keywords": [
      {"keyword": "great battery life", "score": 0.92},
      {"keyword": "amazing camera", "score": 0.89}
    ],
    "negative_keywords": [
      {"keyword": "cheap quality", "score": 0.87},
      {"keyword": "broke quickly", "score": 0.80}
    ]
  },
  "total_reviews": 100,
  "average_rating": 3.4
}
```

## 🎨 Frontend Integration Guide

### Sentiment Meters
```javascript
// Use checklist.sentiment for progress bars/pie charts
const sentimentData = response.checklist.sentiment;
const total = sentimentData.positive + sentimentData.neutral + sentimentData.negative;

const positivePercent = (sentimentData.positive / total) * 100;
const neutralPercent = (sentimentData.neutral / total) * 100; 
const negativePercent = (sentimentData.negative / total) * 100;
```

### Emotion Distribution
```javascript
// Use checklist.emotions for emotion meters
const emotions = response.checklist.emotions;
const emotionTotal = Object.values(emotions).reduce((a, b) => a + b, 0);

Object.entries(emotions).forEach(([emotion, count]) => {
  const percentage = (count / emotionTotal) * 100;
  // Update emotion meter for this emotion
});
```

### Keyword Tags
```javascript
// Use keywords for tag clouds or keyword lists
const positiveKeywords = response.keywords.positive_keywords;
const negativeKeywords = response.keywords.negative_keywords;

// Create tag elements with confidence-based styling
positiveKeywords.forEach(kw => {
  const opacity = kw.score; // Use score for tag opacity/size
  createTag(kw.keyword, 'positive', opacity);
});
```

### Summary Statistics
```javascript
// Use for overview cards
const stats = {
  totalReviews: response.total_reviews,
  averageRating: response.average_rating,
  satisfaction: (response.checklist.sentiment.positive / response.total_reviews) * 100
};
```

## 🔧 Technical Features

### Universal Product Support
- Works with **any product category** (electronics, clothing, books, services, etc.)
- No product-specific bias or constraints
- Organic keyword discovery from actual review content

### Intelligent Processing
- **Sentiment Analysis**: 3-class classification (positive/neutral/negative)
- **Emotion Detection**: 7-emotion recognition (joy, anger, sadness, surprise, fear, disgust, neutral)
- **Keyword Extraction**: Universal algorithm with confidence scoring
- **Smart Filtering**: Removes generic terms, boosts descriptive phrases

### Performance Optimized
- Handles up to **200 reviews** per request
- GPU acceleration (CUDA) when available
- Efficient processing with combined analysis pipeline
- Average processing time: 2-5 seconds for 100 reviews

## 📊 Response Structure Details

### Checklist Data
Perfect for frontend meters and progress bars:
- **Sentiment counts**: Raw numbers for each sentiment category
- **Emotion counts**: Distribution across 7 emotion categories
- **Ready for percentage calculations**: Just divide by totals

### Keywords Data  
Optimized for tag clouds and keyword visualization:
- **Confidence scores**: 0.0 to 1.0 for styling/sizing
- **Meaningful phrases**: Multi-word expressions preserved
- **Separate positive/negative**: Easy color coding

### Summary Statistics
Essential dashboard metrics:
- **Total reviews**: Count of processed reviews
- **Average rating**: Calculated from input ratings
- **Easy satisfaction metrics**: Derive from sentiment counts

## 🚀 Quick Start

1. **Start the server**:
   ```bash
   python main.py
   ```

2. **Test the endpoint**:
   ```bash
   python test_dashboard_api.py
   ```

3. **Frontend integration**:
   ```javascript
   fetch('/analyze-dashboard', {
     method: 'POST',
     headers: {'Content-Type': 'application/json'},
     body: JSON.stringify({reviews: reviewData})
   })
   .then(response => response.json())
   .then(data => {
     // Use data.checklist for meters
     // Use data.keywords for tags
     // Use data.total_reviews, data.average_rating for stats
   });
   ```

## 🎯 Use Cases

### E-commerce Dashboard
- Product review sentiment analysis
- Customer satisfaction metrics
- Feature-specific feedback extraction

### Service Quality Monitoring  
- Customer experience tracking
- Issue identification and trending
- Service improvement insights

### Market Research
- Consumer opinion analysis
- Product comparison insights
- Competitive analysis

### Content Moderation
- Review quality assessment
- Spam/fake review detection
- Sentiment-based filtering

## 🔍 Additional Endpoints

The API also provides specialized endpoints for advanced use cases:

- **`/analyze-sentiment`**: Basic sentiment analysis only
- **`/analyze-emotions`**: Emotion detection only  
- **`/extract-keywords`**: Keyword extraction with pre-categorized reviews
- **`/analyze-reviews`**: Full pipeline analysis
- **`/analyze-mixed-reviews`**: Handles mixed review formats

## 💡 Best Practices

1. **Batch Processing**: Send multiple reviews in one request for efficiency
2. **Error Handling**: Check for HTTP 503 (models not loaded) and retry
3. **Rate Limiting**: Respect the 200 review per request limit
4. **Caching**: Cache results for identical review sets
5. **Confidence Scoring**: Use keyword scores for visual emphasis

---

**Ready for Production**: This API is designed for seamless frontend integration with structured, predictable responses optimized for dashboard visualization and user interface components.
#!/usr/bin/env python3
"""
Test script for dashboard-focused review analysis API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_dashboard_api():
    """Test the new dashboard API with the specified input format"""
    
    # Sample data matching your exact specification
    test_data = {
        "reviews": [
            {"id": "r1", "name": "Alice", "rating": 5, "comments": "Amazing camera and great battery life."},
            {"id": "r2", "name": "Bob", "rating": 2, "comments": "Cheap quality, broke quickly."},
            {"id": "r3", "name": "Charlie", "rating": 4, "comments": "Good performance but expensive price."},
            {"id": "r4", "name": "Diana", "rating": 1, "comments": "Terrible build quality, feels weak and flimsy."},
            {"id": "r5", "name": "Eve", "rating": 5, "comments": "Excellent value for money, fast charging works great."},
            {"id": "r6", "name": "Frank", "rating": 3, "comments": "Average product, nothing special but works fine."},
            {"id": "r7", "name": "Grace", "rating": 2, "comments": "Slow performance and poor customer service."},
            {"id": "r8", "name": "Henry", "rating": 5, "comments": "Outstanding quality and durable construction."},
            {"id": "r9", "name": "Iris", "rating": 1, "comments": "Broke after one week, very disappointing experience."},
            {"id": "r10", "name": "Jack", "rating": 4, "comments": "Good features but could be more user-friendly."}
        ]
    }
    
    print("🎯 Testing Dashboard API with Specified Format")
    print("=" * 55)
    print(f"📊 Processing {len(test_data['reviews'])} sample reviews")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-dashboard", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\\n✅ Dashboard Analysis Results:")
            print("=" * 40)
            
            # Display checklist data for frontend meters
            checklist = result["checklist"]
            
            print("📈 SENTIMENT DISTRIBUTION (for frontend meters):")
            sentiment = checklist["sentiment"]
            total_sentiment = sentiment["positive"] + sentiment["neutral"] + sentiment["negative"]
            
            if total_sentiment > 0:
                pos_pct = (sentiment["positive"] / total_sentiment) * 100
                neu_pct = (sentiment["neutral"] / total_sentiment) * 100
                neg_pct = (sentiment["negative"] / total_sentiment) * 100
                
                print(f"   • Positive: {sentiment['positive']} reviews ({pos_pct:.1f}%)")
                print(f"   • Neutral:  {sentiment['neutral']} reviews ({neu_pct:.1f}%)")
                print(f"   • Negative: {sentiment['negative']} reviews ({neg_pct:.1f}%)")
            
            print("\\n🎭 EMOTION DISTRIBUTION (for frontend meters):")
            emotions = checklist["emotions"]
            total_emotions = sum(emotions.values())
            
            emotion_list = [
                ("Joy", emotions["joy"]),
                ("Anger", emotions["anger"]),
                ("Sadness", emotions["sadness"]),
                ("Surprise", emotions["surprise"]),
                ("Fear", emotions["fear"]),
                ("Disgust", emotions["disgust"]),
                ("Neutral", emotions["neutral"])
            ]
            
            for emotion_name, count in emotion_list:
                if count > 0:
                    pct = (count / total_emotions) * 100 if total_emotions > 0 else 0
                    print(f"   • {emotion_name}: {count} reviews ({pct:.1f}%)")
            
            # Display keywords for frontend visualization
            keywords = result["keywords"]
            
            print("\\n✅ TOP POSITIVE KEYWORDS (for frontend tags):")
            for kw in keywords["positive_keywords"][:8]:
                print(f"   • \"{kw['keyword']}\" (confidence: {kw['score']})")
            
            print("\\n❌ TOP NEGATIVE KEYWORDS (for frontend tags):")
            for kw in keywords["negative_keywords"][:8]:
                print(f"   • \"{kw['keyword']}\" (confidence: {kw['score']})")
            
            # Display summary stats
            print("\\n📊 SUMMARY STATISTICS:")
            print(f"   • Total Reviews Processed: {result['total_reviews']}")
            print(f"   • Average Rating: {result['average_rating']}/5.0")
            
            # Show raw JSON structure for frontend integration
            print("\\n🔧 FRONTEND INTEGRATION:")
            print("Raw JSON structure perfect for frontend meters and visualization:")
            print(json.dumps(result, indent=2)[:500] + "..." if len(json.dumps(result, indent=2)) > 500 else json.dumps(result, indent=2))
            
            print("\\n🎉 Dashboard API Test Successful!")
            print("✨ Response format is ready for frontend dashboard integration!")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
        print("Run: python main.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_dashboard_edge_cases():
    """Test edge cases for the dashboard API"""
    
    print("\\n🧪 Testing Dashboard API Edge Cases")
    print("=" * 45)
    
    # Test with minimal data
    minimal_data = {
        "reviews": [
            {"id": "min1", "name": "User1", "rating": 3, "comments": "Okay product."}
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-dashboard", json=minimal_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Minimal data test passed")
            print(f"   • Processed {result['total_reviews']} review")
            print(f"   • Average rating: {result['average_rating']}")
        else:
            print(f"❌ Minimal data test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Minimal data test error: {str(e)}")
    
    # Test with empty comments
    empty_comments_data = {
        "reviews": [
            {"id": "empty1", "name": "User2", "rating": 4, "comments": ""},
            {"id": "empty2", "name": "User3", "rating": 2, "comments": "   "},
            {"id": "valid1", "name": "User4", "rating": 5, "comments": "Great product!"}
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-dashboard", json=empty_comments_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Empty comments handling test passed")
            print(f"   • Processed {result['total_reviews']} valid reviews out of 3")
        else:
            print(f"❌ Empty comments test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Empty comments test error: {str(e)}")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("🚀 Starting Dashboard API Tests")
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    test_dashboard_api()
    test_dashboard_edge_cases()
    
    print("\\n✨ Dashboard API testing completed!")
    print("🎯 API is ready for frontend dashboard integration!")
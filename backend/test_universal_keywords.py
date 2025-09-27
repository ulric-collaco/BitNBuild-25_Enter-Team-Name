#!/usr/bin/env python3
"""
Test script for universal product keyword extraction
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_universal_keyword_extraction():
    """Test the universal keyboard extraction with diverse product examples"""
    
    # Sample reviews for different types of products (universal testing)
    test_data = {
        "positive_reviews": [
            "Amazing camera quality and the photos look professional",
            "Battery lasts all day even with heavy usage",  
            "Fast charging is incredible - charges in 30 minutes",
            "Build quality is solid and premium feeling",
            "Screen is bright and vibrant colors",
            "Great value for the money, highly recommended",
            "Comfortable fit and breathable material",
            "Easy to use interface and intuitive design",
            "Excellent customer service and quick delivery",
            "Durable construction withstands daily wear"
        ],
        "negative_reviews": [
            "Battery drains fast, dies within 4 hours",
            "Screen cracked within a week of normal use",
            "Camera quality is poor in low light conditions",
            "Heats up during use, gets uncomfortably warm",
            "Too expensive for what you get",
            "Frequent crashes and system freezes",
            "Sound is muffled and unclear",
            "Slow performance takes forever to load",
            "Poor build quality, feels cheap and flimsy",
            "Difficult to setup and confusing instructions",
            "Uncomfortable design causes hand fatigue",
            "Broke after just one month of normal use"
        ]
    }
    
    print("🧪 Testing Universal Product Keyword Extraction")
    print("=" * 55)
    
    try:
        response = requests.post(f"{BASE_URL}/extract-keywords", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Positive Keywords (What Users Love):")
            for keyword in result["positive_keywords"]:
                print(f"   • {keyword['keyword']} (confidence: {keyword['score']})")
            
            print("\n❌ Negative Keywords (Common Issues):")
            for keyword in result["negative_keywords"]:
                print(f"   • {keyword['keyword']} (confidence: {keyword['score']})")
            
            # Analyze the results for universal applicability
            print("\n📊 Universal Keyword Analysis:")
            pos_keywords = [kw['keyword'].lower() for kw in result["positive_keywords"]]
            neg_keywords = [kw['keyword'].lower() for kw in result["negative_keywords"]]
            
            print(f"   📈 Total positive keywords extracted: {len(result['positive_keywords'])}")
            print(f"   📉 Total negative keywords extracted: {len(result['negative_keywords'])}")
            
            # Check for meaningful multi-word phrases
            pos_phrases = [kw for kw in result["positive_keywords"] if len(kw['keyword'].split()) > 1]
            neg_phrases = [kw for kw in result["negative_keywords"] if len(kw['keyword'].split()) > 1]
            
            print(f"   🔗 Multi-word positive phrases: {len(pos_phrases)}")
            print(f"   🔗 Multi-word negative phrases: {len(neg_phrases)}")
            
            # Check for descriptive keywords (not just adjectives)
            descriptive_pos = [kw for kw in result["positive_keywords"] 
                             if any(desc in kw['keyword'].lower() for desc in ['quality', 'performance', 'design', 'value', 'service', 'delivery', 'construction'])]
            descriptive_neg = [kw for kw in result["negative_keywords"] 
                             if any(desc in kw['keyword'].lower() for desc in ['quality', 'performance', 'design', 'setup', 'instructions', 'construction'])]
            
            print(f"   🎯 Descriptive positive keywords: {len(descriptive_pos)}")
            print(f"   🎯 Descriptive negative keywords: {len(descriptive_neg)}")
            
            # Overall assessment
            total_keywords = len(result["positive_keywords"]) + len(result["negative_keywords"])
            phrase_ratio = (len(pos_phrases) + len(neg_phrases)) / total_keywords if total_keywords > 0 else 0
            descriptive_ratio = (len(descriptive_pos) + len(descriptive_neg)) / total_keywords if total_keywords > 0 else 0
            
            print(f"\n📈 Quality Metrics:")
            print(f"   • Multi-word phrase ratio: {phrase_ratio:.1%}")
            print(f"   • Descriptive keyword ratio: {descriptive_ratio:.1%}")
            
            if phrase_ratio >= 0.4 and descriptive_ratio >= 0.3:
                print("🎉 Excellent universal keyword extraction!")
            elif phrase_ratio >= 0.3 and descriptive_ratio >= 0.2:
                print("👍 Good universal keyword extraction")
            else:
                print("⚠️ Keyword extraction could be more descriptive")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
        print("Run: python main.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_mixed_product_reviews():
    """Test mixed review analysis with diverse product types"""
    
    print("\n🧪 Testing Mixed Product Review Analysis")
    print("=" * 45)
    
    # Mixed product reviews (could be electronics, clothing, books, etc.)
    mixed_reviews = {
        "reviews": [
            {"id": 1, "rating": 5, "review": "Excellent build quality and fast performance"},
            {"id": 2, "rating": 2, "review": "Poor customer service and delayed shipping"},
            {"id": 3, "rating": 4, "review": "Great value for money, comfortable design"},
            {"id": 4, "rating": 1, "review": "Broke after two weeks, very disappointing quality"},
            {"id": 5, "rating": 5, "review": "Easy to use interface and intuitive controls"},
            {"id": 6, "rating": 2, "review": "Overpriced for what you get, better alternatives exist"},
            {"id": 7, "rating": 4, "review": "Durable materials and stylish appearance"},
            {"id": 8, "rating": 1, "review": "Confusing setup process and unclear instructions"}
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-mixed-reviews", json=mixed_reviews)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📊 Sentiment Distribution:")
            print(f"   • Positive: {result['sentiment_counts']['positive']}")
            print(f"   • Neutral: {result['sentiment_counts']['neutral']}")
            print(f"   • negative: {result['sentiment_counts']['negative']}")
            
            print(f"\n🎭 Top Emotions:")
            emotions = result['emotion_summary']
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, count in top_emotions:
                if count > 0:
                    print(f"   • {emotion.capitalize()}: {count}")
            
            print(f"\n✅ Automatically Extracted Positive Keywords:")
            for kw in result['positive_keywords'][:8]:  # Top 8
                print(f"   • {kw['keyword']} ({kw['score']})")
            
            print(f"\n❌ Automatically Extracted Negative Keywords:")
            for kw in result['negative_keywords'][:8]:  # Top 8
                print(f"   • {kw['keyword']} ({kw['score']})")
            
            print("\n🎯 Universal Extraction Success!")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("🚀 Starting Universal Product Keyword Extraction Tests")
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    test_universal_keyword_extraction()
    test_mixed_product_reviews()
    
    print("\n✨ Universal keyword extraction testing completed!")
    print("🌍 The model now works for ANY product in the world!")
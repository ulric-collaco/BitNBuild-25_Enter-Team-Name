#!/usr/bin/env python3
"""
Test script for phone-specific keyword extraction
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_phone_keyword_extraction():
    """Test the enhanced keyword extraction with phone-specific examples"""
    
    # Sample phone reviews for testing
    test_data = {
        "positive_reviews": [
            "Amazing camera quality and the photos look professional",
            "Battery lasts all day even with heavy usage",  
            "Fast charging is incredible - charges in 30 minutes",
            "Build quality is solid and premium feeling",
            "Screen is bright and vibrant colors",
            "Great value for the money, highly recommended"
        ],
        "negative_reviews": [
            "Battery drains fast, dies within 4 hours",
            "Screen cracked within a week of normal use",
            "Camera quality is poor in low light",
            "Phone heats up during charging",
            "Too expensive for what you get",
            "Frequent crashes and system freezes",
            "Speaker sound is muffled and quiet",
            "Slow charging takes forever to charge"
        ]
    }
    
    print("🧪 Testing Phone-Specific Keyword Extraction")
    print("=" * 50)
    
    try:
        response = requests.post(f"{BASE_URL}/extract-keywords", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Positive Keywords (Phone Features):")
            for keyword in result["positive_keywords"]:
                print(f"   • {keyword['keyword']} (score: {keyword['score']})")
            
            print("\n❌ Negative Keywords (Phone Issues):")
            for keyword in result["negative_keywords"]:
                print(f"   • {keyword['keyword']} (score: {keyword['score']})")
            
            # Analyze the results
            print("\n📊 Analysis:")
            pos_keywords = [kw['keyword'].lower() for kw in result["positive_keywords"]]
            neg_keywords = [kw['keyword'].lower() for kw in result["negative_keywords"]]
            
            # Check for phone-specific features
            phone_features_found = 0
            total_features = 0
            
            feature_checks = {
                'battery': ['battery', 'lasts all day', 'charging'],
                'camera': ['camera', 'photos', 'quality'],
                'screen': ['screen', 'display', 'bright'],
                'build': ['build quality', 'solid', 'premium'],
                'value': ['value', 'money', 'price']
            }
            
            for category, terms in feature_checks.items():
                found_in_pos = any(term in ' '.join(pos_keywords) for term in terms)
                found_in_neg = any(term in ' '.join(neg_keywords) for term in terms)
                total_features += 1
                if found_in_pos or found_in_neg:
                    phone_features_found += 1
                    print(f"   ✓ {category.capitalize()} features detected")
                else:
                    print(f"   ⚠ {category.capitalize()} features NOT detected")
            
            feature_coverage = (phone_features_found / total_features) * 100
            print(f"\n📈 Phone Feature Coverage: {feature_coverage:.1f}%")
            
            if feature_coverage >= 80:
                print("🎉 Excellent phone-specific keyword extraction!")
            elif feature_coverage >= 60:
                print("👍 Good phone-specific keyword extraction")
            else:
                print("⚠️ Keyword extraction needs improvement for phone features")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
        print("Run: python main.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_mixed_reviews_with_test_json():
    """Test mixed review analysis with test.json data"""
    
    print("\n🧪 Testing Mixed Review Analysis with test.json")
    print("=" * 50)
    
    try:
        # Load test.json data
        with open('test.json', 'r') as f:
            test_data = json.load(f)
        
        # Take first 10 reviews for testing
        test_reviews = {"reviews": test_data[:10]}
        
        response = requests.post(f"{BASE_URL}/analyze-mixed-reviews", json=test_reviews)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📊 Sentiment Analysis Results:")
            print(f"   • Positive: {result['sentiment_counts']['positive']}")
            print(f"   • Neutral: {result['sentiment_counts']['neutral']}")
            print(f"   • Negative: {result['sentiment_counts']['negative']}")
            
            print(f"\n🎭 Top Emotions:")
            emotions = result['emotion_summary']
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, count in top_emotions:
                if count > 0:
                    print(f"   • {emotion.capitalize()}: {count}")
            
            print(f"\n✅ Positive Keywords Found: {len(result['positive_keywords'])}")
            for kw in result['positive_keywords'][:5]:  # Top 5
                print(f"   • {kw['keyword']} ({kw['score']})")
            
            print(f"\n❌ Negative Keywords Found: {len(result['negative_keywords'])}")
            for kw in result['negative_keywords'][:5]:  # Top 5
                print(f"   • {kw['keyword']} ({kw['score']})")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print("❌ test.json file not found")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("🚀 Starting Phone-Specific Keyword Extraction Tests")
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    test_phone_keyword_extraction()
    test_mixed_reviews_with_test_json()
    
    print("\n✨ Testing completed!")
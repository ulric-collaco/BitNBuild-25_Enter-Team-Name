# Review Radar - Render Deployment

## 🔹 Project Overview
**Review Radar** is a web tool and browser extension that helps shoppers quickly understand product reviews. It scrapes reviews from e-commerce sites, applies sentiment and emotion analysis, and extracts top positive and negative keywords. The results are shown in a simple dashboard with clear insights into customer opinions.

## 🔹 Team Details
- Ulric Collaco (TL)
- Swar Churi
- Tanush Chavan
- Sherwin Gonsalves

## 🔹 Tech Stack
- **Backend:** FastAPI (Python)
- **AI/ML:** Hugging Face Inference API
  - Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Keyword Extraction: `ml6team/keyphrase-extraction-kbir-inspec`
- **Cloud Services:** Render (Deployment)

## 🔹 Project Structure
```
render/
├── backend/
│   └── main.py          # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔹 API Endpoints
- `GET /` - Root endpoint with service information
- `GET /health` - Health check endpoint
- `POST /analyze-dashboard` - Main endpoint for review sentiment analysis

## 🔹 Environment Variables
Make sure to set the following environment variable in your Render deployment:
```
HF_API_KEY=your_huggingface_api_key_here
```

## 🔹 Deployment on Render
1. Connect your GitHub repository to Render
2. Select the `render` directory as the root directory
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add your `HF_API_KEY` environment variable
6. Deploy!

## 🔹 Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

## 🔹 API Usage Example
```bash
curl -X POST "http://localhost:8000/analyze-dashboard" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {
        "id": "1",
        "name": "Product Review",
        "rating": 5,
        "comments": "This product is amazing! Great quality and fast delivery."
      }
    ]
  }'
```
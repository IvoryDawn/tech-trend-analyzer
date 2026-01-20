import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
import joblib
import httpx
from datetime import datetime
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Literal

load_dotenv()

class AnalysisItem(BaseModel):
    sentiment: str
    pred_popularity_score: float
    issue_title: str = "" 
    issue_url: str = "" 
    repo_name: str = "" 

class TrendResponse(BaseModel):
    tech_name: str
    results: List[AnalysisItem]
    status: Literal["live", "mock"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading models...") # --- Startup: Load the models ---
    # Load your 'sentiment_model.joblib'
    try:
        sentiment_model = joblib.load('models/sentiment_model.joblib')
        app.state.sentiment_model = sentiment_model
        print("✅ SUCCESS: Sentiment model loaded")
    except Exception as e:
        print(f"❌ ERROR loading sentiment model: {e}")
        app.state.sentiment_model = None

    # Load your 'popularity_model.joblib'
    try:
        popularity_model = joblib.load('models/popularity_model.joblib')
        app.state.popularity_model = popularity_model
        print("✅ SUCCESS: Popularity Prediction model loaded")
    except Exception as e:
        print(f"❌ ERROR loading Popularity Prediction model: {e}")
        app.state.popularity_model = None
    
    if app.state.sentiment_model is not None and app.state.popularity_model is not None:
            print("Both models are loaded successfully.")
    else:
            print("Some models failed to load.")

    yield # The app stays here while running
    
    print("🏁 Shutting down...") # --- Shutdown: Clean up resources ---
    app.state.sentiment_model = None
    app.state.popularity_model = None

app = FastAPI(title='Tech Trend API', version='1.0', description='Real-Time Developer Sentiment & Job Trend Analyzer', lifespan=lifespan)

# Dependency Injection
async def get_sentiment_model(request: Request):
    model = request.app.state.sentiment_model
    if model is None:
        raise HTTPException(status_code=503, detail="Sentiment model is not loaded. Service unavailable.")
    return model

async def get_popularity_model(request: Request):
    model = request.app.state.popularity_model
    if model is None:
        raise HTTPException(status_code=503, detail="Popularity Model not loaded. Service Unavailable.")
    return model

# Feature Engineering Logic
def prepare_features(age_days: int, comments: int, sentiment: int) -> np.array:
    features = np.array([[age_days, comments, sentiment]], dtype =np.float64)
    return features

async def fetch_github_issues(keyword: str):
    url = "https://api.github.com/search/issues"
    params = {
        "q": f"{keyword} is:issue state:open",
        "per_page": 5,
        "sort": "created",
        "order": "desc"
    }
    headers = {
        "User-Agent": "Tech-Trend-Analyzer/1.0",
        "Accept": "application/vnd.github.v3+json"
    }
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
        print(f"🔑 Using GitHub token for {keyword}")
    else:
        print(f"⚠️  No GitHub token found for {keyword} (rate limited)")

    try:
        async with httpx.AsyncClient() as client:  # Talk to GitHub
            response = await client.get(url, params=params, headers=headers, timeout=10.0)
            if response.status_code == 200:  # If GitHub says OK
                data = response.json()
                return data.get("items", [])  # Get the list of issues
            else:
                print(f"GitHub API error: {response.status_code}")
                return []
                
    except Exception as e:
        print(f"Network error: {e}")
        return [] 

def process_github_issue(issue):
    try:
        title = issue.get("title","").strip()
        body = issue.get("body","").strip() if issue.get("body") else ""
        text_for_analysis = f"{title}. {body}" if body else title
        comments_count = issue.get("comments", 0)
        created_at_str = issue.get("created_at")
        if not created_at_str:
            return None
        # Convert string to date (GitHub format: "2023-12-01T10:30:00Z")
        created_at_str_clean = created_at_str.replace("Z", "+00:00")   # Remove 'Z' and parse
        created_date = datetime.fromisoformat(created_at_str_clean)
        current_time = datetime.now(created_date.tzinfo)
        days_old = (current_time - created_date).days
        days_old = max(0, days_old)  # Make sure it's not negative
        issue_url = issue.get("html_url", "")
        repo_url = issue.get("repository_url", "").replace("https://api.github.com/repos/", "")
        return {
            "text": text_for_analysis,
            "comments": comments_count,
            "age_days": days_old, 
            "title": title[:100],
            "url": issue_url,      
            "repo": repo_url,
            "issue_number": issue.get("number", 0) 
        }
    except Exception as e:
        print(f"Error processing issue: {e}")
        return None

@app.get("/test-features/{age}/{comments}/{sentiment}")
def test_features(age: int, comments: int, sentiment: int):
    features = prepare_features(age, comments, sentiment)
    return {
        "input": {"age_days": age, "comments": comments, "sentiment": sentiment},
        "output_shape": features.shape,
        "output_data": features.tolist()
    }

@app.get('/health')
def System_health():
    sentiment_loaded = hasattr(app.state, 'sentiment_model') and app.state.sentiment_model is not None
    popularity_loaded = hasattr(app.state, 'popularity_model') and app.state.popularity_model is not None
    
    models_loaded = sentiment_loaded and popularity_loaded
    if models_loaded:
        return {"status": "ready", "models_loaded": True, "details": {
            "sentiment_model": "loaded",
            "popularity_model": "loaded",
            "total_models": 2}}
    else:
         raise HTTPException(status_code=503, detail={
                "status": "unavailable",
                "models_loaded": models_loaded,
                "details": {
                    "sentiment_model": "loaded" if sentiment_loaded else "missing",
                    "popularity_model": "loaded" if popularity_loaded else "missing"
                }})
    
@app.get("/analyze/{tech_name}", response_model=TrendResponse)
async def analyze_tech(tech_name: str, s_model = Depends(get_sentiment_model), p_model = Depends(get_popularity_model)):
    print(f"🔍 Analyzing '{tech_name}' on GitHub...")
    results = []
    status = "mock"
    try:
        github_issues = await fetch_github_issues(tech_name)
        if github_issues and len(github_issues) > 0:
            print(f"✅ Found {len(github_issues)} GitHub issues")
            status = "live"
            for issue in github_issues:
                issue_data = process_github_issue(issue)
                if not issue_data:
                    continue
                try:
                    sentiment_prediction = s_model.predict([issue_data["text"]])[0]
                    sentiment_label = "positive" if sentiment_prediction == 1 else "negative"
                except:
                    sentiment_label = "neutral"
                    sentiment_prediction = 0.5

                try:
                    features = prepare_features(age_days=issue_data["age_days"], comments = issue_data["comments"], sentiment=sentiment_prediction)
                    popularity_score = p_model.predict(features)[0]
                except:
                    popularity_score = 50.0
                    popularity_score = max(1, popularity_score * 3) 
                results.append(AnalysisItem(
                sentiment=sentiment_label,
                pred_popularity_score=float(popularity_score),
                issue_title= issue_data['title'],
                issue_url = issue_data['url'],
                repo_name = issue_data['repo']
                ))
                print(f"   - Issue: '{issue_data['title']}'")
                print(f"     → {sentiment_label} ({popularity_score:.1f}) | 🔗 {issue_data['url']}")
    except Exception as e:
        print(f"🚨 CRITICAL ERROR in analysis pipeline: {e}")
        print("🔄 Falling back to dummy data...")

    if not results:
        print("⚠️  No GitHub data, using dummy example")
        dummy_age = 7
        dummy_comments = 15
        sample_text = f"{tech_name} is amazing for modern applications"
        
        sentiment_pred = s_model.predict([sample_text])[0]
        sentiment_label = "positive" if sentiment_pred == 1 else "negative"
        
        features = prepare_features(dummy_age, dummy_comments, sentiment_pred)
        popularity_score = p_model.predict(features)[0]
        popularity_score = max(1, popularity_score * 3)
        results.append(AnalysisItem(
        sentiment=sentiment_label,
        pred_popularity_score=float(popularity_score), 
        issue_title="Sample Analysis",
        issue_url="https://github.com/example",
        repo_name="example/repo"
        ))
    print(f"📊 Analysis complete: {len(results)} items, status: {status}")
    
    return TrendResponse(
    tech_name=tech_name,
    results=results,
    status=status
    )


@app.get('/')
def root():
    return {
        "message": "Tech Trend Analyzer API",
        "version": "1.0", 
        "architecture": "Dependency Injection (Enterprise Ready)",
        "endpoints" : {
            "GET /" : "Documentation", 
            "GET /health": "Health Check", 
            "GET /analyze/{tech_name}": "Real predictions with injected models",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.get("/test-dependency/{tech_name}")
def test_dependency_injection(
    tech_name: str,
    s_model = Depends(get_sentiment_model),
    p_model = Depends(get_popularity_model)
):
    """Test that dependency injection works"""
    return {
        "tech_name": tech_name,
        "dependency_test": "successful",
        "models_available": True,
        "sentiment_model_type": type(s_model).__name__,
        "popularity_model_type": type(p_model).__name__
    }
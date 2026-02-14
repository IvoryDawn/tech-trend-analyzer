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
from langdetect import detect, DetectorFactory
from sentiment_engine import SentimentAnalyzer    # Import your refactored class
from fastapi.responses import FileResponse

load_dotenv()
DetectorFactory.seed = 0  # Set seed for consistent language detection

class AnalysisItem(BaseModel):
    sentiment: str
    pred_popularity_score: float
    issue_title: str = "" 
    issue_url: str = "" 
    repo_name: str = "" 

class TrendSummary(BaseModel):
    total_analyzed: int
    positive_percentage: float
    average_popularity: float
    verdict: str

class TrendResponse(BaseModel):
    tech_name: str
    summary: TrendSummary
    results: List[AnalysisItem]
    status: Literal["live", "mock"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading models...") 
    
    # Load Sentiment Model via your Class
    try:
        sentiment_analyzer = SentimentAnalyzer(model_path='models/sentiment_model.joblib')
        if sentiment_analyzer.model is not None:
            app.state.sentiment_model = sentiment_analyzer
            print("✅ SUCCESS: Sentiment Engine loaded")
        else:
            raise Exception("Model file missing or failed to load")
    except Exception as e:
        print(f"❌ ERROR loading Sentiment Engine: {e}")
        app.state.sentiment_model = None

    # Load Popularity Model
    try:
        popularity_model = joblib.load('models/popularity_model.joblib')
        app.state.popularity_model = popularity_model
        print("✅ SUCCESS: Popularity Prediction model loaded")
    except Exception as e:
        print(f"❌ ERROR loading Popularity Prediction model: {e}")
        app.state.popularity_model = None
    
    if app.state.sentiment_model is not None and app.state.popularity_model is not None:
        print("✅ Both models are loaded successfully.")
    else:
        print("⚠️ Some models failed to load.")

    yield 
    
    print("🏁 Shutting down...") 
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
        "q": f"{keyword} in:title is:issue state:open comments:>3 created:>2024-01-01",
        "per_page": 30,
        "sort": "comments",
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
        async with httpx.AsyncClient() as client: 
            response = await client.get(url, params=params, headers=headers, timeout=10.0)
            if response.status_code == 200: 
                data = response.json()
                return data.get("items", []) 
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

        # Strict English Filter
        try:   
            if detect(text_for_analysis) != 'en':
                return None
        except:  # If it's pure code or symbols and can't be detected, skip it
            return None
    
        comments_count = issue.get("comments", 0)
        created_at_str = issue.get("created_at")
        if not created_at_str:
            return None
        created_at_str_clean = created_at_str.replace("Z", "+00:00")
        created_date = datetime.fromisoformat(created_at_str_clean)
        current_time = datetime.now(created_date.tzinfo)
        days_old = (current_time - created_date).days
        days_old = max(0, days_old) 
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
            valid_count = 0   # Keep track of valid issues
            for issue in github_issues:
                issue_data = process_github_issue(issue)
                if not issue_data:
                    continue   # Skips non-English or broken issues
                
                # Use the Sentiment Engine
                try:
                    sentiment_result = s_model.predict(issue_data["text"])
                    sentiment_label = sentiment_result.get("sentiment", "neutral")
                    
                    # Convert string back to number for the Popularity Model
                    sentiment_numeric = 1 if sentiment_label == "positive" else 0
                except:
                    sentiment_label = "neutral"
                    sentiment_numeric = 0.5

                # Use Popularity Model
                try:
                    features = prepare_features(age_days=issue_data["age_days"], comments=issue_data["comments"], sentiment=sentiment_numeric)
                    raw_ml_score = p_model.predict(features)[0]
                except:
                    raw_ml_score = 50.0
                
                engagement_floor = issue_data["comments"] * 5.0 
                
                popularity_score = float(max(engagement_floor, raw_ml_score * 3.0, 1.0))
                
                results.append(AnalysisItem(
                    sentiment=sentiment_label,
                    pred_popularity_score=float(popularity_score),
                    issue_title= issue_data['title'],
                    issue_url = issue_data['url'],
                    repo_name = issue_data['repo']
                ))

                valid_count += 1
                if valid_count >= 10:   # Stop when we hit 10 valid English issues
                    break

                print(f"   - Issue: '{issue_data['title'][:30]}...'")
                print(f"     → {sentiment_label} ({popularity_score:.1f}) | 🔗 {issue_data['url']}")
    except Exception as e:
        print(f"🚨 CRITICAL ERROR in analysis pipeline: {e}")
        print("🔄 Falling back to dummy data...")

    if not results:
        # Fallback dummy data logic
        print("⚠️  No GitHub data, using dummy example")
        dummy_age = 7
        dummy_comments = 15
        sample_text = f"{tech_name} is amazing for modern applications"
        
        sentiment_result = s_model.predict(sample_text)
        sentiment_label = sentiment_result.get("sentiment", "positive")
        sentiment_numeric = 1 if sentiment_label == "positive" else 0
        
        features = prepare_features(dummy_age, dummy_comments, sentiment_numeric)
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
    
    # Calculate the Final Verdict
    if len(results) > 0:
        positive_count = sum(1 for r in results if r.sentiment == "positive")
        pos_percent = (positive_count / len(results)) * 100
        avg_pop = sum(r.pred_popularity_score for r in results) / len(results)
        
        # The Decision Logic
        if avg_pop >= 120:
            verdict = "🔥 Highly Trending (Massive Community Engagement)"
        elif avg_pop >= 75:
            verdict = "📈 Stable / Widespread Enterprise Use"
        elif avg_pop >= 40:
            verdict = "🛠️ Legacy Maintenance / Niche Adoption"
        else:
            verdict = "📉 Quiet Community / Low Trend"
    else:
        pos_percent = 0.0
        avg_pop = 0.0
        verdict = "❓ Not enough data to determine"

    print(f"📊 Analysis complete. Verdict: {verdict}")
    
    return TrendResponse(
        tech_name=tech_name,
        summary=TrendSummary(
            total_analyzed=len(results),
            positive_percentage=round(pos_percent, 1),
            average_popularity=round(avg_pop, 1),
            verdict=verdict
        ),
        results=results[:5], # Only return top 5 to user
        status=status
    )

@app.get('/')
def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
Title: Tech Trend Analyzer API.

One-Liner: A Machine Learning API that scrapes GitHub issues, analyzes developer sentiment, and predicts technology trends.

Tech Stack: FastAPI, Scikit-Learn (Logistic Regression), Docker, GitHub API, Pandas.

How to Run (Docker):
docker build -t tech-trend-api .
docker run -p 8000:8000 --env-file .env tech-trend-api

Live Demo: (https://tech-trend-api.onrender.com/docs)

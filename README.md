# 🚀 Tech Trend Analyzer API

An intelligent, containerized Machine Learning API that scrapes live GitHub data, analyzes developer sentiment, and predicts the true popularity of software frameworks. 

Unlike basic keyword counters, this project handles real-world data drift and the "Open Source Hospital Paradox" (where high bug reports actually indicate high adoption) by engineering an ML pipeline with heuristic calibration and strict engagement filters.

## ⚙️ Tech Stack
* **Backend:** FastAPI, Python, Uvicorn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Custom Feature Engineering), Joblib, Langdetect
* **Data Ingestion:** GitHub REST API, HTTPX (Asynchronous fetching)
* **DevOps:** Docker, Render (Cloud Deployment)

## 🧠 Core Features & Engineering Challenges Solved

* **Real-Time Data Pipeline:** Asynchronously fetches live, highly-commented GitHub issues created post-2024 to avoid historical data drift.
* **Language & Noise Filtering:** Utilizes `langdetect` to enforce strict English parsing, stripping out code-only issues and spam.
* **The "Hospital Paradox" Solution:** Re-engineered the query logic to filter by `comments:>3` rather than repository stars, ensuring the ML model analyzes genuine enterprise developer engagement rather than student homework or dead repositories.
* **ML Heuristic Calibration Layer:** Implemented an engagement floor mathematically tied to comment volume, preventing the linear popularity model from flatlining when encountering out-of-distribution inference data.
* **OOP Architecture:** Built with clean Dependency Injection, loading ML models asynchronously during the FastAPI startup lifespan to ensure zero-latency inference.

## 🐳 How to Run Locally (Docker)

1. Clone the repository.
2. Create a `.env` file in the root directory and add your GitHub token: `GITHUB_TOKEN=your_token_here`
3. Build the Docker image:
   ```bash
   docker build -t tech-trend-api .
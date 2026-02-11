# 1. Base Image (Lightweight Python)
FROM python:3.9-slim

# 2. Set Work Directory inside the container
WORKDIR /app

# 3. Copy Requirements first (caching trick to make future builds faster)
COPY requirements.txt .

# 4. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download NLTK stopwords (Required for your text cleaning)
RUN python -m nltk.downloader stopwords punkt punkt_tab

# 6. Copy the rest of your Application Code
COPY . .

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
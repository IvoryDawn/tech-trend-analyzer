import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import os

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """
    Standard cleaning function to be applied to DataFrame.
    """
    if pd.isna(text): return ""
    
    # 1. Lowercase
    text = str(text).lower()
    
    # 2. Remove Punctuation
    text = "".join(u for u in text if u not in (",", "?", ".", ";", "!", ":", '"'))
    
    # 3. Remove Numbers
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    
    return text

def run_training():
    print("🚀 Loading Dataset...")
    # Adjust path if necessary
    try:
        data = pd.read_csv("Data/Dataset.txt", sep='\t', quoting=3, header=None, names=['text', 'sentiment'], encoding='latin1')
    except FileNotFoundError:
        print("❌ Error: 'Data/Dataset.txt' not found.")
        return

    # --- Preprocessing ---
    print(f"📊 Original Data Length: {len(data)}")
    data.drop_duplicates(inplace=True)
    data['text'] = data['text'].apply(clean_text)
    data = data.dropna(subset=['text'])
    print(f"✨ Cleaned Data Length: {len(data)}")

    X = data['text']
    y = data['sentiment']

    # --- Training ---
    print("🧠 Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=100))
    ])

    pipeline.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = pipeline.predict(X_test)
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))

    # --- Saving ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    save_path = 'models/sentiment_model.joblib'
    joblib.dump(pipeline, save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    run_training()
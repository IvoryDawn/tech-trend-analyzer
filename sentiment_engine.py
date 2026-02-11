import joblib
import os
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentAnalyzer:
    """
    Refactored Inference Class.
    Loads the trained model once and handles text cleaning + prediction.
    """

    def __init__(self, model_path="models/sentiment_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self._load_model()

    def _load_model(self):
        """Loads the serialized pipeline (Tfidf + LogisticRegression)."""
        if os.path.exists(self.model_path):
            print(f"📂 Loading model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
        else:
            print(f"⚠️ Warning: Model file {self.model_path} not found.")
            self.model = None

    def _preprocess(self, text: str) -> str:
        """
        Mimics the exact cleaning steps from your training script:
        1. Lowercase
        2. Remove Punctuation
        3. Remove Numbers
        4. Remove Stopwords
        """
        if not text: 
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove Punctuation (using regex is faster/cleaner than loops)
        # Keeps only letters and spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # 3. Remove Numbers
        text = re.sub(r'\d+', '', text)
        
        # 4. Remove Stopwords
        words = text.split()
        clean_words = [w for w in words if w not in self.stop_words]
        
        return " ".join(clean_words)

    def predict(self, text: str) -> dict:
        """
        Returns: {'sentiment': 'positive'/'negative', 'confidence': 0.95}
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        clean_text = self._preprocess(text)
        
        if not clean_text:
            return {"sentiment": "neutral", "confidence": 0.0}

        try:
            prediction = self.model.predict([clean_text])[0]
            probs = self.model.predict_proba([clean_text])[0]
            confidence = max(probs)

            # --- THE FIX: Force conversion to integer ---
            try:
                pred_val = int(prediction)
                sentiment = "positive" if pred_val == 1 else "negative"
            except ValueError:
                # Fallback just in case the model actually outputs 'positive' natively
                sentiment = str(prediction)

            return {
                "text": text,
                "cleaned_text": clean_text,
                "sentiment": sentiment,
                "confidence": round(float(confidence), 2)
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Test it
    analyzer = SentimentAnalyzer()
    print(analyzer.predict("This error is absolutely terrible and broken!"))
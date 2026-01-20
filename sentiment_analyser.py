import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv("Data/Dataset.txt", sep='\t', quoting=3, header=None, names=['text', 'sentiment'], encoding='latin1')
print("Sentiment value count: ", data['sentiment'].value_counts())
# Quick clean 
print("Length of merged dataset: ", len(data))
print("Data set looks like: \n", data.head())
print("Checking for missing values or null values: ", data.isnull().sum())  # No missing or null values
data.drop_duplicates(inplace=True) # Dropping duplicates
data['text'] = data['text'].str.lower()
# Remove punctuation from the string of words 
def remove_punctuation(text):
    final = "".join(u for u in text if u not in (",", "?", ".", ";", "!", ":", '"'))
    return final
data['text'] = data['text'].apply(remove_punctuation)
data = data.dropna(subset=['text'])
# Removing numbers from the text
data['text'] = data['text'].str.replace('\d+', '')
print("Length of merged dataset after cleaning: ", len(data))
print("Data set after cleaning: \n", data.head())

# stopwords
stop_words = stopwords.words('english')
data['text'] = data['text'].apply(lambda text: " ".join(word for word in text.split() if word.lower() not in stop_words))
print("Stopwords: ", stopwords)

# Separate x and y for training
x = data['text']
y = data['sentiment']
print("Total rows for training: ", len(x))

sentiment_pipeline = Pipeline([('vectorizer', TfidfVectorizer(stop_words = 'english', max_features = 5000)), ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=100))])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
sentiment_pipeline.fit(x_train, y_train)
y_pred = sentiment_pipeline.predict(x_test)
class_report = classification_report(y_test, y_pred, target_names=['Negative(0)', 'Positive(1)'])
print("Classification Report: \n", class_report)

# Serializing and saving the model
try:
    joblib.dump(sentiment_pipeline, 'sentiment_model.joblib')
    print("✅ SUCCESS: sentiment_pipeline saved as 'sentiment_model.joblib'")

    import os
    file_size = os.path.getsize('sentiment_model.joblib')
    print(f"File size: {file_size/1024:.2f} KB")
except Exception as e:
    print(f"❌ ERROR saving model: {e}")
    print("Please check file permissions or disk space.")
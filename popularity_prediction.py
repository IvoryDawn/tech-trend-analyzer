import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

np.random.seed(42)
n_samples = 200

print(f"\n📊 Generating {n_samples} synthetic samples...")

# Mock numerical features
synthetic_data = {
    'age_days': np.random.randint(0, 30, n_samples),  # Issues 0-30 days old
    'current_comments': np.random.randint(0, 50, n_samples),  # 0-50 existing comments
    'sentiment_score': np.random.uniform(0, 1, n_samples)  # Sentiment from Stage 1
}
predicted_reactions = (
    # Base: New issues get more attention
    8 * np.exp(-0.2 * synthetic_data['age_days']) +  # DECREASES exponentially with age
    0.5 * synthetic_data['current_comments'] + # Comments: Current engagement matters
    12 * synthetic_data['sentiment_score'] + # Sentiment: Positive gets more reactions
    np.random.normal(0, 2, n_samples) # Random variation
)
# Ensure no negative
predicted_reactions = np.maximum(0.5, predicted_reactions)
synthetic_data['predicted_reactions'] = predicted_reactions
print(f"✓ Created {n_samples} synthetic samples")

# Split and train the model
x = np.column_stack([synthetic_data['age_days'], synthetic_data['current_comments'], synthetic_data['sentiment_score']])
y = synthetic_data['predicted_reactions']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_pred, y_test)
root_mse = np.sqrt(mse)
print("\n📊 MODEL PERFORMANCE:")
print("-" * 40)
print(f"✅ RMSE: {root_mse:.2f} predicted reactions")
print(f"✅ MSE: {mse:.2f}")


try:
    joblib.dump(model, 'popularity_prediction_model.joblib')
    import os
    file_size = os.path.getsize('popularity_prediction_model.joblib')
    print(f"File size is {file_size/1024:.2f} kb")

except Exception as e:
    print(f"❌ ERROR saving model: {e}")
    print("Please check file permissions or disk space.")
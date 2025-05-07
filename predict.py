import pandas as pd
import joblib
from scipy.sparse import hstack

from preprocessing import preprocess_text, add_numeric_features

# Load the saved TF-IDF vectorizer and trained model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('best_model.joblib')

# Load test data (assumes 'Reddit/x_test.csv' has columns: ID, text)
df_test = pd.read_csv('Reddit/x_test.csv')

# Add numeric features for test data
df_test = add_numeric_features(df_test)

# Preprocess text and transform with TF-IDF
df_test['clean_text'] = df_test['text'].apply(preprocess_text)
X_test_tfidf = vectorizer.transform(df_test['clean_text'])

# Combine with numeric features
X_test_num = df_test[['word_count', 'exclamation_count', 'upper_case_count']].values
X_test_full = hstack([X_test_tfidf, X_test_num]).tocsr()

# Generate predicted probabilities (probability of positive class)
proba = model.predict_proba(X_test_full)[:, 1]

# Prepare output DataFrame with ID and predicted probability
df_out = pd.DataFrame({
    'ID': df_test['ID'],
    'label': proba
})

# Save predictions to CSV
df_out.to_csv('Reddit/test_predictions.csv', index=False)
print("Test predictions saved to 'Reddit/test_predictions.csv'.")

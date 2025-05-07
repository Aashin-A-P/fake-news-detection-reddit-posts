import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_text

# Load vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('best_model.joblib')

st.title("Reddit Fake News Detector")
st.write("Enter a Reddit post title or upload a CSV file of titles to get predictions.")

# ----- Single Title Prediction -----
st.subheader("Single Title Prediction")
title_input = st.text_input("Enter a Reddit title", "")

if title_input:
    clean_title = preprocess_text(title_input)
    X_tfidf_single = vectorizer.transform([clean_title])
    word_count = len(title_input.split())
    exclamation_count = title_input.count('!')
    upper_case_count = sum(1 for w in title_input.split() if w.isupper())
    X_num_single = [[word_count, exclamation_count, upper_case_count]]
    X_full_single = hstack([X_tfidf_single, X_num_single]).tocsr()
    proba_single = model.predict_proba(X_full_single)[:, 1][0]
    label_single = int(proba_single > 0.5)

    st.write(f"**Prediction probability of being Fake News:** {proba_single:.2f}")
    st.write("**Predicted label:**", "Fake (1)" if label_single else "Not Fake (0)")

# ----- Batch Prediction -----
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with titles", type=["csv"])

if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)

    if 'text' not in df_batch.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        if 'label' in df_batch.columns:
            df_batch.rename(columns={'label': 'label_true'}, inplace=True)

        df_batch['clean_text'] = df_batch['text'].apply(preprocess_text)
        X_tfidf_batch = vectorizer.transform(df_batch['clean_text'])
        df_batch['word_count'] = df_batch['text'].apply(lambda x: len(str(x).split()))
        df_batch['exclamation_count'] = df_batch['text'].apply(lambda x: str(x).count('!'))
        df_batch['upper_case_count'] = df_batch['text'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
        X_num_batch = df_batch[['word_count', 'exclamation_count', 'upper_case_count']].values
        X_full_batch = hstack([X_tfidf_batch, X_num_batch]).tocsr()

        df_batch['probability'] = model.predict_proba(X_full_batch)[:, 1]
        df_batch['label'] = (df_batch['probability'] > 0.5).astype(int)

        st.write("## Batch Prediction Results")
        st.dataframe(df_batch[['ID', 'text', 'probability', 'label']])

        csv_results = df_batch[['ID', 'label']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv_results,
                           file_name='batch_predictions.csv', mime='text/csv')

        if 'label_true' in df_batch.columns:
            st.subheader("Model Evaluation Metrics")
            y_true = df_batch['label_true']
            y_pred = df_batch['label']
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)

            st.write(f"**Accuracy:** {acc:.2f}")
            st.text("Classification Report:")
            st.text(report)

import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, make_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from preprocessing import preprocess_text, add_numeric_features

# Load data
df_train = pd.read_csv('Reddit/xy_train.csv')

# Add numeric features and preprocess text
df_train = add_numeric_features(df_train)
df_train['clean_text'] = df_train['text'].apply(preprocess_text)

# Vectorize text
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5)
X_tfidf = vectorizer.fit_transform(df_train['clean_text'])
X_num = df_train[['word_count', 'exclamation_count', 'upper_case_count']].values
X_train_full = hstack([X_tfidf, X_num]).tocsr()

# Encode labels (in case labels are non-numeric)
le = LabelEncoder()
y_train = le.fit_transform(df_train['label'])

# Decide whether it's binary or multiclass
is_multiclass = len(le.classes_) > 2
roc_auc = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr' if is_multiclass else 'raise')

# --------------------------
# Logistic Regression
# --------------------------
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr_param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(lr, lr_param_grid, scoring=roc_auc, cv=5, n_jobs=-1)
grid_lr.fit(X_train_full, y_train)
best_lr = grid_lr.best_estimator_
print(f"Best LogisticRegression params: {grid_lr.best_params_}")

y_pred_lr = cross_val_predict(best_lr, X_train_full, y_train, cv=5)
print("Logistic Regression Classification Report:")
print(classification_report(y_train, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_lr))
y_proba_lr = best_lr.predict_proba(X_train_full)
print(f"ROC-AUC (LR): {roc_auc_score(y_train, y_proba_lr, multi_class='ovr' if is_multiclass else 'raise'):.4f}")

# --------------------------
# XGBoost
# --------------------------
print("Training XGBoost...")
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_xgb = GridSearchCV(xgb_clf, xgb_param_grid, scoring=roc_auc, cv=5, n_jobs=-1)
grid_xgb.fit(X_train_full, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"Best XGBoost params: {grid_xgb.best_params_}")

y_pred_xgb = cross_val_predict(best_xgb, X_train_full, y_train, cv=5)
print("XGBoost Classification Report:")
print(classification_report(y_train, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_xgb))
y_proba_xgb = best_xgb.predict_proba(X_train_full)
print(f"ROC-AUC (XGB): {roc_auc_score(y_train, y_proba_xgb, multi_class='ovr' if is_multiclass else 'raise'):.4f}")

# Choose best model
auc_lr = roc_auc_score(y_train, y_proba_lr, multi_class='ovr' if is_multiclass else 'raise')
auc_xgb = roc_auc_score(y_train, y_proba_xgb, multi_class='ovr' if is_multiclass else 'raise')
best_model = best_xgb if auc_xgb >= auc_lr else best_lr
print(f"Selected best model: {best_model.__class__.__name__}")

# Save artifacts
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(best_model, 'best_model.joblib')
joblib.dump(le, 'label_encoder.joblib')  # optional
print("Saved vectorizer, model, and label encoder.")

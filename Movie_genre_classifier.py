# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 2. FUNCTION TO LOAD TXT DATA
def load_data(filepath):
    genres = []
    descriptions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                genre, description = parts
                genres.append(genre)
                descriptions.append(description)
    return pd.DataFrame({'Genre': genres, 'Description': descriptions})

# 3. LOAD DATA
train_df = load_data('Genre Classification Dataset/train_data.txt')
test_df = load_data('Genre Classification Dataset/test_data.txt')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())

# 4. PREPROCESSING

# Fill NaNs in description (should not be necessary, but safe)
train_df['Description'] = train_df['Description'].fillna('')
test_df['Description'] = test_df['Description'].fillna('')

# Lowercase descriptions
train_df['Description'] = train_df['Description'].str.lower()
test_df['Description'] = test_df['Description'].str.lower()

# Encode genre labels
le = LabelEncoder()
train_df['genre_encoded'] = le.fit_transform(train_df['Genre'])

# 5. FEATURE EXTRACTION - TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(train_df['Description']).toarray()
X_test = tfidf.transform(test_df['Description']).toarray()

y_train = train_df['genre_encoded']

# 6. TRAIN-VALIDATION SPLIT
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 7. MODEL TRAINING
model = LogisticRegression(max_iter=1000)
model.fit(X_train_, y_train_)

# 8. EVALUATE ON VALIDATION SET
y_val_pred = model.predict(X_val)
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# 9. PREDICT ON TEST DATA
test_preds = model.predict(X_test)
test_preds_labels = le.inverse_transform(test_preds)

# 10. SAVE PREDICTIONS
test_df['predicted_genre'] = test_preds_labels
test_df[['Description', 'predicted_genre']].to_csv('test_predictions.csv', index=False)

print("\nPredictions saved to 'test_predictions.csv'")

# 11. OPTIONAL: SAVE MODEL AND TFIDF
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nModel, Vectorizer, and LabelEncoder saved!")

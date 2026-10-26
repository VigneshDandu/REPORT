import pandas as pd
import os
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# 1. Load preprocessed dataset
# -----------------------------------
df = pd.read_csv("data/preprocessed_reviews.csv")

print("Dataset shape:", df.shape)


# -----------------------------------
# 2. Make sure text is valid
# -----------------------------------
df["clean_text"] = df["clean_text"].fillna("").astype(str)

df = df[df["clean_text"].str.strip() != ""].copy()


# -----------------------------------
# 3. Prepare input and target
# -----------------------------------
X_text = df["clean_text"]
X_rating = df["rating"]
y = df["label"].map({
    "OR": 0,   # Genuine
    "CG": 1    # Fake
})


print("\nLabel distribution:")
print(y.value_counts())


# -----------------------------------
# 4. Train/Test Split
# -----------------------------------
X_text_train, X_text_test, X_rating_train, X_rating_test, y_train, y_test = train_test_split(
    X_text,
    X_rating,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_text_train))
print("Testing samples:", len(X_text_test))


# -----------------------------------
# 5. TF-IDF for review text
# -----------------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)


# -----------------------------------
# 6. Convert rating to sparse matrix
# -----------------------------------
X_rating_train_sparse = csr_matrix(
    X_rating_train.values.reshape(-1, 1)
)

X_rating_test_sparse = csr_matrix(
    X_rating_test.values.reshape(-1, 1)
)


# -----------------------------------
# 7. Combine text + rating
# -----------------------------------
X_train_final = hstack([
    X_text_train_tfidf,
    X_rating_train_sparse
])

X_test_final = hstack([
    X_text_test_tfidf,
    X_rating_test_sparse
])


print("\nFinal training feature shape:", X_train_final.shape)
print("Final testing feature shape:", X_test_final.shape)


# -----------------------------------
# 8. Train Logistic Regression
# -----------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_final, y_train)

print("\nModel training completed!")


# -----------------------------------
# 9. Prediction
# -----------------------------------
y_pred = model.predict(X_test_final)

probabilities = model.predict_proba(X_test_final)


# -----------------------------------
# 10. Evaluation
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine (OR)", "Fake (CG)"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 11. Save model and vectorizer
# -----------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression_text_rating.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer_text_rating.pkl"
)

print("\nModel saved:")
print("models/logistic_regression_text_rating.pkl")

print("\nVectorizer saved:")
print("models/tfidf_vectorizer_text_rating.pkl")
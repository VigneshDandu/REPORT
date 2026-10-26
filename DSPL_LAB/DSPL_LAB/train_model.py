import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------------
# 1. Load preprocessed dataset
# -----------------------------------
df = pd.read_csv("data/preprocessed_reviews.csv")

print("Dataset shape:", df.shape)

# -----------------------------------
# 2. Select input and target
# -----------------------------------
# Make sure all review text values are valid strings
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Remove completely empty reviews
df = df[df["clean_text"].str.strip() != ""].copy()

X = df["clean_text"]
y = df["label"]

# Convert labels:
# OR = Genuine -> 0
# CG = Fake -> 1
y = y.map({"OR": 0, "CG": 1})

print("\nLabel distribution:")
print(y.value_counts())

# -----------------------------------
# 3. Train/Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------
# 4. TF-IDF
# -----------------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

# -----------------------------------
# 5. Train Logistic Regression
# -----------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

# -----------------------------------
# 6. Prediction
# -----------------------------------
y_pred = model.predict(X_test_tfidf)

# -----------------------------------
# 7. Evaluation
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
# 8. Save model and vectorizer
# -----------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved:")
print("models/logistic_regression.pkl")

print("\nVectorizer saved:")
print("models/tfidf_vectorizer.pkl")
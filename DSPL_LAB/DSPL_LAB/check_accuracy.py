import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score


# Load data
df = pd.read_csv("data/preprocessed_reviews.csv")

df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["clean_text"].str.strip() != ""].copy()

# Prepare data
X_text = df["clean_text"]
X_rating = df["rating"]

y = df["label"].map({
    "OR": 0,
    "CG": 1
})

# Same split used during training
(
    X_text_train,
    X_text_test,
    X_rating_train,
    X_rating_test,
    y_train,
    y_test
) = train_test_split(
    X_text,
    X_rating,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load saved vectorizer
vectorizer = joblib.load(
    "models/tfidf_vectorizer_text_rating.pkl"
)

# Load saved model
model = joblib.load(
    "models/logistic_regression_comparison.pkl"
)

# Convert test text to TF-IDF
X_test_tfidf = vectorizer.transform(X_text_test)

# Add rating
X_test_rating = csr_matrix(
    X_rating_test.values.reshape(-1, 1)
)

X_test_final = hstack([
    X_test_tfidf,
    X_test_rating
])

# Predict — NO TRAINING
y_pred = model.predict(X_test_final)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model: Logistic Regression")
print("Test samples:", len(y_test))
print("Test Accuracy:", round(accuracy * 100, 2), "%")
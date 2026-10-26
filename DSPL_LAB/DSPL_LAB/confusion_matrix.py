import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# -----------------------------------
# 1. Load preprocessed dataset
# -----------------------------------
df = pd.read_csv("data/preprocessed_reviews.csv")

print("Dataset shape:", df.shape)

# Make sure text is valid
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Remove empty reviews
df = df[df["clean_text"].str.strip() != ""].copy()


# -----------------------------------
# 2. Prepare data
# -----------------------------------
X_text = df["clean_text"]
X_rating = df["rating"]

y = df["label"].map({
    "OR": 0,
    "CG": 1
})


# -----------------------------------
# 3. Same train/test split
# -----------------------------------
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


# -----------------------------------
# 4. Load the same TF-IDF settings
# -----------------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)


# -----------------------------------
# 5. Add rating
# -----------------------------------
X_rating_train_sparse = csr_matrix(
    X_rating_train.values.reshape(-1, 1)
)

X_rating_test_sparse = csr_matrix(
    X_rating_test.values.reshape(-1, 1)
)


X_train_final = hstack([
    X_train_tfidf,
    X_rating_train_sparse
])

X_test_final = hstack([
    X_test_tfidf,
    X_rating_test_sparse
])


# -----------------------------------
# 6. Load final Logistic Regression model
# -----------------------------------
model = joblib.load(
    "models/logistic_regression_comparison.pkl"
)

print("Model loaded successfully.")


# -----------------------------------
# 7. Make predictions
# -----------------------------------
y_pred = model.predict(X_test_final)


# -----------------------------------
# 8. Confusion Matrix
# -----------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------------
# 9. Classification Report
# -----------------------------------
report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Genuine (OR)",
        "Fake (CG)"
    ]
)

print("\nClassification Report:")
print(report)


# -----------------------------------
# 10. Create outputs folder
# -----------------------------------
os.makedirs("outputs", exist_ok=True)


# -----------------------------------
# 11. Save confusion matrix graph
# -----------------------------------
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Genuine (OR)",
        "Fake (CG)"
    ]
)

display.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()


# -----------------------------------
# 12. Save classification report
# -----------------------------------
with open(
    "outputs/classification_report.txt",
    "w"
) as file:
    file.write(report)


print("\nConfusion matrix saved:")
print("outputs/confusion_matrix.png")

print("\nClassification report saved:")
print("outputs/classification_report.txt")

print("\nEvaluation completed successfully!")
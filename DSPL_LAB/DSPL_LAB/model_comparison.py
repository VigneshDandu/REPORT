import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# -----------------------------------
# 1. Load dataset
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
    "OR": 0,   # Genuine
    "CG": 1    # Fake
})

print("\nLabel distribution:")
print(y.value_counts())


# -----------------------------------
# 3. Train/Test Split
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

print("\nTraining samples:", len(X_text_train))
print("Testing samples:", len(X_text_test))


# -----------------------------------
# 4. TF-IDF
# -----------------------------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


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

print("\nFinal feature shape:", X_train_final.shape)


# -----------------------------------
# 6. Define models
# -----------------------------------
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=30
    )
}


# -----------------------------------
# 7. Train and evaluate models
# -----------------------------------
results = []

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    # Train
    model.fit(X_train_final, y_train)

    # Predict
    y_pred = model.predict(X_test_final)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1-score :", round(f1 * 100, 2), "%")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Save model
    filename = name.lower().replace(" ", "_") + "_comparison.pkl"

    joblib.dump(
        model,
        f"models/{filename}"
    )


# -----------------------------------
# 8. Create results DataFrame
# -----------------------------------
results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# -----------------------------------
# 9. Save results
# -----------------------------------
results_df.to_csv(
    "outputs/model_comparison_results.csv",
    index=False
)


# -----------------------------------
# 10. Create comparison graph
# -----------------------------------
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

x = range(len(results_df))

width = 0.2

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):

    values = results_df[metric] * 100

    positions = [
        value + (i - 1.5) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    results_df["Model"]
)

plt.ylabel("Score (%)")
plt.xlabel("Machine Learning Model")
plt.title("ML Model Comparison")
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300
)

plt.close()


print("\nComparison graph saved:")
print("outputs/model_comparison.png")

print("\nResults saved:")
print("outputs/model_comparison_results.csv")

print("\nModel comparison completed successfully!")
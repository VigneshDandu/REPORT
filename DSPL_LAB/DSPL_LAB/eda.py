import pandas as pd
import matplotlib

# Prevent graph windows from opening
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/fake reviews dataset.csv")

print("Original dataset shape:", df.shape)

# -----------------------------
# 2. Remove duplicate reviews
# -----------------------------
df = df.drop_duplicates(subset=["text_"]).reset_index(drop=True)

print("After removing duplicate reviews:", df.shape)

# -----------------------------
# 3. Create outputs folder
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 4. Review length
# -----------------------------
df["review_length"] = df["text_"].str.len()

# -----------------------------
# 5. Label distribution
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.title("Review Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/label_distribution.png", dpi=200)
plt.close()

# -----------------------------
# 6. Rating distribution
# -----------------------------
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="rating")
plt.title("Rating Distribution")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png", dpi=200)
plt.close()

# -----------------------------
# 7. Category distribution
# -----------------------------
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    y="category",
    order=df["category"].value_counts().index
)
plt.title("Reviews by Product Category")
plt.xlabel("Number of Reviews")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig("outputs/category_distribution.png", dpi=200)
plt.close()

# -----------------------------
# 8. Review length distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["review_length"], bins=40)
plt.title("Review Length Distribution")
plt.xlabel("Review Length (Characters)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/review_length_distribution.png", dpi=200)
plt.close()

# -----------------------------
# 9. Rating vs Label
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating", hue="label")
plt.title("Rating Distribution by Review Label")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/rating_vs_label.png", dpi=200)
plt.close()

# -----------------------------
# 10. Save cleaned dataset
# -----------------------------
df.to_csv("data/cleaned_reviews.csv", index=False)

# -----------------------------
# 11. Print results
# -----------------------------
print("\nLabel counts:")
print(df["label"].value_counts())

print("\nRating counts:")
print(df["rating"].value_counts())

print("\nCategory counts:")
print(df["category"].value_counts())

print("\nReview length statistics:")
print(df["review_length"].describe())

print("\nEDA completed successfully!")
print("Graphs saved in the outputs folder.")
print("Cleaned dataset saved as data/cleaned_reviews.csv")
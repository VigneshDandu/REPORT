import pandas as pd
import re

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/cleaned_reviews.csv")

print("Dataset loaded:", df.shape)

# -----------------------------
# 2. Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove newline characters
    text = re.sub(r"\n+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------------
# 3. Apply text cleaning
# -----------------------------
df["clean_text"] = df["text_"].apply(clean_text)

# -----------------------------
# 4. Show examples
# -----------------------------
print("\nOriginal vs Cleaned Reviews:\n")

for i in range(5):
    print("Original:")
    print(df.loc[i, "text_"])

    print("\nCleaned:")
    print(df.loc[i, "clean_text"])

    print("-" * 80)

# -----------------------------
# 5. Save preprocessed dataset
# -----------------------------
df.to_csv("data/preprocessed_reviews.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Saved as: data/preprocessed_reviews.csv")
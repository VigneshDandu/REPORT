import pandas as pd

# Load dataset
df = pd.read_csv("data/fake reviews dataset.csv")

print("=" * 100)
print("LONGER DEMONSTRATION REVIEWS")
print("=" * 100)

for label in ["CG", "OR"]:

    print("\n" + "=" * 100)
    print("LABEL:", label)
    print("=" * 100)

    for rating in [1, 2, 3, 4, 5]:

        # Filter by label and rating
        examples = df[
            (df["label"] == label) &
            (df["rating"] == rating)
        ].copy()

        # Calculate review length
        examples["review_length"] = examples["text_"].str.len()

        # Take the longest review
        examples = examples.sort_values(
            "review_length",
            ascending=False
        ).head(1)

        print(f"\nRating: {rating}")

        for _, row in examples.iterrows():
            print("Review:")
            print(row["text_"])
            print("Review length:", row["review_length"], "characters")
            print("-" * 100)
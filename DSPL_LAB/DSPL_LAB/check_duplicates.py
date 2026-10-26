import pandas as pd

df = pd.read_csv("data/fake reviews dataset.csv")

duplicate_texts = df[df["text_"].duplicated(keep=False)].sort_values("text_")

print(duplicate_texts[["text_", "rating", "label"]].to_string(index=False))
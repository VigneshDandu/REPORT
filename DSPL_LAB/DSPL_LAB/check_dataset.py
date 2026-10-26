import pandas as pd

file_path = "data/fake reviews dataset.csv"

df = pd.read_csv(file_path)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nRating distribution:")
print(df["rating"].value_counts())

print("\nNumber of unique categories:")
print(df["category"].nunique())

print("\nCategories:")
print(df["category"].value_counts())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nNumber of duplicate reviews:")
print(df["text_"].duplicated().sum())
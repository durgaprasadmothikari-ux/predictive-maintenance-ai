import pandas as pd

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Check dataset size
print("Dataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check machine failures
print("\nMachine Failure Count:")
print(df["Machine failure"].value_counts())
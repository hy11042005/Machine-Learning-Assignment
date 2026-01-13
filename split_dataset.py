import pandas as pd

# Load original large dataset
df = pd.read_csv("IMDB Dataset.csv")

# Shuffle dataset (important)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 3 equal parts
n = len(df)
part_size = n // 3

df_part1 = df.iloc[:part_size]
df_part2 = df.iloc[part_size:2*part_size]
df_part3 = df.iloc[2*part_size:]

# Save to smaller CSV files
df_part1.to_csv("IMDB_part1.csv", index=False)
df_part2.to_csv("IMDB_part2.csv", index=False)
df_part3.to_csv("IMDB_part3.csv", index=False)

print("Dataset has been split into 3 smaller files.")

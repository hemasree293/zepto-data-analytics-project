import pandas as pd
import seaborn as sns


# ============================================================
# MODULE 2 — EDA
# ============================================================

# Load Titanic dataset ONCE
df = sns.load_dataset("titanic")

# Save offline fallback immediately
df.to_csv("analytics/titanic.csv", index=False)


# ============================================================
# TASK 1 — DATA PROFILING
# ============================================================

print("\n========== DATA INFO ==========")
print(df.info())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# ============================================================
# MISSING VALUE PERCENTAGES
# ============================================================

missing_percentage = df.isnull().mean() * 100

print("\n========== MISSING VALUE PERCENTAGES ==========")

for column, percentage in missing_percentage.items():
    if percentage > 0:
        print(f"{column}: {percentage:.2f}%")

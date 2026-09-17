import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD TITANIC DATASET
# ============================================================

print("\n========== LOADING TITANIC DATASET ==========")

df = sns.load_dataset("titanic")

print(df.head())


# Save immediately as the required offline fallback
df.to_csv("analytics/titanic.csv", index=False)


# ============================================================
# 2. INITIAL DATA PROFILING
# ============================================================

print("\n========== DATA INFO ==========")
print(df.info())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")
missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_percentage)

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ============================================================
# 3. MISSING VALUE CLEANING
# ============================================================

print("\n========== MISSING VALUE CLEANING ==========")

# Age: 19.87% missing → 5–30% → median imputation
age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

# Embarked and embark_town: 0.22% missing → below 5% → drop rows
df = df.dropna(subset=["embarked", "embark_town"])

# Deck: 77.22% missing → very high missingness → drop column
df = df.drop(columns=["deck"])


# Create age groups AFTER age has been imputed
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)

# Family size
df["family_size"] = df["sibsp"] + df["parch"] + 1


print("Missing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset shape:", df.shape)


# Save the final cleaned dataset to the same required fallback file
df.to_csv("analytics/titanic.csv", index=False)

print("\nCleaned Titanic dataset saved to analytics/titanic.csv")


# ============================================================
# 4. BASIC DISTRIBUTIONS
# ============================================================

print("\n========== SEX DISTRIBUTION ==========")
print(df["sex"].value_counts())

print("\n========== PASSENGER CLASS DISTRIBUTION ==========")
print(df["pclass"].value_counts().sort_index())

print("\n========== SURVIVAL DISTRIBUTION ==========")
print(df["survived"].value_counts().sort_index())

print("\n========== SURVIVAL PERCENTAGE ==========")
print(
    (df["survived"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 5. SURVIVAL ANALYSIS
# ============================================================

print("\n========== SURVIVAL BY GENDER ==========")
print(pd.crosstab(df["sex"], df["survived"]))

print("\n========== SURVIVAL RATE BY GENDER ==========")
print(
    df.groupby("sex")["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY PASSENGER CLASS ==========")
print(
    df.groupby("pclass")["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY AGE GROUP ==========")
print(
    df.groupby("age_group", observed=False)["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY GENDER AND CLASS ==========")
print(
    df.groupby(["sex", "pclass"])["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY EMBARKATION TOWN ==========")
print(
    df.groupby("embark_town")["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY TRAVEL STATUS ==========")
print(
    df.groupby("alone")["survived"]
      .mean()
      .mul(100)
      .round(2)
)

print("\n========== SURVIVAL RATE BY FAMILY SIZE ==========")
print(
    df.groupby("family_size")["survived"]
      .mean()
      .mul(100)
      .round(2)
)


# ============================================================
# 6. BOOLEAN MASKING
# ============================================================

print("\n========== BOOLEAN MASKING: FEMALE SURVIVAL ==========")

female_passengers = df[
    df["sex"] == "female"
]

print(
    "Female survival rate:",
    round(female_passengers["survived"].mean() * 100, 2),
    "%"
)


print("\n========== BOOLEAN MASKING: FEMALE + FIRST CLASS ==========")

female_first_class = df[
    (df["sex"] == "female") & (df["pclass"] == 1)
]

print(
    "Number of female first-class passengers:",
    len(female_first_class)
)

print(
    "Survival rate:",
    round(female_first_class["survived"].mean() * 100, 2),
    "%"
)


print("\n========== BOOLEAN MASKING: FEMALE OR FIRST CLASS ==========")

female_or_first_class = df[
    (df["sex"] == "female") | (df["pclass"] == 1)
]

print(
    "Number of female or first-class passengers:",
    len(female_or_first_class)
)

print(
    "Survival rate:",
    round(female_or_first_class["survived"].mean() * 100, 2),
    "%"
)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

print("\n========== CORRELATION MATRIX ==========")

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix.round(2))


# ============================================================
# 8. STRONGEST CORRELATIONS
# ============================================================

print("\n========== STRONGEST CORRELATIONS ==========")

corr_pairs = (
    correlation_matrix
    .where(~correlation_matrix.eq(1))
    .abs()
    .stack()
    .sort_values(ascending=False)
)

print(corr_pairs.head(4))


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

print("\n========== CORRELATION HEATMAP ==========")

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()

plt.savefig("analytics/correlation_heatmap.png")
plt.show()


# ============================================================
# 10. AGE HISTOGRAM
# ============================================================

print("\n========== AGE HISTOGRAM ==========")

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("analytics/age_histogram.png")
plt.show()


# ============================================================
# 11. AGE BOXPLOT
# ============================================================

print("\n========== AGE BOXPLOT ==========")

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="age"
)

plt.title("Boxplot of Passenger Age")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig("analytics/age_boxplot.png")
plt.show()


# ============================================================
# 12. FARE HISTOGRAM
# ============================================================

print("\n========== FARE HISTOGRAM ==========")

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("analytics/fare_histogram.png")
plt.show()


# ============================================================
# 13. FARE BOXPLOT
# ============================================================

print("\n========== FARE BOXPLOT ==========")

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fare"
)

plt.title("Boxplot of Passenger Fare")
plt.xlabel("Fare")
plt.tight_layout()

plt.savefig("analytics/fare_boxplot.png")
plt.show()


# ============================================================
# 14. IQR OUTLIER COUNTS
# ============================================================

print("\n========== IQR OUTLIER COUNTS ==========")


# ---------- AGE OUTLIERS ----------

age_q1 = df["age"].quantile(0.25)
age_q3 = df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = df[
    (df["age"] < age_lower) |
    (df["age"] > age_upper)
]

print("Age outliers:", len(age_outliers))


# ---------- FARE OUTLIERS ----------

fare_q1 = df["fare"].quantile(0.25)
fare_q3 = df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = df[
    (df["fare"] < fare_lower) |
    (df["fare"] > fare_upper)
]

print("Fare outliers:", len(fare_outliers))


# ============================================================
# 15. FARE CENTRAL TENDENCY AND SKEWNESS
# ============================================================

print("\n========== FARE CENTRAL TENDENCY ==========")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("Fare mean:", round(fare_mean, 2))
print("Fare median:", round(fare_median, 2))
print("Fare mode:", round(fare_mode, 2))


print("\nSkewness conclusion:")

if fare_mean > fare_median > fare_mode:
    print(
        "Fare distribution is right-skewed (positively skewed)."
    )
elif fare_mean < fare_median < fare_mode:
    print(
        "Fare distribution is left-skewed (negatively skewed)."
    )
else:
    print(
        "Fare distribution does not follow a simple "
        "mean-median-mode skewness pattern."
    )


# ============================================================
# 16. STANDARDIZATION — EXPLORATORY ONLY
# ============================================================

print("\n========== STANDARDIZATION CHECK ==========")

scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    df[["age", "fare"]]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=["age_z", "fare_z"]
)

print("\nBefore standardization:")
print(
    df[["age", "fare"]]
    .agg(["mean", "std"])
    .round(4)
)

print("\nAfter standardization:")
print(
    standardized_df
    .agg(["mean", "std"])
    .round(4)
)

print(
    "\nStandardization is exploratory only and "
    "is not used as modeling preprocessing."
)


# ============================================================
# END
# ============================================================

print("\n========== EDA PIPELINE COMPLETED ==========")
print("Final dataset shape:", df.shape)
print("Final columns:")
print(df.columns.tolist())

print("\n========== SURVIVAL RATE BY GENDER CHART ==========")

gender_survival = (
    df.groupby("sex")["survived"]
      .mean()
      .mul(100)
)

plt.figure(figsize=(7, 5))

gender_survival.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("analytics/survival_by_gender.png")
plt.show()

print("\n========== SURVIVAL RATE BY CLASS CHART ==========")

class_survival = (
    df.groupby("pclass")["survived"]
      .mean()
      .mul(100)
)

plt.figure(figsize=(7, 5))

class_survival.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("analytics/survival_by_class.png")
plt.show()

print("\n========== SURVIVAL BY GENDER AND CLASS CHART ==========")

gender_class_survival = (
    df.groupby(["sex", "pclass"])["survived"]
      .mean()
      .mul(100)
      .unstack()
)

plt.figure(figsize=(8, 5))

gender_class_survival.plot(kind="bar")

plt.title("Survival Rate by Gender and Passenger Class")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Passenger Class")
plt.tight_layout()

plt.savefig("analytics/survival_by_gender_class.png")
plt.show()

print("\n========== SURVIVAL BY AGE GROUP CHART ==========")

age_group_survival = (
    df.groupby("age_group", observed=False)["survived"]
      .mean()
      .mul(100)
)

plt.figure(figsize=(8, 5))

age_group_survival.plot(kind="bar")

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("analytics/survival_by_age_group.png")
plt.show()

# Interpretation:
# Female passengers had a substantially higher survival rate than male passengers.
# The survival rate was approximately 74.20% for females compared with 18.89% for males.
# This indicates a strong association between gender and survival in the Titanic dataset.

# Interpretation:
# First-class passengers had the highest survival rate, followed by second-class passengers.
# Third-class passengers had the lowest survival rate among the three passenger classes.
# This indicates a clear association between passenger class and survival in the dataset.

# Interpretation:
# Survival rates varied across both gender and passenger class.
# Female passengers generally had higher survival rates than male passengers within the same class.
# The chart also shows that passenger class was associated with differences in survival rates for both genders.

# Interpretation:
# Survival rates differed across the age groups in the Titanic dataset.
# The chart shows that survival was not uniform across age categories.
# This suggests that age group was associated with differences in survival outcomes.




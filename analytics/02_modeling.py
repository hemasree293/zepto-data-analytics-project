import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== LOADING CLEANED TITANIC DATA ==========")

df = pd.read_csv("analytics/titanic.csv")

print("Dataset shape:", df.shape)

print("\nClass balance:")
print(df["survived"].value_counts())

print("\nClass balance percentage:")
print(
    (df["survived"].value_counts(normalize=True) * 100).round(2)
)

from sklearn.model_selection import train_test_split

print("\n========== STRATIFIED TRAIN/TEST SPLIT ==========")

X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining target distribution:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTest target distribution:")
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n========== PREPROCESSING SETUP ==========")

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Preprocessing pipeline created successfully.")

from sklearn.linear_model import LogisticRegression

print("\n========== LOGISTIC REGRESSION ==========")

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

logistic_pipeline.fit(X_train, y_train)

logistic_predictions = logistic_pipeline.predict(X_test)

print("Logistic Regression trained successfully.")
print("Number of test predictions:", len(logistic_predictions))

from sklearn.tree import DecisionTreeClassifier

print("\n========== DECISION TREE ==========")

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]
)

decision_tree_pipeline.fit(X_train, y_train)

decision_tree_predictions = decision_tree_pipeline.predict(X_test)

print("Decision Tree trained successfully.")
print("Number of test predictions:", len(decision_tree_predictions))

from sklearn.ensemble import RandomForestClassifier

print("\n========== RANDOM FOREST ==========")

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

random_forest_pipeline.fit(X_train, y_train)

random_forest_predictions = random_forest_pipeline.predict(X_test)

print("Random Forest trained successfully.")
print("Number of test predictions:", len(random_forest_predictions))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("\n========== CLASSIFICATION METRICS ==========")

models = {
    "Logistic Regression": (
        logistic_pipeline,
        logistic_predictions
    ),
    "Decision Tree": (
        decision_tree_pipeline,
        decision_tree_predictions
    ),
    "Random Forest": (
        random_forest_pipeline,
        random_forest_predictions
    )
}

classification_results = []

for model_name, (model, predictions) in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions)

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(f"\n--- {model_name} ---")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))
    print("Confusion Matrix:")
    print(cm)

classification_results_df = pd.DataFrame(classification_results)

print("\n========== MODEL COMPARISON TABLE ==========")
print(classification_results_df.round(4))

print("\n========== CONFUSION MATRICES ==========")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (model_name, (model, predictions)) in zip(
    axes,
    models.items()
):
    cm = confusion_matrix(y_test, predictions)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig("analytics/confusion_matrices.png")
plt.show()

from sklearn.metrics import roc_curve

print("\n========== ROC CURVES ==========")

plt.figure(figsize=(8, 6))

for model_name, (model, predictions) in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.title("ROC Curves for Titanic Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig("analytics/roc_curves.png")
plt.show()

print("\n========== CLASS BALANCE ANALYSIS ==========")

class_counts = y.value_counts().sort_index()
class_percentages = (
    y.value_counts(normalize=True)
     .sort_index()
     .mul(100)
     .round(2)
)

print("Class counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages)

print(
    "\nThe Titanic target variable contains two classes: "
    "0 = did not survive and 1 = survived."
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("\n========== CLASS IMBALANCE COMPARISON ==========")

# ------------------------------------------------------------
# 1. BASELINE LOGISTIC REGRESSION
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)


# ------------------------------------------------------------
# 2. CLASS_WEIGHT = BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

balanced_pipeline.fit(X_train, y_train)

balanced_pred = balanced_pipeline.predict(X_test)


# ------------------------------------------------------------
# 3. SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

smote_pipeline.fit(X_train, y_train)

smote_pred = smote_pipeline.predict(X_test)


# ------------------------------------------------------------
# COMPARE RESULTS
# ------------------------------------------------------------

imbalance_results = []

for name, predictions in [
    ("Baseline", baseline_pred),
    ("Class Weight Balanced", balanced_pred),
    ("SMOTE", smote_pred)
]:

    imbalance_results.append({
        "Method": name,
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions)
    })


imbalance_results_df = pd.DataFrame(imbalance_results)

print("\n========== IMBALANCE RESULTS ==========")

print(
    imbalance_results_df.round(4)
)

from sklearn.model_selection import GridSearchCV

print("\n========== RANDOM FOREST GRID SEARCH ==========")

grid_rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            oob_score=True
        ))
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=grid_rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(round(grid_search.best_score_, 4))

best_rf_pipeline = grid_search.best_estimator_

print("\nRandom Forest OOB score:")
print(
    round(
        best_rf_pipeline.named_steps["model"].oob_score_,
        4
    )
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n========== FARE REGRESSION ==========")

# Features used to predict fare
regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]

# Train/test split for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Separate preprocessing for regression
regression_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ),
            regression_numeric_features
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            ),
            regression_categorical_features
        )
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", regression_preprocessor),
        ("model", LinearRegression())
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

regression_predictions = regression_pipeline.predict(
    X_reg_test
)

print("Fare regression model trained successfully.")

print("\n========== REGRESSION METRICS ==========")

mae = mean_absolute_error(
    y_reg_test,
    regression_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        regression_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    regression_predictions
)

# Number of predictors after one-hot encoding
X_reg_train_transformed = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_train)
)

p = X_reg_train_transformed.shape[1]
n = len(y_reg_test)

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²:", round(r2, 4))
print("Adjusted R²:", round(adjusted_r2, 4))

print("\n========== RESIDUAL PLOT ==========")

residuals = y_reg_test - regression_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    regression_predictions,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Fare Regression Residual Plot")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig("analytics/fare_residual_plot.png")
plt.show()

print("\n========== HETEROSCEDASTICITY CONCLUSION ==========")

print(
    "The residual plot shows evidence of heteroscedasticity "
    "because the spread of residuals increases as predicted "
    "fare increases. The residual variance is therefore not "
    "constant across the range of predicted fares."
)

print("\n========== FINAL MODEL COMPARISON ==========")

final_classification_table = classification_results_df[
    ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].copy()

print("\nCLASSIFICATION MODELS:")
print(final_classification_table.round(4).to_string(index=False))

print("\nREGRESSION MODEL:")
print("Model: Linear Regression")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²:", round(r2, 4))
print("Adjusted R²:", round(adjusted_r2, 4))

import joblib

print("\n========== SAVE COMPLETE MODEL PIPELINE ==========")

joblib.dump(
    best_rf_pipeline,
    "analytics/titanic_best_pipeline.joblib"
)

print(
    "Complete pipeline saved to "
    "analytics/titanic_best_pipeline.joblib"
)

print("\n========== RELOAD SAVED PIPELINE ==========")

loaded_pipeline = joblib.load(
    "analytics/titanic_best_pipeline.joblib"
)

# Raw passenger input
raw_passenger = X_test[
    ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
].iloc[[0]]

print("\nRaw passenger input:")
print(raw_passenger)

prediction = loaded_pipeline.predict(raw_passenger)

print("\nPrediction:")
print(prediction)

if prediction[0] == 1:
    print("Predicted outcome: Survived")
else:
    print("Predicted outcome: Did not survive")

print("\nSaved pipeline reloaded and prediction completed successfully.")


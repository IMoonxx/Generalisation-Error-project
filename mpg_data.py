
# PHASE 6 — REAL-WORLD VALIDATION
# Auto MPG Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 1. LOAD DATASET

print("=" * 70)
print("PHASE 6 — REAL-WORLD VALIDATION")
print("=" * 70)

column_names = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

df = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    names=column_names,
    na_values="?"
)

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# 2. DATA INSPECTION
print("\n" + "=" * 70)
print("DATA INSPECTION")
print("=" * 70)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())



# 3. CLEANING

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

before = len(df)

# Horsepower contains missing values represented by '?'
df = df.dropna(subset=["horsepower"])

after = len(df)

print("\nRows before cleaning:", before)
print("Rows after cleaning:", after)
print("Rows removed:", before - after)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# 4. DEFINE PREDICTOR AND TARGET

# Predictor:
# Horsepower
# Target:
# Miles per gallon (MPG)

X = df[["horsepower"]]
y = df["mpg"]

print("\n" + "=" * 70)
print("VARIABLE SELECTION")
print("=" * 70)

print("\nPredictor: Horsepower")
print("Target: MPG")

print("\nNumber of observations:", len(df))

# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# 6. MODEL EVALUATION FUNCTION

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    generalization_gap = (
        test_rmse - train_rmse
    )

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "mae": test_mae,
        "r2": test_r2,
        "generalization_gap": generalization_gap
    }



# 7. FIT POLYNOMIAL MODEL

degrees = [1, 2, 5]

results = []

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    metrics = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    metrics["degree"] = degree

    results.append(metrics)

    print(f"\nPolynomial Degree: {degree}")
    print(
        f"Training RMSE: "
        f"{metrics['train_rmse']:.4f}"
    )
    print(
        f"Test RMSE: "
        f"{metrics['test_rmse']:.4f}"
    )
    print(
        f"MAE: "
        f"{metrics['mae']:.4f}"
    )
    print(
        f"R²: "
        f"{metrics['r2']:.4f}"
    )
    print(
        f"Generalization Gap: "
        f"{metrics['generalization_gap']:.4f}"
    )


results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL RESULTS:")
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 8. SAVE MODEL RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    "real_data_model_results.csv",
    index=False
)

print(
    "\nModel results saved as "
    "real_data_model_results.csv"
)


# 9. GRAPH — MODEL COMPLEXITY


plt.figure(figsize=(9, 6))

plt.plot(
    results_df["degree"],
    results_df["test_rmse"],
    marker="o",
    linewidth=2
)

plt.xticks(degrees)

plt.xlabel("Polynomial Degree")
plt.ylabel("Test RMSE")

plt.title(
    "Model Complexity vs Test Error — Auto MPG"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "real_data_model_complexity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 10. GRAPH — TRAINING VS TEST ERROR

x = np.arange(len(degrees))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    results_df["train_rmse"],
    width,
    label="Training RMSE"
)

plt.bar(
    x + width / 2,
    results_df["test_rmse"],
    width,
    label="Test RMSE"
)

plt.xticks(
    x,
    [f"Degree {d}" for d in degrees]
)

plt.ylabel("RMSE")

plt.title(
    "Training vs Test Error — Auto MPG"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "real_data_training_vs_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 11. GRAPH — GENERALIZATION GAP

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["degree"].astype(str),
    results_df["generalization_gap"]
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Generalization Gap")

plt.title(
    "Generalization Gap — Auto MPG"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "real_data_generalization_gap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 12. SAMPLE-SIZE EXPERIMENT


print("\n" + "=" * 70)
print("REAL-DATA SAMPLE-SIZE EXPERIMENT")
print("=" * 70)

sample_sizes = [
    50,
    100,
    150,
    200,
    250,
    len(X_train)
]

# Remove duplicates and values larger than training set
sample_sizes = sorted(
    set(
        n for n in sample_sizes
        if n <= len(X_train)
    )
)

sample_results = []

for n in sample_sizes:

    # Fixed random state makes this reproducible
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=n,
        random_state=42
    )

    for degree in degrees:

        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )

        model.fit(
            X_sub,
            y_sub
        )

        predictions = model.predict(X_test)

        test_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        sample_results.append({
            "sample_size": n,
            "degree": degree,
            "test_rmse": test_rmse
        })


sample_results_df = pd.DataFrame(
    sample_results
)

print(
    "\nSample-size experiment results:"
)

print(
    sample_results_df.to_string(index=False)
)


# ------------------------------------------------------------
# SAVE SAMPLE-SIZE RESULTS
# ------------------------------------------------------------

sample_results_df.to_csv(
    "real_data_sample_size_results.csv",
    index=False
)

print(
    "\nSample-size results saved as "
    "real_data_sample_size_results.csv"
)



# 13. GRAPH — SAMPLE SIZE VS TEST ERROR

plt.figure(figsize=(10, 6))

for degree in degrees:

    temp = sample_results_df[
        sample_results_df["degree"] == degree
    ].sort_values("sample_size")

    plt.plot(
        temp["sample_size"],
        temp["test_rmse"],
        marker="o",
        linewidth=2,
        label=f"Degree {degree}"
    )

plt.xlabel("Training Sample Size")
plt.ylabel("Test RMSE")

plt.title(
    "Sample Size vs Test Error — Auto MPG"
)

plt.legend()

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "real_data_sample_size_effect.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 14. REAL DATA SUMMARY

print("\n" + "=" * 70)
print("PHASE 6 SUMMARY")
print("=" * 70)

best_row = results_df.loc[
    results_df["test_rmse"].idxmin()
]

print(
    "\nLowest test RMSE:"
)

print(
    f"Degree {int(best_row['degree'])}"
)

print(
    f"Test RMSE: "
    f"{best_row['test_rmse']:.4f}"
)

print(
    f"R²: "
    f"{best_row['r2']:.4f}"
)

print(
    f"Generalization Gap: "
    f"{best_row['generalization_gap']:.4f}"
)

print("\nReal-world validation complete!")

print("\nFiles created:")
print("- real_data_model_results.csv")
print("- real_data_sample_size_results.csv")
print("- real_data_model_complexity.png")
print("- real_data_training_vs_test.png")
print("- real_data_generalization_gap.png")
print("- real_data_sample_size_effect.png")

print("\n" + "=" * 70)


# RESEARCH PROJECT
# Phase 3 — Simulation Setup & Data Generation

import numpy as np
import pandas as pd
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

print("Libraries loaded successfully!")


# 2. REPRODUCIBLE RANDOM SEED

SEED = 42
rng = np.random.default_rng(SEED)

print("Random seed set to:", SEED)


# 3. DATA-GENERATING FUNCTION

def generate_data(n, noise_std, rng):
    """
    Generate synthetic data based on the true relationship:

        Y = 3X + 2X² + noise

    Parameters
    ----------
    n : int
        Number of observations.

    noise_std : float
        Standard deviation of the random noise.

    rng : numpy random generator
        Reproducible random-number generator.

    Returns
    -------
    X : numpy array
        Predictor variable.

    Y : numpy array
        Response variable.
    """

    # Generate X from a standard normal distribution
    X = rng.normal(loc=0, scale=1, size=n)

    # Generate random noise
    noise = rng.normal(
        loc=0,
        scale=noise_std,
        size=n
    )

    # True underlying relationship
    Y = 3 * X + 2 * (X ** 2) + noise

    return X, Y



# 4. GENERATE AN EXAMPLE DATASET


X, Y = generate_data(
    n=100,
    noise_std=2,
    rng=rng
)

print("\nFirst 10 X values:")
print(X[:10])

print("\nFirst 10 Y values:")
print(Y[:10])

print("\nNumber of observations:", len(X))



# 5. VISUALIZE THE DATA


plt.figure(figsize=(8, 5))

plt.scatter(X, Y, alpha=0.6)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simulated Data: n = 100, Noise = 2")

plt.grid(alpha=0.2)
plt.show()



# 6. COMPARE DIFFERENT
#    NOISE LEVELS

noise_levels = [0.5, 2, 5]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, noise in zip(axes, noise_levels):

    X_temp, Y_temp = generate_data(
        n=100,
        noise_std=noise,
        rng=rng
    )

    ax.scatter(
        X_temp,
        Y_temp,
        alpha=0.6
    )

    ax.set_title(f"Noise σ = {noise}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.2)

plt.suptitle(
    "Effect of Noise on the Simulated Data",
    fontsize=14
)

plt.tight_layout()
plt.show()



# 7. CHECK DATA


print("\n--- DATA CHECK ---")

print("Missing X values:", np.isnan(X).sum())
print("Missing Y values:", np.isnan(Y).sum())

print("Mean of X:", np.mean(X))
print("Mean of Y:", np.mean(Y))

print("Standard deviation of X:", np.std(X))
print("Standard deviation of Y:", np.std(Y))

print("\nPhase 3 foundation complete!")


# 8. MODEL-TRAINING FUNCTION


def train_model(X_train, y_train, degree):
    """
    Train a polynomial regression model.

    degree = 1  -> linear model
    degree = 2  -> quadratic model
    degree = 5  -> more complex model
    degree = 10 -> highly complex model
    """

    # Convert X into the correct shape for scikit-learn
    X_train = X_train.reshape(-1, 1)

    # Create polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # Train the model
    model.fit(X_train, y_train)

    return model


# 9. MODEL-EVALUATION FUNCTION


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a trained model using:
    RMSE, MAE, R², training error,
    test error and generalization gap.
    """

    # Reshape X for scikit-learn
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # RMSE
    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_predictions)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    # MAE
    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    # R²
    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    # Generalization gap
    generalization_gap = test_rmse - train_rmse

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "mae": test_mae,
        "r2": test_r2,
        "generalization_gap": generalization_gap
    }


# TEST ONE MODEL


# Generate data
X_test_data, Y_test_data = generate_data(
    n=100,
    noise_std=2,
    rng=rng
)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_test_data,
    Y_test_data,
    test_size=0.20,
    random_state=42
)

# Train a degree-2 model
model = train_model(
    X_train,
    y_train,
    degree=2
)

# Evaluate it
results = evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)

print("\n--- SINGLE MODEL TEST ---")
print("Training RMSE:", results["train_rmse"])
print("Test RMSE:", results["test_rmse"])
print("MAE:", results["mae"])
print("R²:", results["r2"])
print("Generalization Gap:", results["generalization_gap"])


# COMPARE MODEL COMPLEXITIES


degrees = [1, 2, 5, 10]

print("\n--- MODEL COMPARISON ---")

for degree in degrees:

    model = train_model(
        X_train,
        y_train,
        degree=degree
    )

    results = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(f"\nPolynomial Degree: {degree}")
    print(f"Training RMSE: {results['train_rmse']:.3f}")
    print(f"Test RMSE: {results['test_rmse']:.3f}")
    print(f"MAE: {results['mae']:.3f}")
    print(f"R²: {results['r2']:.3f}")
    print(
        f"Generalization Gap: "
        f"{results['generalization_gap']:.3f}"
    )


# 9. SINGLE SIMULATION FUNCTION


def run_single_simulation(
    n,
    noise_std,
    degree,
    rng
):
    """
    Generate data, split it, train a model,
    and evaluate its performance.
    """

    # Generate synthetic data
    X, y = generate_data(
        n=n,
        noise_std=noise_std,
        rng=rng
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Train model
    model = train_model(
        X_train,
        y_train,
        degree
    )

    # Evaluate model
    results = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Add experimental conditions
    results["sample_size"] = n
    results["noise"] = noise_std
    results["degree"] = degree

    return results

result = run_single_simulation(
    n=100,
    noise_std=2,
    degree=5,
    rng=rng
)

print(result)


# 10. SMALL EXPERIMENT — PIPELINE TEST


sample_sizes = [50, 100, 250, 500, 1000, 5000]
noise_levels = [0.5, 2, 5]
degrees = [1, 2, 5, 10]
repetitions = 50

small_results = []

for n in sample_sizes:

    for noise in noise_levels:

        for degree in degrees:

            for repetition in range(repetitions):

                result = run_single_simulation(
                    n=n,
                    noise_std=noise,
                    degree=degree,
                    rng=rng
                )

                # Record repetition number
                result["repetition"] = repetition + 1

                small_results.append(result)


# Convert results into a DataFrame
small_results_df = pd.DataFrame(small_results)


# Display results
print("\n--- STARTING FULL SIMULATION ---")

full_results = []

for n in sample_sizes:

    for noise in noise_levels:

        for degree in degrees:

            for repetition in range(repetitions):

                result = run_single_simulation(
                    n=n,
                    noise_std=noise,
                    degree=degree,
                    rng=rng
                )

                result["repetition"] = repetition + 1

                full_results.append(result)


# Convert results to DataFrame
results_df = pd.DataFrame(full_results)


# Save results
results_df.to_csv(
    "simulation_results.csv",
    index=False
)


# Final summary
print("\n--- FULL SIMULATION COMPLETE ---")

print("Number of rows:", len(results_df))

print("\nColumns:")
print(results_df.columns.tolist())

print("\nFirst 5 rows:")
print(results_df.head())

print("\nResults saved as:")
print("simulation_results.csv")


# 11. CHECK SMALL EXPERIMENT FOR ERRORS


print("\n--- ERROR CHECK ---")

# Check expected number of rows
expected_rows = (
    len(sample_sizes)
    * len(noise_levels)
    * len(degrees)
    * repetitions
)

print("Expected rows:", expected_rows)
print("Actual rows:", len(small_results_df))

# Check missing values
print(
    "\nMissing values:",
    small_results_df.isna().sum().sum()
)

# Check infinite values
print(
    "Infinite values:",
    np.isinf(
        small_results_df.select_dtypes(
            include=np.number
        )
    ).sum().sum()
)

# Check that RMSE and MAE are non-negative
print(
    "Negative RMSE values:",
    (small_results_df["train_rmse"] < 0).sum()
    + (small_results_df["test_rmse"] < 0).sum()
)

print(
    "Negative MAE values:",
    (small_results_df["mae"] < 0).sum()
)

# Check that all expected experimental conditions exist
print(
    "\nSample sizes tested:",
    sorted(small_results_df["sample_size"].unique())
)

print(
    "Noise levels tested:",
    sorted(small_results_df["noise"].unique())
)

print(
    "Model degrees tested:",
    sorted(small_results_df["degree"].unique())
)

print(
    "Repetitions tested:",
    sorted(small_results_df["repetition"].unique())
)


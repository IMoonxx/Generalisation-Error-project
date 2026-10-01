# ============================================================
# PHASE 4 — MAIN ANALYSIS
# ============================================================
#
# This file analyses the results generated during Phase 3.
# It does NOT rerun the simulation.
#
# Main analysis models:
#   Degree 1, Degree 2, Degree 5
#
# Degree 10 is excluded from the PRIMARY analysis because
# exploratory analysis showed extreme numerical instability
# at small sample sizes.
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD RESULTS
# ------------------------------------------------------------

print("=" * 70)
print("PHASE 4 — MAIN ANALYSIS")
print("=" * 70)

FILE_NAME = "simulation_results.csv"

df = pd.read_csv(FILE_NAME)

print("\nSimulation results loaded successfully!")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. BASIC DATA CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC DATA CHECK")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSample sizes:")
print(sorted(df["sample_size"].unique()))

print("\nNoise levels:")
print(sorted(df["noise"].unique()))

print("\nModel degrees:")
print(sorted(df["degree"].unique()))

print("\nNumber of repetitions:")
print(df["repetition"].nunique())


# ------------------------------------------------------------
# 3. REMOVE DEGREE 10 FROM PRIMARY ANALYSIS
# ------------------------------------------------------------

main_degrees = [1, 2, 5]

main_df = df[df["degree"].isin(main_degrees)].copy()

print("\n" + "=" * 70)
print("PRIMARY ANALYSIS DATA")
print("=" * 70)

print("\nDegrees included:")
print(main_degrees)

print("\nDegrees excluded from primary analysis:")
print([d for d in sorted(df["degree"].unique()) if d not in main_degrees])

print(
    "\nReason: Degree 10 showed occasional extremely large "
    "test errors at small sample sizes."
)


# ------------------------------------------------------------
# 4. BOOTSTRAP CONFIDENCE INTERVAL
# ------------------------------------------------------------

def bootstrap_median_ci(values, n_boot=2000, confidence=0.95, seed=42):
    """
    Calculate a bootstrap confidence interval for the median.
    """

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    med = np.median(values)

    bootstrap_medians = []

    for _ in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_medians.append(np.median(sample))

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_medians,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_medians,
        100 * (1 - alpha / 2)
    )

    return med, lower, upper


# ------------------------------------------------------------
# 5. FUNCTION TO CREATE SUMMARY TABLES
# ------------------------------------------------------------

def summarize_by(group_columns, metric="test_rmse"):

    rows = []

    grouped = main_df.groupby(group_columns)

    for group_values, group in grouped:

        values = group[metric].values

        median, lower, upper = bootstrap_median_ci(values)

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {}

        for column, value in zip(group_columns, group_values):
            row[column] = value

        row["median"] = median
        row["ci_lower"] = lower
        row["ci_upper"] = upper

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS 1 — EFFECT OF SAMPLE SIZE
# ============================================================

print("\n" + "=" * 70)
print("1. EFFECT OF SAMPLE SIZE")
print("=" * 70)

# Keep noise fixed at 2.
# Compare model degrees 1, 2 and 5.

sample_df = main_df[
    main_df["noise"] == 2
].copy()

sample_summary = summarize_by(
    ["sample_size", "degree"],
    metric="test_rmse"
)

print("\nMedian Test RMSE by sample size and model degree:")
print(sample_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for degree in main_degrees:

    subset = sample_summary[
        sample_summary["degree"] == degree
    ].sort_values("sample_size")

    plt.plot(
        subset["sample_size"],
        subset["median"],
        marker="o",
        label=f"Degree {degree}"
    )

    plt.fill_between(
        subset["sample_size"],
        subset["ci_lower"],
        subset["ci_upper"],
        alpha=0.15
    )

plt.xscale("log")

plt.xlabel("Sample Size")
plt.ylabel("Median Test RMSE")
plt.title("Effect of Sample Size on Prediction Error")

plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    "phase4_sample_size_effect.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 2 — EFFECT OF MODEL COMPLEXITY
# ============================================================

print("\n" + "=" * 70)
print("2. EFFECT OF MODEL COMPLEXITY")
print("=" * 70)

# Fix sample size and noise.
# n = 100 gives us a meaningful setting where
# complexity differences are visible.

complexity_df = main_df[
    (main_df["sample_size"] == 100) &
    (main_df["noise"] == 2)
].copy()

complexity_summary = summarize_by(
    ["degree"],
    metric="test_rmse"
)

print("\nMedian Test RMSE by model complexity:")
print(complexity_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

x = complexity_summary["degree"]
y = complexity_summary["median"]

yerr_lower = y - complexity_summary["ci_lower"]
yerr_upper = complexity_summary["ci_upper"] - y

plt.errorbar(
    x,
    y,
    yerr=[yerr_lower, yerr_upper],
    marker="o",
    capsize=5,
    linewidth=2
)

plt.xticks(main_degrees)

plt.xlabel("Polynomial Degree")
plt.ylabel("Median Test RMSE")
plt.title(
    "Effect of Model Complexity\n"
    "(Sample Size = 100, Noise = 2)"
)

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    "phase4_model_complexity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 3 — EFFECT OF NOISE
# ============================================================

print("\n" + "=" * 70)
print("3. EFFECT OF NOISE")
print("=" * 70)

# Fix sample size and model complexity.
# n = 500 and degree = 2 gives a relatively stable comparison.

noise_df = main_df[
    (main_df["sample_size"] == 500) &
    (main_df["degree"] == 2)
].copy()

noise_summary = summarize_by(
    ["noise"],
    metric="test_rmse"
)

print("\nMedian Test RMSE by noise level:")
print(noise_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

x = noise_summary["noise"]
y = noise_summary["median"]

yerr_lower = y - noise_summary["ci_lower"]
yerr_upper = noise_summary["ci_upper"] - y

plt.errorbar(
    x,
    y,
    yerr=[yerr_lower, yerr_upper],
    marker="o",
    capsize=5,
    linewidth=2
)

plt.xlabel("Noise Level (σ)")
plt.ylabel("Median Test RMSE")
plt.title(
    "Effect of Noise on Prediction Error\n"
    "(Sample Size = 500, Degree = 2)"
)

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    "phase4_noise_effect.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 4 — GENERALIZATION GAP
# ============================================================

print("\n" + "=" * 70)
print("4. GENERALIZATION GAP")
print("=" * 70)

gap_summary = summarize_by(
    ["sample_size", "degree"],
    metric="generalization_gap"
)

gap_summary = gap_summary[
    gap_summary["degree"].isin(main_degrees)
]

print("\nMedian generalization gap:")
print(gap_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for degree in main_degrees:

    subset = gap_summary[
        gap_summary["degree"] == degree
    ].sort_values("sample_size")

    plt.plot(
        subset["sample_size"],
        subset["median"],
        marker="o",
        label=f"Degree {degree}"
    )

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xscale("log")

plt.xlabel("Sample Size")
plt.ylabel("Median Generalization Gap")
plt.title("Generalization Gap Across Sample Sizes")

plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    "phase4_generalization_gap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 5 — DIMINISHING RETURNS FROM MORE DATA
# ============================================================

print("\n" + "=" * 70)
print("5. DIMINISHING RETURNS FROM ADDITIONAL DATA")
print("=" * 70)

# Use degree 2 and noise = 2.
# This isolates the effect of sample size.

diminishing_df = main_df[
    (main_df["degree"] == 2) &
    (main_df["noise"] == 2)
].copy()

diminishing_summary = summarize_by(
    ["sample_size"],
    metric="test_rmse"
).sort_values("sample_size")

print("\nSample-size performance:")
print(diminishing_summary.round(3))


# Calculate percentage improvement
# between consecutive sample sizes.

diminishing_summary["previous_median"] = (
    diminishing_summary["median"].shift(1)
)

diminishing_summary["improvement_percent"] = (
    (
        diminishing_summary["previous_median"]
        - diminishing_summary["median"]
    )
    / diminishing_summary["previous_median"]
    * 100
)

print("\nDiminishing returns analysis:")
print(
    diminishing_summary[
        [
            "sample_size",
            "median",
            "improvement_percent"
        ]
    ].round(3)
)


# ------------------------------------------------------------
# Plot percentage improvement
# ------------------------------------------------------------

improvement_df = diminishing_summary.dropna(
    subset=["improvement_percent"]
)

plt.figure(figsize=(10, 6))

plt.plot(
    improvement_df["sample_size"],
    improvement_df["improvement_percent"],
    marker="o",
    linewidth=2
)

plt.xscale("log")

plt.xlabel("New Sample Size")
plt.ylabel("Improvement in Median Test RMSE (%)")
plt.title("Diminishing Returns from Additional Data")

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    "phase4_diminishing_returns.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 6 — SAMPLE SIZE × MODEL COMPLEXITY
# ============================================================

print("\n" + "=" * 70)
print("6. SAMPLE SIZE × MODEL COMPLEXITY")
print("=" * 70)

interaction_complexity = main_df[
    main_df["noise"] == 2
].copy()

interaction_complexity_summary = summarize_by(
    ["sample_size", "degree"],
    metric="test_rmse"
)

print("\nInteraction summary:")
print(interaction_complexity_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
    sharey=True
)

for ax, degree in zip(axes, main_degrees):

    subset = interaction_complexity_summary[
        interaction_complexity_summary["degree"] == degree
    ].sort_values("sample_size")

    ax.plot(
        subset["sample_size"],
        subset["median"],
        marker="o",
        linewidth=2
    )

    ax.set_xscale("log")

    ax.set_title(f"Polynomial Degree {degree}")
    ax.set_xlabel("Sample Size")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Median Test RMSE")

fig.suptitle(
    "Interaction Between Sample Size and Model Complexity",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "phase4_sample_size_complexity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 7 — SAMPLE SIZE × NOISE
# ============================================================

print("\n" + "=" * 70)
print("7. SAMPLE SIZE × NOISE")
print("=" * 70)

interaction_noise = main_df[
    main_df["degree"] == 2
].copy()

interaction_noise_summary = summarize_by(
    ["sample_size", "noise"],
    metric="test_rmse"
)

print("\nInteraction summary:")
print(interaction_noise_summary.round(3))


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
    sharey=True
)

noise_levels = sorted(
    interaction_noise_summary["noise"].unique()
)

for ax, noise in zip(axes, noise_levels):

    subset = interaction_noise_summary[
        interaction_noise_summary["noise"] == noise
    ].sort_values("sample_size")

    ax.plot(
        subset["sample_size"],
        subset["median"],
        marker="o",
        linewidth=2
    )

    ax.set_xscale("log")

    ax.set_title(f"Noise σ = {noise}")
    ax.set_xlabel("Sample Size")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("Median Test RMSE")

fig.suptitle(
    "Interaction Between Sample Size and Noise",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "phase4_sample_size_noise.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ANALYSIS 8 — VARIABILITY ACROSS REPETITIONS
# ============================================================

print("\n" + "=" * 70)
print("8. VARIABILITY ACROSS REPEATED SIMULATIONS")
print("=" * 70)

variability = main_df.groupby(
    ["sample_size", "degree", "noise"]
)["test_rmse"].agg(
    ["mean", "median", "std", "min", "max"]
).reset_index()

print("\nVariability summary:")
print(variability.round(3).head(30))


# ============================================================
# ANALYSIS 9 — IDENTIFY UNEXPECTED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("9. UNEXPECTED RESULTS / EXTREME OBSERVATIONS")
print("=" * 70)

# We investigate the primary analysis only.

extreme_threshold = main_df["test_rmse"].quantile(0.99)

extreme_results = main_df[
    main_df["test_rmse"] > extreme_threshold
].sort_values(
    "test_rmse",
    ascending=False
)

print(
    f"\n99th percentile of Test RMSE: "
    f"{extreme_threshold:.3f}"
)

print("\nLargest primary-analysis Test RMSE values:")

print(
    extreme_results[
        [
            "sample_size",
            "noise",
            "degree",
            "repetition",
            "train_rmse",
            "test_rmse",
            "generalization_gap"
        ]
    ].head(15).round(3)
)


# ============================================================
# ANALYSIS 10 — HYPOTHESIS CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. HYPOTHESIS CHECK")
print("=" * 70)

print(
    """
The following checks are descriptive rather than formal
hypothesis tests.

H1: Increasing sample size should generally reduce test error.

H2: Increasing model complexity may improve fit, but excessive
    complexity can increase instability/generalization error.

H3: Increasing noise should increase prediction error.

H4: The benefit of additional observations should eventually
    become smaller (diminishing returns).

H5: Larger sample sizes should generally reduce variability
    across repeated simulations.
"""
)


# ------------------------------------------------------------
# H1 — SAMPLE SIZE
# ------------------------------------------------------------

degree2_sample = diminishing_summary.sort_values(
    "sample_size"
)

first_error = degree2_sample.iloc[0]["median"]
last_error = degree2_sample.iloc[-1]["median"]

print("\nH1 — Sample Size:")
print(
    f"Median Test RMSE at smallest sample size: "
    f"{first_error:.3f}"
)

print(
    f"Median Test RMSE at largest sample size: "
    f"{last_error:.3f}"
)

if last_error < first_error:
    print("Observed pattern: Test error decreased.")
else:
    print("Observed pattern: Test error did not decrease.")


# ------------------------------------------------------------
# H2 — MODEL COMPLEXITY
# ------------------------------------------------------------

print("\nH2 — Model Complexity:")

for _, row in complexity_summary.iterrows():

    print(
        f"Degree {int(row['degree'])}: "
        f"Median Test RMSE = {row['median']:.3f}"
    )


# ------------------------------------------------------------
# H3 — NOISE
# ------------------------------------------------------------

print("\nH3 — Noise:")

for _, row in noise_summary.sort_values("noise").iterrows():

    print(
        f"Noise σ = {row['noise']}: "
        f"Median Test RMSE = {row['median']:.3f}"
    )


# ------------------------------------------------------------
# H4 — DIMINISHING RETURNS
# ------------------------------------------------------------

print("\nH4 — Diminishing Returns:")

print(
    diminishing_summary[
        [
            "sample_size",
            "improvement_percent"
        ]
    ].round(2)
)


# ------------------------------------------------------------
# H5 — VARIABILITY
# ------------------------------------------------------------

print("\nH5 — Variability:")

for degree in main_degrees:

    subset = variability[
        (variability["degree"] == degree) &
        (variability["noise"] == 2)
    ].sort_values("sample_size")

    print(f"\nDegree {degree}:")

    print(
        subset[
            ["sample_size", "std"]
        ].round(3).to_string(index=False)
    )


# ============================================================
# DISTRIBUTION-SHIFT NOTE
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION-SHIFT EXPERIMENT")
print("=" * 70)

print(
    """
The current simulation_results.csv contains:

- sample_size
- noise
- degree
- repetition
- training/test performance metrics

It does NOT contain a distribution-shift condition or a
shift/no-shift indicator.

Therefore, a genuine distribution-shift analysis cannot be
performed from the current Phase 3 results without running
an additional simulation experiment.

No distribution-shift result is being invented here.
"""
)


# ============================================================
# SAVE ANALYSIS TABLES
# ============================================================

sample_summary.to_csv(
    "phase4_sample_size_summary.csv",
    index=False
)

complexity_summary.to_csv(
    "phase4_complexity_summary.csv",
    index=False
)

noise_summary.to_csv(
    "phase4_noise_summary.csv",
    index=False
)

gap_summary.to_csv(
    "phase4_generalization_gap_summary.csv",
    index=False
)

diminishing_summary.to_csv(
    "phase4_diminishing_returns.csv",
    index=False
)

interaction_complexity_summary.to_csv(
    "phase4_interaction_complexity.csv",
    index=False
)

interaction_noise_summary.to_csv(
    "phase4_interaction_noise.csv",
    index=False
)

variability.to_csv(
    "phase4_variability.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 ANALYSIS COMPLETE")
print("=" * 70)

print(
    """
Analysis completed:

✓ Effect of sample size
✓ Effect of model complexity
✓ Effect of noise
✓ Generalization gaps
✓ Diminishing returns
✓ Sample size × model complexity
✓ Sample size × noise
✓ Confidence intervals
✓ Variability across repetitions
✓ Unexpected/extreme results
✓ Descriptive hypothesis checks

Degree 10 was excluded from the primary analysis because
exploratory testing revealed extreme instability.

Distribution-shift analysis requires an additional Phase 3
experiment because the current dataset does not contain
distribution-shift conditions.

All summary tables and graphs have been saved.
"""
)

print("\nGenerated graph files:")
print("1. phase4_sample_size_effect.png")
print("2. phase4_model_complexity.png")
print("3. phase4_noise_effect.png")
print("4. phase4_generalization_gap.png")
print("5. phase4_diminishing_returns.png")
print("6. phase4_sample_size_complexity.png")
print("7. phase4_sample_size_noise.png")

print("\nDone! 🎉")
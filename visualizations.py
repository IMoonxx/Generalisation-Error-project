# ============================================================
# PHASE 5 — VISUALISATION
# Polynomial Regression Simulation Study
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("simulation_results.csv")

print("=" * 60)
print("PHASE 5 — VISUALISATION")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. BASIC SETTINGS
# ------------------------------------------------------------

# Degree 10 is retained for diagnostics but excluded from
# the main visual story because of severe numerical instability.

MAIN_DEGREES = [1, 2, 5]

plot_df = df[df["degree"].isin(MAIN_DEGREES)].copy()


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def confidence_interval(series):
    """
    Approximate 95% confidence interval for the median-based
    visualisation using the standard error of the mean.

    Used here mainly to show uncertainty across repetitions.
    """
    series = series.dropna()

    mean = series.mean()
    se = series.std(ddof=1) / np.sqrt(len(series))

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    return mean, lower, upper


def grouped_summary(data, group_cols, metric="test_rmse"):

    summary = (
        data.groupby(group_cols)[metric]
        .agg(
            median="median",
            mean="mean",
            std="std",
            count="count"
        )
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    summary["ci_lower"] = (
        summary["mean"] - 1.96 * summary["se"]
    )

    summary["ci_upper"] = (
        summary["mean"] + 1.96 * summary["se"]
    )

    return summary


# ============================================================
# FIGURE 1
# EFFECT OF SAMPLE SIZE
# ============================================================

summary = grouped_summary(
    plot_df,
    ["sample_size", "degree"],
    "test_rmse"
)

plt.figure(figsize=(10, 6))

for degree in MAIN_DEGREES:

    temp = summary[summary["degree"] == degree].sort_values(
        "sample_size"
    )

    plt.plot(
        temp["sample_size"],
        temp["median"],
        marker="o",
        linewidth=2,
        label=f"Polynomial Degree {degree}"
    )

    plt.fill_between(
        temp["sample_size"],
        temp["ci_lower"],
        temp["ci_upper"],
        alpha=0.12
    )

plt.xscale("log")

plt.xlabel("Sample Size")
plt.ylabel("Median Test RMSE")

plt.title(
    "Effect of Sample Size on Prediction Error"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_1_sample_size_effect.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 2
# TRAINING VS TEST ERROR
# ============================================================

train_summary = (
    plot_df
    .groupby(["degree"])["train_rmse"]
    .median()
)

test_summary = (
    plot_df
    .groupby(["degree"])["test_rmse"]
    .median()
)

x = np.arange(len(MAIN_DEGREES))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    [train_summary[d] for d in MAIN_DEGREES],
    width,
    label="Training RMSE"
)

plt.bar(
    x + width / 2,
    [test_summary[d] for d in MAIN_DEGREES],
    width,
    label="Test RMSE"
)

plt.xticks(
    x,
    [f"Degree {d}" for d in MAIN_DEGREES]
)

plt.ylabel("Median RMSE")

plt.title(
    "Training vs Test Prediction Error"
)

plt.legend()
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_2_training_vs_test.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 3
# GENERALIZATION GAP
# ============================================================

gap_summary = grouped_summary(
    plot_df,
    ["sample_size", "degree"],
    "generalization_gap"
)

plt.figure(figsize=(10, 6))

for degree in MAIN_DEGREES:

    temp = gap_summary[
        gap_summary["degree"] == degree
    ].sort_values("sample_size")

    plt.plot(
        temp["sample_size"],
        temp["median"],
        marker="o",
        linewidth=2,
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

plt.title(
    "Generalization Gap Across Sample Sizes"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_3_generalization_gap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 4
# EFFECT OF NOISE
# ============================================================

noise_summary = grouped_summary(
    plot_df,
    ["noise", "degree"],
    "test_rmse"
)

plt.figure(figsize=(10, 6))

for degree in MAIN_DEGREES:

    temp = noise_summary[
        noise_summary["degree"] == degree
    ].sort_values("noise")

    plt.plot(
        temp["noise"],
        temp["median"],
        marker="o",
        linewidth=2,
        label=f"Degree {degree}"
    )

plt.xlabel("Noise Level (σ)")
plt.ylabel("Median Test RMSE")

plt.title(
    "Effect of Noise on Prediction Error"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_4_noise_effect.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 5
# MODEL COMPLEXITY
# ============================================================

complexity_summary = grouped_summary(
    plot_df,
    ["degree"],
    "test_rmse"
)

complexity_summary = complexity_summary.sort_values(
    "degree"
)

plt.figure(figsize=(9, 6))

plt.errorbar(
    complexity_summary["degree"],
    complexity_summary["mean"],
    yerr=1.96 * complexity_summary["se"],
    marker="o",
    linewidth=2,
    capsize=5
)

plt.xticks(MAIN_DEGREES)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Test RMSE")

plt.title(
    "Effect of Model Complexity on Prediction Error"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_5_model_complexity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 6
# SAMPLE SIZE × MODEL COMPLEXITY HEATMAP
# ============================================================

heatmap_data = (
    plot_df
    .groupby(["sample_size", "degree"])["test_rmse"]
    .median()
    .unstack()
)

plt.figure(figsize=(9, 6))

plt.imshow(
    heatmap_data,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(
    label="Median Test RMSE"
)

plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Sample Size")

plt.title(
    "Prediction Error Across Sample Size and Model Complexity"
)

plt.tight_layout()

plt.savefig(
    "figure_6_sample_size_complexity_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 7
# UNCERTAINTY / VARIABILITY
# ============================================================

uncertainty = (
    plot_df
    .groupby(["sample_size", "degree"])["test_rmse"]
    .agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    .reset_index()
)

plt.figure(figsize=(10, 6))

for degree in MAIN_DEGREES:

    temp = uncertainty[
        uncertainty["degree"] == degree
    ].sort_values("sample_size")

    plt.plot(
        temp["sample_size"],
        temp["median"],
        marker="o",
        linewidth=2,
        label=f"Degree {degree}"
    )

    plt.fill_between(
        temp["sample_size"],
        temp["q25"],
        temp["q75"],
        alpha=0.15
    )

plt.xscale("log")

plt.xlabel("Sample Size")
plt.ylabel("Test RMSE")

plt.title(
    "Prediction Error and Variability Across Sample Sizes"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "figure_7_uncertainty.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# DEGREE 10 DIAGNOSTIC
# ============================================================

degree10 = df[df["degree"] == 10]

print("\n" + "=" * 60)
print("DEGREE 10 DIAGNOSTIC")
print("=" * 60)

print(
    "\nMedian Test RMSE:",
    degree10["test_rmse"].median()
)

print(
    "Mean Test RMSE:",
    degree10["test_rmse"].mean()
)

print(
    "Maximum Test RMSE:",
    degree10["test_rmse"].max()
)

print(
    "Maximum Generalization Gap:",
    degree10["generalization_gap"].max()
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

final_summary = (
    plot_df
    .groupby("degree")
    .agg(
        median_test_rmse=("test_rmse", "median"),
        mean_test_rmse=("test_rmse", "mean"),
        median_train_rmse=("train_rmse", "median"),
        median_generalization_gap=(
            "generalization_gap",
            "median"
        )
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(final_summary.to_string(index=False))


print("\n" + "=" * 60)
print("PHASE 5 VISUALISATION COMPLETE")
print("=" * 60)
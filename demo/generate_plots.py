import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import ast

df = pd.read_csv("data-5.csv")
df.head()
# df[df["decoded_output"].isna()]

protocol = "FK12"
noise_model = "malicious"
restricted_df = df[
    (df["protocol"]==protocol)
    &
    (df["noise_model"]==noise_model)
    ]

threshold = 0.15
aborted_df = restricted_df[
    (restricted_df["n_failed_test_rounds"]>threshold*100)
    |
    (restricted_df["decoded_output"].isna())
    ]

aborted_df.to_csv("aborted.csv", index=False)

accepted_df = restricted_df[
    (restricted_df["n_failed_test_rounds"]<= threshold*100)
    &
    (restricted_df["decoded_output"].notna())
]
accepted_df.to_csv("accepted.csv", index=False)

corrupted_df = restricted_df[
    (restricted_df["decoded_output"].notna())
    &
    (restricted_df["match"] == "✗")
]
corrupted_df.to_csv("corrupted.csv", index=False)

# restricted_df[(restricted_df["n_failed_test_rounds"]<threshold*100) & (restricted_df["decoded_output"].notna())]

# === Parameters ===
threshold_values = [0.05, 0.08, 0.10, 0.15, 0.25, 0.5, 1]
total_test_rounds = 100

def not_none(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() != "none"

def plot_failure_and_wrong_accept_filtered(initial_df, thresholds, total_test_rounds):
    df0 = initial_df.copy()
    df0["failed_ratio"] = df0["n_failed_test_rounds"] / float(total_test_rounds)

    # Mean failed ratio per parameter (blue, plotted once)
    mean_failed_ratio = df0.groupby("parameter", as_index=True)["failed_ratio"].mean()
    xs_all = np.array(sorted(mean_failed_ratio.index))
    y_failed = mean_failed_ratio.loc[xs_all].values

    plt.figure(figsize=(8, 5))
    # Blue: mean failed-round ratio (always plotted)
    plt.scatter(xs_all, y_failed, label="Mean failed-round ratio", s=50, color="blue")

    # Precompute decoded presence
    decoded_present = df0["decoded_output"].apply(not_none)

    # Palette for thresholds
    cmap = plt.cm.get_cmap("tab10", len(thresholds))

    for i, thr in enumerate(thresholds):
        df_t = df0.copy()
        df_t["accepted"] = df_t["n_failed_test_rounds"] < thr * total_test_rounds
        df_t["wrongly_accepted"] = (
            df_t["accepted"] & decoded_present & (df_t["decoded_output"] != df_t["correct_value"])
        )

        grp = df_t.groupby("parameter", as_index=True)
        accepted_counts = grp["accepted"].sum()
        wrongly_accepted_counts = grp["wrongly_accepted"].sum()
        wrong_accept_ratio = wrongly_accepted_counts / accepted_counts.replace(0, np.nan)

        # Filter: only show points where the *mean failed ratio* < threshold
        mask = mean_failed_ratio.loc[xs_all].values < thr
        xs = xs_all[mask]
        if xs.size == 0:
            continue  # nothing to plot for this threshold

        y_wrong = wrong_accept_ratio.loc[xs].values
        # Drop parameters where there are no accepted circuits (NaN after division)
        valid = ~np.isnan(y_wrong)
        xs = xs[valid]
        y_wrong = y_wrong[valid]

        if xs.size == 0:
            continue

        plt.scatter(xs, y_wrong, s=50, color=cmap(i),
                    label=f"Wrongly-accepted / accepted @ thr={thr:.2f}")

    # Optional: draw threshold lines to give visual reference (for the largest thr only)
    plt.axhline(max(thresholds), linestyle="--", linewidth=1.2, alpha=0.4,
                label=f"ref line = {max(thresholds):.0%}")

    plt.xlabel("Parameter")
    plt.ylabel("Ratio")
    plt.title("Failure vs. Wrongly-Accepted Ratios by Parameter (filtered by threshold)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Example call
plot_failure_and_wrong_accept_filtered(restricted_df, threshold_values, total_test_rounds)

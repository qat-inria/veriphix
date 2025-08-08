import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load CSV file
df = pd.read_csv("noise_protocol_analysis.csv")

# Convert relevant columns
df["parameter"] = df["parameter"].astype(float)
df["n_failed_test_rounds"] = df["n_failed_test_rounds"].astype(int)

# List of protocols to generate separate plots for
protocols = df["protocol"].unique()

# Generate one plot per protocol, with two subplots side-by-side
for protocol in protocols:
    df_protocol = df[df["protocol"] == protocol]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Failed Test Rounds vs Noise Parameter — Protocol: {protocol}", fontsize=16)

    for i, noise_model in enumerate(["depolarising", "malicious"]):
        df_noise = df_protocol[df_protocol["noise_model"] == noise_model]

        axes[i].plot(
            df_noise["parameter"],
            df_noise["n_failed_test_rounds"],
            marker='o',
            linestyle='-',
        )

        axes[i].set_title(f"{noise_model.capitalize()} Noise")
        axes[i].set_xlabel("Noise Parameter")
        if i == 0:
            axes[i].set_ylabel("Failed Test Rounds")
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

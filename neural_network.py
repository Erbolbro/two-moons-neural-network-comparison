
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# 1. Create the Two Moons dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.25,
    random_state=42
)

# Store all experiment results
results = []


# 2. Repeat the experiment 10 times
for seed in range(10):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    # Scale using training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Compare three architectures
    for neurons in [4, 16, 64]:

        model = MLPClassifier(
            hidden_layer_sizes=(neurons,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            alpha=0.0001,
            max_iter=1000,
            n_iter_no_change=30,
            random_state=seed
        )

        # Measure training time
        start = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start

        # Evaluate accuracy
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_acc = accuracy_score(
            y_train, train_predictions
        )

        test_acc = accuracy_score(
            y_test, test_predictions
        )

        # Save each experiment result
        results.append({
            "seed": seed,
            "neurons": neurons,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "generalization_gap": train_acc - test_acc,
            "epochs": model.n_iter_,
            "correct_test": int(
                (test_predictions == y_test).sum()
            )
        })


# 4. Create DataFrame
df = pd.DataFrame(results)

# Each test set contains 200 samples
# correct_test contains the number of correct predictions

# Save results
df.to_csv("two_moons_results.csv", index=False)


# 5. Display all 30 experiments
print("\n========== ALL 30 EXPERIMENTS ==========\n")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# 6. Calculate mean and standard deviation
summary = df.groupby("neurons")[
    [
        "train_accuracy",
        "test_accuracy",
        "training_time",
        "generalization_gap",
        "epochs"
    ]
].agg(["mean", "std"])


# 7. Display full statistics
print("\n========== FULL STATISTICS ==========\n")

# Transpose to prevent wide table truncation
print(
    summary.T.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# 8. Easy-to-read model comparison
print("\n========== MODEL COMPARISON ==========\n")

for neurons, group in df.groupby("neurons"):

    print(f"NEURAL NETWORK: {neurons} neurons")

    print(
        f"Training accuracy: "
        f"{group['train_accuracy'].mean() * 100:.4f}%"
    )

    print(
        f"Test accuracy: "
        f"{group['test_accuracy'].mean() * 100:.4f}%"
    )

    print(
        f"Test accuracy SD: "
        f"{group['test_accuracy'].std() * 100:.4f}"
        f" percentage points"
    )

    print(
        f"Training time: "
        f"{group['training_time'].mean():.6f} seconds"
    )

    print(
        f"Training time SD: "
        f"{group['training_time'].std():.6f} seconds"
    )

    print(
        f"Average epochs: "
        f"{group['epochs'].mean():.2f}"
    )

    print(
        f"Generalization gap: "
        f"{group['generalization_gap'].mean() * 100:.4f}"
        f" percentage points"
    )

    print("-" * 45)


# 9. Simple accuracy chart in terminal
print("\n========== TERMINAL ACCURACY CHART ==========\n")

for neurons, group in df.groupby("neurons"):

    accuracy = group["test_accuracy"].mean() * 100
    bars = "█" * round(accuracy / 2)

    print(
        f"{neurons:>2} neurons | "
        f"{bars} {accuracy:.2f}%"
    )


# 10. Create and save graphs
for metric, ylabel, filename in [
    ("test_accuracy", "Test accuracy", "accuracy.png"),
    ("training_time", "Training time (seconds)", "time.png")
]:

    grouped = df.groupby("neurons")[metric]

    means = grouped.mean()
    stds = grouped.std()

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        means.index,
        means.values,
        yerr=stds.values,
        marker="o",
        capsize=5,
        linewidth=2
    )

    plt.xticks([4, 16, 64])
    plt.xlabel("Number of hidden neurons")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Hidden Neurons")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save graph as PNG
    plt.savefig(filename, dpi=300)

    # Display graph
    plt.show()

    # Close figure
    plt.close()


print("\nExperiment completed successfully!")
print("Results saved to two_moons_results.csv")
print("Graphs saved to accuracy.png and time.png")

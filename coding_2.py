import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Loading train and test data from CSV files
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Ensuring the train data is structured correctly
if "x" not in train_data.columns or train_data.shape[1] < 2:
    raise ValueError("Train data must include 'x' and at least one 'y' column.")
if "x" not in test_data.columns or "y" not in test_data.columns:
    raise ValueError("Test data must include 'x' and 'y' columns.")

# Step 1: Interpolation to align test x-values with train x-values
ideal_functions = train_data.set_index("x")  # Set 'x' as the index for interpolation
interpolated_ideal = ideal_functions.reindex(test_data["x"], method="nearest")  # Nearest interpolation

# Step 2: Calculating deviations
deviations = abs(interpolated_ideal.values - test_data["y"].values[:, None])
closest_function_idx = np.argmin(deviations, axis=1)
test_data["mapped_function"] = interpolated_ideal.columns[closest_function_idx]
test_data["deviation"] = deviations[np.arange(len(test_data)), closest_function_idx]

# Saving the results to a new CSV file
output_file = "mapped_results.csv"
test_data.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Step 3: Random Forest Model

# Training a Random Forest model on the training data
X_train = train_data[["x"]]
y_train = train_data.drop(columns=["x"])

# Train the RandomForestRegressor for each ideal function
rf_models = {}
predictions_rf = {}
for col in y_train.columns:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train[col])
    rf_models[col] = rf
    predictions_rf[col] = rf.predict(test_data[["x"]])

# Compute the mean squared error for each model (ideal function vs. Random Forest)
mse_rf = {col: mean_squared_error(test_data["y"], predictions_rf[col]) for col in predictions_rf}

# Step 4: Visualization
plt.figure(figsize=(12, 6))

# Plotting the ideal functions
for col in ideal_functions.columns:
    plt.plot(train_data["x"], train_data[col], label=f"Ideal Function {col}", linestyle="--")

# Scattering test data points
plt.scatter(test_data["x"], test_data["y"], color="red", label="Test Data (y)", zorder=5)

# Annotate mapped functions
for i, row in test_data.iterrows():
    plt.annotate(
        row["mapped_function"],
        (row["x"], row["y"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
        fontsize=8,
    )

plt.title("Ideal Functions and Test Data Mapping")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Residual plot (Test vs. Ideal)
plt.figure(figsize=(12, 6))
plt.scatter(test_data["x"], test_data["deviation"], color="green", label="Residuals (Deviations)")
plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
plt.title("Residual Plot (Test Data vs. Ideal Functions)")
plt.xlabel("x")
plt.ylabel("Deviation (Error)")
plt.legend()
plt.grid(True)

# Deviation distribution (Histogram)
plt.figure(figsize=(12, 6))
plt.hist(test_data["deviation"], bins=30, color="purple", alpha=0.7)
plt.title("Distribution of Deviations (Errors)")
plt.xlabel("Deviation")
plt.ylabel("Frequency")
plt.grid(True)

# Ideal Functions vs Test Data
plt.figure(figsize=(12, 6))
for col in ideal_functions.columns:
    plt.plot(train_data["x"], train_data[col], label=f"Ideal Function {col}", linestyle="--")
plt.scatter(test_data["x"], test_data["y"], color="red", label="Test Data (y)", zorder=5)
plt.title("Ideal Functions vs Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Mapping Accuracy (Test Data and Mapped Functions)
plt.figure(figsize=(12, 6))
for i, row in test_data.iterrows():
    plt.plot([row["x"], row["x"]], [0, row["y"]], color="blue", linestyle=":", alpha=0.6)
    plt.scatter(row["x"], row["y"], color="red", zorder=5)
    plt.scatter(row["x"], ideal_functions[row["mapped_function"]].loc[row["x"]], color="green", marker="x", zorder=5)
plt.title("Mapping Accuracy (Test Data vs Mapped Functions)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Mapping Accuracy", "Test Data", "Mapped Function"], loc="upper left")
plt.grid(True)

# Cumulative Sum of Deviations
cumulative_deviations = np.cumsum(test_data["deviation"])
plt.figure(figsize=(12, 6))
plt.plot(test_data["x"], cumulative_deviations, color="orange", label="Cumulative Sum of Deviations")
plt.title("Cumulative Sum of Deviations")
plt.xlabel("x")
plt.ylabel("Cumulative Deviation")
plt.legend()
plt.grid(True)

# Random Forest Predictions vs Ideal Functions
plt.figure(figsize=(12, 6))
for col in predictions_rf:
    plt.plot(test_data["x"], predictions_rf[col], label=f"RF Prediction for Ideal Function {col}", linestyle="-.")
plt.scatter(test_data["x"], test_data["y"], color="red", label="Test Data (y)", zorder=5)
plt.title("Random Forest Predictions vs Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Show all plots
plt.show()


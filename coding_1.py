import pandas as pd
import numpy as np

data = pd.read_csv("D:\Data\ideal.csv")

# Separating the columns into x values and y values
x_values = data["x"].values
y_columns = data.columns[1:]

# Defining a function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Placeholder to store RMSE values for each function
rmse_results = {}

# Assuming the ideal functions are in columns y1 to y50
for column in y_columns:
    y_actual = data[column].values
    rmse = calculate_rmse(x_values, y_actual)
    rmse_results[column] = rmse

# Sorting the functions by RMSE in ascending order
sorted_rmse = sorted(rmse_results.items(), key=lambda x: x[1])

# Selecting the top 4 functions with the lowest RMSE
ideal_functions = sorted_rmse[:4]

print("The four ideal functions are:")
for func, error in ideal_functions:
    print(f"{func}: RMSE = {error}")

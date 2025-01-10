import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, Integer
import numpy as np

# InitialisingSQLite database
engine = create_engine("sqlite:///functions_database.db")
metadata = MetaData()

# Training Data Table
training_table = Table(
    "training_data", metadata,
    Column("x", Float, primary_key=True),
    Column("y1", Float),
    Column("y2", Float),
    Column("y3", Float),
    Column("y4", Float),
)

# Ideal Functions Table
ideal_table = Table(
    "ideal_functions", metadata,
    Column("x", Float, primary_key=True),
    *[Column(f"y{i+1}", Float) for i in range(50)]  # y1 to y50
)

# Test Results Table
test_results_table = Table(
    "test_results", metadata,
    Column("x", Float),
    Column("y", Float),
    Column("delta_y", Float),  # Deviation
    Column("ideal_function_no", Integer)  # Ideal function index
)

# Creating tables in the database
metadata.create_all(engine)

# Loading CSV data into the database
def load_csv_to_table(csv_file, table_name):
    """Loads a CSV file into the specified SQLite table."""
    data = pd.read_csv(csv_file)
    data.to_sql(table_name, con=engine, if_exists="replace", index=False)

# Loading training and ideal functions into the database
load_csv_to_table("train.csv", "training_data")
load_csv_to_table("ideal.csv", "ideal_functions")

# Processing test data
test_data = pd.read_csv("test.csv")
ideal_df = pd.read_sql_table("ideal_functions", con=engine).set_index("x")

# Matching test data to ideal functions
results = []
for _, row in test_data.iterrows():
    x, y = row["x"], row["y"]
    if x in ideal_df.index:  
        deviations = abs(ideal_df.loc[x] - y)
        ideal_function_no = deviations.idxmin()  
        delta_y = deviations.min()
        results.append({
            "x": x,
            "y": y,
            "delta_y": delta_y,
            "ideal_function_no": int(ideal_function_no[1:]) 
        })

# Saving results to the test_results table
results_df = pd.DataFrame(results)
results_df.to_sql("test_results", con=engine, if_exists="replace", index=False)


import matplotlib.pyplot as plt


train_data = pd.read_sql_table("training_data", con=engine)
plt.figure(figsize=(12, 6))
for col in ["y1", "y2", "y3", "y4"]:
    plt.plot(train_data["x"], train_data[col], label=f"Training {col}", linestyle="--")


plt.scatter(test_data["x"], test_data["y"], color="red", label="Test Data (y)", zorder=5)


for _, row in results_df.iterrows():
    plt.annotate(
        f"f{row['ideal_function_no']}",
        (row["x"], row["y"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
        fontsize=8,
    )

plt.title("Training, Ideal Functions, and Test Data Mapping")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

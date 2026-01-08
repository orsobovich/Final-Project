import pandas as pd
from exploration import run_check

'''
Semi-automated manual test for the data loading / validation stage.

Goal:
Verify that the dataset was loaded correctly and meets
basic assumptions required for further analysis.
'''


# Load the dataset inside the test file
try:
    df = pd.read_csv("Mental_Health_Lifestyle_Dataset.csv")
except Exception as e:
    print("Failed to load dataset in validation test.")
    raise e


print("\nData Loading / Validation Checks:")

# Check 1: Verify that df is a pandas DataFrame
run_check("df is a pandas DataFrame",
          isinstance(df, pd.DataFrame))

# Check 2: Verify that the DataFrame is not empty
run_check("df is not empty",
          not df.empty)

# Check 3: Verify that the dataset has at least one row
run_check("df has at least one row",
          df.shape[0] > 0)

# Check 4: Verify that required columns exist
required_columns = [
    "Age",
    "Gender",
    "Sleep Hours",
    "Stress Level",
    "Happiness Score"
]

for col in required_columns:
    run_check(f"Required column '{col}' exists",
              col in df.columns)

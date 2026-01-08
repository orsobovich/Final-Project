import numpy as np
import pandas as pd
from exploration import numeric_ranges, run_check

# Create a small test DataFrame with known values
# Column A has different values, column B has the same value in all rows
test_df = pd.DataFrame({
    "A": [1, 2, 3, 4],
    "B": [10, 10, 10, 10]
})

# Run the function we want to test on the test DataFrame
result = numeric_ranges(test_df)

# Print the output so we can see the result clearly
print("\nOutput of numeric_ranges(test_df):")
print(result)

# Print a title before running the checks
print("\nChecks:")

# Check that the function returned a pandas DataFrame
run_check("Result is a DataFrame", isinstance(result, pd.DataFrame))

# Check that the output contains a 'min' column
run_check("Has column 'min'", "min" in result.columns)

# Check that the output contains a 'max' column
run_check("Has column 'max'", "max" in result.columns)

# Check that the output contains a 'mean' column
run_check("Has column 'mean'", "mean" in result.columns)

# Check that the output contains a 'std' column
run_check("Has column 'std'", "std" in result.columns)

# Check that the minimum value of column A is correct
run_check("A min == 1", result.loc["A", "min"] == 1)

# Check that the maximum value of column A is correct
run_check("A max == 4", result.loc["A", "max"] == 4)

# Check that the mean of column A is between its minimum and maximum
run_check(
    "A mean between min and max",
    result.loc["A", "min"] <= result.loc["A", "mean"] <= result.loc["A", "max"]
)

# Check that the standard deviation of column A is not negative
run_check("A std >= 0", result.loc["A", "std"] >= 0)

# Check that a constant column (B) has a standard deviation of 0
run_check("B std == 0 (constant column)", result.loc["B", "std"] == 0)

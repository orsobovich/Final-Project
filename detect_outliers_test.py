import numpy as np
import pandas as pd
from exploration import detect_outliers, run_check

'''
Assumption:
- detect_outliers(df, threshold) returns a pandas Series.
- The Series index contains the numeric column names.
- The Series values are counts of outliers in each numeric column.
'''

# Create a small test DataFrame for outlier testing
# X: has one very large value (100) compared to the others (0), so we expect at least 1 outlier
# Const: constant column (all 1s), so it should have 0 outliers
test_df = pd.DataFrame({
    "X": [0, 0, 0, 0, 100],
    "Const": [1, 1, 1, 1, 1]
})

# Run the outlier detection function on the test DataFrame
# threshold=2 makes the test more sensitive than the default 3
result = detect_outliers(test_df, threshold=2)

# Print the output so we can see what the function returned
print("\nOutput of detect_outliers(test_df, threshold=2):")
print(result)

# Print a title before running the checks
print("\nChecks:")

# Check 1: Make sure the function returned a pandas Series (expected output type)
run_check("Result is a Series", isinstance(result, pd.Series))

# Check 2: Make sure all outlier counts are non-negative (counts cannot be negative)
# fillna(0) is used just in case any column returned NaN
run_check("All counts are non-negative", (result.fillna(0) >= 0).all())

# Check 3: Constant column should have 0 outliers because all values are identical
run_check("Const outliers == 0", result.loc["Const"] == 0)

# Check 4: Column X should have at least 1 outlier due to the extreme value (100)
# We use >= 1 because depending on the exact Z-score calculation, it might count 1 or more
run_check("X outliers >= 1 (expected at least one)", result.loc["X"] >= 1)

'''
How to interpret the results:
- If all checks print True, the function behaves as expected on this test case.
- If one check prints False, it means the function output does not match expectations.
  Then we inspect the printed result to understand why.
'''

import numpy as np
import pandas as pd
from exploration import data_info, run_check
# Import numpy for using NaN values
import numpy as np

# Import pandas for creating and working with DataFrames
import pandas as pd

# Import the function we want to test (data_info)
# and the helper function used to print test results (run_check)
from exploration import data_info, run_check


'''
Semi-automated manual testing for data_info(df)

This script tests the function data_info(df) by:
- Running it on small DataFrames with known properties
- Checking expected results using simple True / False conditions
- Covering both a normal case and several edge cases
'''


'''
-------------------------
Test 1: Standard case
-------------------------
This is a normal DataFrame with:
- Numeric and categorical columns
- One missing value in the Age column
'''

# Create a small DataFrame with known values
test_df = pd.DataFrame({
    "Age": [20, 21, np.nan, 23],      # One missing value (np.nan)
    "Gender": ["F", "M", "F", "M"],   # Two unique categories
    "Score": [1, 2, 3, 4]             # All values are unique
})

# Run the function data_info on the test DataFrame
result = data_info(test_df)

# Print the output so we can see the summary table
print("\nTest 1 Output (standard case):")
print(result)

# Run checks to verify the output values
print("\nTest 1 Checks:")

# Check that Age has exactly 1 missing value
run_check("Age missing_count == 1",
          result.loc["Age", "missing_count"] == 1)

# Check that the missing percentage for Age is 25%
# (1 missing value out of 4 rows)
run_check("Age missing_percent == 25.0",
          result.loc["Age", "missing_percent"] == 25.0)

# Check that Age has 3 unique non-missing values
run_check("Age unique_values == 3",
          result.loc["Age", "unique_values"] == 3)

# Check that Gender has no missing values
run_check("Gender missing_count == 0",
          result.loc["Gender", "missing_count"] == 0)

# Check that Gender has exactly 2 unique categories (F and M)
run_check("Gender unique_values == 2",
          result.loc["Gender", "unique_values"] == 2)

# Check that Score has 4 unique values
run_check("Score unique_values == 4",
          result.loc["Score", "unique_values"] == 4)


'''
-------------------------
Edge Case 1: Empty DataFrame
-------------------------
This DataFrame has no rows and no columns.
The goal is to check that the function does not crash.
'''

# Create an empty DataFrame
empty_df = pd.DataFrame()

# Run data_info on the empty DataFrame
empty_result = data_info(empty_df)

# Print the result
print("\nEdge Case 1 Output (empty DataFrame):")
print(empty_result)

print("\nEdge Case 1 Checks:")

# Check that the function still returns a DataFrame
run_check("Returned object is a DataFrame",
          isinstance(empty_result, pd.DataFrame))

# Check that there are no rows in the output
# (no variables to summarize)
run_check("Output is empty (no rows)",
          empty_result.shape[0] == 0)


'''
-------------------------
Edge Case 2: Rows but no columns
-------------------------
This DataFrame has rows but no variables.
'''

# Create a DataFrame with 5 rows and no columns
no_cols_df = pd.DataFrame(index=range(5))

# Run data_info on it
no_cols_result = data_info(no_cols_df)

# Print the result
print("\nEdge Case 2 Output (rows but no columns):")
print(no_cols_result)

print("\nEdge Case 2 Checks:")

# Check that the output is still a DataFrame
run_check("Returned object is a DataFrame",
          isinstance(no_cols_result, pd.DataFrame))

# Check that there are no variables summarized
run_check("Output is empty (no rows)",
          no_cols_result.shape[0] == 0)


'''
-------------------------
Edge Case 3: Column with all missing values
-------------------------
All values are NaN.
'''

# Create a DataFrame where the column contains only missing values
all_missing_df = pd.DataFrame({
    "AllMissing": [np.nan, np.nan, np.nan]
})

# Run data_info
all_missing_result = data_info(all_missing_df)

# Print the result
print("\nEdge Case 3 Output (all missing column):")
print(all_missing_result)

print("\nEdge Case 3 Checks:")

# Check that all 3 values are counted as missing
run_check("AllMissing missing_count == 3",
          all_missing_result.loc["AllMissing", "missing_count"] == 3)

# Check that missing percentage is 100%
run_check("AllMissing missing_percent == 100.0",
          all_missing_result.loc["AllMissing", "missing_percent"] == 100.0)

# Check that there are 0 unique non-missing values
run_check("AllMissing unique_values == 0",
          all_missing_result.loc["AllMissing", "unique_values"] == 0)


'''
-------------------------
Edge Case 4: Constant column
-------------------------
All values are the same.
'''

# Create a DataFrame with a constant column
constant_df = pd.DataFrame({
    "Const": [1, 1, 1, 1]
})

# Run data_info
constant_result = data_info(constant_df)

# Print the result
print("\nEdge Case 4 Output (constant column):")
print(constant_result)

print("\nEdge Case 4 Checks:")

# Check that there are no missing values
run_check("Const missing_count == 0",
          constant_result.loc["Const", "missing_count"] == 0)

# Check that only one unique value is counted
run_check("Const unique_values == 1",
          constant_result.loc["Const", "unique_values"] == 1)


'''
-------------------------
Edge Case 5: Categorical column with None
-------------------------
None should be treated as a missing value.
'''

# Create a DataFrame with a missing value in a categorical column
none_in_cat_df = pd.DataFrame({
    "Gender": ["F", None, "M", "F"]
})

# Run data_info
none_in_cat_result = data_info(none_in_cat_df)

# Print the result
print("\nEdge Case 5 Output (categorical with None):")
print(none_in_cat_result)

print("\nEdge Case 5 Checks:")

# Check that None is counted as a missing value
run_check("Gender missing_count == 1",
          none_in_cat_result.loc["Gender", "missing_count"] == 1)

# Check that missing percentage is 25%
run_check("Gender missing_percent == 25.0",
          none_in_cat_result.loc["Gender", "missing_percent"] == 25.0)

# Check that only F and M are counted as unique values
run_check("Gender unique_values == 2",
          none_in_cat_result.loc["Gender", "unique_values"] == 2)


'''
-------------------------
Edge Case 6: Mixed data types
-------------------------
A column with numbers and strings together.
'''

# Create a DataFrame with mixed data types
mixed_df = pd.DataFrame({
    "Mixed": [1, "a", 3]
})

# Run data_info
mixed_result = data_info(mixed_df)

# Print the result
print("\nEdge Case 6 Output (mixed types):")
print(mixed_result)

print("\nEdge Case 6 Checks:")

# Check that all unique non-missing values are counted
run_check("Mixed unique_values == 3",
          mixed_result.loc["Mixed", "unique_values"] == 3)

# Check that there are no missing values
run_check("Mixed missing_count == 0",
          mixed_result.loc["Mixed", "missing_count"] == 0)


'''
If all or most checks return True,
the data_info(df) function behaves as expected
for both normal usage and common edge cases.
'''


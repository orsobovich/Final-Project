import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_check(label, condition):
    '''
    Helper function for testing.
    Prints whether a condition passed (True) or failed (False).
    '''
    try:
        # Print the label and the boolean result of the condition
        print(f"{label}: {bool(condition)}")
    except Exception as e:
        # Catch unexpected errors during the check
        print(f"{label}: Error -> {e}")



'''
Load the dataset from a CSV file.
Basic error handling is used to prevent the script from crashing
if the file path or name is incorrect.
'''
try:
    df = pd.read_csv("Mental_Health_Lifestyle_Dataset.csv")
except Exception as e:
    print("Could not load the dataset. Please check the file path.")
    print("Error:", e)

def overview(df, n=5):
    '''
    Prints a basic overview of the dataset:
    - dataset size (rows and columns)
    - column names
    - first n rows
    '''
    try:
        # Print the shape of the DataFrame
        print("Shape:", df.shape)

        # Print all column names
        print("\nColumns:")
        print(df.columns)

        # Print the first n rows for a quick inspection
        print("\nFirst rows:")
        print(df.head(n))
    except Exception as e:
        # Handle errors if df is not valid
        print("Error in overview():", e)


def data_info(df):
    '''
    Creates and returns a summary table for each variable:
    - data type
    - number of missing values
    - percentage of missing values
    - number of unique values
    '''
    try:
        # Build a summary DataFrame with key information per column
        info = pd.DataFrame({
            "dtype": df.dtypes,                      # Data type of each column
            "missing_count": df.isna().sum(),        # Number of missing values
            "missing_percent": df.isna().mean() * 100,  # Percentage of missing values
            "unique_values": df.nunique()            # Number of unique values
        })

        # Sort variables by missing percentage (descending)
        return info.sort_values("missing_percent", ascending=False)
    except Exception as e:
        print("Error in data_info():", e)
        return None

def descriptive_stats(df):
    '''
    Computes descriptive statistics separately for:
    - numeric variables
    - categorical variables
    '''
    try:
        # Select numeric columns and compute statistics (mean, std, etc.)
        numeric = df.select_dtypes(include=[np.number]).describe()

        # Select categorical columns and compute statistics (count, unique, top, freq)
        categorical = df.select_dtypes(exclude=[np.number]).describe(include="all")

        return numeric, categorical
    except Exception as e:
        print("Error in descriptive_stats():", e)
        return None, None

def categorical_frequencies(df, top_n=10):
    '''
    Prints frequency tables for categorical variables.
    Shows the top N most frequent values for each column.
    '''
    try:
        # Identify categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        # Loop over each categorical column
        for col in cat_cols:
            print(f"\n{col} (top {top_n})")

            # Print value counts, including missing values
            print(df[col].value_counts(dropna=False).head(top_n))
    except Exception as e:
        print("Error in categorical_frequencies():", e)


def numeric_ranges(df):
    '''
    Computes basic statistics for numeric variables:
    minimum, maximum, mean, and standard deviation.
    '''
    try:
        # Select only numeric columns
        num = df.select_dtypes(include=[np.number])

        # Aggregate statistics and transpose for readability
        return num.agg(["min", "max", "mean", "std"]).T
    except Exception as e:
        print("Error in numeric_ranges():", e)
        return None


def detect_outliers(df, threshold=3):
    '''
    Detects potential outliers using Z-scores.
    Counts values with absolute Z-score above the threshold.
    '''
    try:
        # Select numeric column names
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        # Loop over numeric columns
        for col in num_cols:
            # Remove missing values
            s = df[col].dropna()

            # If standard deviation is zero, no outliers are possible
            if s.std() == 0:
                outliers[col] = 0
                continue

            # Compute Z-scores
            z = (s - s.mean()) / s.std()

            # Count values exceeding the threshold
            outliers[col] = (np.abs(z) > threshold).sum()

        # Return results sorted by number of outliers
        return pd.Series(outliers).sort_values(ascending=False)
    except Exception as e:
        print("Error in detect_outliers():", e)
        return None


def plot_distribution(df, col):
    '''
    Plots a histogram for a selected numeric variable.
    Used for visual inspection of the distribution.
    '''
    try:
        # Check if the column exists
        if col not in df.columns:
            print(f"Column '{col}' not found.")
            return

        # Plot histogram
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()
    except Exception as e:
        print("Error in plot_distribution():", e)


'''
Example usage
'''
overview(df)

info_table = data_info(df)
print("\nMissing values summary:")
print(info_table.head(10))

num_desc, cat_desc = descriptive_stats(df)
print("\nNumeric descriptive statistics:")
print(num_desc)

print("\nCategorical descriptive statistics:")
print(cat_desc)

categorical_frequencies(df)

ranges = numeric_ranges(df)
print("\nNumeric ranges:")
print(ranges)

outliers = detect_outliers(df)
print("\nOutlier counts:")
print(outliers.head(10))

# Example plot (replace with an existing numeric column)
# plot_distribution(df, "Age")

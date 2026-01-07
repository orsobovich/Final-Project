import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    Provides a quick overview of the dataset.
    Prints the dataset shape, column names, and the first n rows.
    '''
    try:
        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns)
        print("\nFirst rows:")
        print(df.head(n))
    except Exception as e:
        print("Error in overview():", e)

def data_info(df):
    '''
    Returns a summary table including:
    - data type of each variable
    - number and percentage of missing values
    - number of unique values per variable
    '''
    try:
        info = pd.DataFrame({
            "dtype": df.dtypes,
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean() * 100,
            "unique_values": df.nunique()
        }).sort_values("missing_percent", ascending=False)
        return info
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
        numeric = df.select_dtypes(include=[np.number]).describe()
        categorical = df.select_dtypes(exclude=[np.number]).describe(include="all")
        return numeric, categorical
    except Exception as e:
        print("Error in descriptive_stats():", e)
        return None, None

def categorical_frequencies(df, top_n=10):
    '''
    Prints frequency tables for categorical variables.
    By default, shows the top N most frequent values.
    '''
    try:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            print(f"\n{col} (top {top_n})")
            print(df[col].value_counts(dropna=False).head(top_n))
    except Exception as e:
        print("Error in categorical_frequencies():", e)

def numeric_ranges(df):
    '''
    Returns minimum, maximum, mean, and standard deviation
    for all numeric variables.
    '''
    try:
        num = df.select_dtypes(include=[np.number])
        return num.agg(["min", "max", "mean", "std"]).T
    except Exception as e:
        print("Error in numeric_ranges():", e)
        return None

def detect_outliers(df, threshold=3):
    '''
    Detects potential outliers in numeric variables using Z-scores.
    Values with absolute Z-score greater than the threshold
    are counted as outliers.
    '''
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in num_cols:
            s = df[col].dropna()
            if s.std() == 0:
                outliers[col] = 0
                continue

            z = (s - s.mean()) / s.std()
            outliers[col] = (np.abs(z) > threshold).sum()

        return pd.Series(outliers).sort_values(ascending=False)
    except Exception as e:
        print("Error in detect_outliers():", e)
        return None

def plot_distribution(df, col):
    '''
    Plots a histogram to visually inspect the distribution
    of a selected numeric variable.
    '''
    try:
        if col not in df.columns:
            print(f"Column '{col}' not found.")
            return

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

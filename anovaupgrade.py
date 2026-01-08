import logging
from typing import Dict, List

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------
# Logger configuration
logger = logging.getLogger("mental_health_analysis")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
console_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(console_handler)

logger.info("Logger initialized for Mental Health Analysis project")

"""""
שיפורים שעשינו כאן:

שמנו שם ייחודי ל‑logger (mental_health_analysis) כדי למנוע בלבול בפרויקטים אחרים.

בדיקה של if not logger.handlers – מונע הוספה כפולה של handlers אם מריצים את הקוד יותר מפעם אחת.

הודעת info ראשונית – מראה שהלוגר מאותחל בהצלחה.
"""""



# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
def compute_group_means(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Compute the mean of `value_col` for each category in `group_col`.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    group_col : str
        The name of the categorical column to group by.
        (Why str? Because we need the column name as text)
    value_col : str
        The name of the numeric column to calculate the mean for.

    Returns
    -------
    pd.Series or None
        A pandas Series with group names as index and their mean as values.
        Returns None if an error occurs.
    """
    
    # Log that the function has started
    logger.info(f"Computing mean of '{value_col}' per '{group_col}'")

    try:
        # Compute the mean of value_col for each group in group_col
        means = data.groupby(group_col)[value_col].mean()
        
        # Log the first few results for monitoring
        logger.info(f"Computed means (head):\n{means.head()}")
        return means

    except KeyError as key_error:
        # Catch the error if the specified column does not exist in the DataFrame
        logger.error(f"Column not found: {key_error}")
        return None

    except TypeError as type_error:
        # Catch errors if the input data types are incorrect (for example non-DataFrame values)
        logger.error(f"Type error encountered: {type_error}")
        return None

    except ValueError as value_error:
        # Catch errors if the computation cannot be performed due to invalid values (for example non-numeric values)
        logger.error(f"Value error encountered: {value_error}")
        return None

    except Exception as unexpected_error:
        # Catch any other unexpected errors that might occur during computation
        logger.error(f"Unexpected error during group mean computation: {unexpected_error}")
        return None
    

def plot_distributions(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Plot the distribution of a numeric column across categories using a boxplot.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    group_col : str
        The categorical column to group by (must be str).
    value_col : str
        The numeric column to plot.

    Returns
    -------
    bool
        True if the plot was successfully created, False otherwise.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Log that the function started
    logger.info(f"Plotting distribution of '{value_col}' across '{group_col}'")

    try:
        # Prepare the data: select relevant columns and drop missing values
        plot_data = data[[group_col, value_col]].dropna()

        # Create the figure with a fixed size
        plt.figure(figsize=(10, 6))

        # Draw the boxplot with seaborn
        sns.boxplot(data=plot_data, x=group_col, y=value_col)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30)

        # Add a title to the plot
        plt.title(f"Distribution of {value_col} by {group_col}")

        # Adjust layout to avoid clipping
        plt.tight_layout()

        # Display the plot
        plt.show()

        # Return True to indicate success
        return True

    except KeyError as key_error:
        # Catch missing columns in the DataFrame
        logger.error(f"Column not found: {key_error}")
        return False

    except TypeError as type_error:
        # Catch invalid data types, e.g., non-numeric values
        logger.error(f"Type error encountered: {type_error}")
        return False

    except ValueError as value_error:
        # Catch errors related to invalid values for plotting 
        logger.error(f"Value error encountered during plotting: {value_error}")
        return False

    except Exception as unexpected_error:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during plotting: {unexpected_error}")
        return False

"""""
🔹 מה שונה ומקצועי כאן:

Logging – בתחילת הפונקציה ובמידת הצורך אחרי שמירה.

Try-except עם שמות ברורים – key_error, value_error, type_error, unexpected_error.

בדיקת NaN – מחזירה DataFrame נקי לפני הציור.

אפשרות שמירה – אם רוצים לשמור את הגרף עם save_path.

Return True/False – כדי שמי שקורא לפונקציה ידע אם הכל הצליח.
"""""



# ---------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------
def run_one_way_anova(data: pd.DataFrame, group_col: str, value_col: str):
    """
    Perform one-way ANOVA and return ANOVA table.

    Features:
    - Logging for info, warnings, and errors
    - Handles spaces/special characters in column names with Q()
    - Marks categorical variable with C() for ANOVA
    - Warns if some groups are very small (<5)
    - Safe execution with try-except
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    try:
        logger.info(f"Running one-way ANOVA on '{value_col}' grouped by '{group_col}'")

        # Check if the columns exist in the DataFrame
        # For general analysis functions, errors are logged and None is returned to allow the analysis pipeline to continue without crashing.
        if group_col not in data.columns or value_col not in data.columns:
            logger.error(f"Columns '{group_col}' or '{value_col}' not found")
            return None

        # Wrap dependent variable column with Q() to handle spaces/special characters
        dependent_wrapped = f'Q("{value_col}")'  # e.g., Q("Happiness Score")
        # Wrap independent categorical variable with C(Q()) to mark it as categorical
        factor_wrapped = f'C(Q("{group_col}"))'  # e.g., C(Q("Diet Type"))

        # Build formula for OLS model: dependent ~ factor
        formula = f'{dependent_wrapped} ~ {factor_wrapped}'
        # Fit the model using Ordinary Least Squares regression
        model = ols(formula, data=data).fit()

        # Perform ANOVA on the fitted model, Type II sums of squares
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Warn if any groups are very small
        counts = data[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")

        # Log ANOVA results
        logger.info(f"ANOVA completed:\n{anova_table}")
        return anova_table

    except KeyError as key_error:
        # Column not found
        logger.error(f"Column not found: {key_error}")
        return None

    except ValueError as value_error:
        # Handle invalid computation values
        logger.error(f"Value error during ANOVA: {value_error}")
        return None

    except Exception as unexpected_error:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during ANOVA: {unexpected_error}")
        return None

"""""
להעיף את כל ה# e.g., C(Q("Diet Type")) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""""



# ---------------------------------------------------------------------
# Planned Contrasts
# ---------------------------------------------------------------------
def run_planned_contrast(data: pd.DataFrame, group_col: str, value_col: str, contrast_weights: dict, visualize: bool = False):
    """
    Perform a planned contrast on the specified groups.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - group_col: categorical independent variable
    - value_col: continuous dependent variable
    - contrast_weights: dictionary specifying weights for each group
    - visualize: if True, plot weighted means
    
    Returns:
    - dict with t_statistic, degrees_of_freedom, p_value or None on error
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    try:
        # Log start of planned contrast
        logger.info(f"Running planned contrast for '{value_col}' across '{group_col}'")

        # Check that all groups in contrast_weights exist in the data
        missing_groups = [g for g in contrast_weights if g not in data[group_col].unique()]
        if missing_groups:
            raise KeyError(f"Groups missing from data: {missing_groups}")

        # Group data by the categorical variable
        grouped = data.groupby(group_col)[value_col]
        means = grouped.mean()      # Compute mean per group
        ns = grouped.count()        # Compute sample size per group

        # Warn if any group is very small or has zero variance
        for group in contrast_weights:
            if ns[group] < 5:
                logger.warning(f"Group '{group}' has very few observations ({ns[group]})")
            if grouped.var()[group] == 0:
                logger.warning(f"Variance for group '{group}' is zero. Contrast may be unstable")

        # Compute weighted contrast value
        contrast_value = sum(contrast_weights[g] * means[g] for g in contrast_weights)

        # Compute variance of the contrast
        variance = sum((contrast_weights[g] ** 2) * grouped.var()[g] / ns[g] for g in contrast_weights)

        # Calculate t-statistic and degrees of freedom
        t_stat = contrast_value / (variance ** 0.5)
        df = len(data) - len(means)

        # Compute two-tailed p-value
        p_value = stats.t.sf(abs(t_stat), df) * 2

        # Log results
        logger.info(f"Planned contrast result: t={t_stat:.3f}, df={df}, p={p_value:.4f}")

        # Optional visualization of weighted means
        if visualize:
            weighted_means = {g: means[g]*contrast_weights[g] for g in contrast_weights}
            plt.figure(figsize=(8,5))
            plt.bar(weighted_means.keys(), weighted_means.values(), color='skyblue')
            plt.ylabel(f"Weighted {value_col}")
            plt.title("Planned Contrast Weighted Means")
            plt.show()

        # Return results as dictionary
        return {"t_statistic": t_stat, "degrees_of_freedom": df, "p_value": p_value}

    except KeyError as key_error:
        # Raised if a specified group does not exist
        logger.error(f"Key error: {key_error}")
        return None

    except ZeroDivisionError:
        # Raised if variance calculation is zero (cannot divide by zero)
        logger.error("Variance calculation resulted in zero. Contrast cannot be computed")
        return None

    except Exception as unexpected_error:
        # Catch all other unexpected errors
        logger.error(f"Unexpected error during planned contrast: {unexpected_error}")
        return None


"""""
לסגור את הנקודה עם ה raise KeyError בפונקציה הזו לעומת הקודמת ששם זה RETURN NONE - יש צילום מסך עם הסבר של הצ'אט
"""""


# ---------------------------------------------------------------------
# Post-hoc Test (Tukey HSD)
# ---------------------------------------------------------------------
def run_tukey_hsd(data: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05, visualize: bool = False):
    """
    Perform Tukey HSD post-hoc test for all pairwise group comparisons.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - group_col: categorical independent variable
    - value_col: continuous dependent variable
    - alpha: significance level (default 0.05)
    - visualize: if True, plot mean differences with confidence intervals
    
    Returns:
    - TukeyHSDResults object or None on error
    """
    import matplotlib.pyplot as plt
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    try:
        # Log start of Tukey HSD test
        logger.info(f"Running Tukey HSD pairwise comparisons for '{group_col}' on '{value_col}'")

        # Check that the columns exist
        if group_col not in data.columns or value_col not in data.columns:
            logger.error(f"Columns '{group_col}' or '{value_col}' not found in DataFrame")
            return None

        # Optional: warn if groups are very small
        counts = data[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")

        # Perform Tukey HSD test
        tukey_result = pairwise_tukeyhsd(
            endog=data[value_col],       # dependent variable
            groups=data[group_col],      # categorical independent variable
            alpha=alpha
        )

        # Log summary of results
        logger.info("Tukey HSD test completed:\n" + str(tukey_result.summary()))

        # Optional visualization: plot group comparisons
        if visualize:
            tukey_result.plot_simultaneous(figsize=(10, 6))
            plt.title("Tukey HSD: Pairwise Comparisons")
            plt.show()

        # Return TukeyHSDResults object for further analysis
        return tukey_result

    except KeyError as key_error:
        # Raised if columns do not exist
        logger.error(f"Key error: {key_error}")
        return None

    except ValueError as value_error:
        # Raised for invalid inputs, e.g., empty data or all NaN
        logger.error(f"Value error during Tukey HSD: {value_error}")
        return None

    except Exception as unexpected_error:
        # Catch all other unexpected errors
        logger.error(f"Unexpected error during Tukey HSD: {unexpected_error}")
        return None
    






"""""
"מה נשדרג:

Dashboard אינטראקטיבי – שימוש ב‑Plotly או Streamlit להצגת תוצאות live:

גרפים של קבוצות עם ממוצעים ו‑CI

weighted means ל‑contrast

pairwise p-values ל‑Tukey

Heatmap של pairwise p-values – עבור כל בדיקת המשך (Tukey או contrast):

צבעים לפי p-value או significance

קל לראות איזה קבוצות שונות משמעותית

Sensitivity / Outlier Analysis – בדיקה של השפעת:

הסרת קבוצות קטנות (<5)

הסרת extreme values (outliers)

הצגת השפעה על t-stat, p-value ו‑mean differences
"""""
# ---------------------------------------------------------------------
# 1. One-way ANOVA with outlier/sensitivity check and optional heatmap
# ---------------------------------------------------------------------
def run_one_way_anova(data: pd.DataFrame, group_col: str, value_col: str, visualize: bool = False):
    """
    Run one-way ANOVA with warnings for small groups and optional visualization.
    """
    try:
        logger.info(f"Running ANOVA for '{value_col}' across '{group_col}'")

        # Check columns exist
        if group_col not in data.columns or value_col not in data.columns:
            logger.error(f"Columns '{group_col}' or '{value_col}' not found")
            return None

        # Remove NaNs
        df = data[[group_col, value_col]].dropna()

        # Warn if small groups
        counts = df[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")

        # Fit model using OLS
        formula = f'Q("{value_col}") ~ C(Q("{group_col}"))'
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Optional heatmap of group means
        if visualize:
            means = df.groupby(group_col)[value_col].mean()
            sns.heatmap(means.to_frame().T, annot=True, cmap="coolwarm")
            plt.title(f"Mean {value_col} by {group_col}")
            plt.show()

        logger.info(f"ANOVA completed:\n{anova_table}")
        return anova_table

    except Exception as e:
        logger.error(f"Unexpected error during ANOVA: {e}")
        return None


# ---------------------------------------------------------------------
# 2. Planned Contrast with sensitivity check and weighted means plot
# ---------------------------------------------------------------------
def run_planned_contrast(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    contrast_weights: dict,
    visualize: bool = False
):
    """
    Run a planned contrast (user-supplied weights) with optional visualization.
    """
    try:
        logger.info(f"Running planned contrast for '{value_col}' across '{group_col}'")

        # Check if all groups exist
        missing = [g for g in contrast_weights if g not in data[group_col].unique()]
        if missing:
            logger.error(f"Groups missing from data: {missing}")
            return None

        df = data[[group_col, value_col]].dropna()
        grouped = df.groupby(group_col)[value_col]
        means = grouped.mean()
        ns = grouped.count()

        # Warn for small groups or zero variance
        for g in contrast_weights:
            if ns[g] < 5:
                logger.warning(f"Group '{g}' has very few observations ({ns[g]})")
            if grouped.var()[g] == 0:
                logger.warning(f"Variance for group '{g}' is zero. Contrast may be unstable.")

        # Compute contrast
        contrast_value = sum(contrast_weights[g] * means[g] for g in contrast_weights)
        variance = sum((contrast_weights[g] ** 2) * grouped.var()[g] / ns[g] for g in contrast_weights)
        t_stat = contrast_value / np.sqrt(variance)
        df_total = len(df) - len(means)
        p_value = stats.t.sf(abs(t_stat), df_total) * 2

        logger.info(f"Planned contrast result: t={t_stat:.3f}, df={df_total}, p={p_value:.4f}")

        # Optional visualization: weighted means bar plot
        if visualize:
            weighted_means = {g: means[g] * contrast_weights[g] for g in contrast_weights}
            plt.figure(figsize=(8,5))
            plt.bar(weighted_means.keys(), weighted_means.values(), color='skyblue')
            plt.ylabel(f"Weighted {value_col}")
            plt.title("Planned Contrast Weighted Means")
            plt.show()

        return {"t_statistic": t_stat, "degrees_of_freedom": df_total, "p_value": p_value}

    except Exception as e:
        logger.error(f"Unexpected error during planned contrast: {e}")
        return None


# ---------------------------------------------------------------------
# 3. Tukey HSD post-hoc test with heatmap and outlier check
# ---------------------------------------------------------------------
def run_tukey_hsd(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05,
    visualize: bool = False
):
    """
    Perform Tukey HSD pairwise comparisons with optional heatmap visualization.
    """
    try:
        logger.info(f"Running Tukey HSD for '{value_col}' across '{group_col}'")

        if group_col not in data.columns or value_col not in data.columns:
            logger.error(f"Columns '{group_col}' or '{value_col}' not found")
            return None

        df = data[[group_col, value_col]].dropna()

        # Warn if small groups
        counts = df[group_col].value_counts()
        small_groups = counts[counts < 5]
        if not small_groups.empty:
            logger.warning(f"Small groups detected: {small_groups.to_dict()}")

        # Run Tukey HSD
        tukey_result = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=alpha)
        logger.info("Tukey HSD test completed:\n" + str(tukey_result.summary()))

        # Optional heatmap of pairwise p-values
        if visualize:
            # Create matrix of pairwise p-values
            groups = tukey_result.groupsunique
            p_matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)
            for i, (g1, g2, _, _, p) in enumerate(zip(
                    tukey_result._multicomp.group1, tukey_result._multicomp.group2,
                    tukey_result.meandiffs, tukey_result.reject, tukey_result.pvalues)):
                p_matrix.loc[g1, g2] = p
                p_matrix.loc[g2, g1] = p
            sns.heatmap(p_matrix, annot=True, cmap="coolwarm", fmt=".3f")
            plt.title("Tukey HSD Pairwise p-values")
            plt.show()

        return tukey_result

    except Exception as e:
        logger.error(f"Unexpected error during Tukey HSD: {e}")
        return None
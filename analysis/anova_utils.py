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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
def compute_group_means(
    data: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.Series:
    """
    Compute mean of value_col for each category in group_col.
    """
    logger.info(f"Computing mean of '{value_col}' per '{group_col}'")
    return data.groupby(group_col)[value_col].mean()


def plot_distributions(
    data: pd.DataFrame,
    group_col: str,
    value_col: str
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info(f"Plotting distributions for '{value_col}' across '{group_col}'")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=group_col, y=value_col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------
def run_one_way_anova(
    data: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    Perform one-way ANOVA using OLS with support for column names with spaces.
    """
    logger.info(f"Running one-way ANOVA on {value_col} grouped by {group_col}")
    
    # Use f-string to insert the column names into the Q() wrapper
    # This results in a formula like: Q("Happiness Score") ~ C(Q("Diet Type"))
    formula = f'Q("{value_col}") ~ C(Q("{group_col}"))'
    
    model = ols(formula, data=data).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


# ---------------------------------------------------------------------
# Planned Contrasts
# ---------------------------------------------------------------------
def run_planned_contrast(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    contrast_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Perform a planned contrast comparing
    """
    logger.info(f"Running planned contrast analysis for '{group_col}' on '{value_col}'")

    grouped = data.groupby(group_col)[value_col]
    means = grouped.mean()
    ns = grouped.count()

    contrast_value = sum(
        contrast_weights[group] * means[group]
        for group in contrast_weights
    )

    variance = sum(
        (contrast_weights[group] ** 2) *
        grouped.var()[group] / ns[group]
        for group in contrast_weights
    )

    t_stat = contrast_value / (variance ** 0.5)
    df = len(data) - len(means)

    p_value = stats.t.sf(abs(t_stat), df) * 2

    return {
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value": p_value
    }


# ---------------------------------------------------------------------
# Post-hoc Test (Tukey HSD)
# ---------------------------------------------------------------------
def run_tukey_hsd(
    data: pd.DataFrame,
    group_col: str,
    value_col: str
) -> pairwise_tukeyhsd:
    """
    Run Tukey HSD post-hoc test for all pairwise diet comparisons.
    """
    logger.info(f"Running Tukey HSD pairwise comparisons for '{group_col}'")
    
    return pairwise_tukeyhsd(
        endog=data[value_col],
        groups=data[group_col],
        alpha=0.05
    )

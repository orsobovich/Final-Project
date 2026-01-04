import pandas as pd

from analysis.anova_utils import (
    compute_group_means,
    plot_distributions,
    run_one_way_anova,
    run_planned_contrast,
    run_tukey_hsd
)


def analyze_social_interaction_by_mental_health(
    data: pd.DataFrame
) -> None:
    """
    Analyze differences in social interaction scores across mental health groups.
    """

    group_col = "Mental Health Condition"
    value_col = "Social Interaction Score"

    # -----------------------------
    # EDA
    # -----------------------------
    means = compute_group_means(
        data=data,
        group_col=group_col,
        value_col=value_col
    )
    print(means)

    plot_distributions(
        data=data,
        group_col=group_col,
        value_col=value_col
    )

    # -----------------------------
    # One-way ANOVA
    # -----------------------------
    anova_table = run_one_way_anova(
        data=data,
        group_col=group_col,
        value_col=value_col
    )
    print(anova_table)

    # -----------------------------
    # Planned contrast
    # -----------------------------
    contrast_weights = {
        "None": 1.0,        # No mental health condition
        "Anxiety": -0.25,
        "Depression": -0.25,
        "PTSD": -0.25,
        "Bipolar": -0.25
    }

    contrast_result = run_planned_contrast(
        data=data,
        group_col=group_col,
        value_col=value_col,
        contrast_weights=contrast_weights
    )

    print(contrast_result)
    
    tukey_results = run_tukey_hsd(data, group_col, value_col)
    print("\nTukey HSD Results:")
    print(tukey_results)

import pandas as pd

from analysis.anova_utils import (
    compute_group_means,
    plot_distributions,
    run_one_way_anova,
    run_planned_contrast,
    run_tukey_hsd
)

def analyze_diet_happiness(data: pd.DataFrame) -> None:
    group_col = "Diet Type"
    value_col = "Happiness Score"

    # ... existing EDA and ANOVA code ...
    anova_table = run_one_way_anova(data, group_col, value_col)
    print("\nANOVA Table:")
    print(anova_table)

    # -----------------------------
    # Post-hoc Test (Tukey HSD)
    # -----------------------------
    # Only necessary if ANOVA is significant (p < 0.05)
    tukey_results = run_tukey_hsd(data, group_col, value_col)
    print("\nTukey HSD Results:")
    print(tukey_results)

    # -----------------------------
    # Planned Contrast
    # -----------------------------

    contrast_weights = {
        "Vegan": 3.0,
        "Vegetarian": 3.0,
        "Balanced": -2.0,
        "Junk Food": -2.0,
        "Keto": -2.0
    }

    contrast_result = run_planned_contrast(
        data=data,
        group_col=group_col,
        value_col=value_col,
        contrast_weights=contrast_weights
    )
    print("\nPlanned Contrast (Plant-Based vs Others):")
    print(contrast_result)

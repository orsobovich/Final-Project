import pandas as pd

from analysis.diet_happiness import analyze_diet_happiness
from analysis.mental_health_social import analyze_social_interaction_by_mental_health



def main() -> None:
    data = pd.read_csv("Mental_Health_Lifestyle_Dataset.csv", keep_default_na=False, na_values=[''])

    analyze_diet_happiness(data)
    analyze_social_interaction_by_mental_health(data)


if __name__ == "__main__":
    main()

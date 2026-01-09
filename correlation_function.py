import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
sns.set_theme()
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout,force=True)

df=pd.read_csv("Mental_Health_Lifestyle_Dataset.csv") 

def is_valid_level(series): #checks if the series is ordinal 
    ordinal_levels = {"Low", "Moderate", "High"} #from the first part of the project we know this is the ordinal levels
    values = set(series.dropna().unique()) #create a new set that includes only the values without NaN to compare them
    return values.issubset(ordinal_levels) #return true if all the values from the column are "ordinal_levels" 

#The dictionary maps the categorical levels to numeric ranks
# to establish the ordinal relationship required for Spearman's rank correlation analysis.
def level_to_numeric(series):
    mapping = {"Low": 1, "Moderate": 2, "High": 3}  
    return series.map(mapping) #Changing to numeric ranks

  # Create a funqtion that checks significance          
def find_sig(p_value, alpha=0.05):
    if p_value < alpha:
        logging.info(f"p-value is significant: {p_value}")
        return True
    else:
        logging.info(f"p-value isn't significant: {p_value}")
        return False
    
 # Create a correlation plot
def plot_correlation(cor_1, cor_2, p_value): 
    if find_sig(p_value):
        sns.regplot(x=cor_1, y=cor_2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title("Correlation Plot")
        plt.xlabel(cor_1.name) #Takes the name of the column
        plt.ylabel(cor_2.name) #Takes the name of the column
        plt.show()
        

def visualize_correlation(var_1, var_2):
#Validates that the input variables meet the type criteria (numeric or ordinal)
# required for the selected correlation method.
    try:
        if pd.api.types.is_numeric_dtype(var_1) and pd.api.types.is_numeric_dtype(var_2):
            r, p_value = pearsonr(var_1, var_2)
            plot_correlation(var_1,var_2,p_value)
            logging.info(f"Pearson correlation: {r}")

        elif pd.api.types.is_numeric_dtype(var_1) and is_valid_level(var_2):
            level_num = level_to_numeric(var_2)
            corr, p_value = spearmanr(var_1, level_num)
            plot_correlation(var_1,level_num,p_value)
            logging.info(f"Spearman correlation: {corr}")

        elif pd.api.types.is_numeric_dtype(var_2) and is_valid_level(var_1):
            level_num = level_to_numeric(var_1)
            corr, p_value = spearmanr(var_2, level_num)
            plot_correlation(level_num,var_2,p_value)
            logging.info(f"Spearman correlation: {corr}")

        #raise an exception so it gets caught by the general exception handler
        else:
            logging.error("Unsupported variable types for correlation")
            raise TypeError("Invalid input types") 
    except ValueError as e:
        logging.error(f"Value error in correlation: {e}")
        raise

    except Exception as e: #catch all the exception except ValueError
        logging.exception("Unexpected error in correlation")
        raise

#Checks correlation       
logging.info("correlation between Stress Level and Sleep Hours:")        
visualize_correlation(df['Stress Level'], df['Sleep Hours'])

logging.info("correlation between Age and Sleep Hours:")
visualize_correlation(df["Age"], df["Sleep Hours"]) 

logging.info("correlation between Social Interaction Score and Stress Level:")
visualize_correlation(df["Social Interaction Score"], df["Stress Level"])

logging.info("correlation between Age and Social Interaction Score:")
visualize_correlation(df["Age"], df["Social Interaction Score"])
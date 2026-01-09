import pytest
import pandas as pd
from correlation_function import is_valid_level,level_to_numeric, find_sig, visualize_correlation
from unittest.mock import patch

def test_on_valid_levels_should_return_true():
    test_1=pd.Series(['Low', 'Moderate', 'High', 'Moderate', 'Low'], name="satisfaction")
    assert is_valid_level(test_1) == True

def test_on_invalid_levels_should_return_false():
    test_2=pd.Series([25, 30, 35, 40, 45], name="age")
    assert is_valid_level(test_2) == False

def test_on_valid_with_nan_should_return_true():
    test_3=pd.Series(['Low', None, 'Moderate', 'High', None, 'Moderate', 'Low', None], name="satisfaction")
    assert is_valid_level(test_3) == True
    
def test_on_valid_levels_should_return_correct_numbers():
    test_4=pd.Series(["Low","Moderate","High"], name="satisfaction")
    assert level_to_numeric(test_4).to_list() == [1,2,3] 

def test_on_significant_pvalue_should_return_true():
    assert find_sig(0.04) == True

def test_on_insignificant_pvalue_should_return_false():
    assert find_sig(1) == False

# לבדוקקקק
def test_correlation_pearson_on_both_numeric_should_not_raise_exception():
    var1 = pd.Series([1, 2, 3, 4, 5])
    var2 = pd.Series([2, 4, 6, 8, 10])
    with patch('matplotlib.pyplot.show'):
        try:
            visualize_correlation(var1, var2)
        except Exception as e:
            pytest.fail(f"Pearson correlation failed unexpectedly: {e}")        


def test_correlation_pearson_on_one_numeric_should_not_raise_exception():
    var_numeric = pd.Series([1, 2, 3])
    var_ordinal = pd.Series(['Low', 'Moderate', 'High'])
    
    try:
        visualize_correlation(var_numeric, var_ordinal) 
        visualize_correlation(var_ordinal, var_numeric) 
    except Exception as e:
        pytest.fail(f"Spearman correlation failed unexpectedly: {e}")


def test_correlation_pearson_on_invalid_levels_should_raise_exception():
    
    var_numeric = pd.Series([1, 2, 3])
    var_invalid = pd.Series(['Apple', 'Banana', 'Orange']) 
    # The function won't raise an exception when it gets TypeError
    with pytest.raises(TypeError): 
        visualize_correlation(var_numeric, var_invalid)
        
        
def test_correlation_on_different_lengths_should_raise_value_error():
    var1 = pd.Series([1, 2, 3])   
    var2 = pd.Series([1, 2])       
    # The function won't raise an exception when it gets ValueError
    with pytest.raises(ValueError):
        visualize_correlation(var1, var2)
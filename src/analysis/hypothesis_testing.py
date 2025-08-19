import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

def test_province_risk_differences(df):
    """Test if there are risk differences across provinces"""
    model = ols('LossRatio ~ C(Province)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

def test_zipcode_risk_differences(df):
    """Test if there are risk differences between zipcodes"""
    # Sample top 10 zipcodes for demonstration
    top_zips = df['PostalCode'].value_counts().nlargest(10).index
    subset = df[df['PostalCode'].isin(top_zips)]
    model = ols('LossRatio ~ C(PostalCode)', data=subset).fit()
    return sm.stats.anova_lm(model, typ=2)

def test_gender_risk_differences(df):
    """Test if there are risk differences between genders"""
    male = df[df['Gender'] == 'Male']['LossRatio']
    female = df[df['Gender'] == 'Female']['LossRatio']
    t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
    return {'t-statistic': t_stat, 'p-value': p_val}

def test_zipcode_margin_differences(df):
    """Test if there are margin differences between zipcodes (top 10 by count)"""
    top_zips = df['PostalCode'].value_counts().nlargest(10).index
    subset = df[df['PostalCode'].isin(top_zips)]
    model = ols('Margin ~ C(PostalCode)', data=subset).fit()
    return sm.stats.anova_lm(model, typ=2)

def run_all_tests(df):
    """Run all hypothesis tests"""
    results = {
        'province_risk': test_province_risk_differences(df),
        'gender_risk': test_gender_risk_differences(df)
    }
    
    try:
        results['zipcode_risk'] = test_zipcode_risk_differences(df)
    except Exception as e:
        print(f"Zipcode test failed: {str(e)}")
    
    return results
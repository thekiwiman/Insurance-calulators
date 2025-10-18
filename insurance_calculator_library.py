import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly as go


# Page configuration
st.set_page_config(
    page_title="Endowment Life Insurance Calculator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #F0FDF4, #DBEAFE);
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Insurance Library Functions
def accumulated_annuity(periods, i, type=1):
    """Calculate accumulated value of annuity"""
    if type == 1:
        return (math.pow(1 + i, periods) - 1) / i
    elif type == 2:
        return (math.pow(1 + i, periods) - 1) / (math.pow((1 - 1 / i), -1))
    else:
        return 0

@st.cache_data
def load_death_probabilities():
    """Load and parse the death probability CSV files"""
    import os
    
    # Try multiple possible locations
    possible_paths = [
        ('DeathProbsE_F_Alt2_TR2025.csv', 'DeathProbsE_M_Alt2_TR2025.csv'),  # Root directory
        ('data/DeathProbsE_F_Alt2_TR2025.csv', 'data/DeathProbsE_M_Alt2_TR2025.csv'),  # data subdirectory
        ('./data/DeathProbsE_F_Alt2_TR2025.csv', './data/DeathProbsE_M_Alt2_TR2025.csv'),  # explicit relative path
        (r'life-insurance-calculators/DeathProbsE_F_Alt2_TR2025.csv',r'life-insurance-calculators/DeathProbsE_M_Alt2_TR2025.csv',)
    ]
    
    # Try to find the files
    for female_path, male_path in possible_paths:
        if os.path.exists(female_path) and os.path.exists(male_path):
            try:
                # Load female data
                female_df = pd.read_csv(female_path, skiprows=1)
                
                # Load male data
                male_df = pd.read_csv(male_path, skiprows=1)
                
                # Get 2025 data (most recent)
                female_2025 = female_df[female_df['Year'] == 2025].iloc[0]
                male_2025 = male_df[male_df['Year'] == 2025].iloc[0]
                
                return {
                    'female': female_2025,
                    'male': male_2025
                }
            except Exception as e:
                continue
    
    # If we get here, files weren't found
    #st.error("❌ Unable to load actuarial data. Please ensure the CSV files are in the repository.")
    return None

def get_death_probability(data, age, gender='female'):
    """Get death probability for a specific age and gender"""
    if data is None:
        return 0.0

    # If a dict was passed, pick the gender-specific entry
    if isinstance(data, dict):
        key = str(gender).lower()
        if key in data:
            data = data[key]
        else:
            data = data.get('female') or data.get('male')

    prob = 0.0
    # If data is a pandas Series, try column lookup
    if isinstance(data, pd.Series):
        for lookup in (str(age), int(age) if isinstance(age, (str, float)) and str(age).isdigit() else None, age):
            if lookup is None:
                continue
            try:
                if lookup in data.index:
                    val = data[lookup]
                    if pd.notna(val):
                        prob = float(val)
                        break
            except Exception:
                continue
    else:
        # For list/array/tuple-like data
        try:
            idx = int(age)
            if idx >= 0 and idx < len(data):
                val = data[idx]
                if pd.notna(val):
                    prob = float(val)
        except Exception:
            prob = 0.0

    return prob

def calculate_premium(current_age, payout_age, interest, payout, gender='female', risk_margin=1.0):
    """
    Calculate premium for endowment insurance
    
    Args:
        risk_margin: Multiplier for mortality rates (1.0 = standard, >1.0 = more conservative)
                     - 1.0: Standard actuarial rates
                     - 1.2: 20% safety margin (conservative)
                     - 1.5: 50% safety margin (very conservative)
                     - 0.8: 20% discount (aggressive/preferred risk)
    """
    weighted_total_annuity = 0
    death_data = load_death_probabilities()
    
    if death_data is None:
        return None, None
    
    prob_death_given_age_is_x = 0
    prob_death_and_age_is_x = 0
    prob_age_is_x = 1
    death_cdf = 0
    
    for evaluation_age in range(current_age, payout_age):
        prob_age_is_x = (1 - prob_death_given_age_is_x) * prob_age_is_x
        
        # Get base mortality rate and apply risk margin
        base_death_prob = get_death_probability(death_data, evaluation_age, gender)
        prob_death_given_age_is_x = min(base_death_prob * risk_margin, 1.0)  # Cap at 100%
        
        if evaluation_age < payout_age - 1:
            prob_death_and_age_is_x = prob_age_is_x * prob_death_given_age_is_x
        else:
            prob_death_and_age_is_x = prob_age_is_x
            
        death_cdf += prob_death_and_age_is_x
        weighted_total_annuity += accumulated_annuity(evaluation_age - current_age, interest, 1) * prob_death_and_age_is_x
    
    premium = payout / weighted_total_annuity if weighted_total_annuity > 0 else 0
    return premium, death_cdf

def calculate_asset_growth(premium, current_age, payout_age, interest, payout, gender):
    """
    Calculate how accumulated premiums grow over time compared to payout amount.
    Returns data for visualization.
    """
    death_data = load_death_probabilities()
    if death_data is None:
        return None
    
    ages = []
    accumulated_values = []
    survival_probs = []
    
    prob_death_given_age_is_x = 0
    prob_age_is_x = 1
    
    for age in range(current_age, payout_age + 1):
        years_elapsed = age - current_age
        
        # Calculate accumulated value of premiums
        if years_elapsed > 0:
            accumulated_value = premium * accumulated_annuity(years_elapsed, interest, 1)
        else:
            accumulated_value = 0
        
        # Calculate survival probability
        if age < payout_age:
            prob_death_given_age_is_x = get_death_probability(death_data, age, gender)
            prob_age_is_x = prob_age_is_x * (1 - prob_death_given_age_is_x)
        
        ages.append(age)
        accumulated_values.append(accumulated_value)
        survival_probs.append(prob_age_is_x * 100)
    
    return pd.DataFrame({
        'Age': ages,
        'Accumulated_Value': accumulated_values,
        'Survival_Probability': survival_probs,
        'Payout_Amount': [payout] * len(ages)
    })

def calculate_risk_tolerance(premium, payout, current_age, payout_age, interest, gender):
    """Calculate probability of death before premiums exceed payout"""
    death_cdf = 0
    prob_death_given_age_is_x = 0
    prob_age_is_x = 1
    death_data = load_death_probabilities()
    
    if death_data is None:
        return 0
    
    for x in range(current_age, payout_age):
        prob_age_is_x = (1 - prob_death_given_age_is_x) * prob_age_is_x
        prob_death_given_age_is_x = get_death_probability(death_data, x, gender)
        prob_death_and_age_is_x = prob_age_is_x * prob_death_given_age_is_x
        death_cdf += prob_death_and_age_is_x
        
        s = premium * accumulated_annuity(x - current_age, interest, 1)
        
        if s > payout:
            return death_cdf
    
    return death_cdf

def solve_premium_for_risk_target(target_risk, payout, current_age, payout_age, interest, gender, max_iterations=50):
    """
    Solve for premium that achieves a specific risk tolerance target using binary search.
    
    Args:
        target_risk: Desired probability (0-1) of death before break-even
        payout: Desired payout amount
        current_age: Current age
        payout_age: Maturity age
        interest: Annual interest rate
        gender: 'male' or 'female'
        max_iterations: Maximum iterations for convergence
    
    Returns:
        premium: Annual premium that achieves target risk
        actual_risk: Actual risk tolerance achieved
    """
    # Initial bounds for premium search
    # Lower bound: minimal premium (almost zero)
    # Upper bound: premium that covers full payout in first year
    base_premium=calculate_premium(current_age,payout_age,interest,payout,gender)[0]
    base_risk=calculate_risk_tolerance(base_premium, payout,current_age,payout_age,interest,gender )
    if(target_risk > base_risk):
            return base_premium, base_risk
    min_premium = 1
    max_premium = payout * 2  # Conservative upper bound
    
    tolerance = 0.001  # 0.1% tolerance on risk
    
    for iteration in range(max_iterations):
        # Try midpoint premium
        test_premium = (min_premium + max_premium) / 2
        
        # Calculate risk tolerance for this premium
        actual_risk = calculate_risk_tolerance(test_premium, payout, current_age, payout_age, interest, gender)
        
        # Check if we're close enough
        if abs(actual_risk - target_risk) < tolerance:
            return test_premium, actual_risk
        
        # Adjust search bounds
        # If actual risk > target, we need higher premium (death happens before break-even more often)
        # If actual risk < target, we need lower premium (premiums accumulate faster)
        if actual_risk > target_risk:
            min_premium = test_premium
        else:
            max_premium = test_premium
    
    # Return best estimate even if not perfectly converged
    return test_premium, actual_risk
 
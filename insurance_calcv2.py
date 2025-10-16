import streamlit as st
import pandas as pd
import numpy as np
import math

# Page configuration
st.set_page_config(
    page_title="Endowment Insurance Calculator",
    page_icon="💰",
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
    
    # Debug: Show current directory and files
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    
    # st.write(f"**Debug Info:**")
    # st.write(f"Current directory: {current_dir}")
    # st.write(f"Files found: {files_in_dir}")
    
    try:
        # Load female data
        female_df = pd.read_csv('life-insurance-calculators/DeathProbsE_F_Alt2_TR2025.csv', skiprows=1)
        # st.success("✅ Female data loaded successfully")
        
        # Load male data
        male_df = pd.read_csv('life-insurance-calculators/DeathProbsE_M_Alt2_TR2025.csv', skiprows=1)
        # st.success("✅ Male data loaded successfully")
        
        # Get 2025 data (most recent)
        female_2025 = female_df[female_df['Year'] == 2025].iloc[0]
        male_2025 = male_df[male_df['Year'] == 2025].iloc[0]
        
        return {
            'female': female_2025,
            'male': male_2025
        }
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.error("Make sure CSV files are in your GitHub repository root directory")
        return None
    except Exception as e:
        st.error(f"❌ Error loading death probability data: {e}")
        st.error(f"Error type: {type(e).__name__}")
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

def calculate_premium(current_age, payout_age, interest, payout, gender='female'):
    """Calculate premium for endowment insurance"""
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
        prob_death_given_age_is_x = get_death_probability(death_data, evaluation_age, gender)
        
        if evaluation_age < payout_age - 1:
            prob_death_and_age_is_x = prob_age_is_x * prob_death_given_age_is_x
        else:
            prob_death_and_age_is_x = prob_age_is_x
            
        death_cdf += prob_death_and_age_is_x
        weighted_total_annuity += accumulated_annuity(evaluation_age - current_age, interest, 1) * prob_death_and_age_is_x
    
    premium = payout / weighted_total_annuity if weighted_total_annuity > 0 else 0
    return premium, death_cdf

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

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>💰 Endowment Insurance Premium Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Calculate premiums for policies that guarantee a payout at maturity or death</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Info box
    st.info("""
    **What is Endowment Insurance?**
    
    Endowment insurance combines life insurance with a savings component. It pays out a lump sum either:
    - At the end of the policy term (maturity age), OR
    - Upon death, whichever occurs first
    
    This calculator determines the annual premium needed to guarantee your desired payout.
    """)
    
    # Load death probability data
    death_prob_data = load_death_probabilities()
    
    if death_prob_data is None:
        st.error("Unable to load actuarial data. Please ensure the CSV files are in the same directory.")
        return
    
    st.markdown("---")
    st.markdown("### 📋 Enter Policy Details")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        current_age = st.number_input(
            "👤 Current Age",
            min_value=18,
            max_value=80,
            value=25,
            step=1,
            help="Your age when the policy starts"
        )
        
        payout_age = st.number_input(
            "🎯 Maturity Age",
            min_value=current_age + 5,
            max_value=100,
            value=60,
            step=1,
            help="Age when the policy matures (must be at least 5 years from current age)"
        )
        
        gender = st.selectbox(
            "Gender",
            options=['female', 'male'],
            format_func=lambda x: x.capitalize(),
            help="Used for mortality probability calculations"
        )
    
    with col2:
        payout = st.number_input(
            "💵 Desired Payout Amount ($)",
            min_value=10000,
            max_value=5000000,
            value=100000,
            step=10000,
            help="Amount you want to receive at maturity or upon death"
        )
        
        interest = st.slider(
            "📈 Expected Annual Return (%)",
            min_value=1.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="Expected annual interest rate on accumulated premiums"
        ) / 100
        
    # Validation
    if payout_age <= current_age:
        st.error("Maturity age must be greater than current age!")
        return
    
    # Calculate premium
    st.markdown("---")
    
    with st.spinner("Calculating premium..."):
        premium, death_cdf = calculate_premium(current_age, payout_age, interest, payout, gender)
        
        if premium is None:
            st.error("Unable to calculate premium. Please check your inputs.")
            return
        
        risk_tolerance = calculate_risk_tolerance(premium, payout, current_age, payout_age, interest, gender)
    
    # Display results
    st.markdown("<h2 style='text-align: center; color: #059669;'>Your Premium Calculation</h2>", unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    years = payout_age - current_age
    total_paid = premium * years
    accumulated_value = premium * accumulated_annuity(years, interest, 1)
    
    with col1:
        st.metric("Annual Premium", f"${premium:,.2f}")
    
    with col2:
        st.metric(f"Total Paid ({years} years)", f"${total_paid:,.2f}")
    
    with col3:
        st.metric("Accumulated Value at Maturity", f"${accumulated_value:,.2f}")
    
    # Additional metrics
    st.markdown("---")
    st.markdown("### 📊 Policy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Break-Even Risk",
            f"{risk_tolerance * 100:.3f}%",
            help="Probability of death before accumulated premiums exceed payout"
        )
        
        st.info(f"""
        **Break-Even Analysis:**
        
        There is a **{risk_tolerance * 100:.3f}%** probability that you would pass away before 
        your accumulated premium payments (with interest) exceed the policy payout of ${payout:,}.
        
        This represents the "insurance" component of the endowment policy.
        """)
    
    with col2:
        st.metric(
            "Policy Period",
            f"{years} years",
            help="Time from current age to maturity"
        )
        
        roi = ((payout / total_paid) - 1) * 100 if total_paid > 0 else 0
        
        st.success(f"""
        **Investment Component:**
        
        If you survive to age {payout_age}, you'll receive ${payout:,} after paying 
        ${total_paid:,.2f} in total premiums.
        
        This represents a **{roi:.1f}%** total return on your premium payments 
        (not accounting for the time value of money or interest earned).
        """)
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 🔍 How This Works")
    
    with st.expander("View Calculation Details"):
        st.markdown(f"""
        **Premium Calculation Method:**
        
        1. **Expected Present Value**: The premium is calculated so that the expected present value 
           of all premium payments equals the expected present value of the payout.
        
        2. **Mortality Adjustment**: Uses 2025 Social Security Administration mortality tables 
           to calculate the probability of death at each age from {current_age} to {payout_age}.
        
        3. **Interest Accumulation**: Premiums accumulate with {interest*100:.1f}% annual interest, 
           creating a growing fund over time.
        
        4. **Risk Pooling**: The premium accounts for the fact that some policyholders will die 
           early (receiving the payout from limited premiums) while others survive to maturity 
           (having paid all premiums).
        
        **Key Formula Components:**
        - Accumulated Annuity Factor: {accumulated_annuity(years, interest, 1):.4f}
        - Total Death Probability: {death_cdf * 100:.3f}%
        - Survival Probability: {(1 - death_cdf) * 100:.3f}%
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Important Disclaimer:**
    
    This calculator provides educational estimates based on actuarial principles and 2025 mortality data. 
    Actual insurance premiums will vary based on:
    - Medical underwriting and health conditions
    - Insurance company expense loadings and profit margins
    - Policy riders and additional features
    - State regulations and taxes
    - Company-specific pricing models
    
    **This is not a quote.** Contact a licensed insurance agent for actual pricing.
    """)

if __name__ == "__main__":
    main()
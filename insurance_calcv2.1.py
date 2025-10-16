import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Endowment Insurance Calculator",
    page_icon="💰",
    layout="wide"
)

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

# ==============================================================================
# ACTUARIAL CALCULATION FUNCTIONS
# ==============================================================================

def accumulated_annuity(periods, interest_rate, annuity_type=1):
    """
    Calculate accumulated value of annuity.
    
    Args:
        periods: Number of payment periods
        interest_rate: Interest rate per period
        annuity_type: 1 for ordinary annuity, 2 for annuity due
    
    Returns:
        Accumulated value factor
    """
    if annuity_type == 1:
        return (math.pow(1 + interest_rate, periods) - 1) / interest_rate
    elif annuity_type == 2:
        return (math.pow(1 + interest_rate, periods) - 1) / (math.pow((1 - 1 / interest_rate), -1))
    else:
        return 0


@st.cache_data
def load_death_probabilities():
    """
    Load and parse the death probability CSV files.
    
    Returns:
        Dictionary with 'female' and 'male' pandas Series containing 2025 mortality data,
        or None if files cannot be loaded
    """
    import os
    
    possible_paths = [
        ('DeathProbsE_F_Alt2_TR2025.csv', 'DeathProbsE_M_Alt2_TR2025.csv'),
        ('data/DeathProbsE_F_Alt2_TR2025.csv', 'data/DeathProbsE_M_Alt2_TR2025.csv'),
        ('./data/DeathProbsE_F_Alt2_TR2025.csv', './data/DeathProbsE_M_Alt2_TR2025.csv'),
    ]
    
    for female_path, male_path in possible_paths:
        if os.path.exists(female_path) and os.path.exists(male_path):
            try:
                female_df = pd.read_csv(female_path, skiprows=1)
                male_df = pd.read_csv(male_path, skiprows=1)
                
                female_2025 = female_df[female_df['Year'] == 2025].iloc[0]
                male_2025 = male_df[male_df['Year'] == 2025].iloc[0]
                
                return {
                    'female': female_2025,
                    'male': male_2025
                }
            except Exception:
                continue
    
    st.error("❌ Unable to load actuarial data. Please ensure the CSV files are in the repository.")
    return None


def get_death_probability(data, age, gender='female'):
    """
    Get death probability for a specific age and gender.
    
    Args:
        data: Mortality data (dict or pandas Series)
        age: Age to lookup
        gender: 'female' or 'male'
    
    Returns:
        Death probability as float (0.0 if not found)
    """
    if data is None:
        return 0.0

    # Handle dictionary input
    if isinstance(data, dict):
        key = str(gender).lower()
        data = data.get(key) or data.get('female') or data.get('male')

    prob = 0.0
    
    # Handle pandas Series
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
        # Handle list/array/tuple-like data
        try:
            idx = int(age)
            if 0 <= idx < len(data):
                val = data[idx]
                if pd.notna(val):
                    prob = float(val)
        except Exception:
            prob = 0.0

    return prob


def calculate_premium(current_age, payout_age, interest, payout, gender='female', risk_margin=1.0):
    """
    Calculate premium for endowment insurance using actuarial principles.
    
    Args:
        current_age: Age when policy starts
        payout_age: Age when policy matures
        interest: Annual interest rate (as decimal, e.g., 0.06 for 6%)
        payout: Desired payout amount
        gender: 'female' or 'male'
        risk_margin: Multiplier for mortality rates (1.0 = standard, >1.0 = conservative)
    
    Returns:
        Tuple of (annual_premium, cumulative_death_probability)
    """
    death_data = load_death_probabilities()
    
    if death_data is None:
        return None, None
    
    weighted_total_annuity = 0
    death_cdf = 0
    prob_age_is_x = 1
    prob_death_given_age_is_x = 0
    
    for evaluation_age in range(current_age, payout_age):
        # Calculate survival probability to this age
        prob_age_is_x *= (1 - prob_death_given_age_is_x)
        
        # Get mortality rate and apply risk margin
        base_death_prob = get_death_probability(death_data, evaluation_age, gender)
        prob_death_given_age_is_x = min(base_death_prob * risk_margin, 1.0)
        
        # Calculate probability of death at this specific age
        if evaluation_age < payout_age - 1:
            prob_death_and_age_is_x = prob_age_is_x * prob_death_given_age_is_x
        else:
            prob_death_and_age_is_x = prob_age_is_x
            
        death_cdf += prob_death_and_age_is_x
        
        # Weight the annuity by probability of death at this age
        annuity_factor = accumulated_annuity(evaluation_age - current_age, interest, 1)
        weighted_total_annuity += annuity_factor * prob_death_and_age_is_x
    
    premium = payout / weighted_total_annuity if weighted_total_annuity > 0 else 0
    
    return premium, death_cdf


def calculate_risk_tolerance(premium, payout, current_age, payout_age, interest, gender):
    """
    Calculate probability of death before premiums exceed payout (break-even risk).
    
    Returns:
        Probability that death occurs before accumulated premiums exceed payout amount
    """
    death_data = load_death_probabilities()
    
    if death_data is None:
        return 0
    
    death_cdf = 0
    prob_age_is_x = 1
    prob_death_given_age_is_x = 0
    
    for age in range(current_age, payout_age):
        # Update survival probability
        prob_age_is_x *= (1 - prob_death_given_age_is_x)
        prob_death_given_age_is_x = get_death_probability(death_data, age, gender)
        
        # Accumulate death probability
        prob_death_and_age_is_x = prob_age_is_x * prob_death_given_age_is_x
        death_cdf += prob_death_and_age_is_x
        
        # Check if accumulated premiums exceed payout
        accumulated_premiums = premium * accumulated_annuity(age - current_age, interest, 1)
        
        if accumulated_premiums > payout:
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
        Tuple of (premium, actual_risk_tolerance)
    """
    # Check if target risk is achievable
    base_premium, _ = calculate_premium(current_age, payout_age, interest, payout, gender)
    base_risk = calculate_risk_tolerance(base_premium, payout, current_age, payout_age, interest, gender)
    
    if target_risk > base_risk:
        return base_premium, base_risk
    
    # Binary search bounds
    min_premium = 1
    max_premium = payout * 2
    tolerance = 0.001  # 0.1% tolerance on risk
    
    for _ in range(max_iterations):
        test_premium = (min_premium + max_premium) / 2
        actual_risk = calculate_risk_tolerance(test_premium, payout, current_age, payout_age, interest, gender)
        
        # Check convergence
        if abs(actual_risk - target_risk) < tolerance:
            return test_premium, actual_risk
        
        # Adjust search bounds
        if actual_risk > target_risk:
            min_premium = test_premium
        else:
            max_premium = test_premium
    
    return test_premium, actual_risk


def calculate_asset_growth(premium, current_age, payout_age, interest, payout, gender):
    """
    Calculate how accumulated premiums grow over time compared to payout amount.
    
    Returns:
        DataFrame with columns: Age, Accumulated_Value, Survival_Probability, Payout_Amount
    """
    death_data = load_death_probabilities()
    
    if death_data is None:
        return None
    
    ages = []
    accumulated_values = []
    survival_probs = []
    
    prob_age_is_x = 1
    prob_death_given_age_is_x = 0
    
    for age in range(current_age, payout_age + 1):
        years_elapsed = age - current_age
        
        # Calculate accumulated value of premiums
        accumulated_value = premium * accumulated_annuity(years_elapsed, interest, 1) if years_elapsed > 0 else 0
        
        # Calculate survival probability
        if age < payout_age:
            prob_death_given_age_is_x = get_death_probability(death_data, age, gender)
            prob_age_is_x *= (1 - prob_death_given_age_is_x)
        
        ages.append(age)
        accumulated_values.append(accumulated_value)
        survival_probs.append(prob_age_is_x * 100)
    
    return pd.DataFrame({
        'Age': ages,
        'Accumulated_Value': accumulated_values,
        'Survival_Probability': survival_probs,
        'Payout_Amount': [payout] * len(ages)
    })

# ==============================================================================
# USER INTERFACE
# ==============================================================================

def render_header():
    """Render page header and introduction."""
    st.markdown("<h1 style='text-align: center;'>💰 Endowment Insurance Premium Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Calculate premiums for policies that guarantee a payout at maturity or death</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("""
    **What is Endowment Insurance?**
    
    Endowment insurance combines life insurance with a savings component. It pays out a lump sum either:
    - At the end of the policy term (maturity age), OR
    - Upon death, whichever occurs first
    
    This calculator determines the annual premium needed to guarantee your desired payout.
    """)


def render_input_form():
    """
    Render input form and return user selections.
    
    Returns:
        Dictionary with keys: current_age, payout_age, gender, payout, interest, target_risk
    """
    st.markdown("---")
    st.markdown("### 📋 Enter Policy Details")
    
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
        
        target_risk = st.slider(
            "🎯 Target Risk Tolerance (%)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Desired probability of death before premiums exceed payout"
        ) / 100
    
    return {
        'current_age': current_age,
        'payout_age': payout_age,
        'gender': gender,
        'payout': payout,
        'interest': interest,
        'target_risk': target_risk
    }


def render_results(premium, risk_tolerance, actuarial_premium, inputs):
    """Render calculation results and metrics."""
    target_risk = inputs['target_risk']
    payout = inputs['payout']
    payout_age = inputs['payout_age']
    current_age = inputs['current_age']
    interest = inputs['interest']
    gender = inputs['gender']
    
    years = payout_age - current_age
    total_paid = premium * years
    accumulated_value = premium * accumulated_annuity(years, interest, 1)
    
    # Header
    st.markdown("<h2 style='text-align: center; color: #059669;'>Your Premium Calculation</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #6B7280; font-style: italic;'>Premium calculated to achieve {target_risk*100:.1f}% risk tolerance</p>", unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annual Premium", f"${premium:,.2f}")
    
    with col2:
        st.metric(f"Total Paid ({years} years)", f"${total_paid:,.2f}")
    
    with col3:
        st.metric("Accumulated Value at Maturity", f"${accumulated_value:,.2f}")
    
    # Policy analysis
    st.markdown("---")
    st.markdown("### 📊 Policy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Break-Even Risk",
            f"{risk_tolerance * 100:.3f}%",
            help="Probability of death before accumulated premiums exceed payout"
        )
        
        actuarial_risk = calculate_risk_tolerance(actuarial_premium, payout, current_age, payout_age, interest, gender)
        
        st.success(f"""
        **Risk-Target Pricing:**
        
        Premium calculated to achieve **{target_risk*100:.1f}%** target risk tolerance.
        
        Actual achieved risk: **{risk_tolerance * 100:.3f}%**
        
        This means there's a {risk_tolerance * 100:.3f}% chance you would pass away before 
        your accumulated premiums exceed ${payout:,}.
        
        For comparison, standard actuarial pricing would be **${actuarial_premium:,.2f}/year**
        with a risk tolerance of {actuarial_risk*100:.3f}%.
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


def render_growth_chart(premium, inputs):
    """Render asset growth visualization."""
    st.markdown("---")
    st.markdown("### 📈 Asset Growth Over Time")
    
    growth_data = calculate_asset_growth(
        premium, 
        inputs['current_age'], 
        inputs['payout_age'], 
        inputs['interest'], 
        inputs['payout'], 
        inputs['gender']
    )
    
    if growth_data is None:
        return
    
    fig = go.Figure()
    
    # Add accumulated value line
    fig.add_trace(go.Scatter(
        x=growth_data['Age'],
        y=growth_data['Accumulated_Value'],
        name='Accumulated Premiums',
        mode='lines',
        line=dict(color='#059669', width=3),
        hovertemplate='Age: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add payout line
    fig.add_trace(go.Scatter(
        x=growth_data['Age'],
        y=growth_data['Payout_Amount'],
        name='Payout Amount',
        mode='lines',
        line=dict(color='#DC2626', width=2, dash='dash'),
        hovertemplate='Age: %{x}<br>Payout: $%{y:,.0f}<extra></extra>'
    ))
    
    # Find break-even point
    break_even_age = None
    for i in range(len(growth_data)):
        if growth_data['Accumulated_Value'].iloc[i] >= inputs['payout']:
            break_even_age = growth_data['Age'].iloc[i]
            break_even_value = growth_data['Accumulated_Value'].iloc[i]
            break
    
    # Add break-even marker
    if break_even_age:
        fig.add_trace(go.Scatter(
            x=[break_even_age],
            y=[break_even_value],
            name='Break-Even Point',
            mode='markers',
            marker=dict(color='#F59E0B', size=12, symbol='star'),
            hovertemplate=f'Break-Even Age: {break_even_age}<br>Value: ${break_even_value:,.0f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Premium Accumulation vs Payout Amount',
        xaxis_title='Age',
        yaxis_title='Dollar Amount ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_yaxes(tickformat='$,.0f')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    if break_even_age:
        st.info(f"""
        📍 **Break-Even Analysis:**
        
        Your accumulated premiums will exceed the payout amount at **age {break_even_age}**.
        
        - If you pass away before age {break_even_age}, your beneficiaries receive more than you paid in
        - If you pass away after age {break_even_age}, you've paid more in premiums than the payout
        - The probability of death before age {break_even_age} is **{calculate_risk_tolerance(premium, inputs['payout'], inputs['current_age'], inputs['payout_age'], inputs['interest'], inputs['gender'])*100:.3f}%**
        """)
    else:
        st.info(f"""
        📍 **Growth Analysis:**
        
        Your accumulated premiums will reach **${growth_data['Accumulated_Value'].iloc[-1]:,.0f}** 
        by age {inputs['payout_age']}, which is **below** the payout amount of ${inputs['payout']:,}.
        
        This means the payout always exceeds what you've paid, making this policy favorable from 
        a pure financial perspective at all ages.
        """)


def render_calculation_details(inputs, death_cdf):
    """Render calculation methodology details."""
    st.markdown("---")
    st.markdown("### 🔍 How This Works")
    
    years = inputs['payout_age'] - inputs['current_age']
    interest = inputs['interest']
    
    with st.expander("View Calculation Details"):
        st.markdown(f"""
        **Premium Calculation Method:**
        
        1. **Expected Present Value**: The premium is calculated so that the expected present value 
           of all premium payments equals the expected present value of the payout.
        
        2. **Mortality Adjustment**: Uses 2025 Social Security Administration mortality tables 
           to calculate the probability of death at each age from {inputs['current_age']} to {inputs['payout_age']}.
        
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


def render_disclaimer():
    """Render disclaimer."""
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

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application entry point."""
    # Render header
    render_header()
    
    # Load death probability data
    death_prob_data = load_death_probabilities()
    if death_prob_data is None:
        st.error("Unable to load actuarial data. Please ensure the CSV files are in the same directory.")
        return
    
    # Get user inputs
    inputs = render_input_form()
    
    # Validate inputs
    if inputs['payout_age'] <= inputs['current_age']:
        st.error("Maturity age must be greater than current age!")
        return
    
    # Calculate premium
    st.markdown("---")
    
    with st.spinner("Calculating premium..."):
        premium, risk_tolerance = solve_premium_for_risk_target(
            inputs['target_risk'],
            inputs['payout'],
            inputs['current_age'],
            inputs['payout_age'],
            inputs['interest'],
            inputs['gender']
        )
        
        actuarial_premium, death_cdf = calculate_premium(
            inputs['current_age'],
            inputs['payout_age'],
            inputs['interest'],
            inputs['payout'],
            inputs['gender'],
            1.0
        )
    
    # Render results
    render_results(premium, risk_tolerance, actuarial_premium, inputs)
    render_growth_chart(premium, inputs)
    render_calculation_details(inputs, death_cdf)
    render_disclaimer()


if __name__ == "__main__":
    main()

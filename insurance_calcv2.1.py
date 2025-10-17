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
    st.error("❌ Unable to load actuarial data. Please ensure the CSV files are in the repository.")
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
    
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>Endowment Life Insurance Premium Calculator</h1>" \
    "<h2 style='text-align: center;'> By Joshua Tse</h2>", unsafe_allow_html=True)
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
    
    # Pricing method selection
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        current_age = st.number_input(
            "👤 Current Age",
            min_value=18,
            max_value=90,
            value=25,
            step=1,
            help="Your age when the policy starts"
        )
        
        payout_age = st.number_input(
            "🎯 Maturity Age",
            min_value=current_age + 5,
            max_value=100,
            value=99,
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
        
    # Validation
    if payout_age <= current_age:
        st.error("Maturity age must be greater than current age!")
        return
    
    # Calculate premium
    st.markdown("---")
    
    with st.spinner("Calculating premium..."):
        
            # Risk-target pricing: solve for premium that achieves target risk
            premium, risk_tolerance = solve_premium_for_risk_target(
                target_risk, payout, current_age, payout_age, interest, gender
            )
            
            # Also calculate what the actuarial premium would be for comparison
            actuarial_premium, death_cdf = calculate_premium(current_age, payout_age, interest, payout, gender, 1.0)
            pricing_note = f"Premium calculated to achieve {target_risk*100:.1f}% risk tolerance"
    
    # Display results
    st.markdown("<h2 style='text-align: center; color: #059669;'>Your Premium Calculation</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #6B7280; font-style: italic;'>{pricing_note}</p>", unsafe_allow_html=True)
    
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
        
        # Show pricing method interpretation
        
        st.success(f"""
            **Risk-Target Pricing:**
            
            Premium calculated to achieve **{target_risk*100:.1f}%** target risk tolerance.
            
            Actual achieved risk: **{risk_tolerance * 100:.3f}%**
            
            This means there's a {risk_tolerance * 100:.3f}% chance you would pass away before 
            your accumulated premiums exceed ${payout:,}.
            
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
        
        If you survive to age {payout_age}, you'll receive ${payout:,}
          after paying

        ${total_paid:,.2f} in total premiums.
        
        This represents a **{roi:.1f}%** total return on your premium payments 
        (not accounting for the time value of money or interest earned).
        """)
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 📈 Asset Growth Over Time")
    
    # Generate asset growth data
    growth_data = calculate_asset_growth(premium, current_age, payout_age, interest, payout, gender)
    
    if growth_data is not None:
        import plotly.graph_objects as go
        
        # Create figure with secondary y-axis
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
        
        # Find intersection point (break-even age)
        break_even_age = None
        for i in range(len(growth_data)):
            if growth_data['Accumulated_Value'].iloc[i] >= payout:
                break_even_age = growth_data['Age'].iloc[i]
                break_even_value = growth_data['Accumulated_Value'].iloc[i]
                break
        
        # Add break-even marker if it exists
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
            title=f'Premium Accumulation vs Payout Amount',
            xaxis_title='Age',
            yaxis_title='Dollar Amount ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickformat='$,.0f')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        if break_even_age:
            st.info(f"""
            📍 **Break-Even Analysis:**
            
            Your accumulated premiums will exceed the payout amount at **age {break_even_age}**.
            
            - If you pass away before age {break_even_age}, your beneficiaries receive more than you paid in
            - If you pass away after age {break_even_age}, you've paid more in premiums than the payout
            - The probability of death before age {break_even_age} is **{risk_tolerance*100:.3f}%**
            """)
        else:
            st.info(f"""
            📍 **Growth Analysis:**
            
            Your accumulated premiums will reach **${growth_data['Accumulated_Value'].iloc[-1]:,.0f}** 
            by age {payout_age}, which is **below** the payout amount of ${payout:,}.
            
            This means the payout always exceeds what you've paid, making this policy favorable from 
            a pure financial perspective at all ages.
            """)
    
    st.markdown("---")
    st.markdown("### 🔍 How This Works")
    
    with st.expander("View Calculation Details"):
        st.markdown(f"""
        **Premium Calculation Method:**
        
        1. **Expected Present Value**: The premium is calculated so that the expected accumulated value at time of death/maturity is equal to desired payout ammount.
        
        2. **Mortality Adjustment**: Uses 2025 Social Security Administration mortality tables 
           to calculate the probability of death at each age from {current_age} to {payout_age}.
        
        3. **Interest Accumulation**: Premiums accumulate with {interest*100:.1f}% annual interest, 
           creating a growing fund over time.
        
        4. **Risk Pooling**: The premium accounts for the fact that some policyholders will die 
           early (receiving the payout from limited premiums) while others survive to maturity 
           (having paid all premiums).
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
    st.markdown("---")
    st.link_button("Vist My Site :)","https://joshtse.fit/")

if __name__ == "__main__":
    main()

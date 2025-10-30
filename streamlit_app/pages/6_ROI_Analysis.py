import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from utils.data_loader import load_combined_metrics, load_facilities

st.set_page_config(page_title="ROI Analysis", page_icon="💰", layout="wide")

st.markdown('<style>.main-header {font-size: 2.5rem; font-weight: 700; color: #27ae60;}</style>', unsafe_allow_html=True)
st.markdown('<p class="main-header">💰 ROI & Financial Analysis</p>', unsafe_allow_html=True)

st.markdown("""
## Return on Investment for Decarbonization Strategies

Quantify the financial impact of efficiency improvements, renewable energy investments, 
and carbon pricing scenarios.

**Analysis Includes:**
- **Efficiency Investments**: PUE improvements, water conservation
- **Renewable Energy**: PPA costs vs grid electricity + carbon pricing
- **Carbon Pricing Scenarios**: EU ETS, California Cap-and-Trade, Internal carbon price
- **Decarbonization Cost Curves**: Cost per tonne CO₂e reduced
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load combined metrics"""
    df = load_combined_metrics()
    return df.dropna(subset=['electricity_mwh', 'pue', 'energy_cost_usd'])

try:
    data = load_data()
    facilities = load_facilities()
    
    if len(data) == 0:
        st.error("No data available")
        st.stop()
    
    st.success(f"✅ Loaded {len(data):,} records for financial analysis")
    
    # Sidebar - Analysis Selection
    st.sidebar.header("💼 Analysis Type")
    
    analysis_type = st.sidebar.radio(
        "Select Analysis",
        [
            "PUE Improvement ROI",
            "Renewable Energy ROI", 
            "Water Efficiency ROI",
            "Carbon Pricing Impact",
            "Decarbonization Cost Curve"
        ]
    )
    
    st.markdown("---")
    
    # ====================
    # 1. PUE IMPROVEMENT ROI
    # ====================
    if analysis_type == "PUE Improvement ROI":
        st.markdown("### ⚡ Data Center Efficiency Investment Analysis")
        st.markdown("""
        **PUE (Power Usage Effectiveness)** measures data center efficiency. Lower PUE means more energy 
        goes to IT equipment vs cooling/infrastructure.
        
        **Typical improvements:**
        - HVAC optimization: 1.15 → 1.10 (Cost: $500K, Payback: 2-3 years)
        - Hot aisle containment: 1.20 → 1.15 (Cost: $200K, Payback: 1-2 years)
        - Free cooling: 1.25 → 1.12 (Cost: $1M, Payback: 3-4 years)
        """)
        
        # Filter to data centers only
        dc_data = data[data['facility_type'] == 'Data Center'].copy()
        
        if len(dc_data) == 0:
            st.warning("No data center data available")
            st.stop()
        
        # Current state
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_pue = dc_data['pue'].mean()
            st.metric("Current Fleet PUE", f"{avg_pue:.3f}")
        
        with col2:
            total_electricity = dc_data['electricity_mwh'].sum()
            st.metric("Annual Electricity", f"{total_electricity:,.0f} MWh")
        
        with col3:
            total_energy_cost = dc_data['energy_cost_usd'].sum()
            st.metric("Annual Energy Cost", f"${total_energy_cost/1000:,.0f}M")
        
        st.markdown("---")
        
        # ROI Calculator
        st.markdown("### 🧮 PUE Improvement Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_pue = st.slider(
                "Target PUE",
                min_value=1.00,
                max_value=float(avg_pue),
                value=max(1.08, avg_pue - 0.02),
                step=0.01,
                help="Industry best practice: 1.08-1.10"
            )
            
            capital_investment = st.number_input(
                "Capital Investment ($M)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="One-time cost for efficiency upgrades"
            )
            
            energy_price = st.number_input(
                "Energy Price ($/MWh)",
                min_value=30.0,
                max_value=150.0,
                value=70.0,
                step=5.0,
                help="Average electricity cost"
            )
        
        with col2:
            # Calculate savings
            pue_reduction = avg_pue - target_pue
            pue_reduction_pct = (pue_reduction / avg_pue) * 100
            
            # Energy saved = (current_energy * (1 - target_pue/current_pue))
            current_total_energy = total_electricity * (avg_pue / 1.0)  # Back-calculate total
            new_total_energy = current_total_energy * (target_pue / avg_pue)
            energy_saved = current_total_energy - new_total_energy
            
            annual_cost_savings = energy_saved * energy_price / 1000  # Convert to $M
            
            # Carbon savings (assume 0.3 kg/kWh average)
            carbon_saved = energy_saved * 0.3  # tonnes CO2e
            
            # Payback period
            if annual_cost_savings > 0:
                payback_years = capital_investment / annual_cost_savings
            else:
                payback_years = 999
            
            st.metric("PUE Reduction", f"{pue_reduction:.3f} ({pue_reduction_pct:.1f}%)")
            st.metric("Annual Energy Saved", f"{energy_saved:,.0f} MWh")
            st.metric("Annual Cost Savings", f"${annual_cost_savings:.2f}M")
            st.metric("Annual Carbon Saved", f"{carbon_saved:,.0f} tonnes CO₂e")
            st.metric("Payback Period", f"{payback_years:.1f} years")
        
        # 10-year NPV analysis
        st.markdown("---")
        st.markdown("### 📊 10-Year Financial Projection")
        
        discount_rate = st.slider("Discount Rate (%)", 3.0, 10.0, 6.0, 0.5)
        
        years = np.arange(0, 11)
        cash_flows = np.zeros(11)
        cash_flows[0] = -capital_investment * 1000  # Convert to thousands
        
        for year in range(1, 11):
            # Annual savings with 2% energy price escalation
            escalation = 1.02 ** year
            cash_flows[year] = annual_cost_savings * 1000 * escalation
        
        # Calculate NPV
        npv = sum(cash_flows[i] / ((1 + discount_rate/100) ** i) for i in range(11))
        
        # Calculate cumulative cash flow
        cumulative_cf = np.cumsum(cash_flows)
        
        fig_cf = go.Figure()
        
        fig_cf.add_trace(go.Bar(
            x=years,
            y=cash_flows,
            name='Annual Cash Flow',
            marker_color=['red' if cf < 0 else 'green' for cf in cash_flows]
        ))
        
        fig_cf.add_trace(go.Scatter(
            x=years,
            y=cumulative_cf,
            name='Cumulative Cash Flow',
            mode='lines+markers',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))
        
        fig_cf.update_layout(
            title=f'Cash Flow Analysis (NPV: ${npv:,.0f}K)',
            xaxis_title='Year',
            yaxis_title='Annual Cash Flow ($K)',
            yaxis2=dict(title='Cumulative Cash Flow ($K)', overlaying='y', side='right'),
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cf, width = 'stretch')
        
        if npv > 0:
            st.success(f"✅ **Positive NPV: ${npv:,.0f}K** - Financially attractive investment")
        else:
            st.warning(f"⚠️ **Negative NPV: ${npv:,.0f}K** - May not meet financial hurdle rate")
        
        # Sensitivity analysis
        st.markdown("---")
        st.markdown("### 🎯 Sensitivity Analysis")
        
        # Vary energy price and capital cost
        energy_prices = np.linspace(40, 120, 20)
        capital_costs = np.linspace(0.5, 2.0, 20)
        
        payback_matrix = np.zeros((len(capital_costs), len(energy_prices)))
        
        for i, cap in enumerate(capital_costs):
            for j, price in enumerate(energy_prices):
                savings = energy_saved * price / 1000
                if savings > 0:
                    payback_matrix[i, j] = cap / savings
                else:
                    payback_matrix[i, j] = 999
        
        fig_sens = go.Figure(data=go.Heatmap(
            z=payback_matrix,
            x=energy_prices,
            y=capital_costs,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Payback<br>Years'),
            zmin=0,
            zmax=10
        ))
        
        fig_sens.update_layout(
            title='Payback Period Sensitivity',
            xaxis_title='Energy Price ($/MWh)',
            yaxis_title='Capital Investment ($M)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_sens, width = 'stretch')
        
        st.info("""
        **Investment Decision Framework:**
        - Payback < 3 years: **Strong investment** (typical for operational efficiency)
        - Payback 3-5 years: **Good investment** (meets most corporate hurdle rates)
        - Payback > 5 years: **Marginal** (may require strategic justification beyond ROI)
        """)
    
    # ====================
    # 2. RENEWABLE ENERGY ROI
    # ====================
    elif analysis_type == "Renewable Energy ROI":
        st.markdown("### 🌞 Renewable Energy Investment Analysis")
        st.markdown("""
        Compare the cost of renewable energy (PPAs, on-site solar) vs grid electricity 
        with carbon pricing considerations.
        
        **Typical PPA structures:**
        - 10-15 year contracts
        - Fixed price or escalating
        - Virtual (financial) or physical delivery
        """)
        
        # Current state
        total_electricity = data['electricity_mwh'].sum() / 12  # Monthly average
        current_renewable_mwh = data['renewable_electricity_mwh'].sum() / 12
        current_renewable_pct = (current_renewable_mwh / total_electricity) * 100 if total_electricity > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Electricity", f"{total_electricity:,.0f} MWh")
        
        with col2:
            st.metric("Current Renewable", f"{current_renewable_pct:.0f}%")
        
        with col3:
            avg_emissions = data.groupby('date')['scope2_market_tonnes'].sum().mean()
            st.metric("Avg Monthly Scope 2", f"{avg_emissions:,.0f} tonnes")
        
        st.markdown("---")
        
        # ROI Calculator
        st.markdown("### 🧮 Renewable Energy Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Handle case where already at 100%
            if current_renewable_pct >= 99:
                st.success("✅ Already at 100% renewable electricity match!")
                target_renewable_pct = 100
                
                st.info("""
                **Next Steps:**
                - Move from annual matching to 24/7 hourly matching
                - Increase Carbon-Free Energy (CFE) percentage
                - Add battery storage for grid stability
                - Explore renewable hydrogen for hard-to-abate sectors
                """)
            else:
                target_renewable_pct = st.slider(
                    "Target Renewable %",
                    min_value=int(current_renewable_pct),
                    max_value=100,
                    value=min(100, int(current_renewable_pct) + 20),
                    step=5
                )
            
            ppa_price = st.number_input(
                "PPA Price ($/MWh)",
                min_value=20.0,
                max_value=100.0,
                value=45.0,
                step=5.0,
                help="Long-term contract price for renewable energy"
            )
            
            grid_price = st.number_input(
                "Grid Price ($/MWh)",
                min_value=30.0,
                max_value=150.0,
                value=70.0,
                step=5.0,
                help="Current grid electricity price"
            )
            
            carbon_price = st.number_input(
                "Carbon Price ($/tonne CO₂e)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=10.0,
                help="EU ETS, California, or internal carbon price"
            )
        
        with col2:
            # Calculate additional renewable energy needed
            additional_renewable_pct = target_renewable_pct - current_renewable_pct
            additional_renewable_mwh = (additional_renewable_pct / 100) * total_electricity
            
            # Cost comparison
            renewable_cost = additional_renewable_mwh * ppa_price / 1000  # $K
            grid_cost = additional_renewable_mwh * grid_price / 1000  # $K
            
            # Carbon savings
            grid_ef = 0.35  # kg CO2e/kWh average
            carbon_avoided = additional_renewable_mwh * grid_ef  # tonnes
            carbon_value = carbon_avoided * carbon_price / 1000  # $K
            
            # Net cost
            net_cost = renewable_cost - grid_cost - carbon_value
            cost_per_tonne = (renewable_cost - grid_cost) / carbon_avoided if carbon_avoided > 0 else 0
            
            st.metric("Additional Renewable", f"{additional_renewable_mwh:,.0f} MWh/month")
            st.metric("Renewable Cost", f"${renewable_cost:,.0f}K/month")
            st.metric("Grid Cost Avoided", f"${grid_cost:,.0f}K/month")
            st.metric("Carbon Avoided", f"{carbon_avoided:,.0f} tonnes/month")
            st.metric("Carbon Value", f"${carbon_value:,.0f}K/month")
            
            if net_cost < 0:
                st.success(f"✅ Net Savings: ${abs(net_cost):,.0f}K/month")
            else:
                st.warning(f"⚠️ Net Cost: ${net_cost:,.0f}K/month (${cost_per_tonne:.2f}/tonne)")
        
        # Cost breakdown
        st.markdown("---")
        st.markdown("### 💵 Cost Comparison")
        
        cost_breakdown = pd.DataFrame({
            'Component': ['Renewable Energy Cost', 'Grid Cost Avoided', 'Carbon Value', 'Net Position'],
            'Amount': [renewable_cost, -grid_cost, -carbon_value, net_cost]
        })
        
        fig_cost = go.Figure(go.Waterfall(
            x=cost_breakdown['Component'],
            y=cost_breakdown['Amount'],
            text=cost_breakdown['Amount'].apply(lambda x: f"${x:,.0f}K"),
            textposition='outside',
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
            decreasing={'marker': {'color': 'green'}},
            increasing={'marker': {'color': 'red'}},
            totals={'marker': {'color': 'blue'}}
        ))
        
        fig_cost.update_layout(
            title='Monthly Cost Breakdown',
            yaxis_title='Cost ($K)',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_cost, width = 'stretch')
        
        # Carbon pricing scenarios
        st.markdown("---")
        st.markdown("### 🌍 Carbon Pricing Scenarios")
        
        carbon_prices = np.array([0, 25, 50, 75, 100, 150, 200])
        net_costs = []
        
        for cp in carbon_prices:
            cv = carbon_avoided * cp / 1000
            nc = renewable_cost - grid_cost - cv
            net_costs.append(nc)
        
        fig_carbon = go.Figure()
        
        fig_carbon.add_trace(go.Scatter(
            x=carbon_prices,
            y=net_costs,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig_carbon.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Break-even')
        
        fig_carbon.update_layout(
            title='Net Cost vs Carbon Price',
            xaxis_title='Carbon Price ($/tonne CO₂e)',
            yaxis_title='Net Monthly Cost ($K)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_carbon, width = 'stretch')
        
        # Find break-even carbon price
        if ppa_price > grid_price:
            breakeven_carbon = ((renewable_cost - grid_cost) / carbon_avoided) * 1000 if carbon_avoided > 0 else 999
            st.info(f"""
            **Break-even Carbon Price: ${breakeven_carbon:.2f}/tonne**
            
            - Below this price: Renewables cost more than grid + carbon
            - Above this price: Renewables are financially attractive
            - Current EU ETS: ~€85/tonne (~$92/tonne)
            - California Cap-and-Trade: ~$30/tonne
            """)
        else:
            st.success("✅ Renewables already cheaper than grid electricity - no carbon price needed!")
    
    # ====================
    # 3. WATER EFFICIENCY ROI
    # ====================
    elif analysis_type == "Water Efficiency ROI":
        st.markdown("### 💧 Water Conservation Investment Analysis")
        st.markdown("""
        Water efficiency in data centers reduces:
        1. **Direct costs**: Water purchase and wastewater treatment
        2. **Indirect costs**: Energy for pumping and treatment
        3. **Risk**: Water scarcity and regulatory compliance
        """)
        
        # Current state
        dc_data = data[data['facility_type'] == 'Data Center'].copy()
        
        total_water = dc_data['water_consumption_gallons'].sum() / 12  # Monthly avg
        total_water_cost = dc_data['water_cost_usd'].sum() / 12
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Water Use", f"{total_water/1_000_000:.1f}M gallons")
        
        with col2:
            st.metric("Monthly Water Cost", f"${total_water_cost:,.0f}K")
        
        with col3:
            water_cost_per_kgal = (total_water_cost * 1000) / (total_water / 1000) if total_water > 0 else 0
            st.metric("Cost per 1K gallons", f"${water_cost_per_kgal:.2f}")
        
        st.markdown("---")
        
        # ROI Calculator
        st.markdown("### 🧮 Water Efficiency Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            reduction_pct = st.slider(
                "Water Reduction Target (%)",
                min_value=10,
                max_value=50,
                value=30,
                step=5,
                help="Typical closed-loop cooling: 20-40% reduction"
            )
            
            capital_cost = st.number_input(
                "Capital Investment ($M)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Closed-loop systems, water recycling"
            )
            
            water_unit_cost = st.number_input(
                "Water Cost ($/1000 gallons)",
                min_value=5.0,
                max_value=20.0,
                value=8.5,
                step=0.5
            )
        
        with col2:
            # Calculate savings
            water_saved = total_water * (reduction_pct / 100)
            direct_savings = (water_saved / 1000) * water_unit_cost / 1000  # $K per month
            
            # Indirect energy savings (pumping, treatment)
            # Assume 0.3 kWh per 1000 gallons
            energy_saved = water_saved * 0.3 / 1000  # MWh
            energy_savings = energy_saved * 70 / 1000  # $K at $70/MWh
            
            total_monthly_savings = direct_savings + energy_savings
            annual_savings = total_monthly_savings * 12
            
            payback = capital_cost / annual_savings if annual_savings > 0 else 999
            
            st.metric("Water Saved", f"{water_saved/1_000_000:.2f}M gallons/month")
            st.metric("Direct Savings", f"${direct_savings:.1f}K/month")
            st.metric("Energy Savings", f"${energy_savings:.1f}K/month")
            st.metric("Total Monthly Savings", f"${total_monthly_savings:.1f}K")
            st.metric("Payback Period", f"{payback:.1f} years")
        
        st.info("""
        **Beyond Financial ROI:**
        - **Water risk mitigation**: Reduces vulnerability to drought and regulation
        - **ESG performance**: Improves water stewardship ratings
        - **Community relations**: Less competition for scarce water resources
        - **Regulatory compliance**: Proactive preparation for water restrictions
        
        Many companies value these strategic benefits at 2-3x the direct financial savings.
        """)
    
    # ====================
    # 4. CARBON PRICING IMPACT
    # ====================
    elif analysis_type == "Carbon Pricing Impact":
        st.markdown("### 💸 Carbon Pricing Scenario Analysis")
        st.markdown("""
        Model the financial impact of carbon pricing mechanisms:
        - **EU ETS**: ~€85/tonne ($92/tonne)
        - **California Cap-and-Trade**: ~$30/tonne  
        - **Internal Carbon Price**: Company-set shadow price
        """)
        
        # Calculate monthly emissions by region
        regional_emissions = data.groupby(['region', pd.Grouper(key='date', freq='M')]).agg({
            'scope1_tonnes': 'sum',
            'scope2_market_tonnes': 'sum',
            'total_emissions': 'sum'
        }).reset_index()
        
        # Average monthly by region
        avg_by_region = regional_emissions.groupby('region').agg({
            'scope1_tonnes': 'mean',
            'scope2_market_tonnes': 'mean',
            'total_emissions': 'mean'
        }).reset_index()
        
        st.markdown("### 📊 Current Emissions by Region")
        
        fig_region = go.Figure()
        
        fig_region.add_trace(go.Bar(
            name='Scope 1',
            x=avg_by_region['region'],
            y=avg_by_region['scope1_tonnes'],
            marker_color='#ff7f0e'
        ))
        
        fig_region.add_trace(go.Bar(
            name='Scope 2',
            x=avg_by_region['region'],
            y=avg_by_region['scope2_market_tonnes'],
            marker_color='#2ca02c'
        ))
        
        fig_region.update_layout(
            title='Average Monthly Emissions by Region',
            xaxis_title='Region',
            yaxis_title='Emissions (tonnes CO₂e)',
            barmode='stack',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_region, width = 'stretch')
        
        st.markdown("---")
        
        # Carbon pricing scenarios
        st.markdown("### 💰 Carbon Cost Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Low Price Scenario**")
            low_price = st.number_input("Low ($/tonne)", 0, 100, 25, 5, key='low')
        
        with col2:
            st.markdown("**Medium Price Scenario**")
            med_price = st.number_input("Medium ($/tonne)", 0, 150, 50, 5, key='med')
        
        with col3:
            st.markdown("**High Price Scenario**")
            high_price = st.number_input("High ($/tonne)", 0, 200, 100, 10, key='high')
        
        # Calculate costs for each region under each scenario
        scenarios = {'Low': low_price, 'Medium': med_price, 'High': high_price}
        
        cost_data = []
        for region in avg_by_region['region'].unique():
            region_emissions = avg_by_region[avg_by_region['region'] == region]
            operational_emissions = region_emissions['scope1_tonnes'].values[0] + region_emissions['scope2_market_tonnes'].values[0]
            
            for scenario_name, price in scenarios.items():
                monthly_cost = operational_emissions * price / 1000  # $K
                annual_cost = monthly_cost * 12
                cost_data.append({
                    'Region': region,
                    'Scenario': scenario_name,
                    'Monthly Cost ($K)': monthly_cost,
                    'Annual Cost ($M)': annual_cost / 1000
                })
        
        cost_df = pd.DataFrame(cost_data)
        
        # Visualize
        fig_scenarios = px.bar(
            cost_df,
            x='Region',
            y='Annual Cost ($M)',
            color='Scenario',
            barmode='group',
            title='Annual Carbon Cost by Region and Scenario',
            color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        )
        
        fig_scenarios.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_scenarios, width = 'stretch')
        
        # Total impact
        st.markdown("---")
        st.markdown("### 🌍 Total Company Impact")
        
        total_by_scenario = cost_df.groupby('Scenario')['Annual Cost ($M)'].sum().reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        for idx, row in total_by_scenario.iterrows():
            with [col1, col2, col3][idx]:
                st.metric(
                    f"{row['Scenario']} Scenario",
                    f"${row['Annual Cost ($M)']:.1f}M/year",
                    help=f"At ${scenarios[row['Scenario']]}/tonne CO₂e"
                )
        
        st.warning("""
        **Strategic Implications:**
        - Carbon pricing creates financial incentive for decarbonization
        - EU and California already have mandatory pricing
        - Many companies use internal carbon price ($40-100/tonne) for investment decisions
        - ROI calculations should include carbon cost avoidance
        """)
    
    # ====================
    # 5. DECARBONIZATION COST CURVE
    # ====================
    elif analysis_type == "Decarbonization Cost Curve":
        st.markdown("### 📉 Marginal Abatement Cost Curve (MACC)")
        st.markdown("""
        Prioritize decarbonization initiatives by cost-effectiveness ($/tonne CO₂e reduced).
        
        **Typical initiatives ranked by cost:**
        1. **Negative cost** (saves money): Energy efficiency, waste heat recovery
        2. **Low cost** (<$50/tonne): Renewable PPAs, LED lighting
        3. **Medium cost** ($50-100/tonne): On-site solar, HVAC upgrades
        4. **High cost** (>$100/tonne): Carbon capture, sustainable aviation fuel
        """)
        
        # Define decarbonization initiatives with costs and potential
        initiatives = pd.DataFrame({
            'Initiative': [
                'LED Lighting Upgrades',
                'HVAC Optimization',
                'Server Virtualization',
                'Renewable PPA (Wind)',
                'Hot Aisle Containment',
                'Free Cooling Systems',
                'On-site Solar',
                'Renewable PPA (Solar)',
                'Water Recycling',
                'Battery Storage',
                'Green Hydrogen',
                'Direct Air Capture'
            ],
            'Cost per tCO2e': [-45, -30, -25, 15, 20, 35, 50, 45, 40, 75, 150, 350],
            'Annual Reduction (k tonnes)': [5, 12, 8, 150, 10, 25, 30, 200, 3, 50, 100, 500],
            'Capital Required ($M)': [2, 5, 3, 0, 8, 15, 25, 0, 4, 40, 200, 800],
            'Timeframe': ['Immediate', '1 year', 'Immediate', '2 years', '1 year', 
                         '2-3 years', '2-3 years', '2 years', '1-2 years', '3-4 years', 
                         '5+ years', '5+ years']
        })
        
        # Sort by cost per tonne
        initiatives = initiatives.sort_values('Cost per tCO2e')
        initiatives['Cumulative Reduction'] = initiatives['Annual Reduction (k tonnes)'].cumsum()
        
        # Create MACC
        fig_macc = go.Figure()
        
        colors = ['green' if cost < 0 else 'orange' if cost < 50 else 'red' 
                 for cost in initiatives['Cost per tCO2e']]
        
        fig_macc.add_trace(go.Bar(
            x=initiatives['Cumulative Reduction'],
            y=initiatives['Cost per tCO2e'],
            text=initiatives['Initiative'],
            marker_color=colors,
            textposition='inside',
            textangle=0,
            width=initiatives['Annual Reduction (k tonnes)'],
            customdata=initiatives[['Initiative', 'Annual Reduction (k tonnes)', 'Capital Required ($M)', 'Timeframe']],
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Cost: $%{y}/tonne<br>' +
                         'Reduction: %{customdata[1]}K tonnes/year<br>' +
                         'Capital: $%{customdata[2]}M<br>' +
                         'Timeframe: %{customdata[3]}<extra></extra>'
        ))
        
        fig_macc.add_hline(y=0, line_dash='dash', line_color='black', annotation_text='Break-even')
        
        fig_macc.update_layout(
            title='Marginal Abatement Cost Curve (MACC)',
            xaxis_title='Cumulative Annual Emissions Reduction (K tonnes CO₂e)',
            yaxis_title='Cost per Tonne Reduced ($/tCO₂e)',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_macc, width = 'stretch')
        
        st.markdown("---")
        
        # Initiative details
        st.markdown("### 📋 Initiative Portfolio")
        
        # Color code the table
        def color_cost(val):
            if val < 0:
                return 'background-color: #d4edda'
            elif val < 50:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        styled_df = initiatives[['Initiative', 'Cost per tCO2e', 'Annual Reduction (k tonnes)', 
                                'Capital Required ($M)', 'Timeframe']].style.applymap(
            color_cost, subset=['Cost per tCO2e']
        ).format({
            'Cost per tCO2e': '${:.0f}',
            'Annual Reduction (k tonnes)': '{:.0f}',
            'Capital Required ($M)': '${:.0f}'
        })
        
        st.dataframe(styled_df, width = 'stretch')
        
        st.markdown("---")
        
        # Portfolio optimization
        st.markdown("### 🎯 Portfolio Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.slider(
                "Available Budget ($M)",
                min_value=0,
                max_value=500,
                value=100,
                step=10,
                help="Total capital budget for decarbonization"
            )
            
            max_cost_per_tonne = st.slider(
                "Max Cost per Tonne ($/tCO₂e)",
                min_value=-50,
                max_value=200,
                value=100,
                step=10,
                help="Maximum acceptable cost per tonne reduced"
            )
        
        with col2:
            # Filter initiatives by constraints
            feasible = initiatives[
                (initiatives['Capital Required ($M)'] <= budget) & 
                (initiatives['Cost per tCO2e'] <= max_cost_per_tonne)
            ].copy()
            
            # Simple greedy optimization: pick lowest cost first
            feasible = feasible.sort_values('Cost per tCO2e')
            cumulative_budget = 0
            selected = []
            
            for idx, row in feasible.iterrows():
                if cumulative_budget + row['Capital Required ($M)'] <= budget:
                    selected.append(row)
                    cumulative_budget += row['Capital Required ($M)']
            
            if len(selected) > 0:
                selected_df = pd.DataFrame(selected)
                total_reduction = selected_df['Annual Reduction (k tonnes)'].sum()
                total_capital = selected_df['Capital Required ($M)'].sum()
                avg_cost = (selected_df['Cost per tCO2e'] * selected_df['Annual Reduction (k tonnes)']).sum() / total_reduction if total_reduction > 0 else 0
                
                st.metric("Initiatives Selected", len(selected))
                st.metric("Total Annual Reduction", f"{total_reduction:.0f}K tonnes")
                st.metric("Total Capital Required", f"${total_capital:.0f}M")
                st.metric("Weighted Avg Cost", f"${avg_cost:.0f}/tonne")
            else:
                st.warning("No initiatives meet the budget and cost constraints")
        
        if len(selected) > 0:
            st.markdown("**Selected Initiatives:**")
            st.dataframe(
                selected_df[['Initiative', 'Cost per tCO2e', 'Annual Reduction (k tonnes)', 
                            'Capital Required ($M)', 'Timeframe']].style.format({
                    'Cost per tCO2e': '${:.0f}',
                    'Annual Reduction (k tonnes)': '{:.0f}',
                    'Capital Required ($M)': '${:.0f}'
                }),
                width = 'stretch'
            )
            
            # Implementation timeline
            st.markdown("---")
            st.markdown("### 📅 Implementation Timeline")
            
            timeline_data = []
            for _, row in selected_df.iterrows():
                timeline_data.append({
                    'Initiative': row['Initiative'],
                    'Start': 0,
                    'Duration': {'Immediate': 0.5, '1 year': 1, '1-2 years': 1.5, 
                               '2 years': 2, '2-3 years': 2.5, '3-4 years': 3.5}.get(row['Timeframe'], 2),
                    'Reduction': row['Annual Reduction (k tonnes)']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig_timeline = go.Figure()
            
            for idx, row in timeline_df.iterrows():
                fig_timeline.add_trace(go.Bar(
                    name=row['Initiative'],
                    x=[row['Duration']],
                    y=[row['Initiative']],
                    orientation='h',
                    marker_color=px.colors.qualitative.Set2[idx % len(px.colors.qualitative.Set2)],
                    text=f"{row['Reduction']:.0f}K tonnes/yr",
                    textposition='inside',
                    hovertemplate=f"<b>{row['Initiative']}</b><br>Duration: {row['Duration']:.1f} years<br>Reduction: {row['Reduction']:.0f}K tonnes/yr<extra></extra>"
                ))
            
            fig_timeline.update_layout(
                title='Implementation Timeline',
                xaxis_title='Years from Start',
                yaxis_title='',
                height=max(300, len(timeline_df) * 40),
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_timeline, width = 'stretch')
            
            # Cumulative impact
            st.markdown("---")
            st.markdown("### 📈 Cumulative Impact Projection")
            
            years = np.arange(0, 6)
            cumulative_reduction = np.zeros(6)
            
            for _, row in timeline_df.iterrows():
                start_year = int(row['Duration'])
                for year in range(start_year, 6):
                    cumulative_reduction[year] += row['Reduction']
            
            fig_impact = go.Figure()
            
            fig_impact.add_trace(go.Scatter(
                x=years,
                y=cumulative_reduction,
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
            
            fig_impact.update_layout(
                title='Cumulative Annual Emissions Reduction',
                xaxis_title='Year',
                yaxis_title='Annual Reduction (K tonnes CO₂e)',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_impact, width = 'stretch')
            
            st.success(f"""
            **Portfolio Summary:**
            - By Year 5: **{cumulative_reduction[-1]:.0f}K tonnes/year** reduction
            - Total investment: **${total_capital:.0f}M**
            - Average cost: **${avg_cost:.0f}/tonne** CO₂e reduced
            - Payback: Most initiatives pay back in **2-4 years** through energy savings
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Prioritization Strategy:**
            
            1. **Quick Wins** (Negative cost)
               - LED lighting, HVAC optimization
               - Immediate implementation
               - Self-funding through savings
            
            2. **Core Investments** ($0-50/tonne)
               - Renewable PPAs, efficiency upgrades
               - Strong business case
               - 2-4 year payback
            
            3. **Strategic Bets** ($50-100/tonne)
               - On-site generation, storage
               - Longer payback but strategic value
               - Reduces grid dependence
            """)
        
        with col2:
            st.markdown("""
            **Beyond Cost:**
            
            **Co-benefits to consider:**
            - Energy security and reliability
            - ESG ratings improvement
            - Attracting sustainability-focused customers
            - Employee engagement and retention
            - Future-proofing against regulations
            
            **Risk factors:**
            - Technology maturity
            - Permitting and approvals
            - Supply chain availability
            - Financing terms
            """)

        st.markdown("---")
    
    # ====================
    # HISTORICAL ROI ANALYSIS
    # ====================
    st.markdown("### 📊 Historical Performance Analysis")
    st.markdown("""
    Analyze actual operational improvements and cost trends from 2020-2024.
    """)
    
    # Calculate historical improvements
    try:
        yearly_data = data.groupby(data['date'].dt.year).agg({
            'pue': 'mean',
            'cfe_pct': 'mean',
            'water_consumption_gallons': 'sum',
            'electricity_mwh': 'sum',
            'total_emissions': 'sum',
            'energy_cost_usd': 'sum',
            'water_cost_usd': 'sum'
        }).reset_index()
        
        yearly_data.columns = ['year', 'pue', 'cfe_pct', 'water_consumption', 'electricity', 
                               'emissions', 'energy_cost', 'water_cost']
        
        # Filter to complete years
        yearly_data = yearly_data[(yearly_data['year'] >= 2020) & (yearly_data['year'] <= 2024)]
        
        if len(yearly_data) >= 2:
            first_year = yearly_data.iloc[0]
            last_year = yearly_data.iloc[-1]
            years_span = int(last_year['year'] - first_year['year'])
            
            st.markdown(f"#### 💼 Portfolio Trends ({int(first_year['year'])}-{int(last_year['year'])})")
            
            # Calculate changes
            pue_change = first_year['pue'] - last_year['pue']
            cfe_change = last_year['cfe_pct'] - first_year['cfe_pct']
            water_change = last_year['water_consumption'] - first_year['water_consumption']
            water_change_pct = (water_change / first_year['water_consumption']) * 100
            electricity_change = last_year['electricity'] - first_year['electricity']
            electricity_change_pct = (electricity_change / first_year['electricity']) * 100
            
            # Display trends
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if abs(pue_change) > 0.005:
                    st.metric("PUE Change", f"{pue_change:+.3f}")
                    st.caption(f"{first_year['pue']:.3f} → {last_year['pue']:.3f}")
                else:
                    st.metric("PUE", "Stable")
                    st.caption(f"Maintained {last_year['pue']:.3f}")
            
            with col2:
                st.metric("CFE Change", f"{cfe_change*100:+.0f}%")
                st.caption(f"{first_year['cfe_pct']*100:.0f}% → {last_year['cfe_pct']*100:.0f}%")
            
            with col3:
                st.metric("Water Growth", f"{water_change_pct:+.0f}%")
                st.caption(f"{water_change/1_000_000:+.0f}M gal/year")
            
            with col4:
                st.metric("Electricity Growth", f"{electricity_change_pct:+.0f}%")
                st.caption("Business expansion")
            
            st.markdown("---")
            
            # Contextual analysis
            st.markdown("#### 📊 Performance Context")
            
            # Check if there were meaningful improvements
            has_pue_improvement = pue_change > 0.01
            has_cfe_increase = cfe_change > 0.05
            has_water_reduction = water_change < 0

            #count data centers
            num_dc = len(data[data['facility_type'] == 'Data Center']['facility_id'].unique())
            
            if has_pue_improvement or has_cfe_increase or has_water_reduction:
                # Calculate ROI for actual improvements
                st.markdown("**Efficiency Gains & Financial Impact:**")
                
                
                returns_breakdown = []
                investment_breakdown = []
                
                # PUE improvement ROI
                if has_pue_improvement:
                    annual_electricity = last_year['electricity']
                    energy_saved = annual_electricity * (pue_change / first_year['pue'])
                    energy_cost_savings = energy_saved * 70 / 1000  # $K at $70/MWh
                    
                    # Estimate investment
                    pue_investment = pue_change * 400 * num_dc  # $K
                    
                    investment_breakdown.append(('PUE Efficiency', pue_investment))
                    returns_breakdown.append(('Energy Savings', energy_cost_savings))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ **PUE Improvement:** {pue_change:.3f}")
                        st.markdown(f"- Energy Saved: {energy_saved:,.0f} MWh/year")
                        st.markdown(f"- Cost Savings: ${energy_cost_savings:,.0f}K/year")
                    
                    with col2:
                        st.info(f"💵 **Est. Investment:** ${pue_investment:,.0f}K")
                        if energy_cost_savings > 0:
                            payback = pue_investment / energy_cost_savings
                            st.markdown(f"- Payback: {payback:.1f} years")
                            st.markdown(f"- Annual ROI: {(energy_cost_savings/pue_investment)*100:.0f}%")
                
                # CFE increase impact
                if has_cfe_increase:
                    carbon_avoided = last_year['electricity'] * 0.35 * cfe_change  # tonnes
                    carbon_value_conservative = carbon_avoided * 50 / 1000  # $K at $50/tonne
                    carbon_value_eu = carbon_avoided * 85 / 1000  # $K at €85/tonne
                    
                    renewable_investment = 100  # $K for contracting
                    investment_breakdown.append(('Renewable Contracts', renewable_investment))
                    returns_breakdown.append(('Carbon Value', carbon_value_conservative))
                    
                    st.success(f"✅ **CFE Increase:** {cfe_change*100:.0f}%")
                    st.markdown(f"- Carbon Avoided: {carbon_avoided:,.0f} tonnes/year")
                    st.markdown(f"- Carbon Value: ${carbon_value_conservative:,.0f}K/year (at $50/tonne)")
                    st.markdown(f"- EU ETS Value: ${carbon_value_eu:,.0f}K/year (at €85/tonne)")
                
                # Calculate total ROI if there were improvements
                if len(investment_breakdown) > 0:
                    total_investment = sum(inv[1] for inv in investment_breakdown)
                    total_returns = sum(ret[1] for ret in returns_breakdown)
                    
                    if total_returns > 0:
                        st.markdown("---")
                        st.markdown("**📈 Combined ROI:**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Investment", f"${total_investment:,.0f}K")
                        
                        with col2:
                            st.metric("Annual Returns", f"${total_returns:,.0f}K/year")
                        
                        with col3:
                            payback = total_investment / total_returns
                            st.metric("Payback Period", f"{payback:.1f} years")
            
            else:
                # No major improvements, focus on growth management
                st.info("""
                **Portfolio Scale & Efficiency:**
                
                While absolute emissions and resource consumption increased due to business growth, 
                the portfolio maintained stable operational efficiency metrics:
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Growth Context:**")
                    st.markdown(f"- Electricity: +{electricity_change_pct:.0f}% (business expansion)")
                    st.markdown(f"- Water: +{water_change_pct:.0f}% (proportional to growth)")
                    st.markdown(f"- PUE: Maintained at {last_year['pue']:.3f} (industry-leading)")
                    st.markdown(f"- CFE: {last_year['cfe_pct']*100:.0f}% (100% renewable match)")
                
                with col2:
                    st.markdown("**Efficiency Metrics:**")
                    
                    # Calculate intensity
                    emissions_2020 = first_year['emissions']
                    emissions_2024 = last_year['emissions']
                    electricity_2020 = first_year['electricity']
                    electricity_2024 = last_year['electricity']
                    
                    intensity_2020 = (emissions_2020 / electricity_2020) * 1000  # kg/MWh
                    intensity_2024 = (emissions_2024 / electricity_2024) * 1000
                    intensity_improvement = ((intensity_2020 - intensity_2024) / intensity_2020) * 100
                    
                    st.markdown(f"- Emissions Intensity: {intensity_improvement:+.1f}%")
                    st.markdown(f"- From {intensity_2020:.0f} → {intensity_2024:.0f} kg CO₂e/MWh")
                    
                    if intensity_improvement > 0:
                        st.success("✅ Improved efficiency per unit of energy")
                    else:
                        st.info("→ Maintained efficiency despite growth")
            
            st.markdown("---")
            
            # Future opportunities
            st.markdown("#### 🎯 Opportunities for Further Improvement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**High-Impact Initiatives:**")
                
                # Calculate potential from best practice
                best_pue = 1.08
                current_pue = last_year['pue']
                
                if current_pue > best_pue:
                    potential_improvement = current_pue - best_pue
                    potential_energy = last_year['electricity'] * (potential_improvement / current_pue)
                    potential_savings = potential_energy * 70 / 1000
                    
                    st.markdown(f"**1. PUE Optimization**")
                    st.markdown(f"- Target: {best_pue:.3f} (from {current_pue:.3f})")
                    st.markdown(f"- Potential: {potential_energy:,.0f} MWh/year")
                    st.markdown(f"- Value: ${potential_savings:,.0f}K/year")
                    st.markdown(f"- Est. Investment: ${potential_improvement * 400 * num_dc:,.0f}K")
                else:
                    st.markdown("**1. PUE Leadership**")
                    st.markdown(f"- Already at best-in-class {current_pue:.3f}")
                    st.markdown("- Focus: Maintain excellence")
                
                # CFE opportunity
                current_cfe = last_year['cfe_pct']
                if current_cfe < 0.80:
                    cfe_opportunity = 0.80 - current_cfe
                    carbon_opportunity = last_year['electricity'] * 0.35 * cfe_opportunity
                    
                    st.markdown(f"**2. CFE Acceleration**")
                    st.markdown(f"- Target: 80% (from {current_cfe*100:.0f}%)")
                    st.markdown(f"- Carbon Avoided: {carbon_opportunity:,.0f} tonnes/year")
                    st.markdown(f"- Value: ${carbon_opportunity * 50 / 1000:,.0f}K/year at $50/tonne")
                else:
                    st.markdown(f"**2. CFE Excellence**")
                    st.markdown(f"- Already at {current_cfe*100:.0f}%")
                    st.markdown("- Path to 100% 24/7 matching")
            
            with col2:
                st.markdown("**Strategic Recommendations:**")
                
                st.markdown("""
                **Continuous Improvement:**
                - Benchmark against industry leaders
                - Pilot emerging efficiency technologies
                - Optimize cooling systems seasonally
                - Monitor PUE trends monthly
                
                **Renewable Acceleration:**
                - Additional PPA contracts in high-grid regions
                - On-site solar for suitable locations
                - Battery storage for 24/7 CFE matching
                
                **Water Strategy:**
                - Closed-loop cooling upgrades
                - Water replenishment programs
                - Drought-resilient technologies
                """)
            
            st.info("""
            **Methodology:**
            - All metrics calculated from actual operational data (2020-2024)
            - Investment estimates based on industry-standard costs
            - Energy costs at $70/MWh market average
            - Carbon value at $50/tonne (conservative) or €85/tonne (EU ETS)
            - ROI calculations use simple payback for clarity
            """)
        
        else:
            st.warning("Need at least 2 years of data for historical analysis")
    
    except Exception as e:
        st.warning(f"Could not complete historical analysis: {e}")
        import traceback
        st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
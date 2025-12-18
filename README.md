#  Corporate Sustainability Analytics Platform

A production-ready analytics dashboard for corporate carbon accounting, operational efficiency tracking, and decarbonization strategy planning. Built with real environmental report data (2020-2024).

Live app found at: **carbon-dashboard-qiuk6yiazz2jwumlz9bjul.streamlit.app**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Project Overview

This platform demonstrates enterprise-grade sustainability analytics capabilities:

- **Data Engineering**: ETL pipeline processing real environmental reports
- **Machine Learning**: Time series forecasting with Prophet, ARIMA, and Holt-Winters
- **Financial Modeling**: ROI analysis, cost curves, scenario planning
- **Operational Metrics**: PUE, CFE, water intensity, waste circularity
- **Interactive Dashboards**: 5 comprehensive analysis pages

---

##  Project Structure
```
carbon-dashboard/
├── README.md                          # This file
├── PORTFOLIO_OVERVIEW.md              # Detailed project documentation
├── INTERVIEW_PREP.md                  # Interview talking points
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── generate_sample_data.py        # Data generation from environmental reports
│   └── sustainability_data.db         # Generated SQLite database
│
├── streamlit_app/
│   ├── app.py                          # Landing page
│   │
│   ├── pages/                          # Dashboard pages
│   │   ├── 1__Emissions_Overview.py  # Portfolio-wide tracking
│   │   ├── 2__Operational_Metrics.py # PUE, CFE, water, waste
│   │   ├── 3__Scope_Analysis.py      # GHG Protocol breakdown
│   │   ├── 4__ML_Forecasting.py      # Prophet, ARIMA, Holt-Winters
│   │   └── 5__ROI_Analysis.py        # Financial modeling
│   │
│   └── utils/
│       └── data_loader.py              # Reusable data utilities
│
└── docs/                               # Additional documentation (optional)
```

---

##  Dashboard Features

### 1.  Emissions Overview
- **Real-time KPIs**: Total emissions, YoY changes, renewable percentage
- **Multi-dimensional breakdowns**: By scope, region, facility type, time
- **Intensity metrics**: Emissions per MWh, per water consumed
- **Historical trends**: 2020-2024 performance analysis

### 2.  Operational Metrics
- **PUE Tracking**: Power Usage Effectiveness (industry benchmark: 1.09)
- **Carbon-Free Energy**: Hourly CFE matching (66% in 2024)
- **Water Management**: Consumption, replenishment (64% in 2024), intensity
- **Waste Circularity**: Diversion rates (84% in 2024)
- **Historical Performance**: Year-over-year improvements with financial impact

### 3.  Scope Analysis
- **GHG Protocol Compliant**: Scope 1, 2 (location & market-based), 3
- **Category Breakdown**: 15 Scope 3 categories
- **SBTi Tracking**: Science-Based Targets (50% Scope 1&2, 30% Scope 3 by 2030)
- **Visualizations**: Waterfall charts, Pareto analysis

### 4.  ML Forecasting
- **Three Models**: Prophet, ARIMA/SARIMA, Holt-Winters
- **Scenario Analysis**: Business-as-usual, Aggressive Decarbonization, Efficiency Focus
- **Forecast Targets**: Emissions, PUE, CFE, water, waste
- **Model Diagnostics**: MAE, RMSE, MAPE, AIC, BIC
- **6-60 Month Horizon**: Adjustable forecast period

### 5.  ROI Analysis
- **PUE Improvement ROI**: NPV, payback, sensitivity analysis
- **Renewable Energy ROI**: PPA costs vs grid + carbon pricing
- **Water Efficiency ROI**: Cost savings + risk mitigation
- **Carbon Pricing Impact**: EU ETS, California Cap-and-Trade scenarios
- **Cost Curves**: Marginal Abatement Cost Curve (MACC) with portfolio optimization
- **Historical Returns**: Actual 2020-2024 improvements with estimated investment costs

---

##  Technical Stack

**Core Technologies:**
- **Python 3.10+**: Primary language
- **Streamlit 1.28+**: Interactive dashboards
- **Plotly**: Interactive visualizations
- **SQLite**: Database

**Data Science:**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

**Time Series:**
- **Prophet**: Facebook's forecasting library
- **Statsmodels**: ARIMA/SARIMA, Holt-Winters
- **Scipy**: Statistical functions

**Other:**
- **SHAP**: Model explainability (optional)

---

##  Data Methodology

### Data Sources

**Real Data (from environmental reports):**
-  Annual emissions by scope (2020-2024): 8.7M → 15.2M tonnes CO₂e
-  Facility-level PUE ratings: 27 data centers (2020-2024)
-  Facility-level water consumption: 27 data centers (2024)
-  Regional CFE percentages: 12% (APAC) to 92% (Latin America)
-  Energy consumption: 15.5M → 32.7M MWh (2020-2024)
-  Waste generation: 31.5K → 58.5K tonnes (2020-2024)

**Estimated/Interpolated:**
-  Monthly disaggregation: Linear interpolation from annual totals
-  Facilities not in reports: Proportional allocation based on facility type
-  Cost estimates: Market rates ($70/MWh energy, $50/tonne CO₂e)

### Key Assumptions
- 100% renewable electricity matching maintained 2020-2024
- Market rates for cost calculations (clearly documented)
- Investment costs from industry benchmarks ($300-500K per 0.01 PUE improvement)
- No fabricated training data or false benchmarks

### Verification
Generated data matches published reports within ±1.2%:
```
2020: Generated=8,735,546 | Actual=8,737,400 | Diff=0.0%
2021: Generated=10,630,118 | Actual=10,520,600 | Diff=1.0%
2022: Generated=11,941,625 | Actual=11,900,300 | Diff=0.3%
2023: Generated=14,387,547 | Actual=14,296,800 | Diff=0.6%
2024: Generated=15,373,083 | Actual=15,185,200 | Diff=1.2%
```

---

##  Learning Outcomes

This project demonstrates:

**Data Engineering:**
- ETL pipeline design and implementation
- Database schema design (normalized tables)
- Data quality validation and error handling
- Time series processing and interpolation

**Machine Learning:**
- Time series forecasting (Prophet, ARIMA, Holt-Winters)
- Model evaluation and comparison
- Scenario analysis and what-if modeling
- Forecasting both emissions and operational metrics

**Financial Modeling:**
- NPV analysis with discount rates
- Break-even and sensitivity analysis
- Cost-benefit modeling
- Portfolio optimization under constraints

**Domain Expertise:**
- GHG Protocol (Scope 1, 2, 3)
- Science-Based Targets (SBTi)
- Operational efficiency metrics (PUE, CFE)
- Sustainability reporting standards

**Software Engineering:**
- Modular, reusable code architecture
- Comprehensive error handling
- Interactive dashboard development
- Production-ready patterns

---

##  Troubleshooting

### Common Issues

**1. Database not found**
```bash
# Regenerate the database
cd data
python generate_sample_data.py
cd ..
```

**2. Import errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**3. ARIMA not available**
- ARIMA may not be available due to NumPy compatibility issues
- The app gracefully degrades to Prophet and Holt-Winters
- Both are fully functional and industry-standard

**4. Streamlit won't start**
```bash
# Check Python version
python --version  # Should be 3.10+

# Ensure you're in the correct directory
cd streamlit_app
streamlit run app.py
```

**5. Charts not displaying**
- Clear Streamlit cache: Press 'C' in the browser
- Refresh the page: Press 'R' in the browser
- Check browser console for JavaScript errors

---

##  Usage Examples

### Regenerate Database with Custom Data
Edit `data/generate_sample_data.py` to modify:
- Annual emissions targets
- Facility configurations
- Cost assumptions
- Seasonal patterns

Then regenerate:
```bash
cd data
python generate_sample_data.py
```

### Export Data
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/sustainability_data.db')
df = pd.read_sql_query("SELECT * FROM emissions_monthly", conn)
df.to_csv('emissions_export.csv', index=False)
```

### Run Specific Pages Only
```bash
# Run just the forecasting page
streamlit run streamlit_app/pages/4__ML_Forecasting.py
```

---

##  Future Enhancements

**Deployment:**
- [ ] Deploy to Streamlit Cloud
- [ ] Containerize with Docker
- [ ] Add CI/CD pipeline

**Features:**
- [ ] User authentication
- [ ] PDF report generation
- [ ] Data export functionality
- [ ] Real-time API integrations
- [ ] Email alerting

**ML Enhancements:**
- [ ] Ensemble forecasting
- [ ] Anomaly detection alerts
- [ ] Transfer learning
- [ ] Uncertainty quantification

**Additional Analysis:**
- [ ] Facility-level deep dives
- [ ] Supply chain Scope 3 tracking
- [ ] Renewable energy certificate (REC) management
- [ ] Carbon offset portfolio management

---

##  Contributing

This is a portfolio project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**Adam Ritter**

- GitHub: [github.com/adam-ritter]
- LinkedIn: [linkedin.com/in/adam-ritter-env]
- Email: [adam.h.ritter@gmail.com]

---

##  Acknowledgments

- Environmental report data methodology based on published tech company sustainability reports
- GHG Protocol standards from [ghgprotocol.org](https://ghgprotocol.org)
- Science-Based Targets initiative (SBTi) methodology
- EPA eGRID emission factors
- Industry benchmarks for PUE from Uptime Institute and Google

---

##  Additional Resources

**Learn More:**
- [PORTFOLIO_OVERVIEW.md](PORTFOLIO_OVERVIEW.md) - Detailed project documentation

**Related Topics:**
- [GHG Protocol](https://ghgprotocol.org/) - Emissions accounting standards
- [Science Based Targets](https://sciencebasedtargets.org/) - Corporate climate targets
- [Streamlit Documentation](https://docs.streamlit.io/) - Dashboard framework
- [Prophet Documentation](https://facebook.github.io/prophet/) - Time series forecasting

---

##  Support

For questions or issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [PORTFOLIO_OVERVIEW.md](PORTFOLIO_OVERVIEW.md)
3. Open an issue on GitHub
4. Contact via email

---

*Built with ❤️ and ☕ to demonstrate data science and sustainability analytics capabilities.*

import streamlit as st

import warnings
import traceback
import sys
from utils.data_loader import get_data_quality_status

# ... existing imports ...

# After page title, add status
data_status = get_data_quality_status()

if data_status['using_cleaned']:
    st.info("ℹ️ **Data Status:** Using cleaned database (quality issues resolved)")
else:
    st.warning("⚠️ **Data Note:** Using raw database with intentional quality issues (~5%). Visit **Data Quality** page to generate cleaned data for accurate analysis.")

def warning_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warning_with_traceback

st.set_page_config(
    page_title="Corporate Sustainability Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .tech-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🌱 Corporate Sustainability Analytics Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Environmental Accounting & GHG Protocol Analysis</p>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to the Sustainability Analytics Platform
    
    This platform demonstrates **enterprise-grade sustainability data management and analytics** 
    using machine learning, SQL databases, and interactive visualizations.
    
    Built to showcase capabilities for **corporate environmental accounting programs**, 
    particularly for companies like Google transforming sustainability from 
    reporting-focused to decision-focused intelligence.
    
    ### 🎯 Key Capabilities
    
    - **GHG Protocol Compliant**: Scope 1, 2, and 3 emissions tracking
    - **SQL-Based**: Scalable database architecture for enterprise data
    - **ML-Powered**: 4+ machine learning models for forecasting and analysis
    - **Interactive**: Streamlit dashboards + Tableau integration
    - **Real-time**: Live calculations and scenario modeling
    """)

with col2:
    st.info("""
    ### 📊 Quick Stats
    
    **Technology Stack:**
    - Python 3.10+
    - SQLite/PostgreSQL
    - Scikit-learn
    - Prophet & SHAP
    - Streamlit
    - Plotly
    
    **ML Models:**
    - Time Series Forecasting
    - Anomaly Detection
    - Regression Analysis
    - K-means Clustering
    """)

st.markdown("---")

# Features
st.markdown("## 🚀 Platform Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
    <h3>🤖 AI Forecasting</h3>
    <p>Machine learning models predict future emissions using Prophet and Holt-Winters algorithms. 
    Enables proactive planning and early warning of target misses.</p>
    <span class="tech-badge">Prophet</span>
    <span class="tech-badge">Time Series</span>
    <span class="tech-badge">95% CI</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h3>📈 Driver Analysis</h3>
    <p>Regression models with SHAP explainability identify which business metrics 
    drive emissions. Enables data-driven decarbonization strategies.</p>
    <span class="tech-badge">Random Forest</span>
    <span class="tech-badge">SHAP</span>
    <span class="tech-badge">Feature Importance</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <h3>🚨 Anomaly Detection</h3>
    <p>Isolation Forest and statistical methods automatically flag data quality issues 
    across thousands of monthly data points.</p>
    <span class="tech-badge">Isolation Forest</span>
    <span class="tech-badge">Z-Score</span>
    <span class="tech-badge">Auto QA</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h3>🏭 Facility Clustering</h3>
    <p>K-means clustering segments facilities into peer groups for benchmarking 
    and tailored reduction strategies.</p>
    <span class="tech-badge">K-Means</span>
    <span class="tech-badge">PCA</span>
    <span class="tech-badge">Segmentation</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Technical architecture
st.markdown("## 🏗️ Technical Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Data Layer
    - **SQL Database**: GHG Protocol schema
    - **Tables**: Scope 1/2/3, facilities, targets
    - **Queries**: Optimized for analytics
    - **Integration**: REST API ready
    """)

with col2:
    st.markdown("""
    ### 🤖 ML Layer
    - **Forecasting**: Prophet, ARIMA
    - **Anomaly**: Isolation Forest
    - **Regression**: Ensemble methods
    - **Clustering**: K-means, DBSCAN
    """)

with col3:
    st.markdown("""
    ### 📈 Presentation Layer
    - **Streamlit**: Interactive analysis
    - **Tableau**: Executive reporting
    - **Plotly**: Dynamic visualizations
    - **Export**: CSV, PDF, reports
    """)

st.markdown("---")

# Navigation
st.markdown("## 📍 Navigate the Platform")

st.markdown("""
Use the **sidebar** to access different analysis modules:

1. **📊 Emissions Overview** - Dashboard with key metrics and trends
2. **🔬 Scope Analysis** - Deep dive into Scope 1, 2, and 3 emissions
3. **🤖 AI Forecasting** - 12-month emissions predictions
4. **🚨 Anomaly Detection** - Automated data quality checks
5. **📈 Driver Analysis** - Business metric correlation and SHAP
6. **🏭 Facility Clustering** - Peer group segmentation
7. **💾 Data Quality** - Comprehensive data quality dashboard
""")

# Call to action
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎓 For Hiring Managers
    
    This platform demonstrates:
    - SQL proficiency
    - ML/AI experience
    - GHG Protocol expertise
    - Data visualization skills
    - Program management thinking
    """)

with col2:
    st.markdown("""
    ### 💼 For Sustainability Teams
    
    Use this as a template for:
    - Building your analytics stack
    - Implementing ML forecasting
    - Automating data quality
    - Scaling GHG accounting
    """)

with col3:
    st.markdown("""
    ### 🔧 For Developers
    
    Explore the codebase:
    - Modular ML utilities
    - Reusable SQL queries
    - Clean architecture
    - Production-ready patterns
    """)

st.markdown("---")

# Footer
st.markdown("""
### 📧 Contact & Links

**Developer:** Adam Ritter  
**Email:** adam.h.ritter@gmail.com  
**LinkedIn:** [linkedin.com/in/adam-ritter-env](https://linkedin.com/in/adam-ritter-env)  
**GitHub:** [github.com/Adam-Ritter](https://github.com/Adam-Ritter)  

**Built with:** Python • SQL • Streamlit • Scikit-learn • Prophet • SHAP • Plotly

---

*This platform was developed as part of a climate tech portfolio showcasing 
technical skills in sustainability data analytics and machine learning.*
""")
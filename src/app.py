import streamlit as st

from components.healthcare import show_healthcare_dashboard
from components.banking import show_banking_dashboard
from components.enterprise import show_enterprise_dashboard
from components.mri import show_mri_dashboard


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Synthetic Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLOBAL DASHBOARD THEME
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F7F9FC;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    h1, h2, h3 {
        color: #1F2A44;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(31, 42, 68, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #667085;
    }

    div[data-testid="stMetricValue"] {
        color: #1F2A44;
    }

    .dashboard-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E5EAF0;
        box-shadow: 0 2px 8px rgba(31, 42, 68, 0.05);
        margin-bottom: 15px;
    }

    .section-title {
        color: #1F2A44;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div style="
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #E5EAF0;
        margin-bottom: 25px;
    ">

        <div style="
            color: #1F2A44;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        ">
            🧠 Synthetic Intelligence Platform
        </div>

        <div style="
            color: #667085;
            font-size: 16px;
        ">
            Synthetic data generation, analytics and intelligence platform
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Platform")

st.sidebar.markdown(
    "### Select Domain"
)

domain = st.sidebar.radio(
    "",
    [
        "🏥 Healthcare",
        "🧠 MRI Intelligence",
        "🏦 Banking",
        "🏢 Enterprise"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.caption(
    "Synthetic Intelligence Platform"
)

st.sidebar.caption(
    "Analytics • Synthetic Data • AI Intelligence"
)


# =========================================================
# HEALTHCARE
# =========================================================

if domain == "🏥 Healthcare":
    show_healthcare_dashboard()


# =========================================================
# MRI INTELLIGENCE
# =========================================================

elif domain == "🧠 MRI Intelligence":
    show_mri_dashboard()


# =========================================================
# BANKING
# =========================================================

elif domain == "🏦 Banking":
    show_banking_dashboard()


# =========================================================
# ENTERPRISE
# =========================================================

elif domain == "🏢 Enterprise":
    show_enterprise_dashboard()
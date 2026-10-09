import streamlit as st

from components.healthcare import show_healthcare_dashboard
from components.banking import show_banking_dashboard
from components.enterprise import show_enterprise_dashboard
from components.mri import show_mri_dashboard


st.set_page_config(
    page_title="Synthetic Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# =========================
# PLATFORM HEADER
# =========================

st.title("🧠 Synthetic Intelligence Platform")
st.caption("Synthetic data generation, analytics and intelligence platform")


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Platform")

domain = st.sidebar.radio(
    "Select Domain",
    [
        "🏥 Healthcare",
        "🧠 MRI Intelligence",
        "🏦 Banking",
        "🏢 Enterprise"
    ]
)


# =========================
# DOMAIN ROUTING
# =========================

if domain == "🏥 Healthcare":
    show_healthcare_dashboard()

elif domain == "🧠 MRI Intelligence":
    show_mri_dashboard()

elif domain == "🏦 Banking":
    show_banking_dashboard()

elif domain == "🏢 Enterprise":
    show_enterprise_dashboard()
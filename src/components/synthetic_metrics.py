import streamlit as st
import os
import pandas as pd


# =========================================================
# METRIC FILE READER
# =========================================================

def read_metric(file_path, default=0.0):

    try:

        if not os.path.exists(file_path):
            return default

        with open(file_path, "r", encoding="utf-8") as f:
            value = f.read().strip()

        value = value.split(":")[-1]
        value = value.replace("%", "").strip()

        return float(value)

    except Exception:
        return default


# =========================================================
# LOAD HEALTHCARE METRICS
# =========================================================

def load_healthcare_metrics():

    quality = read_metric(
        "outputs/quality_score.txt"
    )

    privacy = read_metric(
        "outputs/privacy_score.txt"
    )

    correlation = read_metric(
        "outputs/correlation_score.txt"
    )

    utility = read_metric(
        "outputs/utility_score.txt"
    )

    return {
        "quality": quality,
        "privacy": privacy,
        "correlation": correlation,
        "utility": utility
    }


# =========================================================
# LOAD DOMAIN METRICS
# =========================================================

def load_domain_metrics(domain):

    domain = domain.lower()

    # -----------------------------------------------------
    # HEALTHCARE
    # -----------------------------------------------------

    if domain == "healthcare":

        return load_healthcare_metrics()

    # -----------------------------------------------------
    # BANKING
    # -----------------------------------------------------

    elif domain == "banking":

        base = "data/banking/comparison"

        return {
            "quality": read_metric(
                f"{base}/quality_score.txt"
            ),
            "privacy": read_metric(
                f"{base}/privacy_score.txt"
            ),
            "correlation": read_metric(
                f"{base}/correlation_score.txt"
            ),
            "utility": read_metric(
                f"{base}/utility_score.txt"
            )
        }

    # -----------------------------------------------------
    # ENTERPRISE
    # -----------------------------------------------------

    elif domain == "enterprise":

        base = "data/enterprise/comparison"

        return {
            "quality": read_metric(
                f"{base}/quality_score.txt"
            ),
            "privacy": read_metric(
                f"{base}/privacy_score.txt"
            ),
            "correlation": read_metric(
                f"{base}/correlation_score.txt"
            ),
            "utility": read_metric(
                f"{base}/utility_score.txt"
            )
        }

    return {
        "quality": 0,
        "privacy": 0,
        "correlation": 0,
        "utility": 0
    }


# =========================================================
# METRIC CARD
# =========================================================

def metric_card(
    title,
    value,
    description
):

    st.markdown(
        f"""
        <div style="
            background-color:white;
            border:1px solid #E5EAF0;
            border-radius:12px;
            padding:18px;
            box-shadow:0 2px 8px rgba(31,42,68,0.05);
            min-height:145px;
        ">

            <div style="
                color:#667085;
                font-size:14px;
                font-weight:600;
                margin-bottom:8px;
            ">
                {title}
            </div>

            <div style="
                color:#1F2A44;
                font-size:30px;
                font-weight:700;
                margin-bottom:8px;
            ">
                {value:.2f}%
            </div>

            <div style="
                color:#667085;
                font-size:12px;
                line-height:1.5;
            ">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SHOW METRIC CARDS
# =========================================================

def show_synthetic_metrics(domain):

    metrics = load_domain_metrics(domain)

    st.divider()

    st.markdown(
        f"""
        <div class="section-title">
            Synthetic Data Quality — {domain.title()}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Metrics generated from the comparison and evaluation "
        "results of real and synthetic datasets."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "Quality Score",
            metrics["quality"],
            "Measures how closely the synthetic "
            "data preserves the statistical "
            "properties of the real dataset."
        )

    with col2:

        metric_card(
            "Privacy Score",
            metrics["privacy"],
            "Project-level indicator based on "
            "duplicate record detection between "
            "real and synthetic data."
        )

    with col3:

        metric_card(
            "Correlation Similarity",
            metrics["correlation"],
            "Measures how closely relationships "
            "between numerical variables are "
            "preserved in synthetic data."
        )

    with col4:

        metric_card(
            "Utility Retention",
            metrics["utility"],
            "Indicates how much of the usefulness "
            "of the original data is retained "
            "for analytical or ML tasks."
        )
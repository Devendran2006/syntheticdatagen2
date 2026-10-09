import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/banking/banking_combined.csv"

# Original smaller synthetic dataset
SYNTHETIC_PATH = "data/banking/synthetic_banking.csv"

# Newly generated large synthetic dataset
LARGE_SYNTHETIC_PATH = "data/banking/synthetic_banking_large.csv"

# Original + synthetic expanded dataset
EXPANDED_PATH = "data/banking/banking_expanded.csv"

COMPARISON_PATH = "data/banking/comparison"


# =========================================================
# DESIGN SYSTEM
# =========================================================
# Same consistent professional palette used everywhere across the
# suite — KPI cards, section headers, chart colorways and badges.

COLORS = {
    "primary": "#0B5FA5",      # deep clinical blue
    "primary_dark": "#08406F",
    "accent": "#12A594",       # teal accent
    "accent_soft": "#E4F5F3",
    "warning": "#D97706",
    "danger": "#DC2626",
    "success": "#059669",
    "text": "#1F2A44",
    "text_muted": "#5B6B85",
    "border": "#E4E9F2",
    "surface": "#FFFFFF",
    "surface_alt": "#F6F8FC",
}

# Consistent series colors so "Real" vs "Synthetic" always render the
# same way across every chart in the dashboard.
SERIES_COLOR_MAP = {
    "Real": COLORS["primary"],
    "Synthetic": COLORS["accent"],
}

CATEGORICAL_SEQUENCE = [
    COLORS["primary"],
    COLORS["accent"],
    "#6D5DD3",
    "#E08E45",
    "#3AA6A6",
    "#B0559A",
    "#4C8DBF",
    "#8C9EB2",
]


def inject_custom_css():
    """Injects only *safe* styling — never overrides text color, so it
    always stays correct in both light and dark Streamlit themes."""

    st.markdown(
        f"""
        <style>
        /* ---- Page header banner (fixed white text on a colored
             gradient — safe in any theme because the background
             itself is always dark enough) ---- */
        .hc-header {{
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            background: linear-gradient(135deg, {COLORS["primary_dark"]} 0%, {COLORS["primary"]} 55%, {COLORS["accent"]} 100%);
            margin-bottom: 1.1rem;
            box-shadow: 0 6px 18px rgba(11, 95, 165, 0.18);
        }}
        .hc-header h1 {{
            margin: 0;
            font-size: 1.55rem;
            font-weight: 700;
            color: #FFFFFF !important;
        }}
        .hc-header p {{
            margin: 0.25rem 0 0 0;
            font-size: 0.92rem;
            color: rgba(255,255,255,0.88) !important;
        }}

        /* ---- Badge (fixed pastel bg + fixed dark text — legible on
             both themes since the badge carries its own background) ---- */
        .hc-badge {{
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .hc-badge-success {{ background: #E4F7EE; color: {COLORS["success"]} !important; }}
        .hc-badge-warning {{ background: #FEF3E2; color: {COLORS["warning"]} !important; }}
        .hc-badge-danger  {{ background: #FDEAEA; color: {COLORS["danger"]} !important; }}

        /* ---- Dataframe / expander polish (border only, no text color) ---- */
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            overflow: hidden;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
        }}

        /* ---- KPI card spacing (border container already themed
             correctly by Streamlit — just add breathing room) ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 0.4rem 0.2rem;
        }}

        div[data-testid="column"] {{
            padding: 0 0.35rem;
        }}

        /* ---- Download buttons (brand color, explicit white text) ---- */
        div[data-testid="stDownloadButton"] button {{
            background: {COLORS["primary"]} !important;
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        div[data-testid="stDownloadButton"] button:hover {{
            background: {COLORS["primary_dark"]} !important;
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon, title, caption=None):
    """Section header using Streamlit's own heading — always theme-correct."""

    st.markdown(f"#### {icon} {title}")

    if caption:
        st.caption(caption)


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hc-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text, kind="success"):
    return f'<span class="hc-badge hc-badge-{kind}">{text}</span>'


# =========================================================
# CHART THEME
# =========================================================

def apply_banking_chart_theme(fig):

    theme_type = get_theme_type()

    if theme_type == "dark":
        text_color = "#E7ECF7"
        line_color = "rgba(255, 255, 255, 0.25)"
    else:
        text_color = COLORS["text"]
        line_color = COLORS["border"]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=text_color,
            family="Segoe UI, Helvetica Neue, Arial, sans-serif",
            size=12,
        ),
        title=dict(
            font=dict(
                color=text_color,
                size=17,
                family="Segoe UI, Helvetica Neue, Arial, sans-serif",
            ),
            x=0.01,
            xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=70
        ),
        colorway=CATEGORICAL_SEQUENCE,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Segoe UI, Helvetica Neue, Arial, sans-serif",
        ),
        bargap=0.25,
        height=380,
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=line_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color),
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor=line_color,
        zerolinewidth=1,
        showline=True,
        linecolor=line_color,
        tickfont=dict(color=text_color),
        title_font=dict(color=text_color),
    )

    return fig


def get_theme_type():
    """Detects whether Streamlit is currently rendering in light or dark
    mode, so chart text/axis colors can be switched to stay visible."""

    try:
        return st.context.theme.type
    except Exception:
        return "light"


def apply_series_colors(fig, df, color_col="Dataset"):
    """Force 'Real' / 'Synthetic' series to consistent brand colors, when present."""

    if color_col in df.columns:
        for trace in fig.data:
            name = getattr(trace, "name", None)
            if name in SERIES_COLOR_MAP:
                trace.marker.color = SERIES_COLOR_MAP[name]

    return fig


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    real_df = pd.read_csv(REAL_PATH)

    # Prefer the newly generated large synthetic dataset
    if os.path.exists(LARGE_SYNTHETIC_PATH):
        synthetic_df = pd.read_csv(LARGE_SYNTHETIC_PATH)
    else:
        synthetic_df = pd.read_csv(SYNTHETIC_PATH)

    return real_df, synthetic_df


# =========================================================
# LOAD EXPANDED DATA
# =========================================================

@st.cache_data
def load_expanded_data():

    if not os.path.exists(EXPANDED_PATH):
        return None

    return pd.read_csv(EXPANDED_PATH)


# =========================================================
# SAFE COMPARISON FILE LOADER
# =========================================================

def load_comparison_file(filename):

    path = os.path.join(
        COMPARISON_PATH,
        filename
    )

    if os.path.exists(path):

        try:
            return pd.read_csv(path)

        except Exception:
            return None

    return None


# =========================================================
# KPI HELPER
# =========================================================

def show_kpi(
    label,
    value,
    description
):
    """Renders a KPI as a native Streamlit bordered card — always
    renders correctly and stays theme-safe in light and dark mode."""

    with st.container(border=True):

        st.metric(label, value)

        st.caption(description)


# =========================================================
# BANKING INTELLIGENCE
# =========================================================

def show_banking_intelligence(df):

    section_title(
        "🏦",
        "Banking Intelligence",
        "Customer, financial, loan, transaction and fraud intelligence."
    )

    # =====================================================
    # KPI
    # =====================================================

    total_customers = len(df)

    avg_balance = (
        df["avg_balance"].mean()
        if "avg_balance" in df.columns
        else 0
    )

    avg_credit = (
        df["avg_credit_score"].mean()
        if "avg_credit_score" in df.columns
        else 0
    )

    if "churn_risk" in df.columns:

        high_churn = (
            df["churn_risk"]
            .astype(str)
            .str.lower()
            .eq("high")
            .sum()
        )

    else:

        high_churn = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        show_kpi(
            "Total Customers",
            f"{total_customers:,}",
            "Number of customer records in the banking dataset."
        )

    with c2:

        show_kpi(
            "Average Balance",
            f"{avg_balance:,.0f}",
            "Average customer account balance across the dataset."
        )

    with c3:

        show_kpi(
            "Average Credit Score",
            f"{avg_credit:.0f}",
            "Average credit score across all customers."
        )

    with c4:

        show_kpi(
            "High Churn Risk",
            f"{high_churn:,}",
            "Customers classified with high churn risk."
        )

    st.divider()

    # =====================================================
    # CUSTOMER ANALYTICS
    # =====================================================

    section_title("👤", "Customer Analytics")

    c1, c2 = st.columns(2)

    with c1:

        if "segment" in df.columns:

            segment = (
                df["segment"]
                .value_counts()
                .reset_index()
            )

            segment.columns = [
                "Segment",
                "Customers"
            ]

            fig = px.bar(
                segment,
                x="Segment",
                y="Customers",
                title="Customer Segment Distribution",
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        if "churn_risk" in df.columns:

            churn = (
                df["churn_risk"]
                .value_counts()
                .reset_index()
            )

            churn.columns = [
                "Risk",
                "Customers"
            ]

            fig = px.bar(
                churn,
                x="Risk",
                y="Customers",
                title="Churn Risk Distribution",
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # FINANCIAL ANALYTICS
    # =====================================================

    section_title("💳", "Financial Analytics")

    c1, c2 = st.columns(2)

    with c1:

        if (
            "avg_income" in df.columns
            and "avg_balance" in df.columns
        ):

            fig = px.scatter(
                df,
                x="avg_income",
                y="avg_balance",
                title="Income vs Average Balance",
                color_discrete_sequence=[COLORS["primary"]],
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        if (
            "avg_credit_score" in df.columns
            and "avg_dti_ratio" in df.columns
        ):

            fig = px.scatter(
                df,
                x="avg_credit_score",
                y="avg_dti_ratio",
                title="Credit Score vs DTI Ratio",
                color_discrete_sequence=[COLORS["accent"]],
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # LOAN ANALYTICS
    # =====================================================

    section_title("🏷️", "Loan & Credit Analytics")

    c1, c2 = st.columns(2)

    with c1:

        if "total_loan_amount" in df.columns:

            fig = px.histogram(
                df,
                x="total_loan_amount",
                nbins=20,
                title="Total Loan Amount Distribution",
                color_discrete_sequence=[COLORS["primary"]],
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        if (
            "avg_loan_amount" in df.columns
            and "avg_credit_score" in df.columns
        ):

            fig = px.scatter(
                df,
                x="avg_loan_amount",
                y="avg_credit_score",
                title="Loan Amount vs Credit Score",
                color_discrete_sequence=[COLORS["accent"]],
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # TRANSACTION ANALYTICS
    # =====================================================

    section_title("🔐", "Transaction & Fraud Analytics")

    c1, c2 = st.columns(2)

    with c1:

        if (
            "transaction_count" in df.columns
            and "total_transaction_amount" in df.columns
        ):

            fig = px.scatter(
                df,
                x="transaction_count",
                y="total_transaction_amount",
                title="Transactions vs Transaction Amount",
                color_discrete_sequence=[COLORS["primary"]],
            )

            fig = apply_banking_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with c2:

        fraud = (
            df["fraud_count"].sum()
            if "fraud_count" in df.columns
            else 0
        )

        devices = (
            df["new_device_count"].sum()
            if "new_device_count" in df.columns
            else 0
        )

        fraud_data = pd.DataFrame(
            {
                "Metric": [
                    "Fraud Events",
                    "New Devices"
                ],
                "Count": [
                    fraud,
                    devices
                ]
            }
        )

        fig = px.bar(
            fraud_data,
            x="Metric",
            y="Count",
            title="Fraud & New Device Activity",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# SYNTHETIC DATA PAGE
# =========================================================

def show_synthetic_data(
    real_df,
    synthetic_df
):

    section_title(
        "🧬",
        "Synthetic Banking Data",
        "CTGAN-generated banking records and the final expanded dataset."
    )

    # =====================================================
    # LOAD FINAL EXPANDED DATASET
    # =====================================================

    expanded_df = load_expanded_data()

    if expanded_df is None:

        st.error(
            "Expanded banking dataset not found. "
            "Please generate banking_expanded.csv first."
        )

        return

    # =====================================================
    # DATASET METRICS
    # =====================================================

    original_records = len(
        real_df
    )

    synthetic_records = len(
        synthetic_df
    )

    expanded_records = len(
        expanded_df
    )

    if original_records > 0:

        expansion_factor = (
            expanded_records
            / original_records
        )

    else:

        expansion_factor = 0

    synthetic_percentage = 0

    if expanded_records > 0:

        synthetic_percentage = (
            synthetic_records
            / expanded_records
        ) * 100

    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        show_kpi(
            "Original Records",
            f"{original_records:,}",
            "Records available before synthetic generation."
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "New records generated using CTGAN."
        )

    with c3:

        show_kpi(
            "Expanded Records",
            f"{expanded_records:,}",
            "Original records plus generated synthetic records."
        )

    with c4:

        show_kpi(
            "Expansion Factor",
            f"{expansion_factor:.1f}×",
            "Final dataset size compared with the original dataset."
        )

    with c5:

        show_kpi(
            "Dataset Columns",
            f"{len(expanded_df.columns):,}",
            "Features available in the final expanded dataset."
        )

    st.divider()

    # =====================================================
    # GENERATION SUMMARY
    # =====================================================

    section_title("🧪", "Synthetic Data Generation")

    with st.container(border=True):

        c1, c2 = st.columns([4, 1])

        with c1:

            st.caption(
                f"Original {original_records:,} records were used as the "
                f"basis for CTGAN synthetic data generation, producing "
                f"{synthetic_records:,} new records and a final expanded "
                f"dataset of {expanded_records:,} records."
            )

        with c2:

            st.markdown(
                badge(f"{expanded_records:,} total", "success"),
                unsafe_allow_html=True
            )

    st.divider()

    # =====================================================
    # SYNTHETIC DATA ANALYTICS
    # =====================================================

    section_title(
        "📈",
        "Synthetic Banking Analytics",
        f"Analytics are based on the final {expanded_records:,}-record expanded dataset."
    )

    # =====================================================
    # AGE DISTRIBUTION
    # =====================================================

    age_column = None

    for column in [
        "age",
        "Age",
        "customer_age",
        "Customer_Age"
    ]:

        if column in expanded_df.columns:

            age_column = column
            break

    if age_column:

        chart_df = expanded_df[
            [age_column]
        ].dropna()

        fig = px.histogram(
            chart_df,
            x=age_column,
            nbins=20,
            title="Customer Age Distribution",
            labels={
                age_column: "Customer Age"
            },
            color_discrete_sequence=[COLORS["primary"]],
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CREDIT SCORE
    # =====================================================

    credit_column = None

    for column in [
        "avg_credit_score",
        "credit_score",
        "Credit_Score",
        "Credit Score"
    ]:

        if column in expanded_df.columns:

            credit_column = column
            break

    if credit_column:

        chart_df = expanded_df[
            [credit_column]
        ].dropna()

        fig = px.histogram(
            chart_df,
            x=credit_column,
            nbins=20,
            title="Credit Score Distribution",
            labels={
                credit_column: "Credit Score"
            },
            color_discrete_sequence=[COLORS["accent"]],
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL SYNTHETIC ANALYTICS
    # =====================================================

    financial_columns = [
        "avg_balance",
        "avg_income",
        "total_loan_amount",
        "avg_loan_amount",
        "total_transaction_amount"
    ]

    available_financial = [
        column
        for column in financial_columns
        if column in expanded_df.columns
    ]

    if available_financial:

        section_title("💰", "Synthetic Financial Metrics")

        financial_values = []

        for column in available_financial:

            financial_values.append(
                {
                    "Metric": column.replace(
                        "_",
                        " "
                    ).title(),
                    "Average": expanded_df[
                        column
                    ].mean()
                }
            )

        financial_df = pd.DataFrame(
            financial_values
        )

        fig = px.bar(
            financial_df,
            x="Metric",
            y="Average",
            title="Average Financial Metrics",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CATEGORICAL DISTRIBUTIONS
    # =====================================================

    categorical_candidates = [
        "gender",
        "Gender",
        "segment",
        "customer_segment",
        "Customer_Segment",
        "churn_risk",
        "account_type",
        "Account_Type",
        "loan_type",
        "Loan_Type",
        "primary_channel",
        "employment_status",
        "Employment_Status"
    ]

    available_categories = []

    for column in categorical_candidates:

        if (
            column in expanded_df.columns
            and column not in available_categories
        ):

            available_categories.append(
                column
            )

    if available_categories:

        section_title("🗂️", "Banking Category Distribution")

        chart_columns = st.columns(2)

        for index, column in enumerate(
            available_categories[:6]
        ):

            counts = (
                expanded_df[column]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Category",
                "Count"
            ]

            fig = px.bar(
                counts,
                x="Category",
                y="Count",
                title=column.replace(
                    "_",
                    " "
                ).title(),
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_banking_chart_theme(fig)

            with chart_columns[
                index % 2
            ]:

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    st.divider()

    # =====================================================
    # EXPANDED DATASET PREVIEW
    # =====================================================

    section_title("🔎", "Expanded Banking Dataset Preview")

    with st.container(border=True):

        c1, c2 = st.columns([4, 1])

        with c1:

            st.caption(
                f"Preview of the currently loaded expanded dataset — "
                f"{expanded_records:,} records available."
            )

        with c2:

            st.markdown(
                badge(f"{expanded_records:,} records", "success"),
                unsafe_allow_html=True
            )

        with st.expander(
            "🔎 View sample rows"
        ):

            st.dataframe(
                expanded_df.head(20),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # =====================================================
    # DOWNLOAD BUTTONS
    # =====================================================

    section_title(
        "⬇️",
        "Download Banking Datasets",
        "Download either the newly generated synthetic records or the complete expanded dataset."
    )

    download_col1, download_col2 = st.columns(2)

    # -----------------------------------------------------
    # LARGE SYNTHETIC DOWNLOAD
    # -----------------------------------------------------

    with download_col1:

        synthetic_csv = (
            synthetic_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                f"⬇️ Download Synthetic Banking Data "
                f"({synthetic_records:,} records)"
            ),
            data=synthetic_csv,
            file_name="synthetic_banking_large.csv",
            mime="text/csv",
            use_container_width=True,
            key="banking_large_synthetic_download"
        )

    # -----------------------------------------------------
    # ORIGINAL + SYNTHETIC DOWNLOAD
    # -----------------------------------------------------

    with download_col2:

        expanded_csv = (
            expanded_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                f"⬇️ Download Original + Synthetic "
                f"({expanded_records:,} records)"
            ),
            data=expanded_csv,
            file_name="banking_expanded.csv",
            mime="text/csv",
            use_container_width=True,
            key="banking_original_synthetic_expanded_download"
        )

    st.caption(
        f"Combined dataset contains "
        f"{original_records:,} original + "
        f"{synthetic_records:,} synthetic records."
    )


# =========================================================
# REAL VS SYNTHETIC COMPARISON
# =========================================================

def show_comparison(
    real_df,
    synthetic_df
):

    section_title(
        "📊",
        "Real vs Synthetic Banking",
        "Direct comparison between the original banking dataset and the generated synthetic dataset."
    )

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    real_records = len(real_df)
    synthetic_records = len(synthetic_df)

    if real_records > 0:
        generation_ratio = (
            synthetic_records / real_records
        )
    else:
        generation_ratio = 0

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    numeric_columns = [
        "age",
        "tenure_years",
        "avg_balance",
        "monthly_logins",
        "products_held",
        "loan_count",
        "total_loan_amount",
        "avg_loan_amount",
        "avg_income",
        "avg_term_months",
        "avg_dti_ratio",
        "avg_credit_score",
        "avg_employment_years",
        "avg_utilisation",
        "total_missed_payments",
        "default_count",
        "transaction_count",
        "total_transaction_amount",
        "avg_transaction_amount",
        "fraud_count",
        "new_device_count"
    ]

    available = [
        col
        for col in numeric_columns
        if col in real_df.columns
        and col in synthetic_df.columns
    ]

    # =====================================================
    # DISTRIBUTION SCORE
    # =====================================================

    distribution_scores = []

    for column in available:

        real_values = pd.to_numeric(
            real_df[column],
            errors="coerce"
        ).dropna()

        synthetic_values = pd.to_numeric(
            synthetic_df[column],
            errors="coerce"
        ).dropna()

        if len(real_values) == 0:
            continue

        if len(synthetic_values) == 0:
            continue

        real_mean = real_values.mean()
        synthetic_mean = synthetic_values.mean()

        real_std = real_values.std()
        synthetic_std = synthetic_values.std()

        mean_similarity = 1 - (
            abs(
                real_mean
                - synthetic_mean
            )
            /
            (
                abs(real_mean)
                + 1e-8
            )
        )

        std_similarity = 1 - (
            abs(
                real_std
                - synthetic_std
            )
            /
            (
                abs(real_std)
                + 1e-8
            )
        )

        score = (
            max(0, mean_similarity) * 0.5
            +
            max(0, std_similarity) * 0.5
        )

        distribution_scores.append(
            score
        )

    if distribution_scores:

        distribution_score = (
            np.mean(
                distribution_scores
            )
            * 100
        )

    else:

        distribution_score = 0

    distribution_score = max(
        0,
        min(
            100,
            distribution_score
        )
    )

    # =====================================================
    # CORRELATION SCORE
    # =====================================================

    if len(available) >= 2:

        real_corr = (
            real_df[
                available
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .corr()
        )

        synthetic_corr = (
            synthetic_df[
                available
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .corr()
        )

        corr_difference = (
            abs(
                real_corr
                - synthetic_corr
            )
        )

        upper = np.triu(
            np.ones(
                corr_difference.shape
            ),
            k=1
        ).astype(bool)

        values = (
            corr_difference
            .where(upper)
            .stack()
        )

        if len(values) > 0:

            avg_difference = values.mean()

            correlation_score = (
                1
                - avg_difference
            ) * 100

        else:

            correlation_score = 0

    else:

        correlation_score = 0

    correlation_score = max(
        0,
        min(
            100,
            correlation_score
        )
    )

    # =====================================================
    # OVERALL SIMILARITY
    # =====================================================

    overall_score = (
        distribution_score * 0.5
        +
        correlation_score * 0.5
    )

    overall_score = max(
        0,
        min(
            100,
            overall_score
        )
    )

    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        show_kpi(
            "Real Records",
            f"{real_records:,}",
            "Records in the original banking dataset."
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "Records in the currently generated synthetic dataset."
        )

    with c3:

        show_kpi(
            "Generation Ratio",
            f"{generation_ratio:.1f}×",
            "Synthetic records generated relative to the original records."
        )

    with c4:

        show_kpi(
            "Distribution Score",
            f"{distribution_score:.1f}%",
            "Measures how closely synthetic numerical distributions match the real dataset."
        )

    with c5:

        show_kpi(
            "Overall Similarity",
            f"{overall_score:.1f}%",
            "Combined similarity score based on distribution and correlation patterns."
        )

    st.divider()

    # =====================================================
    # QUALITY SCORE DETAILS
    # =====================================================

    section_title("✅", "Synthetic Data Quality")

    if overall_score >= 80:
        quality_badge = badge("Excellent match", "success")
    elif overall_score >= 60:
        quality_badge = badge("Reasonable match", "warning")
    else:
        quality_badge = badge("Needs review", "danger")

    st.markdown(quality_badge, unsafe_allow_html=True)
    st.write("")

    q1, q2, q3 = st.columns(3)

    with q1:

        show_kpi(
            "Distribution Similarity",
            f"{distribution_score:.1f}%",
            "Similarity of numerical means and standard deviations between real and synthetic data."
        )

    with q2:

        show_kpi(
            "Correlation Similarity",
            f"{correlation_score:.1f}%",
            "Similarity of relationships between numerical banking variables."
        )

    with q3:

        show_kpi(
            "Common Numeric Features",
            f"{len(available):,}",
            "Number of numerical features available in both real and synthetic datasets."
        )

    st.divider()

    # =====================================================
    # MEAN COMPARISON TABLE
    # =====================================================

    comparison = pd.DataFrame(
        {
            "Metric": available,

            "Real": [
                pd.to_numeric(
                    real_df[col],
                    errors="coerce"
                ).mean()
                for col in available
            ],

            "Synthetic": [
                pd.to_numeric(
                    synthetic_df[col],
                    errors="coerce"
                ).mean()
                for col in available
            ]
        }
    )

    # =====================================================
    # KEY FINANCIAL METRICS
    # =====================================================

    section_title("🔑", "Key Financial Metrics")

    selected = [
        "avg_balance",
        "avg_income",
        "total_loan_amount",
        "avg_credit_score",
        "total_transaction_amount"
    ]

    selected = [
        col
        for col in selected
        if col in available
    ]

    chart_df = comparison[
        comparison["Metric"].isin(selected)
    ]

    if not chart_df.empty:

        chart_long = chart_df.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace("_", " ")
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            title="Key Financial Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CUSTOMER METRICS
    # =====================================================

    section_title("👤", "Customer Metrics")

    selected = [
        "age",
        "tenure_years",
        "monthly_logins",
        "products_held",
        "loan_count"
    ]

    selected = [
        col
        for col in selected
        if col in available
    ]

    chart_df = comparison[
        comparison["Metric"].isin(selected)
    ]

    if not chart_df.empty:

        chart_long = chart_df.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace("_", " ")
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            title="Customer Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig = apply_banking_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL DISTRIBUTION COMPARISON
    # =====================================================

    section_title("📉", "Financial Distribution Comparison")

    distribution_columns = [
        "avg_balance",
        "avg_income",
        "avg_credit_score"
    ]

    distribution_columns = [
        col
        for col in distribution_columns
        if col in real_df.columns
        and col in synthetic_df.columns
    ]

    if distribution_columns:

        c1, c2 = st.columns(2)

        for index, column in enumerate(
            distribution_columns
        ):

            real_values = pd.to_numeric(
                real_df[column],
                errors="coerce"
            ).dropna()

            synthetic_values = pd.to_numeric(
                synthetic_df[column],
                errors="coerce"
            ).dropna()

            distribution_df = pd.DataFrame(
                {
                    "Value": pd.concat(
                        [
                            real_values,
                            synthetic_values
                        ],
                        ignore_index=True
                    ),

                    "Dataset": (
                        ["Real"] * len(real_values)
                        +
                        ["Synthetic"] * len(synthetic_values)
                    )
                }
            )

            fig = px.histogram(
                distribution_df,
                x="Value",
                color="Dataset",
                barmode="overlay",
                nbins=25,
                opacity=0.65,
                title=column.replace(
                    "_",
                    " "
                ).title() + " Distribution",
                color_discrete_map=SERIES_COLOR_MAP,
            )

            fig = apply_banking_chart_theme(fig)

            with (
                c1
                if index % 2 == 0
                else c2
            ):

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # =====================================================
    # CATEGORICAL COMPARISON
    # =====================================================

    section_title("🗂️", "Categorical Distribution")

    categorical = [
        "segment",
        "churn_risk",
        "primary_channel"
    ]

    categorical = [
        col
        for col in categorical
        if col in real_df.columns
        and col in synthetic_df.columns
    ]

    if categorical:

        c1, c2 = st.columns(2)

        for index, column in enumerate(categorical):

            real_counts = (
                real_df[column]
                .astype(str)
                .value_counts(
                    normalize=True
                )
                .mul(100)
                .reset_index()
            )

            real_counts.columns = [
                "Category",
                "Percentage"
            ]

            real_counts["Dataset"] = "Real"

            synthetic_counts = (
                synthetic_df[column]
                .astype(str)
                .value_counts(
                    normalize=True
                )
                .mul(100)
                .reset_index()
            )

            synthetic_counts.columns = [
                "Category",
                "Percentage"
            ]

            synthetic_counts["Dataset"] = "Synthetic"

            combined = pd.concat(
                [
                    real_counts,
                    synthetic_counts
                ],
                ignore_index=True
            )

            fig = px.bar(
                combined,
                x="Category",
                y="Percentage",
                color="Dataset",
                barmode="group",
                title=column.replace(
                    "_",
                    " "
                ).title() +
                " — Real vs Synthetic",
                color_discrete_map=SERIES_COLOR_MAP,
            )

            fig.update_yaxes(
                title="Percentage (%)"
            )

            fig = apply_banking_chart_theme(fig)

            if index % 2 == 0:

                with c1:

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            else:

                with c2:

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

    # =====================================================
    # FULL NUMERIC COMPARISON
    # =====================================================

    with st.expander(
        "🔎 View Full Numeric Comparison"
    ):

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    section_title("📐", "Dataset Statistics")

    statistics = pd.DataFrame(
        {
            "Metric": [
                "Total Records",
                "Total Columns",
                "Numeric Columns",
                "Missing Values"
            ],

            "Real": [
                len(real_df),
                len(real_df.columns),
                len(real_df.select_dtypes(
                    include="number"
                ).columns),
                int(real_df.isna().sum().sum())
            ],

            "Synthetic": [
                len(synthetic_df),
                len(synthetic_df.columns),
                len(synthetic_df.select_dtypes(
                    include="number"
                ).columns),
                int(synthetic_df.isna().sum().sum())
            ]
        }
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # COMPARISON DOWNLOAD
    # =====================================================

    st.divider()

    comparison_csv = (
        comparison
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Banking Comparison",
        data=comparison_csv,
        file_name="banking_real_vs_synthetic.csv",
        mime="text/csv",
        use_container_width=True,
        key="banking_comparison_download"
    )


# =========================================================
# MAIN BANKING DASHBOARD
# =========================================================

def show_banking_dashboard():

    inject_custom_css()

    try:

        real_df, synthetic_df = (
            load_data()
        )

    except Exception as e:

        st.error(
            f"Unable to load banking datasets: {e}"
        )

        return

    # =====================================================
    # HEADER
    # =====================================================

    page_header(
        "🏦 Banking Intelligence",
        "Banking analytics, synthetic data generation results and real-vs-synthetic evaluation."
    )

    synthetic_source = (
        LARGE_SYNTHETIC_PATH
        if os.path.exists(LARGE_SYNTHETIC_PATH)
        else SYNTHETIC_PATH
    )

    st.caption(
        f"Current synthetic dataset: "
        f"{len(synthetic_df):,} records — source `{synthetic_source}`"
    )

    st.divider()

    # =====================================================
    # MAIN TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Banking Intelligence",
            "🧬 Synthetic Data",
            "📊 Real vs Synthetic"
        ]
    )

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        show_banking_intelligence(
            real_df
        )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        show_synthetic_data(
            real_df,
            synthetic_df
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        show_comparison(
            real_df,
            synthetic_df
        )
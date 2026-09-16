import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/healthcare_dataset.csv"

SYNTHETIC_LARGE_PATHS = [
    "data/synthetic_healthcare_large.csv",
    "data/healthcare/synthetic_healthcare_large.csv",
    "outputs/synthetic_healthcare_large.csv",
    "outputs/synthetic_healthcare_111000.csv",
]

OLD_SYNTHETIC_PATHS = [
    "data/synthetic_healthcare.csv",
    "outputs/synthetic_healthcare.csv",
]

HEALTHCARE_EXPANDED_PATHS = [
    "data/healthcare_expanded.csv",
    "data/healthcare/healthcare_expanded.csv",
    "outputs/healthcare_expanded.csv",
]

# =========================================================
# DESIGN SYSTEM
# =========================================================
# A single, consistent professional palette used everywhere:
# KPI cards, section headers, chart colorways and badges.

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
# PAGE STYLE (CHART THEME)
# =========================================================

def apply_healthcare_chart_theme(fig):

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
# DATASET PATH FINDER
# =========================================================

def find_existing_path(paths):

    for path in paths:

        if os.path.exists(path):
            return path

    return None


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    if not os.path.exists(REAL_PATH):

        raise FileNotFoundError(
            f"Healthcare real dataset not found: {REAL_PATH}"
        )

    real_df = pd.read_csv(
        REAL_PATH
    )

    # -----------------------------------------------------
    # Prefer NEW expanded synthetic dataset
    # -----------------------------------------------------

    synthetic_path = find_existing_path(
        SYNTHETIC_LARGE_PATHS
    )

    # -----------------------------------------------------
    # Fallback to old synthetic dataset
    # -----------------------------------------------------

    if synthetic_path is None:

        synthetic_path = find_existing_path(
            OLD_SYNTHETIC_PATHS
        )

    if synthetic_path is None:

        raise FileNotFoundError(
            "No healthcare synthetic dataset found."
        )

    synthetic_df = pd.read_csv(
        synthetic_path
    )

    return (
        real_df,
        synthetic_df,
        synthetic_path
    )


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
# HEALTHCARE ANALYTICS
# =========================================================

def healthcare_analytics(
    real_df,
    synthetic_df
):

    section_title(
        "📈",
        "Healthcare Analytics",
        "Healthcare patient analytics and synthetic dataset overview."
    )

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    section_title("📋", "Dataset Overview")

    real_records = len(
        real_df
    )

    synthetic_records = len(
        synthetic_df
    )

    expanded_records = (
        real_records
        + synthetic_records
    )

    synthetic_columns = len(
        synthetic_df.columns
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        show_kpi(
            "Original Records",
            f"{real_records:,}",
            "Number of records available in the original healthcare dataset."
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "Number of healthcare records generated using synthetic data generation."
        )

    with c3:

        show_kpi(
            "Expanded Records",
            f"{expanded_records:,}",
            "Total records after combining original and synthetic healthcare data."
        )

    with c4:

        show_kpi(
            "Synthetic Features",
            f"{synthetic_columns:,}",
            "Number of features available in the generated synthetic healthcare dataset."
        )

    st.divider()

    # =====================================================
    # BASIC PATIENT ANALYTICS
    # =====================================================

    section_title("🩺", "Patient Analytics")

    c1, c2 = st.columns(2)

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    if "Age" in real_df.columns:

        with c1:

            fig = px.histogram(
                real_df,
                x="Age",
                nbins=20,
                title="Patient Age Distribution",
                color_discrete_sequence=[COLORS["primary"]],
            )

            fig = apply_healthcare_chart_theme(
                fig
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # -----------------------------------------------------
    # MEDICAL CONDITION
    # -----------------------------------------------------

    if "Medical Condition" in real_df.columns:

        with c2:

            condition = (
                real_df[
                    "Medical Condition"
                ]
                .value_counts()
                .reset_index()
            )

            condition.columns = [
                "Medical Condition",
                "Patients"
            ]

            fig = px.bar(
                condition,
                x="Medical Condition",
                y="Patients",
                title="Medical Condition Distribution",
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_healthcare_chart_theme(
                fig
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # BILLING ANALYTICS
    # =====================================================

    if "Billing Amount" in real_df.columns:

        section_title("💳", "Financial Healthcare Analytics")

        c1, c2 = st.columns(2)

        with c1:

            avg_bill = real_df[
                "Billing Amount"
            ].mean()

            show_kpi(
                "Average Billing Amount",
                f"{avg_bill:,.2f}",
                "Average healthcare billing amount across all original patient records."
            )

        with c2:

            total_bill = real_df[
                "Billing Amount"
            ].sum()

            show_kpi(
                "Total Billing Amount",
                f"{total_bill:,.2f}",
                "Total billing amount recorded across the original healthcare dataset."
            )

    # =====================================================
    # ADMISSION ANALYTICS
    # =====================================================

    if "Admission Type" in real_df.columns:

        admission = (
            real_df[
                "Admission Type"
            ]
            .value_counts()
            .reset_index()
        )

        admission.columns = [
            "Admission Type",
            "Patients"
        ]

        fig = px.bar(
            admission,
            x="Admission Type",
            y="Patients",
            title="Admission Type Distribution",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig = apply_healthcare_chart_theme(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # SYNTHETIC DATA PREVIEW
    # =====================================================

    section_title(
        "🧬",
        "Generated Synthetic Dataset"
    )

    with st.container(border=True):

        c1, c2 = st.columns([4, 1])

        with c1:

            st.caption(
                f"Preview of the currently loaded synthetic dataset — "
                f"{len(synthetic_df):,} records available."
            )

        with c2:

            st.markdown(
                badge(f"{len(synthetic_df):,} records", "success"),
                unsafe_allow_html=True
            )

        with st.expander(
            "🔎 View sample rows"
        ):

            st.dataframe(
                synthetic_df.head(20),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # =====================================================
    # DOWNLOAD SECTION
    # =====================================================

    section_title("⬇️", "Download Healthcare Datasets")

    synthetic_csv = (
        synthetic_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label=(
            f"⬇️ Download Synthetic Healthcare Data "
            f"({len(synthetic_df):,} records)"
        ),
        data=synthetic_csv,
        file_name="synthetic_healthcare_large.csv",
        mime="text/csv",
        use_container_width=True,
        key="healthcare_synthetic_large_download"
    )

    # -----------------------------------------------------
    # ORIGINAL + SYNTHETIC
    # -----------------------------------------------------

    combined_df = pd.concat(
        [
            real_df,
            synthetic_df
        ],
        ignore_index=True
    )

    combined_csv = (
        combined_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label=(
            f"⬇️ Download Original + Synthetic "
            f"({len(combined_df):,} records)"
        ),
        data=combined_csv,
        file_name="healthcare_original_plus_synthetic.csv",
        mime="text/csv",
        use_container_width=True,
        key="healthcare_original_synthetic_download"
    )

    st.caption(
        f"Combined dataset contains "
        f"{len(real_df):,} original + "
        f"{len(synthetic_df):,} synthetic records."
    )


# =========================================================
# REAL VS SYNTHETIC COMPARISON
# =========================================================

def healthcare_comparison(
    real_df,
    synthetic_df
):

    section_title(
        "📊",
        "Real vs Synthetic Healthcare",
        "Direct comparison between the original healthcare dataset and the generated synthetic dataset."
    )

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    real_records = len(
        real_df
    )

    synthetic_records = len(
        synthetic_df
    )

    total_records = (
        real_records
        + synthetic_records
    )

    if real_records > 0:

        generation_ratio = (
            synthetic_records
            / real_records
        )

    else:

        generation_ratio = 0

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    numeric_columns = real_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    common_numeric = [
        column
        for column in numeric_columns
        if column in synthetic_df.columns
    ]

    # =====================================================
    # DISTRIBUTION SCORE
    # =====================================================

    distribution_scores = []

    for column in common_numeric:

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

    if len(common_numeric) >= 2:

        real_corr = (
            real_df[
                common_numeric
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .corr()
        )

        synthetic_corr = (
            synthetic_df[
                common_numeric
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
            "Number of records available in the original healthcare dataset."
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "Number of records generated by the synthetic data generation process."
        )

    with c3:

        show_kpi(
            "Generation Ratio",
            f"{generation_ratio:.1f}×",
            "Synthetic records generated relative to the original healthcare records."
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
            "Similarity of relationships between numerical healthcare variables."
        )

    with q3:

        show_kpi(
            "Common Numeric Features",
            f"{len(common_numeric):,}",
            "Number of numerical features available in both real and synthetic datasets."
        )

    st.divider()

    # =====================================================
    # KEY HEALTHCARE METRICS
    # =====================================================

    section_title("🔑", "Key Healthcare Metrics")

    comparison = pd.DataFrame(
        {
            "Metric": common_numeric,

            "Real": [
                pd.to_numeric(
                    real_df[column],
                    errors="coerce"
                ).mean()
                for column in common_numeric
            ],

            "Synthetic": [
                pd.to_numeric(
                    synthetic_df[column],
                    errors="coerce"
                ).mean()
                for column in common_numeric
            ]
        }
    )

    selected = [
        "Age",
        "Billing Amount",
        "Room Number",
        "Length_of_Stay"
    ]

    selected = [
        column
        for column in selected
        if column in common_numeric
    ]

    if selected:

        chart_df = comparison[
            comparison["Metric"].isin(
                selected
            )
        ]

        chart_long = chart_df.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            title="Key Healthcare Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig = apply_healthcare_chart_theme(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # PATIENT METRICS
    # =====================================================

    section_title("👤", "Patient Metrics")

    selected_patient = [
        "Age",
        "Room Number",
        "Length_of_Stay"
    ]

    selected_patient = [
        column
        for column in selected_patient
        if column in common_numeric
    ]

    if selected_patient:

        chart_df = comparison[
            comparison["Metric"].isin(
                selected_patient
            )
        ]

        chart_long = chart_df.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            title="Patient Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig = apply_healthcare_chart_theme(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL METRICS
    # =====================================================

    section_title("💰", "Financial Healthcare Metrics")

    financial_metrics = [
        "Billing Amount"
    ]

    available_financial = [
        column
        for column in financial_metrics
        if column in common_numeric
    ]

    if available_financial:

        chart_df = comparison[
            comparison["Metric"].isin(
                available_financial
            )
        ]

        chart_long = chart_df.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            title="Healthcare Billing — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig = apply_healthcare_chart_theme(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # DISTRIBUTION COMPARISON
    # =====================================================

    section_title("📉", "Numerical Distribution Comparison")

    distribution_columns = [
        "Age",
        "Billing Amount",
        "Room Number",
        "Length_of_Stay"
    ]

    distribution_columns = [
        column
        for column in distribution_columns
        if column in real_df.columns
        and column in synthetic_df.columns
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
                title=(
                    column.replace(
                        "_",
                        " "
                    )
                    +
                    " Distribution"
                ),
                color_discrete_map=SERIES_COLOR_MAP,
            )

            fig = apply_healthcare_chart_theme(
                fig
            )

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

    categorical_columns = [
        column
        for column in real_df.columns
        if column in synthetic_df.columns
        and column not in common_numeric
    ]

    display_categories = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results"
    ]

    display_categories = [
        column
        for column in display_categories
        if column in categorical_columns
    ]

    if display_categories:

        c1, c2 = st.columns(2)

        for index, column in enumerate(
            display_categories
        ):

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
                column,
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
                column,
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
                x=column,
                y="Percentage",
                color="Dataset",
                barmode="group",
                title=(
                    column.replace(
                        "_",
                        " "
                    )
                    +
                    " — Real vs Synthetic"
                ),
                color_discrete_map=SERIES_COLOR_MAP,
            )

            fig.update_yaxes(
                title="Percentage (%)"
            )

            fig = apply_healthcare_chart_theme(
                fig
            )

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
                "Categorical Columns",
                "Missing Values"
            ],

            "Real": [
                len(real_df),

                len(
                    real_df.columns
                ),

                len(
                    real_df.select_dtypes(
                        include="number"
                    ).columns
                ),

                len(
                    real_df.select_dtypes(
                        exclude="number"
                    ).columns
                ),

                int(
                    real_df
                    .isna()
                    .sum()
                    .sum()
                )
            ],

            "Synthetic": [
                len(synthetic_df),

                len(
                    synthetic_df.columns
                ),

                len(
                    synthetic_df.select_dtypes(
                        include="number"
                    ).columns
                ),

                len(
                    synthetic_df.select_dtypes(
                        exclude="number"
                    ).columns
                ),

                int(
                    synthetic_df
                    .isna()
                    .sum()
                    .sum()
                )
            ]
        }
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.divider()

    comparison_download = (
        comparison
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Healthcare Comparison",
        data=comparison_download,
        file_name="healthcare_real_vs_synthetic.csv",
        mime="text/csv",
        use_container_width=True,
        key="healthcare_comparison_download"
    )


# =========================================================
# MAIN HEALTHCARE DASHBOARD
# =========================================================

def show_healthcare_dashboard():

    inject_custom_css()

    try:

        real_df, synthetic_df, synthetic_path = (
            load_data()
        )

    except Exception as e:

        st.error(
            f"Unable to load healthcare datasets: {e}"
        )

        return

    # =====================================================
    # HEADER
    # =====================================================

    page_header(
        "🏥 Healthcare Intelligence",
        "Healthcare analytics, synthetic data evaluation and clinical intelligence."
    )

    st.caption(
        f"Current synthetic dataset: "
        f"{len(synthetic_df):,} records"
    )

    st.divider()

    # =====================================================
    # MAIN TABS
    # =====================================================

    tab1, tab2 = st.tabs(
        [
            "📈 Healthcare Analytics",
            "📊 Real vs Synthetic"
        ]
    )

    # =====================================================
    # ANALYTICS
    # =====================================================

    with tab1:

        healthcare_analytics(
            real_df,
            synthetic_df
        )

    # =====================================================
    # COMPARISON
    # =====================================================

    with tab2:

        healthcare_comparison(
            real_df,
            synthetic_df
        )

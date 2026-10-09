import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/enterprise/enterprise_combined.csv"

# NEW LARGE SYNTHETIC DATASET
SYNTHETIC_PATH = "data/enterprise/synthetic_enterprise_large.csv"

# ORIGINAL + SYNTHETIC EXPANDED DATASET
EXPANDED_PATH = "data/enterprise/enterprise_expanded.csv"


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

def apply_enterprise_chart_theme(fig):

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

    real_df = pd.read_csv(
        REAL_PATH
    )

    synthetic_df = pd.read_csv(
        SYNTHETIC_PATH
    )

    # Expanded dataset is optional
    if os.path.exists(EXPANDED_PATH):

        expanded_df = pd.read_csv(
            EXPANDED_PATH
        )

    else:

        expanded_df = pd.concat(
            [
                real_df,
                synthetic_df
            ],
            ignore_index=True
        )

    return real_df, synthetic_df, expanded_df


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
# ENTERPRISE INTELLIGENCE
# =========================================================

def show_enterprise_intelligence(df):

    section_title(
        "🏢",
        "Enterprise Intelligence",
        "Workforce, project, financial and asset intelligence."
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    # -----------------------------------------------------
    # EMPLOYEES
    # -----------------------------------------------------

    with c1:

        show_kpi(
            "Employees",
            f"{len(df):,}",
            "Employee records in the enterprise dataset."
        )

    # -----------------------------------------------------
    # PROJECTS
    # -----------------------------------------------------

    with c2:

        total_projects = 0

        if "project_count" in df.columns:

            total_projects = df[
                "project_count"
            ].sum()

        show_kpi(
            "Projects",
            f"{total_projects:,.0f}",
            "Total projects represented across departments."
        )

    # -----------------------------------------------------
    # ASSETS
    # -----------------------------------------------------

    with c3:

        if (
            "total_assets" in df.columns
            and len(df) > 0
        ):

            total_assets = df[
                "total_assets"
            ].iloc[0]

        else:

            total_assets = 0

        show_kpi(
            "Assets",
            f"{total_assets:,.0f}",
            "Enterprise assets represented in the dataset."
        )

    # -----------------------------------------------------
    # OVERRUN RATE
    # -----------------------------------------------------

    with c4:

        if (
            "status_on_track" in df.columns
            and "status_overrun" in df.columns
        ):

            on_track = df[
                "status_on_track"
            ].sum()

            overrun = df[
                "status_overrun"
            ].sum()

            total_project_status = (
                on_track + overrun
            )

            if total_project_status > 0:

                overrun_rate = (
                    overrun
                    / total_project_status
                ) * 100

            else:

                overrun_rate = 0

        else:

            overrun_rate = 0

        show_kpi(
            "Project Overrun Rate",
            f"{overrun_rate:.1f}%",
            "Share of projects classified as overrun."
        )

    st.divider()

    # =====================================================
    # WORKFORCE INTELLIGENCE
    # =====================================================

    section_title("👥", "Workforce Intelligence")

    c1, c2 = st.columns(2)

    # =====================================================
    # EMPLOYEES BY DEPARTMENT
    # =====================================================

    with c1:

        if "department" in df.columns:

            department = (
                df[
                    "department"
                ]
                .value_counts()
                .reset_index()
            )

            department.columns = [
                "Department",
                "Employees"
            ]

            fig = px.bar(
                department,
                x="Department",
                y="Employees",
                text="Employees",
                title="Employees by Department",
                color_discrete_sequence=[COLORS["primary"]],
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = apply_enterprise_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # ATTRITION BY DEPARTMENT
    # =====================================================

    with c2:

        if (
            "department" in df.columns
            and "attrition" in df.columns
        ):

            attrition = (
                df.groupby(
                    "department"
                )["attrition"]
                .sum()
                .reset_index()
            )

            attrition.columns = [
                "Department",
                "Attrition"
            ]

            fig = px.bar(
                attrition,
                x="Department",
                y="Attrition",
                text="Attrition",
                title="Attrition by Department",
                color_discrete_sequence=[COLORS["danger"]],
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = apply_enterprise_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # PROJECT INTELLIGENCE
    # =====================================================

    section_title("📁", "Project Intelligence")

    c1, c2 = st.columns(2)

    # =====================================================
    # PLANNED VS ACTUAL
    # =====================================================

    with c1:

        required = [
            "department",
            "avg_planned_days",
            "avg_actual_days"
        ]

        if all(
            column in df.columns
            for column in required
        ):

            project_df = df[
                required
            ].copy()

            project_long = project_df.melt(
                id_vars="department",
                var_name="Metric",
                value_name="Days"
            )

            project_long["Metric"] = (
                project_long["Metric"]
                .replace(
                    {
                        "avg_planned_days": "Planned",
                        "avg_actual_days": "Actual"
                    }
                )
            )

            fig = px.bar(
                project_long,
                x="department",
                y="Days",
                color="Metric",
                barmode="group",
                title="Planned vs Actual Project Days",
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_enterprise_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # BUDGET VS SPEND
    # =====================================================

    with c2:

        required = [
            "department",
            "total_budget",
            "total_spend"
        ]

        if all(
            column in df.columns
            for column in required
        ):

            budget_df = df[
                required
            ].copy()

            budget_long = budget_df.melt(
                id_vars="department",
                var_name="Metric",
                value_name="Amount"
            )

            budget_long["Metric"] = (
                budget_long["Metric"]
                .replace(
                    {
                        "total_budget": "Budget",
                        "total_spend": "Spend"
                    }
                )
            )

            fig = px.bar(
                budget_long,
                x="department",
                y="Amount",
                color="Metric",
                barmode="group",
                title="Budget vs Spend",
                color_discrete_sequence=CATEGORICAL_SEQUENCE,
            )

            fig = apply_enterprise_chart_theme(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # PROJECT STATUS
    # =====================================================

    section_title("🚦", "Project Status")

    if (
        "status_on_track" in df.columns
        and "status_overrun" in df.columns
    ):

        status_df = pd.DataFrame(
            {
                "Status": [
                    "On Track",
                    "Overrun"
                ],

                "Projects": [
                    df[
                        "status_on_track"
                    ].sum(),

                    df[
                        "status_overrun"
                    ].sum()
                ]
            }
        )

        fig = px.bar(
            status_df,
            x="Status",
            y="Projects",
            text="Projects",
            title="Project Status Distribution",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # ASSET RISK
    # =====================================================

    section_title("⚠️", "Asset Risk Intelligence")

    required = [
        "asset_risk_low_count",
        "asset_risk_medium_count",
        "asset_risk_high_count"
    ]

    if all(
        column in df.columns
        for column in required
    ) and len(df) > 0:

        asset_df = pd.DataFrame(
            {
                "Risk": [
                    "Low",
                    "Medium",
                    "High"
                ],

                "Assets": [
                    df[
                        "asset_risk_low_count"
                    ].iloc[0],

                    df[
                        "asset_risk_medium_count"
                    ].iloc[0],

                    df[
                        "asset_risk_high_count"
                    ].iloc[0]
                ]
            }
        )

        fig = px.bar(
            asset_df,
            x="Risk",
            y="Assets",
            text="Assets",
            title="Asset Risk Distribution",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# SYNTHETIC DATA
# =========================================================

def show_synthetic_data(
    real_df,
    synthetic_df,
    expanded_df
):

    section_title(
        "🧬",
        "Synthetic Enterprise Data",
        "CTGAN-generated enterprise records and expanded dataset."
    )

    # =====================================================
    # DATASET METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    # -----------------------------------------------------
    # ORIGINAL
    # -----------------------------------------------------

    with c1:

        show_kpi(
            "Original Records",
            f"{len(real_df):,}",
            "Records in the original enterprise dataset."
        )

    # -----------------------------------------------------
    # SYNTHETIC
    # -----------------------------------------------------

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{len(synthetic_df):,}",
            "New records generated by CTGAN."
        )

    # -----------------------------------------------------
    # EXPANDED
    # -----------------------------------------------------

    with c3:

        show_kpi(
            "Expanded Dataset",
            f"{len(expanded_df):,}",
            "Original + synthetic enterprise records."
        )

    # -----------------------------------------------------
    # COLUMNS
    # -----------------------------------------------------

    with c4:

        show_kpi(
            "Dataset Columns",
            f"{len(synthetic_df.columns):,}",
            "Features available in generated data."
        )

    st.divider()

    # =====================================================
    # GENERATION SUMMARY
    # =====================================================

    section_title("🧪", "Synthetic Data Generation Summary")

    c1, c2, c3 = st.columns(3)

    with c1:

        show_kpi(
            "Generation Multiplier",
            "2×",
            "Synthetic records generated relative to original data."
        )

    with c2:

        increase = 0

        if len(real_df) > 0:

            increase = (
                len(synthetic_df)
                / len(real_df)
            ) * 100

        show_kpi(
            "Additional Data",
            f"{increase:.0f}%",
            "Synthetic records added beyond the original dataset."
        )

    with c3:

        show_kpi(
            "Total Available",
            f"{len(expanded_df):,}",
            "Complete expanded enterprise dataset."
        )

    st.divider()

    # =====================================================
    # SYNTHETIC DATA PREVIEW
    # =====================================================

    section_title("🔎", "Synthetic Dataset Preview")

    with st.container(border=True):

        c1, c2 = st.columns([4, 1])

        with c1:

            st.caption(
                f"Preview of the newly generated {len(synthetic_df):,} synthetic enterprise records."
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

    section_title(
        "⬇️",
        "Download Enterprise Datasets",
        "Choose whether you want only generated synthetic records or the complete expanded dataset."
    )

    c1, c2 = st.columns(2)

    # =====================================================
    # DOWNLOAD SYNTHETIC
    # =====================================================

    with c1:

        synthetic_csv = (
            synthetic_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                f"⬇️ Download Synthetic Records "
                f"({len(synthetic_df):,} records)"
            ),

            data=synthetic_csv,

            file_name=(
                "synthetic_enterprise_large.csv"
            ),

            mime="text/csv",

            use_container_width=True,

            key=(
                "enterprise_synthetic_large_download"
            )
        )

    # =====================================================
    # DOWNLOAD EXPANDED
    # =====================================================

    with c2:

        expanded_csv = (
            expanded_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=(
                f"⬇️ Download Original + Synthetic "
                f"({len(expanded_df):,} records)"
            ),

            data=expanded_csv,

            file_name=(
                "enterprise_expanded.csv"
            ),

            mime="text/csv",

            use_container_width=True,

            key=(
                "enterprise_expanded_download"
            )
        )

    st.caption(
        f"Combined dataset contains "
        f"{len(real_df):,} original + "
        f"{len(synthetic_df):,} synthetic records."
    )


# =========================================================
# REAL VS SYNTHETIC ENTERPRISE COMPARISON
# =========================================================

def show_comparison(
    real_df,
    synthetic_df
):

    section_title(
        "📊",
        "Real vs Synthetic Enterprise",
        "Comparison of original enterprise data against CTGAN-generated synthetic data."
    )

    # =====================================================
    # SIMILARITY SCORES
    # =====================================================

    # Calculate distribution similarity dynamically
    numeric_columns = [
        "age",
        "tenure_years",
        "salary",
        "overtime_pct",
        "last_promotion_years",
        "engagement_score",
        "project_count",
        "avg_planned_days",
        "avg_actual_days",
        "total_budget",
        "total_spend",
        "avg_team_size",
        "avg_scope_changes",
        "status_on_track",
        "status_overrun",
        "total_assets",
        "avg_asset_age_years",
        "avg_runtime_hours",
        "avg_temperature",
        "avg_vibration",
        "avg_tickets_90d",
        "avg_last_service_days"
    ]

    available = [
        col
        for col in numeric_columns
        if col in real_df.columns
        and col in synthetic_df.columns
    ]

    # -----------------------------------------------------
    # Distribution similarity
    # -----------------------------------------------------

    distribution_scores = []

    for col in available:

        real_mean = real_df[col].mean()
        synthetic_mean = synthetic_df[col].mean()

        real_std = real_df[col].std()
        synthetic_std = synthetic_df[col].std()

        if real_mean != 0:

            mean_similarity = (
                1 -
                abs(real_mean - synthetic_mean)
                / abs(real_mean)
            ) * 100

        else:

            mean_similarity = 100

        if real_std != 0:

            std_similarity = (
                1 -
                abs(real_std - synthetic_std)
                / abs(real_std)
            ) * 100

        else:

            std_similarity = 100

        score = (
            mean_similarity * 0.6
            + std_similarity * 0.4
        )

        score = max(
            0,
            min(100, score)
        )

        distribution_scores.append(score)

    if distribution_scores:

        distribution_score = sum(
            distribution_scores
        ) / len(distribution_scores)

    else:

        distribution_score = 0

    # -----------------------------------------------------
    # Correlation similarity
    # -----------------------------------------------------

    if len(available) >= 2:

        real_corr = real_df[
            available
        ].corr()

        synthetic_corr = synthetic_df[
            available
        ].corr()

        correlation_difference = (
            real_corr - synthetic_corr
        ).abs().mean().mean()

        correlation_score = (
            1 - correlation_difference
        ) * 100

        correlation_score = max(
            0,
            min(100, correlation_score)
        )

    else:

        correlation_score = 0

    # -----------------------------------------------------
    # Overall score
    # -----------------------------------------------------

    overall_score = (
        distribution_score
        + correlation_score
    ) / 2

    # =====================================================
    # SCORE CARDS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        show_kpi(
            "Distribution Similarity",
            f"{distribution_score:.1f}%",
            "Similarity of statistical distributions between real and synthetic data."
        )

    with c2:

        show_kpi(
            "Correlation Similarity",
            f"{correlation_score:.1f}%",
            "Similarity of relationships between numeric enterprise features."
        )

    with c3:

        show_kpi(
            "Overall Similarity",
            f"{overall_score:.1f}%",
            "Combined real-vs-synthetic data quality score."
        )

    st.divider()

    # =====================================================
    # QUALITY BADGE
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

    st.divider()

    # =====================================================
    # MEAN COMPARISON DATA
    # =====================================================

    comparison = pd.DataFrame(
        {
            "Metric": available,

            "Real": [
                real_df[col].mean()
                for col in available
            ],

            "Synthetic": [
                synthetic_df[col].mean()
                for col in available
            ]
        }
    )

    # =====================================================
    # EMPLOYEE COMPARISON
    # =====================================================

    section_title("👤", "Employee Metrics Comparison")

    employee_metrics = [
        "age",
        "tenure_years",
        "salary",
        "overtime_pct",
        "engagement_score"
    ]

    employee_metrics = [
        col
        for col in employee_metrics
        if col in available
    ]

    employee_chart = comparison[
        comparison["Metric"].isin(
            employee_metrics
        )
    ]

    if not employee_chart.empty:

        chart_long = employee_chart.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace(
                "_",
                " ",
                regex=False
            )
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            text_auto=".2s",
            title="Employee Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # PROJECT COMPARISON
    # =====================================================

    section_title("📁", "Project Metrics Comparison")

    project_metrics = [
        "project_count",
        "avg_planned_days",
        "avg_actual_days",
        "avg_team_size",
        "avg_scope_changes"
    ]

    project_metrics = [
        col
        for col in project_metrics
        if col in available
    ]

    project_chart = comparison[
        comparison["Metric"].isin(
            project_metrics
        )
    ]

    if not project_chart.empty:

        chart_long = project_chart.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace(
                "_",
                " ",
                regex=False
            )
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            text_auto=".2s",
            title="Project Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL COMPARISON
    # =====================================================

    section_title("💰", "Financial Metrics Comparison")

    financial_metrics = [
        "total_budget",
        "total_spend"
    ]

    financial_metrics = [
        col
        for col in financial_metrics
        if col in available
    ]

    financial_chart = comparison[
        comparison["Metric"].isin(
            financial_metrics
        )
    ]

    if not financial_chart.empty:

        chart_long = financial_chart.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Amount"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace(
                "_",
                " ",
                regex=False
            )
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Amount",
            color="Dataset",
            barmode="group",
            text_auto=".2s",
            title="Financial Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # ASSET COMPARISON
    # =====================================================

    section_title("🛠️", "Asset & Operational Comparison")

    asset_metrics = [
        "total_assets",
        "avg_asset_age_years",
        "avg_runtime_hours",
        "avg_temperature",
        "avg_vibration",
        "avg_tickets_90d",
        "avg_last_service_days"
    ]

    asset_metrics = [
        col
        for col in asset_metrics
        if col in available
    ]

    asset_chart = comparison[
        comparison["Metric"].isin(
            asset_metrics
        )
    ]

    if not asset_chart.empty:

        chart_long = asset_chart.melt(
            id_vars="Metric",
            var_name="Dataset",
            value_name="Value"
        )

        chart_long["Metric"] = (
            chart_long["Metric"]
            .str.replace(
                "_",
                " ",
                regex=False
            )
            .str.title()
        )

        fig = px.bar(
            chart_long,
            x="Metric",
            y="Value",
            color="Dataset",
            barmode="group",
            text_auto=".2s",
            title="Asset & Operational Metrics — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # PROJECT STATUS COMPARISON
    # =====================================================

    status_columns = [
        "status_on_track",
        "status_overrun"
    ]

    if all(
        col in real_df.columns
        and col in synthetic_df.columns
        for col in status_columns
    ):

        section_title("🚦", "Project Status Comparison")

        status_df = pd.DataFrame(
            {
                "Status": [
                    "On Track",
                    "Overrun"
                ],

                "Real": [
                    real_df[
                        "status_on_track"
                    ].mean(),

                    real_df[
                        "status_overrun"
                    ].mean()
                ],

                "Synthetic": [
                    synthetic_df[
                        "status_on_track"
                    ].mean(),

                    synthetic_df[
                        "status_overrun"
                    ].mean()
                ]
            }
        )

        status_long = status_df.melt(
            id_vars="Status",
            var_name="Dataset",
            value_name="Average"
        )

        fig = px.bar(
            status_long,
            x="Status",
            y="Average",
            color="Dataset",
            barmode="group",
            text_auto=".2f",
            title="Project Status — Real vs Synthetic",
            color_discrete_map=SERIES_COLOR_MAP,
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = apply_enterprise_chart_theme(fig)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CORRELATION COMPARISON
    # =====================================================

    if len(available) >= 2:

        section_title("🔗", "Feature Relationship Comparison")

        corr_columns = [
            col
            for col in [
                "age",
                "salary",
                "engagement_score",
                "project_count",
                "total_budget",
                "total_spend",
                "avg_runtime_hours",
                "avg_tickets_90d"
            ]
            if col in available
        ]

        if len(corr_columns) >= 2:

            real_corr = real_df[
                corr_columns
            ].corr()

            synthetic_corr = synthetic_df[
                corr_columns
            ].corr()

            correlation_difference = (
                real_corr
                - synthetic_corr
            ).abs()

            corr_long = (
                correlation_difference
                .stack()
                .reset_index()
            )

            corr_long.columns = [
                "Feature 1",
                "Feature 2",
                "Difference"
            ]

            corr_long = corr_long[
                corr_long["Feature 1"]
                != corr_long["Feature 2"]
            ]

            if not corr_long.empty:

                fig = px.bar(
                    corr_long.head(12),
                    x="Feature 1",
                    y="Difference",
                    color="Feature 2",
                    title="Correlation Difference — Real vs Synthetic",
                    color_discrete_sequence=CATEGORICAL_SEQUENCE,
                )

                fig = apply_enterprise_chart_theme(fig)

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
    # DOWNLOAD COMPARISON
    # =====================================================

    st.divider()

    section_title("⬇️", "Download Comparison Results")

    comparison_csv = (
        comparison
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Enterprise Comparison",
        data=comparison_csv,
        file_name="enterprise_real_vs_synthetic.csv",
        mime="text/csv",
        use_container_width=True,
        key="enterprise_comparison_download"
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

def show_enterprise_dashboard():

    inject_custom_css()

    try:

        real_df, synthetic_df, expanded_df = load_data()

    except Exception as e:

        st.error(
            f"Unable to load enterprise datasets: {e}"
        )

        return

    # =====================================================
    # PAGE HEADER
    # =====================================================

    page_header(
        "🏢 Enterprise Intelligence",
        "Enterprise workforce, project, financial and asset analytics."
    )

    st.caption(
        f"Current synthetic dataset: "
        f"{len(synthetic_df):,} records"
    )

    st.divider()

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Enterprise Intelligence",
            "🧬 Synthetic Data",
            "📊 Real vs Synthetic"
        ]
    )

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        show_enterprise_intelligence(
            real_df
        )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        show_synthetic_data(
            real_df,
            synthetic_df,
            expanded_df
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        show_comparison(
            real_df,
            synthetic_df
        )
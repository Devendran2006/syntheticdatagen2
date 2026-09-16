import streamlit as st
import pandas as pd
import plotly.express as px
import os


# =========================================================
# PATHS
# =========================================================

REAL_PATH = "data/enterprise/enterprise_combined.csv"

# NEW LARGE SYNTHETIC DATASET
SYNTHETIC_PATH = "data/enterprise/synthetic_enterprise_large.csv"

# ORIGINAL + SYNTHETIC EXPANDED DATASET
EXPANDED_PATH = "data/enterprise/enterprise_expanded.csv"


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
# SCORE CARD COMPONENT
# =========================================================

def score_card(label, value, description):

    st.metric(
        label=label,
        value=value
    )

    st.caption(description)
# =========================================================
# CHART THEME
# =========================================================

def style_chart(fig, title=None):

    if title:

        fig.update_layout(
            title={
                "text": title,
                "x": 0,
                "xanchor": "left",
                "font": {
                    "size": 18,
                    "color": "#1F2A44"
                }
            }
        )

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font={
            "color": "#1F2A44"
        },

        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 30
        },

        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        }
    )

    # X axis
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont={
            "color": "#667085"
        }
    )

    # Y axis
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont={
            "color": "#667085"
        }
    )

    return fig


# =========================================================
# ENTERPRISE INTELLIGENCE
# =========================================================

def show_enterprise_intelligence(df):

    st.subheader("🏢 Enterprise Intelligence")

    st.caption(
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

        st.metric(
            "Employees",
            f"{len(df):,}"
        )

        st.caption(
            "Employee records in the enterprise dataset"
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

        st.metric(
            "Projects",
            f"{total_projects:,.0f}"
        )

        st.caption(
            "Total projects represented across departments"
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

        st.metric(
            "Assets",
            f"{total_assets:,.0f}"
        )

        st.caption(
            "Enterprise assets represented in the dataset"
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

        st.metric(
            "Project Overrun Rate",
            f"{overrun_rate:.1f}%"
        )

        st.caption(
            "Share of projects classified as overrun"
        )

    st.divider()

    # =====================================================
    # WORKFORCE INTELLIGENCE
    # =====================================================

    st.subheader("Workforce Intelligence")

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
                text="Employees"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = style_chart(
                fig,
                "Employees by Department"
            )

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
                text="Attrition"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig = style_chart(
                fig,
                "Attrition by Department"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # PROJECT INTELLIGENCE
    # =====================================================

    st.subheader("Project Intelligence")

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
                barmode="group"
            )

            fig = style_chart(
                fig,
                "Planned vs Actual Project Days"
            )

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
                barmode="group"
            )

            fig = style_chart(
                fig,
                "Budget vs Spend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # PROJECT STATUS
    # =====================================================

    st.subheader("Project Status")

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
            text="Projects"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Project Status Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # ASSET RISK
    # =====================================================

    st.subheader("Asset Risk Intelligence")

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
            text="Assets"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Asset Risk Distribution"
        )

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

    st.subheader(
        "🧬 Synthetic Enterprise Data"
    )

    st.caption(
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

        st.metric(
            "Original Records",
            f"{len(real_df):,}"
        )

        st.caption(
            "Records in the original enterprise dataset"
        )

    # -----------------------------------------------------
    # SYNTHETIC
    # -----------------------------------------------------

    with c2:

        st.metric(
            "Synthetic Records",
            f"{len(synthetic_df):,}"
        )

        st.caption(
            "New records generated by CTGAN"
        )

    # -----------------------------------------------------
    # EXPANDED
    # -----------------------------------------------------

    with c3:

        st.metric(
            "Expanded Dataset",
            f"{len(expanded_df):,}"
        )

        st.caption(
            "Original + synthetic enterprise records"
        )

    # -----------------------------------------------------
    # COLUMNS
    # -----------------------------------------------------

    with c4:

        st.metric(
            "Dataset Columns",
            f"{len(synthetic_df.columns):,}"
        )

        st.caption(
            "Features available in generated data"
        )

    st.divider()

    # =====================================================
    # GENERATION SUMMARY
    # =====================================================

    st.subheader(
        "Synthetic Data Generation Summary"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Generation Multiplier",
            "2×"
        )

        st.caption(
            "Synthetic records generated relative to original data"
        )

    with c2:

        increase = 0

        if len(real_df) > 0:

            increase = (
                len(synthetic_df)
                / len(real_df)
            ) * 100

        st.metric(
            "Additional Data",
            f"{increase:.0f}%"
        )

        st.caption(
            "Synthetic records added beyond the original dataset"
        )

    with c3:

        st.metric(
            "Total Available",
            f"{len(expanded_df):,}"
        )

        st.caption(
            "Complete expanded enterprise dataset"
        )

    st.divider()

    # =====================================================
    # SYNTHETIC DATA PREVIEW
    # =====================================================

    st.subheader(
        "Synthetic Dataset Preview"
    )

    st.caption(
        f"Preview of the newly generated {len(synthetic_df):,} synthetic enterprise records."
    )

    st.dataframe(
        synthetic_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # DOWNLOAD SECTION
    # =====================================================

    st.subheader(
        "Download Enterprise Datasets"
    )

    st.caption(
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
                "⬇️ Download 1,400 Synthetic Records"
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

        st.caption(
            f"Downloads only the {len(synthetic_df):,} newly generated synthetic records."
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
                "⬇️ Download 2,100 Original + Synthetic Records"
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
            f"Downloads the complete {len(expanded_df):,}-record enterprise dataset."
        )


# =========================================================
# REAL VS SYNTHETIC ENTERPRISE COMPARISON
# =========================================================

def show_comparison(
    real_df,
    synthetic_df
):

    st.subheader(
        "📊 Real vs Synthetic Enterprise"
    )

    st.caption(
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

        score_card(
            "Distribution Similarity",
            f"{distribution_score:.2f}",
            "Similarity of statistical distributions between real and synthetic data"
        )

    with c2:

        score_card(
            "Correlation Similarity",
            f"{correlation_score:.2f}",
            "Similarity of relationships between numeric enterprise features"
        )

    with c3:

        score_card(
            "Overall Similarity",
            f"{overall_score:.2f}",
            "Combined real-vs-synthetic data quality score"
        )

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

    st.subheader(
        "Employee Metrics Comparison"
    )

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
            text_auto=".2s"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Employee Metrics — Real vs Synthetic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # PROJECT COMPARISON
    # =====================================================

    st.subheader(
        "Project Metrics Comparison"
    )

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
            text_auto=".2s"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Project Metrics — Real vs Synthetic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL COMPARISON
    # =====================================================

    st.subheader(
        "Financial Metrics Comparison"
    )

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
            text_auto=".2s"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Financial Metrics — Real vs Synthetic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # ASSET COMPARISON
    # =====================================================

    st.subheader(
        "Asset & Operational Comparison"
    )

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
            text_auto=".2s"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Asset & Operational Metrics — Real vs Synthetic"
        )

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

        st.subheader(
            "Project Status Comparison"
        )

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
            text_auto=".2f"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig = style_chart(
            fig,
            "Project Status — Real vs Synthetic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CORRELATION COMPARISON
    # =====================================================

    if len(available) >= 2:

        st.subheader(
            "Feature Relationship Comparison"
        )

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
                    title="Correlation Difference — Real vs Synthetic"
                )

                fig = style_chart(
                    fig
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # =====================================================
    # DOWNLOAD COMPARISON
    # =====================================================

    st.divider()

    st.subheader(
        "Download Comparison Results"
    )

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

    st.title(
        "🏢 Enterprise Intelligence"
    )

    st.caption(
        "Enterprise workforce, project, financial and asset analytics."
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
import streamlit as st
import pandas as pd
import plotly.express as px
import os


def show_enterprise_charts():

    st.header("Enterprise — Real vs Synthetic Data Comparison")

    # =========================================================
    # PATHS
    # =========================================================

    numeric_path = "data/enterprise/comparison/numeric_comparison.csv"

    # =========================================================
    # CHECK FILE
    # =========================================================

    if not os.path.exists(numeric_path):
        st.error("Enterprise comparison file not found.")
        st.info(
            "Run: python src/enterprise_comparison.py"
        )
        return

    # =========================================================
    # LOAD DATA
    # =========================================================

    df = pd.read_csv(numeric_path)

    # =========================================================
    # QUALITY SCORES
    # =========================================================

    st.subheader("Synthetic Data Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Distribution Score",
            "93.34"
        )

    with col2:
        st.metric(
            "Correlation Score",
            "83.85"
        )

    with col3:
        st.metric(
            "Overall Similarity",
            "88.59"
        )

    st.divider()

    # =========================================================
    # REAL VS SYNTHETIC MEAN
    # =========================================================

    st.subheader("Real vs Synthetic — Mean Comparison")

    mean_df = df[
        [
            "Column",
            "Real Mean",
            "Synthetic Mean"
        ]
    ].copy()

    mean_long = mean_df.melt(
        id_vars="Column",
        value_vars=[
            "Real Mean",
            "Synthetic Mean"
        ],
        var_name="Dataset",
        value_name="Mean"
    )

    fig_mean = px.bar(
        mean_long,
        x="Column",
        y="Mean",
        color="Dataset",
        barmode="group",
        title="Real vs Synthetic Mean Values"
    )

    fig_mean.update_layout(
        xaxis_title="Enterprise Metrics",
        yaxis_title="Mean Value",
        xaxis_tickangle=-45,
        height=550
    )

    st.plotly_chart(
        fig_mean,
        use_container_width=True
    )

    # =========================================================
    # REAL VS SYNTHETIC STANDARD DEVIATION
    # =========================================================

    st.subheader(
        "Real vs Synthetic — Standard Deviation"
    )

    std_df = df[
        [
            "Column",
            "Real Std",
            "Synthetic Std"
        ]
    ].copy()

    std_long = std_df.melt(
        id_vars="Column",
        value_vars=[
            "Real Std",
            "Synthetic Std"
        ],
        var_name="Dataset",
        value_name="Standard Deviation"
    )

    fig_std = px.bar(
        std_long,
        x="Column",
        y="Standard Deviation",
        color="Dataset",
        barmode="group",
        title="Real vs Synthetic Standard Deviation"
    )

    fig_std.update_layout(
        xaxis_title="Enterprise Metrics",
        yaxis_title="Standard Deviation",
        xaxis_tickangle=-45,
        height=550
    )

    st.plotly_chart(
        fig_std,
        use_container_width=True
    )

    # =========================================================
    # MEAN DIFFERENCE
    # =========================================================

    st.subheader(
        "Mean Difference Between Real and Synthetic Data"
    )

    difference_df = df[
        [
            "Column",
            "Mean Difference %"
        ]
    ].copy()

    difference_df = difference_df.sort_values(
        "Mean Difference %",
        ascending=False
    )

    fig_difference = px.bar(
        difference_df,
        x="Column",
        y="Mean Difference %",
        title="Mean Difference Percentage",
        text="Mean Difference %"
    )

    fig_difference.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_difference.update_layout(
        xaxis_title="Enterprise Metrics",
        yaxis_title="Mean Difference (%)",
        xaxis_tickangle=-45,
        height=550
    )

    st.plotly_chart(
        fig_difference,
        use_container_width=True
    )

    # =========================================================
    # COMPARISON TABLE
    # =========================================================

    st.subheader(
        "Detailed Real vs Synthetic Comparison"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


import streamlit as st
import pandas as pd
import os
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
# CHART STYLE
# =========================================================

def apply_chart_style(fig):

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        ),
        font=dict(
            color="#1F2A44"
        ),
        title=dict(
            font=dict(
                color="#1F2A44",
                size=18
            )
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        )
    )

    return fig


# =========================================================
# KPI HELPER
# =========================================================

def show_kpi(
    label,
    value,
    description
):

    st.metric(
        label,
        value
    )

    st.caption(
        description
    )


# =========================================================
# BANKING INTELLIGENCE
# =========================================================

def show_banking_intelligence(df):

    st.subheader(
        "🏦 Banking Intelligence"
    )

    st.caption(
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
            "Number of customer records in the banking dataset"
        )

    with c2:

        show_kpi(
            "Average Balance",
            f"{avg_balance:,.0f}",
            "Average customer account balance"
        )

    with c3:

        show_kpi(
            "Average Credit Score",
            f"{avg_credit:.0f}",
            "Average credit score across customers"
        )

    with c4:

        show_kpi(
            "High Churn Risk",
            f"{high_churn:,}",
            "Customers classified with high churn risk"
        )

    st.divider()

    # =====================================================
    # CUSTOMER ANALYTICS
    # =====================================================

    st.subheader(
        "Customer Analytics"
    )

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
                title="Customer Segment Distribution"
            )

            fig = apply_chart_style(fig)

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
                title="Churn Risk Distribution"
            )

            fig = apply_chart_style(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # FINANCIAL ANALYTICS
    # =====================================================

    st.subheader(
        "Financial Analytics"
    )

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
                title="Income vs Average Balance"
            )

            fig = apply_chart_style(fig)

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
                title="Credit Score vs DTI Ratio"
            )

            fig = apply_chart_style(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # LOAN ANALYTICS
    # =====================================================

    st.subheader(
        "Loan & Credit Analytics"
    )

    c1, c2 = st.columns(2)

    with c1:

        if "total_loan_amount" in df.columns:

            fig = px.histogram(
                df,
                x="total_loan_amount",
                nbins=20,
                title="Total Loan Amount Distribution"
            )

            fig = apply_chart_style(fig)

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
                title="Loan Amount vs Credit Score"
            )

            fig = apply_chart_style(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # =====================================================
    # TRANSACTION ANALYTICS
    # =====================================================

    st.subheader(
        "Transaction & Fraud Analytics"
    )

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
                title="Transactions vs Transaction Amount"
            )

            fig = apply_chart_style(fig)

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
            title="Fraud & New Device Activity"
        )

        fig = apply_chart_style(fig)

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

    st.subheader(
        "🧬 Synthetic Banking Data"
    )

    st.caption(
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
            "Records available before synthetic generation"
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "New records generated using CTGAN"
        )

    with c3:

        show_kpi(
            "Expanded Records",
            f"{expanded_records:,}",
            "Original records plus generated synthetic records"
        )

    with c4:

        show_kpi(
            "Expansion Factor",
            f"{expansion_factor:.1f}×",
            "Final dataset size compared with original dataset"
        )

    with c5:

        show_kpi(
            "Dataset Columns",
            f"{len(expanded_df.columns):,}",
            "Features available in the final expanded dataset"
        )

    st.divider()

    # =====================================================
    # GENERATION SUMMARY
    # =====================================================

    st.subheader(
        "Synthetic Data Generation"
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.info(
            f"""
            **Original Banking Dataset**

            {original_records:,} records

            Source records used as the basis
            for synthetic data generation.
            """
        )

    with summary_col2:

        st.info(
            f"""
            **CTGAN Generated Data**

            {synthetic_records:,} records

            New synthetic banking records
            generated from learned data patterns.
            """
        )

    with summary_col3:

        st.success(
            f"""
            **Final Expanded Dataset**

            {expanded_records:,} records

            Original and synthetic records
            combined into one final dataset.
            """
        )

    st.divider()

    # =====================================================
    # SYNTHETIC DATA ANALYTICS
    # =====================================================

    st.subheader(
        "Synthetic Banking Analytics"
    )

    st.caption(
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
            }
        )

        fig = apply_chart_style(fig)

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
            }
        )

        fig = apply_chart_style(fig)

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

        st.subheader(
            "Synthetic Financial Metrics"
        )

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
            title="Average Financial Metrics"
        )

        fig = apply_chart_style(fig)

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

        st.subheader(
            "Banking Category Distribution"
        )

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
                ).title()
            )

            fig = apply_chart_style(fig)

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

    st.subheader(
        "Expanded Banking Dataset Preview"
    )

    st.caption(
        f"Showing the first 20 records from the final {expanded_records:,}-record dataset."
    )

    st.dataframe(
        expanded_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # DOWNLOAD BUTTONS
    # =====================================================

    st.subheader(
        "Download Banking Datasets"
    )

    st.caption(
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
            label="⬇️ Download Synthetic Banking Data",
            data=synthetic_csv,
            file_name="synthetic_banking_large.csv",
            mime="text/csv",
            use_container_width=True,
            key="banking_large_synthetic_download"
        )

        st.caption(
            f"{synthetic_records:,} CTGAN-generated synthetic records"
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
            label="⬇️ Download Original + Synthetic Data",
            data=expanded_csv,
            file_name="banking_expanded.csv",
            mime="text/csv",
            use_container_width=True,
            key="banking_original_synthetic_expanded_download"
        )

        st.caption(
            f"{expanded_records:,} total records: original + synthetic"
        )


# =========================================================
# REAL VS SYNTHETIC COMPARISON
# =========================================================

def show_comparison(
    real_df,
    synthetic_df
):

    st.subheader(
        "📊 Real vs Synthetic Banking"
    )

    st.caption(
        "Direct comparison between the original banking dataset and the currently generated synthetic dataset."
    )

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    real_records = len(real_df)
    synthetic_records = len(synthetic_df)

    c1, c2, c3 = st.columns(3)

    with c1:

        show_kpi(
            "Real Records",
            f"{real_records:,}",
            "Records in the original banking dataset"
        )

    with c2:

        show_kpi(
            "Synthetic Records",
            f"{synthetic_records:,}",
            "Records in the currently generated synthetic dataset"
        )

    with c3:

        if real_records > 0:
            generation_ratio = (
                synthetic_records / real_records
            )
        else:
            generation_ratio = 0

        show_kpi(
            "Generation Ratio",
            f"{generation_ratio:.1f}×",
            "Synthetic records generated relative to original records"
        )

    st.divider()

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
    # MEAN COMPARISON
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

    st.subheader(
        "Key Financial Metrics"
    )

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
            text_auto=".2s",
            title="Key Financial Metrics — Real vs Current Synthetic"
        )

        fig = apply_chart_style(fig)

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # CUSTOMER METRICS
    # =====================================================

    st.subheader(
        "Customer Metrics"
    )

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
            text_auto=".2s",
            title="Customer Metrics — Real vs Current Synthetic"
        )

        fig = apply_chart_style(fig)

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # FINANCIAL DISTRIBUTION COMPARISON
    # =====================================================

    st.subheader(
        "Financial Distribution Comparison"
    )

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
                ).title() + " Distribution"
            )

            fig = apply_chart_style(fig)

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

    st.subheader(
        "Categorical Distribution"
    )

    categorical = [
        "segment",
        "churn_risk",
        "primary_channel"
    ]

    c1, c2 = st.columns(2)

    chart_index = 0

    for column in categorical:

        if column not in real_df.columns:
            continue

        if column not in synthetic_df.columns:
            continue

        # -----------------------------------------------
        # REAL
        # -----------------------------------------------

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

        # -----------------------------------------------
        # SYNTHETIC
        # -----------------------------------------------

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

        # -----------------------------------------------
        # COMBINE
        # -----------------------------------------------

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
            text_auto=".1f",
            title=column.replace(
                "_",
                " "
            ).title() +
            " — Real vs Synthetic"
        )

        fig.update_yaxes(
            title="Percentage (%)"
        )

        fig = apply_chart_style(fig)

        fig.update_traces(
            textposition="outside"
        )

        if chart_index % 2 == 0:

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

        chart_index += 1

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    st.subheader(
        "Dataset Statistics"
    )

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
    )# =========================================================
# MAIN BANKING DASHBOARD
# =========================================================

def show_banking_dashboard():

    try:

        real_df, synthetic_df = load_data()

    except Exception as e:

        st.error(
            f"Unable to load banking datasets: {e}"
        )

        return

    st.title(
        "🏦 Banking Intelligence"
    )

    st.caption(
        "Banking analytics, synthetic data generation results and real-vs-synthetic evaluation."
    )

    st.divider()

    # =====================================================
    # THREE TABS
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


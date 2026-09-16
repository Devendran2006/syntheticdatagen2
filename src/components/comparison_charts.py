
import streamlit as st
import os


def show_comparison_charts(
    domain_name,
    chart_directory
):
    """
    Display all generated Real vs Synthetic comparison
    charts for a specific domain.
    """

    st.divider()

    st.header(
        f"📊 {domain_name} — Real vs Synthetic Comparison"
    )

    st.markdown(
        f"""
        Visual comparison between the original real dataset
        and CTGAN-generated synthetic dataset for the
        **{domain_name}** domain.
        """
    )

    # --------------------------------------------------
    # CHECK DIRECTORY
    # --------------------------------------------------

    if not os.path.exists(chart_directory):

        st.warning(
            f"Comparison charts directory not found: "
            f"{chart_directory}"
        )

        st.info(
            "Run the corresponding *_charts.py file first."
        )

        return

    # --------------------------------------------------
    # GET CHARTS
    # --------------------------------------------------

    chart_files = sorted([
        file
        for file in os.listdir(chart_directory)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not chart_files:

        st.warning(
            "No comparison charts found."
        )

        return

    # --------------------------------------------------
    # CHART TITLES
    # --------------------------------------------------

    title_map = {

        # Healthcare
        "age_real_vs_synthetic":
            "Age Distribution — Real vs Synthetic",

        "billing_real_vs_synthetic":
            "Billing Amount — Real vs Synthetic",

        "medical_condition_real_vs_synthetic":
            "Medical Condition Distribution",

        "gender_real_vs_synthetic":
            "Gender Distribution",

        "admission_type_real_vs_synthetic":
            "Admission Type Distribution",

        # Banking
        "customer_profile_real_vs_synthetic":
            "Customer Profile Metrics",

        "loan_credit_real_vs_synthetic":
            "Loan & Credit Metrics",

        "transaction_fraud_real_vs_synthetic":
            "Transaction & Fraud Metrics",

        "segment_distribution_real_vs_synthetic":
            "Customer Segment Distribution",

        "churn_risk_real_vs_synthetic":
            "Churn Risk Distribution",

        "channel_distribution_real_vs_synthetic":
            "Primary Banking Channel",

        # Enterprise
        "employee_metrics_real_vs_synthetic":
            "Employee Metrics",

        "project_metrics_real_vs_synthetic":
            "Project Metrics",

        "asset_metrics_real_vs_synthetic":
            "Asset Metrics",

        "department_distribution_real_vs_synthetic":
            "Department Distribution",

        # Correlations
        "real_correlation_heatmap":
            "Real Data Correlation Matrix",

        "synthetic_correlation_heatmap":
            "Synthetic Data Correlation Matrix"
    }

    # --------------------------------------------------
    # DISPLAY CHARTS
    # --------------------------------------------------

    for i in range(
        0,
        len(chart_files),
        2
    ):

        col1, col2 = st.columns(2)

        # ----------------------------------------------
        # FIRST CHART
        # ----------------------------------------------

        file1 = chart_files[i]

        title1 = title_map.get(
            os.path.splitext(file1)[0],
            os.path.splitext(file1)[0]
                .replace("_", " ")
                .title()
        )

        with col1:

            st.subheader(title1)

            st.image(
                os.path.join(
                    chart_directory,
                    file1
                ),
                use_container_width=True
            )

        # ----------------------------------------------
        # SECOND CHART
        # ----------------------------------------------

        if i + 1 < len(chart_files):

            file2 = chart_files[i + 1]

            title2 = title_map.get(
                os.path.splitext(file2)[0],
                os.path.splitext(file2)[0]
                    .replace("_", " ")
                    .title()
            )

            with col2:

                st.subheader(title2)

                st.image(
                    os.path.join(
                        chart_directory,
                        file2
                    ),
                    use_container_width=True
                )

    # --------------------------------------------------
    # END
    # --------------------------------------------------

    st.success(
        f"{len(chart_files)} comparison charts loaded "
        f"for {domain_name}."
    )


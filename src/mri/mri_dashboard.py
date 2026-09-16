
"""
MRI SYNTHETIC DATA INTELLIGENCE DASHBOARD
=========================================

Purpose:
    Display the complete MRI synthetic-data pipeline results
    in a Streamlit dashboard.

Data sources:

REAL DATA
---------
data/imaging/MRI/Training
data/imaging/MRI/Testing

SYNTHETIC TRAINING DATA
-----------------------
outputs/mri/training_synthetic/synthetic

SYNTHETIC TESTING DATA
----------------------
outputs/mri/gan_testing/synthetic_testing

EVALUATION RESULTS
------------------
outputs/mri/training_synthetic/evaluation
outputs/mri/gan_testing/results

IMPORTANT:
    This dashboard DOES NOT train GANs.
    This dashboard DOES NOT generate images.
    This dashboard DOES NOT modify the original MRI dataset.

    It only reads existing images, CSV files and graphs.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REAL_TRAIN_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Training"
)

REAL_TEST_DIR = (
    BASE_DIR
    / "data"
    / "imaging"
    / "MRI"
    / "Testing"
)

SYNTH_TRAIN_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "synthetic"
)

SYNTH_TEST_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "gan_testing"
    / "synthetic_testing"
)

TRAIN_EVAL_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "evaluation"
)

TRAIN_VIS_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "training_synthetic"
    / "visualization"
)

TEST_RESULTS_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "gan_testing"
    / "results"
)

TEST_PREVIEW_DIR = (
    BASE_DIR
    / "outputs"
    / "mri"
    / "gan_testing"
    / "preview"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}


# ============================================================
# PAGE STYLE
# ============================================================

def apply_style():

    st.markdown(
        """
        <style>

        .main-title {
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .sub-title {
            font-size: 16px;
            color: #667085;
            margin-bottom: 24px;
        }

        .section-title {
            font-size: 24px;
            font-weight: 700;
            margin-top: 28px;
            margin-bottom: 12px;
        }

        .metric-box {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #E4E7EC;
            background-color: #F8FAFC;
        }

        .score-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #D0D5DD;
            background-color: #FFFFFF;
            text-align: center;
        }

        .score-value {
            font-size: 32px;
            font-weight: 700;
        }

        .score-label {
            font-size: 14px;
            color: #667085;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_files(directory):

    if not directory.exists():
        return []

    files = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp"
    ]:
        files.extend(directory.glob(extension))

    return sorted(files)


def count_class_images(base_dir):

    counts = {}

    for class_name in CLASS_NAMES:

        class_dir = base_dir / class_name

        counts[class_name] = len(
            get_image_files(class_dir)
        )

    return counts


def load_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


def display_image_safe(path, caption=None):

    try:

        image = Image.open(path)

        st.image(
            image,
            caption=caption,
            use_container_width=True
        )

    except Exception:

        st.warning(
            f"Unable to load image: {path.name}"
        )


def find_first_image(directory):

    files = get_image_files(directory)

    if files:
        return files[0]

    return None


# ============================================================
# DATASET COUNTS
# ============================================================

def get_dataset_counts():

    real_train = count_class_images(
        REAL_TRAIN_DIR
    )

    synthetic_train = count_class_images(
        SYNTH_TRAIN_DIR
    )

    real_test = count_class_images(
        REAL_TEST_DIR
    )

    synthetic_test = count_class_images(
        SYNTH_TEST_DIR
    )

    return (
        real_train,
        synthetic_train,
        real_test,
        synthetic_test
    )


# ============================================================
# HEADER
# ============================================================

def show_header():

    st.markdown(
        '<div class="main-title">'
        'MRI Synthetic Data Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Real vs Synthetic MRI analysis, quality evaluation, '
        'visual inspection and ML validation'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Testing MRI images are kept completely separate "
        "from GAN training. The Testing dataset is used "
        "only for evaluation."
    )


# ============================================================
# OVERVIEW
# ============================================================

def show_overview():

    st.markdown(
        '<div class="section-title">'
        'MRI Dataset Overview'
        '</div>',
        unsafe_allow_html=True
    )

    (
        real_train,
        synthetic_train,
        real_test,
        synthetic_test
    ) = get_dataset_counts()

    real_train_total = sum(
        real_train.values()
    )

    synthetic_train_total = sum(
        synthetic_train.values()
    )

    real_test_total = sum(
        real_test.values()
    )

    synthetic_test_total = sum(
        synthetic_test.values()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Real Training",
            f"{real_train_total:,}"
        )

    with col2:
        st.metric(
            "Synthetic Training",
            f"{synthetic_train_total:,}"
        )

    with col3:
        st.metric(
            "Real Testing",
            f"{real_test_total:,}"
        )

    with col4:
        st.metric(
            "Synthetic Testing",
            f"{synthetic_test_total:,}"
        )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MRI Classes",
            "4"
        )

    with col2:
        st.metric(
            "Training Synthetic Ratio",
            "35.71%"
        )

    with col3:
        st.metric(
            "Testing Synthetic Ratio",
            "100%"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def show_class_distribution():

    st.markdown(
        '<div class="section-title">'
        'Class-wise MRI Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    (
        real_train,
        synthetic_train,
        real_test,
        synthetic_test
    ) = get_dataset_counts()

    rows = []

    for class_name in CLASS_NAMES:

        rows.append({

            "Class":
                DISPLAY_NAMES[class_name],

            "Real Training":
                real_train[class_name],

            "Synthetic Training":
                synthetic_train[class_name],

            "Real Testing":
                real_test[class_name],

            "Synthetic Testing":
                synthetic_test[class_name]
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    chart_df = df.set_index("Class")

    st.bar_chart(
        chart_df[
            [
                "Real Training",
                "Synthetic Training",
                "Real Testing",
                "Synthetic Testing"
            ]
        ]
    )


# ============================================================
# MRI SAMPLE VIEWER
# ============================================================

def show_sample_images():

    st.markdown(
        '<div class="section-title">'
        'MRI Sample Inspection'
        '</div>',
        unsafe_allow_html=True
    )

    selected_class = st.selectbox(
        "Select MRI Class",
        CLASS_NAMES,
        format_func=lambda x:
            DISPLAY_NAMES[x]
    )

    real_images = get_image_files(
        REAL_TRAIN_DIR / selected_class
    )

    synthetic_images = get_image_files(
        SYNTH_TRAIN_DIR / selected_class
    )

    st.markdown(
        f"### {DISPLAY_NAMES[selected_class]}"
    )

    st.markdown("#### Real Training MRI")

    if real_images:

        cols = st.columns(4)

        for i, image_path in enumerate(
            real_images[:4]
        ):

            with cols[i]:

                display_image_safe(
                    image_path,
                    "Real MRI"
                )

    else:

        st.warning(
            "Real training images not found."
        )

    st.markdown("#### Synthetic Training MRI")

    if synthetic_images:

        cols = st.columns(4)

        for i, image_path in enumerate(
            synthetic_images[:4]
        ):

            with cols[i]:

                display_image_safe(
                    image_path,
                    "Synthetic MRI"
                )

    else:

        st.warning(
            "Synthetic training images not found."
        )


# ============================================================
# REAL VS SYNTHETIC COMPARISON
# ============================================================

def show_visual_comparison():

    st.markdown(
        '<div class="section-title">'
        'Real vs Synthetic Visual Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_path = (
        TRAIN_VIS_DIR
        / "real_vs_synthetic_training_grid.png"
    )

    if comparison_path.exists():

        display_image_safe(
            comparison_path,
            "Real vs Synthetic MRI Training Comparison"
        )

    else:

        st.warning(
            "Real vs Synthetic comparison image "
            "was not found."
        )

    col1, col2 = st.columns(2)

    with col1:

        real_grid = (
            TRAIN_VIS_DIR
            / "real_training_grid.png"
        )

        if real_grid.exists():

            display_image_safe(
                real_grid,
                "Real MRI Training Samples"
            )

    with col2:

        synthetic_grid = (
            TRAIN_VIS_DIR
            / "synthetic_training_grid.png"
        )

        if synthetic_grid.exists():

            display_image_safe(
                synthetic_grid,
                "Synthetic MRI Training Samples"
            )


# ============================================================
# TRAINING QUALITY
# ============================================================

def show_training_quality():

    st.markdown(
        '<div class="section-title">'
        'Synthetic Training Data Quality'
        '</div>',
        unsafe_allow_html=True
    )

    quality_df = load_csv(
        TRAIN_EVAL_DIR
        / "training_quality_summary.csv"
    )

    similarity_df = load_csv(
        TRAIN_EVAL_DIR
        / "training_similarity_scores.csv"
    )

    if quality_df is not None:

        st.dataframe(
            quality_df,
            use_container_width=True,
            hide_index=True
        )

    if similarity_df is not None:

        st.markdown("#### Class-wise Similarity")

        st.dataframe(
            similarity_df,
            use_container_width=True,
            hide_index=True
        )

    col1, col2 = st.columns(2)

    with col1:

        mean_chart = (
            TRAIN_EVAL_DIR
            / "training_mean_comparison.png"
        )

        if mean_chart.exists():

            display_image_safe(
                mean_chart,
                "Mean Pixel Comparison"
            )

    with col2:

        std_chart = (
            TRAIN_EVAL_DIR
            / "training_std_comparison.png"
        )

        if std_chart.exists():

            display_image_safe(
                std_chart,
                "Pixel Standard Deviation Comparison"
            )

    distribution_chart = (
        TRAIN_EVAL_DIR
        / "training_pixel_distribution.png"
    )

    if distribution_chart.exists():

        display_image_safe(
            distribution_chart,
            "Pixel Distribution Comparison"
        )


# ============================================================
# TRAINING SCORE
# ============================================================

def show_training_score():

    quality_df = load_csv(
        TRAIN_EVAL_DIR
        / "training_quality_summary.csv"
    )

    if quality_df is None:
        return

    score_columns = [
        column
        for column in quality_df.columns
        if "Distribution" in column
    ]

    if not score_columns:
        return

    score_column = score_columns[0]

    try:

        score = float(
            quality_df[score_column].mean()
        )

        if score <= 1:
            score *= 100

        st.markdown(
            '<div class="score-box">'
            f'<div class="score-value">'
            f'{score:.2f}%'
            '</div>'
            '<div class="score-label">'
            'Training Distribution Similarity'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    except Exception:
        pass


# ============================================================
# TESTING RESULTS
# ============================================================

def show_testing_results():

    st.markdown(
        '<div class="section-title">'
        'MRI Testing Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    class_df = load_csv(
        TEST_RESULTS_DIR
        / "gan_testing_class_comparison.csv"
    )

    overall_df = load_csv(
        TEST_RESULTS_DIR
        / "gan_testing_overall_score.csv"
    )

    dataset_df = load_csv(
        TEST_RESULTS_DIR
        / "gan_testing_dataset_summary.csv"
    )

    if overall_df is not None:

        score_column = (
            "Overall_Distribution_Score"
        )

        if score_column in overall_df.columns:

            score = float(
                overall_df.iloc[0][score_column]
            )

            if score <= 1:
                score *= 100

            st.markdown(
                '<div class="score-box">'
                f'<div class="score-value">'
                f'{score:.2f}%'
                '</div>'
                '<div class="score-label">'
                'Testing Distribution Similarity'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

    st.write("")

    if class_df is not None:

        st.markdown(
            "#### Testing Class Comparison"
        )

        st.dataframe(
            class_df,
            use_container_width=True,
            hide_index=True
        )

    if dataset_df is not None:

        st.markdown(
            "#### Testing Dataset Summary"
        )

        st.dataframe(
            dataset_df,
            use_container_width=True,
            hide_index=True
        )

    preview_path = (
        TEST_PREVIEW_DIR
        / "synthetic_testing_preview.png"
    )

    if preview_path.exists():

        st.markdown(
            "#### Synthetic Testing Samples"
        )

        display_image_safe(
            preview_path,
            "Synthetic MRI Testing Preview"
        )


# ============================================================
# VISUALIZATION GRAPHS
# ============================================================

def show_visualization_graphs():

    st.markdown(
        '<div class="section-title">'
        'MRI Analysis Graphs'
        '</div>',
        unsafe_allow_html=True
    )

    graph1 = (
        TRAIN_VIS_DIR
        / "training_class_pixel_comparison.png"
    )

    graph2 = (
        TRAIN_VIS_DIR
        / "training_distribution_comparison.png"
    )

    col1, col2 = st.columns(2)

    with col1:

        if graph1.exists():

            display_image_safe(
                graph1,
                "Class-wise Pixel Comparison"
            )

    with col2:

        if graph2.exists():

            display_image_safe(
                graph2,
                "Class-wise Distribution Score"
            )


# ============================================================
# DOWNLOAD SECTION
# ============================================================

def show_downloads():

    st.markdown(
        '<div class="section-title">'
        'Download MRI Results'
        '</div>',
        unsafe_allow_html=True
    )

    files_to_download = [

        (
            "Training Quality Summary",
            TRAIN_EVAL_DIR
            / "training_quality_summary.csv"
        ),

        (
            "Training Class Comparison",
            TRAIN_EVAL_DIR
            / "training_class_comparison.csv"
        ),

        (
            "Training Pixel Statistics",
            TRAIN_EVAL_DIR
            / "training_pixel_statistics.csv"
        ),

        (
            "Training Similarity Scores",
            TRAIN_EVAL_DIR
            / "training_similarity_scores.csv"
        ),

        (
            "Testing Class Comparison",
            TEST_RESULTS_DIR
            / "gan_testing_class_comparison.csv"
        ),

        (
            "Testing Overall Score",
            TEST_RESULTS_DIR
            / "gan_testing_overall_score.csv"
        ),

        (
            "Testing Dataset Summary",
            TEST_RESULTS_DIR
            / "gan_testing_dataset_summary.csv"
        )
    ]

    available_files = [
        item
        for item in files_to_download
        if item[1].exists()
    ]

    if not available_files:

        st.warning(
            "No downloadable MRI result files found."
        )

        return

    for i in range(
        0,
        len(available_files),
        2
    ):

        cols = st.columns(2)

        for j in range(2):

            index = i + j

            if index >= len(
                available_files
            ):
                break

            label, file_path = (
                available_files[index]
            )

            with cols[j]:

                with open(
                    file_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label=f"Download {label}",
                        data=file.read(),
                        file_name=file_path.name,
                        mime="text/csv",
                        use_container_width=True
                    )


# ============================================================
# PIPELINE STATUS
# ============================================================

def show_pipeline_status():

    st.markdown(
        '<div class="section-title">'
        'MRI Pipeline Status'
        '</div>',
        unsafe_allow_html=True
    )

    pipeline = [

        (
            "Real MRI Training Dataset",
            REAL_TRAIN_DIR.exists()
        ),

        (
            "Real MRI Testing Dataset",
            REAL_TEST_DIR.exists()
        ),

        (
            "Synthetic Training Dataset",
            SYNTH_TRAIN_DIR.exists()
        ),

        (
            "Synthetic Testing Dataset",
            SYNTH_TEST_DIR.exists()
        ),

        (
            "Training Evaluation",
            TRAIN_EVAL_DIR.exists()
        ),

        (
            "Training Visualization",
            TRAIN_VIS_DIR.exists()
        ),

        (
            "Testing Evaluation",
            TEST_RESULTS_DIR.exists()
        )
    ]

    rows = []

    for name, exists in pipeline:

        rows.append({

            "Pipeline Component":
                name,

            "Status":
                "Completed" if exists
                else "Missing"
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN DASHBOARD FUNCTION
# ============================================================

def show_mri_dashboard():

    apply_style()

    show_header()

    show_overview()

    show_class_distribution()

    show_sample_images()

    show_visual_comparison()

    show_training_score()

    show_training_quality()

    show_visualization_graphs()

    show_testing_results()

    show_pipeline_status()

    show_downloads()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    show_mri_dashboard()


# ============================================================
# MRI TESTING DATASET DASHBOARD
# Synthetic Intelligence Platform
# ============================================================
#
# PURPOSE:
#   - Analyze ONLY the MRI Testing dataset
#   - Keep Testing completely separate from Training
#   - Display testing KPIs
#   - Display class distribution
#   - Display image statistics
#   - Display MRI testing gallery
#   - Display class-wise testing analysis
#
# IMPORTANT:
#   This file DOES NOT:
#       - modify Training data
#       - modify Training_Preprocessed data
#       - retrain any model
#       - modify existing GAN models
#       - modify existing synthetic MRI outputs
#       - copy Testing images into Training
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MRI_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imaging"
    / "MRI"
)

# ------------------------------------------------------------
# ONLY TESTING DATA IS USED IN THIS FILE
# ------------------------------------------------------------

TESTING_PATH = (
    MRI_ROOT
    / "Testing"
)


# ============================================================
# THEME
# ============================================================

PRIMARY = "#1F2A44"

SOFT_BLUE = "#DCEAF7"
SOFT_TEAL = "#D9F0EE"
SOFT_GREEN = "#E1F0E4"
SOFT_GOLD = "#F7EBCF"
SOFT_RED = "#F6DDDD"
SOFT_PURPLE = "#E9E1F4"

TEXT = "#263238"
MUTED = "#607080"

BORDER = "#E5EAF0"
BACKGROUND = "#F7F9FC"
WHITE = "#FFFFFF"


# ============================================================
# PAGE CSS
# ============================================================

def inject_testing_css():

    st.markdown(
        f"""
        <style>

        .testing-section-title {{
            color: {PRIMARY};
            font-size: 21px;
            font-weight: 700;
            margin-top: 12px;
            margin-bottom: 14px;
        }}

        .testing-section-subtitle {{
            color: {MUTED};
            font-size: 13px;
            margin-bottom: 18px;
        }}

        .testing-metric {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px;
            min-height: 118px;
            box-shadow: 0 2px 8px rgba(31,42,68,0.04);
        }}

        .testing-metric-label {{
            color: {MUTED};
            font-size: 13px;
            font-weight: 600;
        }}

        .testing-metric-value {{
            color: {PRIMARY};
            font-size: 28px;
            font-weight: 750;
            margin-top: 7px;
        }}

        .testing-metric-note {{
            color: {MUTED};
            font-size: 12px;
            margin-top: 4px;
        }}

        .testing-info {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-left: 5px solid {PRIMARY};
            border-radius: 10px;
            padding: 15px 18px;
            margin: 12px 0 20px 0;
        }}

        .testing-success {{
            background: #F0F8F2;
            border: 1px solid #D7E9DA;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 12px 0 20px 0;
        }}

        .testing-warning {{
            background: #FFF9EC;
            border: 1px solid #F1E1B5;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 12px 0 20px 0;
        }}

        .testing-path {{
            background: #F4F6F9;
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 10px 14px;
            color: {MUTED};
            font-size: 12px;
            margin-bottom: 15px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMMON UI
# ============================================================

def section_header(title, subtitle=None):

    st.markdown(
        f"""
        <div class="testing-section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f"""
            <div class="testing-section-subtitle">
                {subtitle}
            </div>
            """,
            unsafe_allow_html=True
        )


def metric_card(
    label,
    value,
    note="",
    background=WHITE
):

    st.markdown(
        f"""
        <div class="testing-metric"
             style="background:{background};">

            <div class="testing-metric-label">
                {label}
            </div>

            <div class="testing-metric-value">
                {value}
            </div>

            <div class="testing-metric-note">
                {note}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHART THEME
# ============================================================

def apply_chart_theme(fig, title=None):

    fig.update_layout(
        title=title,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,

        font=dict(
            family="Arial",
            color=TEXT
        ),

        title_font=dict(
            size=18,
            color=PRIMARY
        ),

        margin=dict(
            l=45,
            r=25,
            t=60,
            b=45
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.85)"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=BORDER
    )

    fig.update_yaxes(
        gridcolor="#EEF1F5",
        linecolor=BORDER
    )

    return fig


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def image_files(folder):

    folder = Path(folder)

    if not folder.exists():

        return []

    patterns = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
        "*.bmp"
    ]

    files = []

    for pattern in patterns:

        files.extend(
            folder.rglob(pattern)
        )

    return sorted(
        set(files)
    )


# ============================================================
# CLASS DISCOVERY
# ============================================================

def get_testing_classes():

    if not TESTING_PATH.exists():

        return []

    classes = []

    for item in TESTING_PATH.iterdir():

        if item.is_dir():

            images = image_files(item)

            if images:

                classes.append(
                    item.name
                )

    return sorted(classes)


# ============================================================
# CLASS IMAGE FILES
# ============================================================

def get_class_images(class_name):

    class_path = (
        TESTING_PATH
        / class_name
    )

    if class_path.exists():

        return image_files(
            class_path
        )

    files = image_files(
        TESTING_PATH
    )

    return [
        file
        for file in files
        if file.parent.name.lower()
        == str(class_name).lower()
    ]


# ============================================================
# CLASS COUNTS
# ============================================================

def get_testing_class_counts():

    classes = get_testing_classes()

    result = {}

    for class_name in classes:

        result[class_name] = len(
            get_class_images(
                class_name
            )
        )

    return result


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(file_path):

    try:

        image = Image.open(
            file_path
        )

        if image.mode not in [
            "RGB",
            "L"
        ]:

            image = image.convert(
                "RGB"
            )

        return image

    except Exception:

        return None


# ============================================================
# IMAGE STATISTICS
# ============================================================

def calculate_image_statistics(
    files,
    max_samples=250
):

    records = []

    for file in files[:max_samples]:

        image = load_image(
            file
        )

        if image is None:

            continue

        array = np.asarray(
            image
        )

        if array.ndim == 2:

            channels = 1

        else:

            channels = array.shape[-1]

        records.append({

            "File":
                file.name,

            "Class":
                file.parent.name,

            "Width":
                image.width,

            "Height":
                image.height,

            "Channels":
                channels,

            "Mean Pixel":
                round(
                    float(
                        array.mean()
                    ),
                    2
                ),

            "Std Pixel":
                round(
                    float(
                        array.std()
                    ),
                    2
                ),

            "Min Pixel":
                int(
                    array.min()
                ),

            "Max Pixel":
                int(
                    array.max()
                )
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# TESTING OVERVIEW
# ============================================================

def show_testing_overview():

    section_header(
        "Testing MRI Dataset",
        "Independent MRI testing dataset analysis."
    )

    st.markdown(
        f"""
        <div class="testing-path">
            <b>Testing Dataset:</b>
            {TESTING_PATH}
        </div>
        """,
        unsafe_allow_html=True
    )

    files = image_files(
        TESTING_PATH
    )

    class_counts = (
        get_testing_class_counts()
    )

    total_images = len(files)

    total_classes = len(
        class_counts
    )

    largest_class = (
        max(
            class_counts.values()
        )
        if class_counts
        else 0
    )

    smallest_class = (
        min(
            class_counts.values()
        )
        if class_counts
        else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Testing Images",
            f"{total_images:,}",
            "Total independent test images",
            SOFT_TEAL
        )

    with c2:

        metric_card(
            "MRI Classes",
            f"{total_classes:,}",
            "Detected testing classes",
            SOFT_BLUE
        )

    with c3:

        metric_card(
            "Largest Class",
            f"{largest_class:,}",
            "Maximum images in one class",
            SOFT_GREEN
        )

    with c4:

        metric_card(
            "Smallest Class",
            f"{smallest_class:,}",
            "Minimum images in one class",
            SOFT_GOLD
        )

    st.write("")

    # --------------------------------------------------------
    # DATASET STATUS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="testing-success">

            <b>Testing dataset status</b><br><br>

            This page reads MRI images from the
            <b>Testing</b> directory only.

            The testing images are kept independent from
            the existing training pipeline.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def show_testing_class_distribution():

    section_header(
        "Testing Class Distribution",
        "Class-wise image distribution within the testing dataset."
    )

    class_counts = (
        get_testing_class_counts()
    )

    if not class_counts:

        st.warning(
            "No testing MRI classes were found."
        )

        return

    df = pd.DataFrame(
        list(
            class_counts.items()
        ),
        columns=[
            "Class",
            "Images"
        ]
    )

    df = df.sort_values(
        "Images",
        ascending=False
    )

    total = df["Images"].sum()

    df["Percentage"] = (
        df["Images"]
        / total
        * 100
    ).round(2)

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    fig = px.bar(
        df,
        x="Class",
        y="Images",
        text="Images"
    )

    fig.update_traces(
        textposition="outside"
    )

    apply_chart_theme(
        fig,
        "Testing MRI Class Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    section_header(
        "Testing Class Statistics"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CLASS QUALITY
# ============================================================

def show_testing_class_analysis():

    section_header(
        "Testing Class Analysis",
        "Detailed class-level analysis of the independent testing dataset."
    )

    class_counts = (
        get_testing_class_counts()
    )

    if not class_counts:

        st.info(
            "No testing class information available."
        )

        return

    total = sum(
        class_counts.values()
    )

    rows = []

    for class_name, count in class_counts.items():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        rows.append({

            "Class":
                class_name,

            "Images":
                count,

            "Percentage":
                round(
                    percentage,
                    2
                ),

            "Dataset Share":
                f"{percentage:.2f}%"
        })

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # CLASS SELECTOR
    # --------------------------------------------------------

    selected_class = st.selectbox(
        "Select testing class",
        df["Class"].tolist(),
        key="mri_testing_analysis_class"
    )

    selected_row = df[
        df["Class"]
        == selected_class
    ]

    if not selected_row.empty:

        row = selected_row.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:

            metric_card(
                "Selected Class",
                selected_class,
                "Current testing class",
                SOFT_BLUE
            )

        with c2:

            metric_card(
                "Images",
                f"{int(row['Images']):,}",
                "Images in selected class",
                SOFT_TEAL
            )

        with c3:

            metric_card(
                "Dataset Share",
                f"{row['Percentage']:.2f}%",
                "Share of testing dataset",
                SOFT_GREEN
            )

    st.write("")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

def show_image_dimensions():

    section_header(
        "Image Dimension Analysis",
        "Resolution and pixel-value statistics from testing MRI images."
    )

    files = image_files(
        TESTING_PATH
    )

    if not files:

        st.info(
            "No testing images found."
        )

        return

    df = calculate_image_statistics(
        files
    )

    if df.empty:

        st.warning(
            "Image statistics could not be calculated."
        )

        return

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    avg_width = df["Width"].mean()
    avg_height = df["Height"].mean()

    avg_mean = df["Mean Pixel"].mean()
    avg_std = df["Std Pixel"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Average Width",
            f"{avg_width:.0f}px",
            "Average test image width",
            SOFT_BLUE
        )

    with c2:

        metric_card(
            "Average Height",
            f"{avg_height:.0f}px",
            "Average test image height",
            SOFT_TEAL
        )

    with c3:

        metric_card(
            "Mean Pixel",
            f"{avg_mean:.2f}",
            "Average pixel intensity",
            SOFT_GREEN
        )

    with c4:

        metric_card(
            "Pixel Std",
            f"{avg_std:.2f}",
            "Average pixel variation",
            SOFT_GOLD
        )

    # --------------------------------------------------------
    # DIMENSION DISTRIBUTION
    # --------------------------------------------------------

    section_header(
        "Image Resolution Distribution"
    )

    dimension_df = (
        df
        .groupby(
            ["Width", "Height"]
        )
        .size()
        .reset_index(
            name="Images"
        )
    )

    dimension_df["Resolution"] = (
        dimension_df["Width"].astype(str)
        + " × "
        + dimension_df["Height"].astype(str)
    )

    fig = px.bar(
        dimension_df,
        x="Resolution",
        y="Images",
        text="Images"
    )

    fig.update_traces(
        textposition="outside"
    )

    apply_chart_theme(
        fig,
        "Testing MRI Resolution Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # STATISTICS TABLE
    # --------------------------------------------------------

    section_header(
        "Image Statistics Sample"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PIXEL DISTRIBUTION
# ============================================================

def show_pixel_distribution():

    section_header(
        "Pixel Intensity Analysis",
        "Pixel-value distribution across the MRI testing images."
    )

    files = image_files(
        TESTING_PATH
    )

    if not files:

        st.info(
            "No testing images found."
        )

        return

    df = calculate_image_statistics(
        files,
        max_samples=250
    )

    if df.empty:

        return

    # --------------------------------------------------------
    # MEAN PIXEL
    # --------------------------------------------------------

    fig = px.histogram(
        df,
        x="Mean Pixel",
        nbins=30
    )

    apply_chart_theme(
        fig,
        "Mean Pixel Intensity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    fig = px.histogram(
        df,
        x="Std Pixel",
        nbins=30
    )

    apply_chart_theme(
        fig,
        "Pixel Variation Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MRI TESTING GALLERY
# ============================================================

def show_testing_gallery():

    section_header(
        "Testing MRI Gallery",
        "Visual exploration of independent testing MRI images."
    )

    class_counts = (
        get_testing_class_counts()
    )

    if not class_counts:

        st.info(
            "No testing MRI images are available."
        )

        return

    selected_class = st.selectbox(
        "Select testing class",
        ["All"] + list(
            class_counts.keys()
        ),
        key="mri_testing_gallery_class"
    )

    # --------------------------------------------------------
    # FILE SELECTION
    # --------------------------------------------------------

    if selected_class == "All":

        files = image_files(
            TESTING_PATH
        )

    else:

        files = get_class_images(
            selected_class
        )

    if not files:

        st.info(
            "No images available for this selection."
        )

        return

    # --------------------------------------------------------
    # GALLERY SIZE
    # --------------------------------------------------------

    max_gallery_images = st.slider(
        "Number of images to display",
        min_value=6,
        max_value=48,
        value=24,
        step=6,
        key="mri_testing_gallery_size"
    )

    files = files[
        :max_gallery_images
    ]

    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------

    columns = 6

    for start in range(
        0,
        len(files),
        columns
    ):

        row = files[
            start:start + columns
        ]

        cols = st.columns(
            columns
        )

        for index, file in enumerate(
            row
        ):

            with cols[index]:

                image = load_image(
                    file
                )

                if image is None:

                    continue

                st.image(
                    image,
                    use_container_width=True
                )

                st.caption(
                    file.parent.name
                )


# ============================================================
# RANDOM SAMPLE VIEW
# ============================================================

def show_random_samples():

    section_header(
        "Testing Sample Explorer",
        "Randomly selected testing MRI images for visual inspection."
    )

    files = image_files(
        TESTING_PATH
    )

    if not files:

        st.info(
            "No testing images found."
        )

        return

    sample_count = st.slider(
        "Number of random samples",
        min_value=4,
        max_value=20,
        value=8,
        step=4,
        key="mri_testing_random_count"
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
        key="mri_testing_random_seed"
    )

    rng = np.random.default_rng(
        seed
    )

    sample_count = min(
        sample_count,
        len(files)
    )

    indices = rng.choice(
        len(files),
        size=sample_count,
        replace=False
    )

    selected_files = [
        files[int(index)]
        for index in indices
    ]

    columns = 5

    for start in range(
        0,
        len(selected_files),
        columns
    ):

        row = selected_files[
            start:start + columns
        ]

        cols = st.columns(
            columns
        )

        for index, file in enumerate(
            row
        ):

            with cols[index]:

                image = load_image(
                    file
                )

                if image is not None:

                    st.image(
                        image,
                        use_container_width=True
                    )

                    st.caption(
                        f"Class: {file.parent.name}"
                    )


# ============================================================
# TESTING DATA TABLE
# ============================================================

def show_testing_data_table():

    section_header(
        "Testing Dataset Details",
        "File-level information extracted from the testing MRI directory."
    )

    files = image_files(
        TESTING_PATH
    )

    if not files:

        st.info(
            "No testing images found."
        )

        return

    rows = []

    for file in files:

        rows.append({

            "File":
                file.name,

            "Class":
                file.parent.name,

            "Extension":
                file.suffix.lower(),

            "Path":
                str(
                    file.relative_to(
                        TESTING_PATH
                    )
                )
        })

    df = pd.DataFrame(
        rows
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------------------

    csv_data = df.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Testing Dataset Index",
        data=csv_data,
        file_name="mri_testing_dataset_index.csv",
        mime="text/csv",
        key="mri_testing_dataset_index_download"
    )


# ============================================================
# TESTING DATASET VALIDATION
# ============================================================

def show_testing_validation():

    section_header(
        "Testing Dataset Validation",
        "Basic integrity checks for the MRI testing dataset."
    )

    files = image_files(
        TESTING_PATH
    )

    classes = (
        get_testing_classes()
    )

    total_images = len(
        files
    )

    valid_images = 0
    invalid_images = 0

    widths = []
    heights = []

    for file in files:

        image = load_image(
            file
        )

        if image is None:

            invalid_images += 1

            continue

        valid_images += 1

        widths.append(
            image.width
        )

        heights.append(
            image.height
        )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Total Files",
            f"{total_images:,}",
            "Detected image files",
            SOFT_BLUE
        )

    with c2:

        metric_card(
            "Valid Images",
            f"{valid_images:,}",
            "Successfully readable images",
            SOFT_GREEN
        )

    with c3:

        metric_card(
            "Invalid Images",
            f"{invalid_images:,}",
            "Images that could not be opened",
            SOFT_RED
        )

    with c4:

        metric_card(
            "Classes",
            f"{len(classes):,}",
            "Detected test classes",
            SOFT_TEAL
        )

    # --------------------------------------------------------
    # VALIDATION STATUS
    # --------------------------------------------------------

    if invalid_images == 0:

        st.markdown(
            """
            <div class="testing-success">
                <b>Validation result:</b>
                All detected testing MRI image files could be read successfully.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="testing-warning">
                <b>Validation result:</b>
                {invalid_images} testing image file(s) could not be read.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # RESOLUTION CONSISTENCY
    # --------------------------------------------------------

    if widths and heights:

        resolution_pairs = list(
            zip(
                widths,
                heights
            )
        )

        unique_resolutions = sorted(
            set(
                resolution_pairs
            )
        )

        st.write("")

        if len(unique_resolutions) == 1:

            st.success(
                "All readable testing images have the same resolution."
            )

        else:

            st.info(
                f"{len(unique_resolutions)} different image resolutions were detected."
            )


# ============================================================
# MAIN TESTING DASHBOARD
# ============================================================

def show_mri_testing_dashboard():

    inject_testing_css()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            font-size:30px;
            font-weight:750;
            color:{PRIMARY};
            margin-bottom:4px;
        ">
            🧪 MRI Testing Dataset
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            color:{MUTED};
            font-size:15px;
            margin-bottom:20px;
        ">
            Independent testing MRI analysis, validation,
            class distribution and image exploration.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMPORTANT NOTICE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="testing-info">

            <b>Testing-only analysis</b><br><br>

            This dashboard reads only from the MRI
            <b>Testing</b> dataset.

            Existing training data, preprocessed training data,
            trained models and synthetic MRI outputs are not
            modified by this module.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TESTING TABS
    # --------------------------------------------------------

    tabs = st.tabs([
        "📊 Overview",
        "📈 Class Distribution",
        "🧩 Class Analysis",
        "🔬 Image Analysis",
        "🖼️ MRI Gallery",
        "🎯 Sample Explorer",
        "✅ Validation",
        "📋 Dataset Details"
    ])

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tabs[0]:

        show_testing_overview()

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    with tabs[1]:

        show_testing_class_distribution()

    # ========================================================
    # CLASS ANALYSIS
    # ========================================================

    with tabs[2]:

        show_testing_class_analysis()

    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    with tabs[3]:

        show_image_dimensions()

        st.divider()

        show_pixel_distribution()

    # ========================================================
    # GALLERY
    # ========================================================

    with tabs[4]:

        show_testing_gallery()

    # ========================================================
    # RANDOM SAMPLE
    # ========================================================

    with tabs[5]:

        show_random_samples()

    # ========================================================
    # VALIDATION
    # ========================================================

    with tabs[6]:

        show_testing_validation()

    # ========================================================
    # DATASET DETAILS
    # ========================================================

    with tabs[7]:

        show_testing_data_table()


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION NAME
# ============================================================

def show_testing_dashboard():

    show_mri_testing_dashboard()


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    show_mri_testing_dashboard()
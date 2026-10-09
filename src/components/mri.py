
"""
MRI Intelligence Dashboard
--------------------------
Professional MRI dashboard designed to match the Healthcare page theme.

Usage from src/app.py:
    from components.mri import show_mri_dashboard
    show_mri_dashboard()

The dashboard reads the project's existing MRI folders and evaluation CSVs.
It does NOT train a GAN and does NOT use real Testing images for training.
"""

import os
import glob
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# =========================================================
# DESIGN SYSTEM
# =========================================================

COLORS = {
    "primary": "#0B5FA5",
    "primary_dark": "#08406F",
    "accent": "#12A594",
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

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

CLASS_COLORS = {
    "glioma": CATEGORICAL_SEQUENCE[4],
    "meningioma": COLORS["primary"],
    "notumor": COLORS["success"],
    "pituitary": CATEGORICAL_SEQUENCE[3],
}


# =========================================================
# PATHS
# =========================================================

REAL_TRAIN = Path(
    "data/imaging/MRI/Training"
)

REAL_TRAIN_PREPROCESSED = Path(
    "data/imaging/MRI/Training_Preprocessed"
)

REAL_TEST = Path(
    "data/imaging/MRI/Testing"
)

V4_SYNTHETIC = Path(
    "outputs/mri/v4/evaluation/synthetic_samples"
)

V5_SYNTHETIC = Path(
    "outputs/mri/v5/evaluation/synthetic_samples"
)

V5_1_SYNTHETIC = Path(
    "outputs/mri/v5_1/evaluation/synthetic_samples"
)

V5_2_SYNTHETIC = Path(
    "outputs/mri/v5_2/evaluation/synthetic_samples"
)

V4_EVAL_RESULTS = Path(
    "outputs/mri/v4/evaluation/results"
)

V5_EVAL_RESULTS = Path(
    "outputs/mri/v5/ml_validation"
)

V5_1_EVAL_RESULTS = Path(
    "outputs/mri/v5_1/ml_validation"
)

V5_2_EVAL_RESULTS = Path(
    "outputs/mri/v5_2/ml_validation"
)

FINAL_SYNTHETIC_ROOT = Path(
    "outputs/mri_testing_v5_6_production_800"
)

FINAL_EVIDENCE_ROOT = Path(
    "outputs/mri_testing_v5_6_quality_evidence_800"
)


# =========================================================
# STYLE HELPERS
# =========================================================

def inject_custom_css():
    """Inject safe dashboard styling."""

    st.markdown(
        f"""
        <style>

        .hc-header {{
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            background: linear-gradient(
                135deg,
                {COLORS["primary_dark"]} 0%,
                {COLORS["primary"]} 55%,
                {COLORS["accent"]} 100%
            );
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

        .hc-badge {{
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .hc-badge-success {{
            background: #E4F7EE;
            color: {COLORS["success"]} !important;
        }}

        .hc-badge-warning {{
            background: #FEF3E2;
            color: {COLORS["warning"]} !important;
        }}

        .hc-badge-danger {{
            background: #FDEAEA;
            color: {COLORS["danger"]} !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            overflow: hidden;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 0.4rem 0.2rem;
        }}

        div[data-testid="column"] {{
            padding: 0 0.35rem;
        }}

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

        /* Compact MRI gallery images */
        .mri-gallery-card {{
            border: 1px solid #E4E9F2;
            border-radius: 10px;
            padding: 0.35rem;
            background: #FFFFFF;
            margin-bottom: 0.7rem;
        }}

        .mri-gallery-caption {{
            font-size: 0.72rem;
            color: #5B6B85;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-top: 0.25rem;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, description=None):
    st.markdown(f"#### {title}")

    if description:
        st.caption(description)


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
    return (
        f'<span class="hc-badge hc-badge-{kind}">{text}</span>'
    )


def metric_card(label, value, caption=None):
    with st.container(border=True):
        st.metric(label, value)

        if caption:
            st.caption(caption)


def get_theme_type():
    try:
        return st.context.theme.type
    except Exception:
        return "light"


def apply_chart_theme(fig, height=380, legend=True):

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
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=70,
        ),
        colorway=CATEGORICAL_SEQUENCE,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Segoe UI, Helvetica Neue, Arial, sans-serif",
        ),
        bargap=0.25,
        height=height,
    )

    if legend:
        fig.update_layout(
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=text_color),
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
            )
        )
    else:
        fig.update_layout(showlegend=False)

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


# =========================================================
# FILE HELPERS
# =========================================================

def image_files(folder):
    if not folder.exists():
        return []

    patterns = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp",
    ]

    files = []

    for pattern in patterns:
        files.extend(folder.rglob(pattern))

    return sorted(set(files))


def class_image_files(root):
    result = {}

    for cls in CLASSES:
        result[cls] = image_files(root / cls)

    return result


def count_class_images(root):
    result = {}

    for cls in CLASSES:
        result[cls] = len(
            image_files(root / cls)
        )

    return result


def find_csv(root, names):
    for name in names:
        path = root / name

        if path.exists():
            return path

    return None


def discover_evaluation_csvs():
    candidates = []

    roots = [
        V4_EVAL_RESULTS,
        V5_EVAL_RESULTS,
        V5_1_EVAL_RESULTS,
        V5_2_EVAL_RESULTS,
        Path("outputs/mri/v4"),
        Path("outputs/mri/v5"),
        Path("outputs/mri/v5_1"),
        Path("outputs/mri/v5_2"),
    ]

    for root in roots:
        if root.exists():
            candidates.extend(
                root.rglob("*.csv")
            )

    return sorted(set(candidates))


def infer_class_from_path(path):
    text = str(path).lower()

    for cls in CLASSES:
        if cls in text:
            return cls

    return "unknown"


def load_image(path):
    try:
        return Image.open(path).convert("L")
    except Exception:
        return None


def normalize_metric_column(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# =========================================================
# DATASET COUNTS
# =========================================================

@st.cache_data(show_spinner=False)
def get_dataset_counts():

    train_counts = count_class_images(
        REAL_TRAIN
    )

    preprocessed_counts = count_class_images(
        REAL_TRAIN_PREPROCESSED
    )

    test_counts = count_class_images(
        REAL_TEST
    )

    synthetic_sources = {
        "V4": V4_SYNTHETIC,
        "V5": V5_SYNTHETIC,
        "V5.1": V5_1_SYNTHETIC,
        "V5.2": V5_2_SYNTHETIC,
    }

    synthetic_counts = {}

    for version, root in synthetic_sources.items():

        counts = count_class_images(root)

        if sum(counts.values()) > 0:
            synthetic_counts[version] = counts

    return (
        train_counts,
        preprocessed_counts,
        test_counts,
        synthetic_counts,
    )


# =========================================================
# OVERVIEW KPI CARDS
# =========================================================

def show_overview_kpis():
    """
    KPI cards moved from directly below the page header
    into the Overview tab.
    """

    real_counts = count_class_images(
        REAL_TEST
    )

    synthetic_counts, _ = get_final_production_counts()

    real_total = sum(
        real_counts.values()
    )

    synthetic_total = sum(
        synthetic_counts.values()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Real Samples",
            f"{real_total:,}",
            "MRI Testing dataset",
        )

    with c2:
        metric_card(
            "Synthetic Samples",
            f"{synthetic_total:,}",
            "Final V5.6 production",
        )

    with c3:
        metric_card(
            "Combined Samples",
            f"{real_total + synthetic_total:,}",
            "Real + synthetic",
        )

    with c4:
        metric_card(
            "MRI Classes",
            "4",
            "Glioma • Meningioma • No Tumor • Pituitary",
        )


# =========================================================
# DATASET OVERVIEW
# =========================================================

def show_dataset_overview():

    section_header(
        "📊 MRI Dataset Overview",
        "Separate real testing and generated synthetic data.",
    )

    (
        train_counts,
        preprocessed_counts,
        test_counts,
        synthetic_counts,
    ) = get_dataset_counts()

    real_train_total = sum(
        train_counts.values()
    )

    real_test_total = sum(
        test_counts.values()
    )

    preprocessed_total = sum(
        preprocessed_counts.values()
    )

    latest_synthetic_total = 0

    if synthetic_counts:

        latest_version = list(
            synthetic_counts.keys()
        )[-1]

        latest_synthetic_total = sum(
            synthetic_counts[latest_version].values()
        )

    else:
        latest_version = "None"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Real Training",
            f"{real_train_total:,}",
            "GAN training source",
        )

    with c2:
        metric_card(
            "Real Testing",
            f"{real_test_total:,}",
            "Evaluation only",
        )

    with c3:
        metric_card(
            "Preprocessed",
            f"{preprocessed_total:,}",
            "128×128 grayscale",
        )

    with c4:
        metric_card(
            "Latest Synthetic",
            f"{latest_synthetic_total:,}",
            latest_version,
        )

    st.divider()

    section_header(
        "🧩 Class Distribution",
        "Each MRI class is handled independently.",
    )

    rows = []

    for cls in CLASSES:

        rows.append({
            "Class": CLASS_LABELS[cls],
            "Real Training": train_counts.get(cls, 0),
            "Preprocessed": preprocessed_counts.get(cls, 0),
            "Real Testing": test_counts.get(cls, 0),
        })

    distribution_df = pd.DataFrame(rows)

    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True,
    )

    long_df = distribution_df.melt(
        id_vars="Class",
        var_name="Dataset",
        value_name="Images",
    )

    fig = px.bar(
        long_df,
        x="Class",
        y="Images",
        color="Dataset",
        barmode="group",
        text="Images",
        color_discrete_sequence=CATEGORICAL_SEQUENCE,
        title="MRI Images by Class and Dataset",
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig = apply_chart_theme(
        fig,
        height=440,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mri_dataset_distribution",
    )


# =========================================================
# SYNTHETIC VERSION SELECTION
# =========================================================

def get_synthetic_versions():

    versions = {
        "V4": V4_SYNTHETIC,
        "V5": V5_SYNTHETIC,
        "V5.1": V5_1_SYNTHETIC,
        "V5.2": V5_2_SYNTHETIC,
    }

    available = {}

    for version, path in versions.items():

        total = len(
            image_files(path)
        )

        if total > 0:
            available[version] = path

    return available


def show_version_selector(
    key="mri_synthetic_version_selector"
):

    versions = get_synthetic_versions()

    if not versions:

        st.warning(
            "No evaluated synthetic MRI image folder was found."
        )

        return None

    version_names = list(
        versions.keys()
    )

    default_index = len(
        version_names
    ) - 1

    selected = st.selectbox(
        "Synthetic MRI Version",
        version_names,
        index=default_index,
        key=key,
    )

    st.caption(
        f"Source: `{versions[selected]}`"
    )

    return versions[selected]


# =========================================================
# REAL MRI TESTING GALLERY
# =========================================================

def show_real_gallery():

    section_header(
        "🖼️ Real MRI Testing Preview",
        "Preview the untouched real MRI Testing images class by class.",
    )

    # IMPORTANT:
    # Training option removed.
    # Only REAL_TEST is displayed here.
    root = REAL_TEST

    selected_class = st.selectbox(
        "MRI class",
        CLASSES,
        format_func=lambda x: CLASS_LABELS[x],
        key="mri_real_gallery_class",
    )

    files = image_files(
        root / selected_class
    )

    if not files:

        st.info(
            f"No {CLASS_LABELS[selected_class]} testing images found."
        )

        return

    max_show = min(
        len(files),
        12,
    )

    sample_count = st.slider(
        "Testing images to display",
        min_value=4,
        max_value=max_show,
        value=min(8, max_show),
        step=4 if max_show >= 8 else 1,
        key="mri_real_gallery_count",
    )

    selected_files = files[:sample_count]

    cols = st.columns(4)

    for index, path in enumerate(selected_files):

        with cols[index % 4]:

            img = load_image(path)

            if img is not None:

                # Fixed compact width prevents oversized MRI images.
                st.image(
                    img,
                    caption=path.name,
                    width=180,
                )


# =========================================================
# SYNTHETIC GALLERY
# =========================================================

def show_synthetic_gallery(root):

    section_header(
        "✨ Synthetic MRI Preview",
        "Preview the final V5.6 generated MRI images independently for each class.",
    )

    selected_class = st.selectbox(
        "Synthetic class",
        CLASSES,
        format_func=lambda x: CLASS_LABELS[x],
        key="mri_synthetic_gallery_class",
    )

    files = image_files(
        root / selected_class
    )

    if not files:

        st.info(
            "No synthetic images found for this class."
        )

        return

    sample_count = min(
        len(files),
        12,
    )

    count = st.slider(
        "Synthetic images to display",
        min_value=4,
        max_value=sample_count,
        value=min(8, sample_count),
        step=4 if sample_count >= 8 else 1,
        key="mri_synthetic_gallery_count",
    )

    selected_files = files[:count]

    cols = st.columns(4)

    for index, path in enumerate(selected_files):

        with cols[index % 4]:

            img = load_image(path)

            if img is not None:

                # Fixed compact width for dashboard fit.
                st.image(
                    img,
                    caption=path.name,
                    width=180,
                )

    st.caption(
        f"{len(files):,} synthetic "
        f"{CLASS_LABELS[selected_class]} images available."
    )


# =========================================================
# SIDE-BY-SIDE REAL VS SYNTHETIC IMAGE COMPARISON
# =========================================================

def show_image_comparison(root):

    section_header(
        "🖼️ Real vs Synthetic MRI Image Comparison",
        "Compare a real Testing MRI with a generated synthetic MRI from the same class.",
    )

    selected_class = st.selectbox(
        "Comparison class",
        CLASSES,
        format_func=lambda x: CLASS_LABELS[x],
        key="mri_side_compare_class",
    )

    # IMPORTANT:
    # Real source is Testing only.
    real_files = image_files(
        REAL_TEST / selected_class
    )

    synthetic_files = image_files(
        root / selected_class
    )

    if not real_files:

        st.warning(
            "Real MRI Testing images are unavailable for this class."
        )

        return

    if not synthetic_files:

        st.warning(
            "Synthetic MRI images are unavailable for this class."
        )

        return

    max_pairs = min(
        len(real_files),
        len(synthetic_files),
    )

    if max_pairs <= 0:
        return

    index = st.slider(
        "Image pair",
        min_value=0,
        max_value=max_pairs - 1,
        value=0,
        key="mri_pair_index",
    )

    # -----------------------------------------------------
    # Compact comparison layout
    # -----------------------------------------------------

    left, middle, right = st.columns(
        [1, 0.12, 1]
    )

    with left:

        st.markdown(
            f"**REAL — {CLASS_LABELS[selected_class]}**"
        )

        real_img = load_image(
            real_files[index]
        )

        if real_img is not None:

            # Compact fixed display size.
            st.image(
                real_img,
                width=320,
            )

            st.caption(
                f"Testing • {real_files[index].name}"
            )

    with middle:

        st.markdown(
            "<div style='height:150px'></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "font-size:22px; color:#5B6B85;'>↔</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            f"**SYNTHETIC — {CLASS_LABELS[selected_class]}**"
        )

        synthetic_img = load_image(
            synthetic_files[index]
        )

        if synthetic_img is not None:

            # Same size as real image.
            st.image(
                synthetic_img,
                width=320,
            )

            st.caption(
                f"V5.6 Synthetic • "
                f"{synthetic_files[index].name}"
            )


# =========================================================
# PIXEL ANALYSIS
# =========================================================

def calculate_image_statistics(
    files,
    limit=200
):

    means = []
    stds = []
    brightness = []
    sharpness = []

    for path in files[:limit]:

        img = load_image(path)

        if img is None:
            continue

        arr = (
            np.asarray(
                img,
                dtype=np.float32,
            )
            / 255.0
        )

        means.append(
            float(arr.mean())
        )

        stds.append(
            float(arr.std())
        )

        brightness.append(
            float(arr.mean())
        )

        gx = np.diff(
            arr,
            axis=1,
        )

        gy = np.diff(
            arr,
            axis=0,
        )

        sharpness.append(
            float(
                gx.var() + gy.var()
            )
        )

    return {
        "mean": np.array(means),
        "std": np.array(stds),
        "brightness": np.array(brightness),
        "sharpness": np.array(sharpness),
    }


def show_pixel_analysis(root):

    section_header(
        "🔬 MRI Pixel & Image Statistics",
        "Engineering-level image statistics shown separately for real and synthetic data.",
    )

    selected_class = st.selectbox(
        "Analysis class",
        CLASSES,
        format_func=lambda x: CLASS_LABELS[x],
        key="mri_pixel_class",
    )

    real_files = image_files(
        REAL_TRAIN / selected_class
    )

    synthetic_files = image_files(
        root / selected_class
    )

    if not real_files or not synthetic_files:

        st.info(
            "Both real and synthetic images are required for this analysis."
        )

        return

    real_stats = calculate_image_statistics(
        real_files
    )

    synthetic_stats = calculate_image_statistics(
        synthetic_files
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Real Mean",
            f"{np.mean(real_stats['mean']):.4f}",
        )

    with c2:
        metric_card(
            "Synthetic Mean",
            f"{np.mean(synthetic_stats['mean']):.4f}",
        )

    with c3:
        metric_card(
            "Real Sharpness",
            f"{np.mean(real_stats['sharpness']):.4f}",
        )

    with c4:
        metric_card(
            "Synthetic Sharpness",
            f"{np.mean(synthetic_stats['sharpness']):.4f}",
        )

    comparison = pd.DataFrame({
        "Metric": [
            "Pixel Mean",
            "Pixel Std",
            "Brightness",
            "Sharpness Proxy",
        ],
        "Real": [
            np.mean(real_stats["mean"]),
            np.mean(real_stats["std"]),
            np.mean(real_stats["brightness"]),
            np.mean(real_stats["sharpness"]),
        ],
        "Synthetic": [
            np.mean(synthetic_stats["mean"]),
            np.mean(synthetic_stats["std"]),
            np.mean(synthetic_stats["brightness"]),
            np.mean(synthetic_stats["sharpness"]),
        ],
    })

    st.dataframe(
        comparison.round(5),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = pd.DataFrame({
        "Value": np.concatenate([
            real_stats["mean"],
            synthetic_stats["mean"],
        ]),
        "Dataset": (
            ["Real"] * len(real_stats["mean"])
            + ["Synthetic"] * len(synthetic_stats["mean"])
        ),
    })

    fig = px.histogram(
        chart_df,
        x="Value",
        color="Dataset",
        barmode="overlay",
        nbins=30,
        opacity=0.65,
        title=(
            f"{CLASS_LABELS[selected_class]} — "
            "Pixel Mean Distribution"
        ),
        color_discrete_map=SERIES_COLOR_MAP,
    )

    fig = apply_chart_theme(
        fig,
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mri_pixel_mean_distribution",
    )


# =========================================================
# V4 EVALUATION
# =========================================================

def load_v4_evaluation():

    detailed = find_csv(
        V4_EVAL_RESULTS,
        [
            "mri_gan_v4_evaluation.csv",
        ],
    )

    summary = find_csv(
        V4_EVAL_RESULTS,
        [
            "mri_gan_v4_summary.csv",
        ],
    )

    detailed_df = None
    summary_df = None

    if detailed:

        try:
            detailed_df = pd.read_csv(
                detailed
            )
        except Exception:
            detailed_df = None

    if summary:

        try:
            summary_df = pd.read_csv(
                summary
            )
        except Exception:
            summary_df = None

    return detailed_df, summary_df


def show_v4_quality():

    section_header(
        "📈 GAN Quality Evaluation",
        "V4 engineering evaluation: distribution, sharpness and diversity.",
    )

    detailed, summary = load_v4_evaluation()

    if detailed is None and summary is None:

        st.info(
            "V4 evaluation CSVs were not found."
        )

        return

    df = (
        detailed
        if detailed is not None
        else summary
    )

    normalized = {
        normalize_metric_column(col): col
        for col in df.columns
    }

    metric_candidates = [
        "distribution_score",
        "sharpness_score",
        "diversity_score",
        "overall_score",
        "overall_diagnostic_score",
    ]

    available_metrics = [
        metric
        for metric in metric_candidates
        if metric in normalized
    ]

    if not available_metrics:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        return

    class_column = None

    for candidate in [
        "class",
        "category",
        "label",
    ]:

        if candidate in normalized:

            class_column = normalized[candidate]

            break

    if class_column is not None:

        metric_df = df.copy()

        rename_map = {
            normalized[m]: m
            for m in available_metrics
        }

        metric_df = metric_df.rename(
            columns=rename_map
        )

        display_columns = [
            col
            for col in [
                class_column,
                "distribution_score",
                "sharpness_score",
                "diversity_score",
                "overall_score",
                "overall_diagnostic_score",
            ]
            if col in metric_df.columns
        ]

        st.dataframe(
            metric_df[
                display_columns
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

        long_df = metric_df.melt(
            id_vars=[class_column],
            value_vars=[
                col
                for col in [
                    "distribution_score",
                    "sharpness_score",
                    "diversity_score",
                    "overall_score",
                    "overall_diagnostic_score",
                ]
                if col in metric_df.columns
            ],
            var_name="Metric",
            value_name="Score",
        )

        long_df["Metric"] = (
            long_df["Metric"]
            .str.replace(
                "_",
                " ",
            )
            .str.title()
        )

        fig = px.bar(
            long_df,
            x=class_column,
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".1f",
            title="V4 Class-wise Quality Scores",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig.update_yaxes(
            range=[0, 100],
            title="Score (%)",
        )

        fig.update_traces(
            textposition="outside",
        )

        fig = apply_chart_theme(
            fig,
            height=470,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="mri_v4_class_quality",
        )

    else:

        st.dataframe(
            df.round(3),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# ML VALIDATION
# =========================================================

def find_ml_validation_files():

    candidates = []

    roots = [
        V5_EVAL_RESULTS,
        V5_1_EVAL_RESULTS,
        V5_2_EVAL_RESULTS,
        Path("outputs/mri/v5"),
        Path("outputs/mri/v5_1"),
        Path("outputs/mri/v5_2"),
    ]

    for root in roots:

        if root.exists():

            candidates.extend(
                root.rglob("*.csv")
            )

    return sorted(
        set(candidates)
    )


def load_ml_results():

    files = find_ml_validation_files()

    selected = []

    for path in files:

        name = path.name.lower()

        if any(
            keyword in name
            for keyword in [
                "summary",
                "results",
                "metrics",
                "validation",
            ]
        ):

            selected.append(path)

    return selected


def classify_ml_file(path):

    text = str(path).lower()

    if "v5_2" in text:
        return "V5.2"

    if "v5_1" in text:
        return "V5.1"

    if "/v5/" in text or "\\v5\\" in text:
        return "V5"

    return "ML Validation"


def show_ml_validation():

    section_header(
        "🤖 ML Utility Validation",
        "Classification utility of synthetic MRI data against the untouched real test set.",
    )

    files = load_ml_results()

    if not files:

        st.info(
            "No ML validation CSV files were found."
        )

        return

    rows = []

    for path in files:

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        normalized = {
            normalize_metric_column(col): col
            for col in df.columns
        }

        accuracy_col = None

        for candidate in [
            "accuracy",
            "test_accuracy",
            "real_test_accuracy",
        ]:

            if candidate in normalized:

                accuracy_col = normalized[candidate]

                break

        if accuracy_col is None:
            continue

        for _, row in df.iterrows():

            experiment = ""

            for candidate in [
                "experiment",
                "experiment_name",
                "model",
                "dataset",
            ]:

                if candidate in normalized:

                    experiment = str(
                        row[normalized[candidate]]
                    )

                    break

            try:

                accuracy = float(
                    row[accuracy_col]
                )

            except Exception:

                continue

            rows.append({
                "Version": classify_ml_file(path),
                "Experiment": experiment,
                "Accuracy": (
                    accuracy * 100
                    if accuracy <= 1
                    else accuracy
                ),
                "Source": path.name,
            })

    if not rows:

        st.info(
            "ML validation files exist, but no compatible accuracy column was detected."
        )

        return

    results = pd.DataFrame(rows)

    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True,
    )

    chart_results = results[
        results["Experiment"]
        .astype(str)
        .str.len()
        > 0
    ]

    if not chart_results.empty:

        fig = px.bar(
            chart_results,
            x="Experiment",
            y="Accuracy",
            color="Version",
            barmode="group",
            text="Accuracy",
            title="Synthetic MRI ML Utility",
            color_discrete_sequence=CATEGORICAL_SEQUENCE,
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_yaxes(
            title="Accuracy (%)",
            range=[0, 100],
        )

        fig = apply_chart_theme(
            fig,
            height=470,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="mri_ml_utility_chart",
        )


# =========================================================
# CLASS QUALITY DASHBOARD
# =========================================================

def show_class_quality():

    section_header(
        "🧩 Class-wise MRI Quality",
        "View each MRI class independently instead of relying only on an overall score.",
    )

    detailed, summary = load_v4_evaluation()

    df = (
        detailed
        if detailed is not None
        else summary
    )

    if df is None:

        st.info(
            "No MRI quality evaluation data available."
        )

        return

    normalized = {
        normalize_metric_column(col): col
        for col in df.columns
    }

    class_col = None

    for candidate in [
        "class",
        "category",
        "label",
    ]:

        if candidate in normalized:

            class_col = normalized[candidate]

            break

    if class_col is None:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        return

    working = df.copy()

    rename = {}

    for metric in [
        "distribution_score",
        "sharpness_score",
        "diversity_score",
        "overall_score",
        "overall_diagnostic_score",
    ]:

        if metric in normalized:

            rename[
                normalized[metric]
            ] = metric

    working = working.rename(
        columns=rename
    )

    available = [
        metric
        for metric in [
            "distribution_score",
            "sharpness_score",
            "diversity_score",
            "overall_score",
            "overall_diagnostic_score",
        ]
        if metric in working.columns
    ]

    if not available:

        st.dataframe(
            working,
            use_container_width=True,
            hide_index=True,
        )

        return

    selected_class = st.selectbox(
        "Select MRI class",
        [
            "All Classes"
        ]
        + [
            str(x)
            for x in working[class_col]
            .dropna()
            .unique()
        ],
        key="mri_quality_class_selector",
    )

    if selected_class != "All Classes":

        selected_df = working[
            working[class_col]
            .astype(str)
            == selected_class
        ]

    else:

        selected_df = working

    metric_rows = []

    for metric in available:

        values = pd.to_numeric(
            selected_df[metric],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        metric_rows.append({
            "Metric": (
                metric
                .replace("_", " ")
                .title()
            ),
            "Score": values.mean(),
        })

    score_df = pd.DataFrame(
        metric_rows
    )

    if score_df.empty:
        return

    cols = st.columns(
        len(score_df)
    )

    for index, row in score_df.iterrows():

        with cols[index]:

            metric_card(
                row["Metric"],
                f"{row['Score']:.2f}%",
            )

    fig = px.bar(
        score_df,
        x="Metric",
        y="Score",
        text="Score",
        title=(
            "MRI Quality Metrics — "
            f"{selected_class}"
        ),
        color_discrete_sequence=[
            COLORS["primary"]
        ],
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[0, 100],
        title="Score (%)",
    )

    fig = apply_chart_theme(
        fig,
        height=420,
        legend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mri_selected_class_quality",
    )


# =========================================================
# DOWNLOADS
# =========================================================

def show_downloads():

    section_header(
        "⬇️ MRI Evaluation Outputs",
        "Download available MRI quality and ML validation results.",
    )

    files = []

    roots = [
        V4_EVAL_RESULTS,
        V5_EVAL_RESULTS,
        V5_1_EVAL_RESULTS,
        V5_2_EVAL_RESULTS,
    ]

    for root in roots:

        if root.exists():

            files.extend(
                root.rglob("*.csv")
            )

    files = sorted(
        set(files)
    )

    if not files:

        st.info(
            "No evaluation CSV files available."
        )

        return

    for index, path in enumerate(files):

        try:
            data = path.read_bytes()
        except Exception:
            continue

        st.download_button(
            f"⬇️ {path.name}",
            data=data,
            file_name=path.name,
            mime="text/csv",
            use_container_width=True,
            key=f"mri_download_{index}_{path.name}",
        )


# =========================================================
# FINAL V5.6 PRODUCTION DASHBOARD
# =========================================================

def find_latest_final_production_run():

    if not FINAL_SYNTHETIC_ROOT.exists():
        return None

    runs = [
        p
        for p in FINAL_SYNTHETIC_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("production_800_")
    ]

    return (
        max(
            runs,
            key=lambda p: p.stat().st_mtime,
        )
        if runs
        else None
    )


def find_latest_final_evidence_run():

    if not FINAL_EVIDENCE_ROOT.exists():
        return None

    runs = [
        p
        for p in FINAL_EVIDENCE_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("validation_")
    ]

    return (
        max(
            runs,
            key=lambda p: p.stat().st_mtime,
        )
        if runs
        else None
    )


def get_final_production_counts():

    run = find_latest_final_production_run()

    if run is None:
        return {
            cls: 0
            for cls in CLASSES
        }, None

    counts = {
        cls: len(
            image_files(run / cls)
        )
        for cls in CLASSES
    }

    if sum(counts.values()) == 0:

        root = run / "synthetic"

        counts = {
            cls: len(
                image_files(root / cls)
            )
            for cls in CLASSES
        }

    return counts, run


def get_final_evidence_files():

    run = find_latest_final_evidence_run()

    if run is None:
        return None, None, None, None

    html_path = (
        run
        / "mri_v5_6_quality_evidence_800_report.html"
    )

    json_path = (
        run
        / "reports"
        / "v5_6_quality_evidence_800_report.json"
    )

    csv_path = (
        run
        / "reports"
        / "v5_6_class_summary_800.csv"
    )

    return (
        html_path if html_path.exists() else None,
        json_path if json_path.exists() else None,
        csv_path if csv_path.exists() else None,
        run,
    )


def load_final_quality_summary():

    _, _, csv_path, evidence_run = (
        get_final_evidence_files()
    )

    if csv_path is None:
        return None, evidence_run

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, evidence_run

    if df.empty:
        return None, evidence_run

    normalized = {
        normalize_metric_column(col): col
        for col in df.columns
    }

    def find_column(candidates):

        for candidate in candidates:

            if candidate in normalized:

                return normalized[candidate]

        return None

    class_col = find_column([
        "class",
        "category",
        "label",
        "mri_class",
    ])

    score_col = find_column([
        "score",
        "quality_score",
        "overall_score",
        "quality",
    ])

    hist_col = find_column([
        "hist",
        "hist_score",
        "histogram",
        "histogram_score",
        "histogram_similarity",
    ])

    struct_col = find_column([
        "struct",
        "struct_score",
        "structural",
        "structural_score",
        "structural_similarity",
    ])

    robust_z_col = find_column([
        "robust_z",
        "robust_z_score",
    ])

    q95_col = find_column([
        "q95_exceed",
        "q95_exceedance",
        "q95_exceed_pct",
    ])

    status_col = find_column([
        "status",
        "result",
        "validation_status",
    ])

    rename = {}

    for source, target in [
        (class_col, "Class"),
        (score_col, "Quality Score"),
        (hist_col, "Histogram Score"),
        (struct_col, "Structural Score"),
        (robust_z_col, "Robust Z"),
        (q95_col, "Q95 Exceedance"),
        (status_col, "Status"),
    ]:

        if source:
            rename[source] = target

    working = df.rename(
        columns=rename
    ).copy()

    if (
        "Class" not in working.columns
        and len(working) == len(CLASSES)
    ):

        working["Class"] = CLASSES

    if "Class" in working.columns:

        working["Class"] = (
            working["Class"]
            .astype(str)
            .str.lower()
        )

        working["Class"] = (
            working["Class"]
            .replace(CLASS_LABELS)
        )

    for col in [
        "Quality Score",
        "Histogram Score",
        "Structural Score",
        "Robust Z",
        "Q95 Exceedance",
    ]:

        if col in working.columns:

            working[col] = pd.to_numeric(
                working[col],
                errors="coerce",
            )

    return working, evidence_run


def show_final_quality_evidence():

    section_header(
        "🧪 Final V5.6 Quality Evidence",
        "Evidence that generated MRI data remains consistent with the real MRI distribution and structure.",
    )

    st.info(
        "These are synthetic-data quality, similarity and consistency metrics. "
        "They are engineering/ML evidence only — not medical accuracy, "
        "clinical validation or diagnostic proof."
    )

    df, evidence_run = (
        load_final_quality_summary()
    )

    if evidence_run is None:

        st.warning(
            "Final V5.6 quality-evidence output was not found."
        )

        return

    if df is None:

        st.warning(
            "The final quality-evidence CSV could not be read."
        )

        return

    status_values = []

    if "Status" in df.columns:

        status_values = (
            df["Status"]
            .dropna()
            .astype(str)
            .str.upper()
            .tolist()
        )

    passed = (
        sum(
            "PASS" in value
            for value in status_values
        )
        if status_values
        else None
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Evidence Classes",
            f"{len(df):,}",
            "Final V5.6 class-wise evidence",
        )

    with c2:

        metric_card(
            "Classes Passed",
            (
                f"{passed:,}/{len(df):,}"
                if passed is not None
                else "Available"
            ),
            (
                "Final evidence result"
                if passed is not None
                else "Status field not present"
            ),
        )

    with c3:

        metric_card(
            "Evidence Type",
            "Engineering",
            "Similarity + consistency",
        )

    display_cols = [
        col
        for col in [
            "Class",
            "Quality Score",
            "Histogram Score",
            "Structural Score",
            "Robust Z",
            "Q95 Exceedance",
            "Status",
        ]
        if col in df.columns
    ]

    if display_cols:

        display_df = df[
            display_cols
        ].copy()

        for col in [
            "Quality Score",
            "Histogram Score",
            "Structural Score",
        ]:

            if col in display_df.columns:

                values = pd.to_numeric(
                    display_df[col],
                    errors="coerce",
                )

                if (
                    not values.dropna().empty
                    and values.dropna().max() <= 1
                ):

                    display_df[col] = (
                        values * 100
                    )

        st.dataframe(
            display_df.round(3),
            use_container_width=True,
            hide_index=True,
        )

    metric_cols = [
        col
        for col in [
            "Quality Score",
            "Histogram Score",
            "Structural Score",
        ]
        if col in df.columns
    ]

    if (
        metric_cols
        and "Class" in df.columns
    ):

        chart_df = df[
            ["Class"] + metric_cols
        ].melt(
            id_vars="Class",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score",
        )

        chart_df["Score"] = pd.to_numeric(
            chart_df["Score"],
            errors="coerce",
        )

        if (
            not chart_df.empty
            and not chart_df["Score"]
            .dropna()
            .empty
        ):

            if (
                chart_df["Score"]
                .dropna()
                .max()
                <= 1
            ):

                chart_df["Score"] = (
                    chart_df["Score"]
                    * 100
                )

            fig = px.bar(
                chart_df,
                x="Class",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title=(
                    "Final V5.6 Synthetic Quality "
                    "Evidence by Class"
                ),
                color_discrete_sequence=(
                    CATEGORICAL_SEQUENCE
                ),
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_yaxes(
                title="Evidence Score (%)",
                range=[0, 100],
            )

            fig = apply_chart_theme(
                fig,
                height=450,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="mri_final_quality_evidence_chart",
            )

    st.caption(
        f"Evidence source: `{evidence_run}`"
    )


# =========================================================
# FINAL DATASET COMPARISON
# =========================================================

def show_final_dataset_comparison():

    section_header(
        "🔄 Real vs Synthetic Dataset Comparison",
        "Final V5.6 production counts compared with the real MRI Testing set.",
    )

    real_counts = count_class_images(
        REAL_TEST
    )

    synthetic_counts, production_run = (
        get_final_production_counts()
    )

    rows = []

    for cls in CLASSES:

        real_count = real_counts.get(
            cls,
            0,
        )

        synthetic_count = synthetic_counts.get(
            cls,
            0,
        )

        expansion = (
            synthetic_count / real_count
            if real_count
            else 0
        )

        rows.append({
            "Class": CLASS_LABELS[cls],
            "Real Testing": real_count,
            "Synthetic Generated": synthetic_count,
            "Total": (
                real_count
                + synthetic_count
            ),
            "Expansion": (
                f"{expansion:.1f}×"
                if real_count
                else "—"
            ),
        })

    comparison_df = pd.DataFrame(
        rows
    )

    total_real = int(
        comparison_df[
            "Real Testing"
        ].sum()
    )

    total_synthetic = int(
        comparison_df[
            "Synthetic Generated"
        ].sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Real Samples",
            f"{total_real:,}",
            "Existing MRI Testing images",
        )

    with c2:

        metric_card(
            "Synthetic Samples",
            f"{total_synthetic:,}",
            "Final V5.6 generated images",
        )

    with c3:

        metric_card(
            "Combined Samples",
            f"{total_real + total_synthetic:,}",
            "Real + synthetic",
        )

    with c4:

        expansion = (
            total_synthetic / total_real
            if total_real
            else 0
        )

        metric_card(
            "Synthetic Expansion",
            f"{expansion:.1f}×",
            "Relative to real Testing",
        )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    chart_df = comparison_df.melt(
        id_vars="Class",
        value_vars=[
            "Real Testing",
            "Synthetic Generated",
        ],
        var_name="Dataset",
        value_name="Images",
    )

    fig = px.bar(
        chart_df,
        x="Class",
        y="Images",
        color="Dataset",
        barmode="group",
        text="Images",
        title="Real vs Synthetic MRI Images by Class",
        color_discrete_map=SERIES_COLOR_MAP,
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig = apply_chart_theme(
        fig,
        height=440,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mri_final_real_vs_synthetic",
    )

    if production_run is not None:

        st.caption(
            f"Production source: `{production_run}`"
        )


# =========================================================
# ZIP DOWNLOAD HELPERS
# =========================================================

def zip_image_dataset(
    root,
    include_real=False
):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for cls in CLASSES:

            for path in image_files(
                root / cls
            ):

                archive.write(
                    path,
                    arcname=(
                        f"synthetic/"
                        f"{cls}/"
                        f"{path.name}"
                    ),
                )

        if include_real:

            for cls in CLASSES:

                for path in image_files(
                    REAL_TEST / cls
                ):

                    archive.write(
                        path,
                        arcname=(
                            f"real/"
                            f"{cls}/"
                            f"{path.name}"
                        ),
                    )

    buffer.seek(0)

    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def get_synthetic_zip(
    root_string
):

    return zip_image_dataset(
        Path(root_string),
        include_real=False,
    )


@st.cache_data(show_spinner=False)
def get_real_and_synthetic_zip(
    root_string
):

    return zip_image_dataset(
        Path(root_string),
        include_real=True,
    )


# =========================================================
# FINAL DOWNLOADS
# =========================================================

def show_final_downloads():

    section_header(
        "⬇️ MRI Data & Quality Evidence Downloads",
        "Download the final V5.6 synthetic dataset, real + synthetic package, and validation evidence.",
    )

    production_run = (
        find_latest_final_production_run()
    )

    if production_run is None:

        st.warning(
            "Final V5.6 production images were not found."
        )

        return

    html_path, json_path, csv_path, _ = (
        get_final_evidence_files()
    )

    synthetic_root = production_run

    if sum(
        len(
            image_files(
                synthetic_root / cls
            )
        )
        for cls in CLASSES
    ) == 0:

        synthetic_root = (
            production_run
            / "synthetic"
        )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "**Synthetic dataset**"
        )

        st.caption(
            "Final V5.6 synthetic MRI images — 800 per class."
        )

        synthetic_data = (
            get_synthetic_zip(
                str(synthetic_root)
            )
        )

        st.download_button(
            "⬇️ Download Synthetic MRI Dataset",
            data=synthetic_data,
            file_name=(
                "mri_v5_6_synthetic_3200.zip"
            ),
            mime="application/zip",
            use_container_width=True,
            key="mri_download_synthetic_3200",
        )

    with c2:

        st.markdown(
            "**Real + synthetic dataset**"
        )

        st.caption(
            "Real MRI Testing images plus the final V5.6 synthetic dataset."
        )

        combined_data = (
            get_real_and_synthetic_zip(
                str(synthetic_root)
            )
        )

        st.download_button(
            "⬇️ Download Real + Synthetic MRI Dataset",
            data=combined_data,
            file_name=(
                "mri_real_1600_plus_synthetic_3200.zip"
            ),
            mime="application/zip",
            use_container_width=True,
            key="mri_download_real_synthetic_4800",
        )

    st.divider()

    section_header(
        "📄 Quality Evidence Files",
        "Final V5.6 engineering/ML quality evidence.",
    )

    evidence_cols = st.columns(3)

    with evidence_cols[0]:

        if html_path:

            st.download_button(
                "⬇️ HTML Report",
                data=html_path.read_bytes(),
                file_name=html_path.name,
                mime="text/html",
                use_container_width=True,
                key="mri_download_final_html",
            )

        else:

            st.caption(
                "HTML report unavailable."
            )

    with evidence_cols[1]:

        if json_path:

            st.download_button(
                "⬇️ JSON Report",
                data=json_path.read_bytes(),
                file_name=json_path.name,
                mime="application/json",
                use_container_width=True,
                key="mri_download_final_json",
            )

        else:

            st.caption(
                "JSON report unavailable."
            )

    with evidence_cols[2]:

        if csv_path:

            st.download_button(
                "⬇️ Class Summary CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
                key="mri_download_final_csv",
            )

        else:

            st.caption(
                "CSV report unavailable."
            )


# =========================================================
# MAIN DASHBOARD
# =========================================================

def show_mri_dashboard():

    inject_custom_css()

    # -----------------------------------------------------
    # PAGE HEADER
    # -----------------------------------------------------
    # KPI cards intentionally removed from here.
    # They now appear at the top of Overview.
    # -----------------------------------------------------

    page_header(
        "🧠 MRI Intelligence",
        "Brain MRI synthetic-data generation, quality evidence and real-vs-synthetic dataset analysis.",
    )

    st.divider()

    # -----------------------------------------------------
    # FOUR MAIN TABS
    # -----------------------------------------------------

    tabs = st.tabs([
        "📊 Overview",
        "🔄 Real vs Synthetic",
        "🧪 Quality Evidence",
        "⬇️ Downloads",
    ])

    # =====================================================
    # OVERVIEW
    # =====================================================

    with tabs[0]:

        # KPI CARDS MOVED HERE
        show_overview_kpis()

        st.divider()

        # -------------------------------------------------
        # REAL TESTING PREVIEW
        # -------------------------------------------------

        show_real_gallery()

        st.divider()

        # -------------------------------------------------
        # FINAL SYNTHETIC PREVIEW
        # -------------------------------------------------

        synthetic_root = (
            find_latest_final_production_run()
        )

        if synthetic_root is not None:

            if sum(
                len(
                    image_files(
                        synthetic_root / cls
                    )
                )
                for cls in CLASSES
            ) == 0:

                synthetic_root = (
                    synthetic_root
                    / "synthetic"
                )

            show_synthetic_gallery(
                synthetic_root
            )

        else:

            st.warning(
                "Final V5.6 production images were not found."
            )

    # =====================================================
    # REAL VS SYNTHETIC
    # =====================================================

    with tabs[1]:

        synthetic_root = (
            find_latest_final_production_run()
        )

        if synthetic_root is not None:

            if sum(
                len(
                    image_files(
                        synthetic_root / cls
                    )
                )
                for cls in CLASSES
            ) == 0:

                synthetic_root = (
                    synthetic_root
                    / "synthetic"
                )

            # -------------------------------------------------
            # DATASET COUNT COMPARISON
            # -------------------------------------------------

            show_final_dataset_comparison()

            st.divider()

            # -------------------------------------------------
            # IMAGE-LEVEL COMPARISON
            # -------------------------------------------------

            show_image_comparison(
                synthetic_root
            )

        else:

            st.warning(
                "Final V5.6 production images were not found."
            )

    # =====================================================
    # QUALITY EVIDENCE
    # =====================================================

    with tabs[2]:

        show_final_quality_evidence()

    # =====================================================
    # DOWNLOADS
    # =====================================================

    with tabs[3]:

        show_final_downloads()

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.divider()

    st.info(
        "⚠️ MRI quality metrics shown here are engineering and ML "
        "evaluation evidence. They do not establish medical realism, "
        "clinical validity or diagnostic safety."
    )


# =========================================================
# DIRECT RUN
# =========================================================

if __name__ == "__main__":

    st.set_page_config(
        page_title="MRI Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    show_mri_dashboard()

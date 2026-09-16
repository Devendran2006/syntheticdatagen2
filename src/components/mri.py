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
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# =========================================================
# THEME
# =========================================================

PRIMARY = "#1F2A44"
SOFT_BLUE = "#6B8FB3"
SOFT_TEAL = "#76A7A3"
SOFT_GREEN = "#88A98F"
SOFT_GOLD = "#C7A76B"
SOFT_RED = "#C98585"
SOFT_PURPLE = "#9A8FB7"
LIGHT_BG = "#F7F9FC"
LIGHT_GRID = "#E8ECF2"

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
    "glioma": SOFT_RED,
    "meningioma": SOFT_BLUE,
    "notumor": SOFT_GREEN,
    "pituitary": SOFT_GOLD,
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


# =========================================================
# HELPERS
# =========================================================

def section_header(title, description=None):
    desc = ""
    if description:
        desc = (
            f"<div style='color:#667085;font-size:13px;"
            f"margin-top:4px;'>{description}</div>"
        )

    st.markdown(
        f"""
        <div style="
            padding:4px 0 10px 0;
            border-bottom:1px solid #E8ECF2;
            margin-bottom:16px;
        ">
            <div style="
                color:#1F2A44;
                font-size:21px;
                font-weight:700;
            ">
                {title}
            </div>
            {desc}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption=None):
    st.metric(label, value)
    if caption:
        st.caption(caption)


def apply_chart_theme(fig, height=400, legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color=PRIMARY,
            size=13,
        ),
        title=dict(
            font=dict(
                color=PRIMARY,
                size=18,
            ),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(
            l=25,
            r=25,
            t=70,
            b=35,
        ),
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_color=PRIMARY,
        ),
    )

    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
            )
        )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=LIGHT_GRID,
        tickfont=dict(color=PRIMARY),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=LIGHT_GRID,
        gridwidth=1,
        zeroline=False,
        tickfont=dict(color=PRIMARY),
    )

    return fig


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
        files.extend(
            folder.rglob(pattern)
        )

    return sorted(set(files))


def class_image_files(root):
    result = {}

    for cls in CLASSES:
        folder = root / cls
        result[cls] = image_files(folder)

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

    return sorted(
        set(candidates)
    )


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
# DATASET OVERVIEW
# =========================================================

def show_dataset_overview():

    section_header(
        "📊 MRI Dataset Overview",
        "Separate real training, real testing and generated synthetic data.",
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
        color_discrete_sequence=[
            PRIMARY,
            SOFT_BLUE,
            SOFT_GOLD,
        ],
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


def show_version_selector(key="mri_synthetic_version_selector"):

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
# REAL MRI GALLERY
# =========================================================

def show_real_gallery():

    section_header(
        "🖼️ Real MRI Gallery",
        "Browse the real MRI training/testing images class by class.",
    )

    source = st.radio(
        "Image source",
        [
            "Training",
            "Testing",
        ],
        horizontal=True,
        key="mri_real_gallery_source",
    )

    root = (
        REAL_TRAIN
        if source == "Training"
        else REAL_TEST
    )

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
            f"No {selected_class} images found."
        )
        return

    max_show = min(
        len(files),
        16,
    )

    sample_count = st.slider(
        "Images to display",
        min_value=4,
        max_value=max_show,
        value=min(8, max_show),
        step=4 if max_show >= 8 else 1,
        key="mri_real_gallery_count",
    )

    selected_files = files[:sample_count]

    cols = st.columns(4)

    for index, path in enumerate(
        selected_files
    ):
        with cols[index % 4]:
            img = load_image(path)

            if img is not None:
                st.image(
                    img,
                    caption=path.name,
                    use_container_width=True,
                )


# =========================================================
# SYNTHETIC GALLERY
# =========================================================

def show_synthetic_gallery(root):

    section_header(
        "✨ Synthetic MRI Gallery",
        "Inspect generated images independently for each brain MRI class.",
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
        16,
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

    for index, path in enumerate(
        selected_files
    ):
        with cols[index % 4]:
            img = load_image(path)

            if img is not None:
                st.image(
                    img,
                    caption=path.name,
                    use_container_width=True,
                )

    st.caption(
        f"{len(files):,} synthetic {CLASS_LABELS[selected_class]} images available."
    )


# =========================================================
# SIDE BY SIDE COMPARISON
# =========================================================

def show_image_comparison(root):

    section_header(
        "🔄 Real vs Synthetic MRI",
        "Compare images from the same class side by side.",
    )

    selected_class = st.selectbox(
        "Comparison class",
        CLASSES,
        format_func=lambda x: CLASS_LABELS[x],
        key="mri_side_compare_class",
    )

    real_files = image_files(
        REAL_TRAIN / selected_class
    )

    synthetic_files = image_files(
        root / selected_class
    )

    if not real_files:
        st.warning(
            "Real MRI images are unavailable for this class."
        )
        return

    if not synthetic_files:
        st.warning(
            "Synthetic MRI images are unavailable for this class."
        )
        return

    index = st.slider(
        "Image pair",
        min_value=0,
        max_value=min(
            len(real_files),
            len(synthetic_files),
        ) - 1,
        value=0,
        key="mri_pair_index",
    )

    left, right = st.columns(2)

    with left:
        real_img = load_image(
            real_files[index]
        )

        if real_img is not None:
            st.image(
                real_img,
                caption=f"REAL — {real_files[index].name}",
                use_container_width=True,
            )

    with right:
        synthetic_img = load_image(
            synthetic_files[index]
        )

        if synthetic_img is not None:
            st.image(
                synthetic_img,
                caption=(
                    f"SYNTHETIC — "
                    f"{synthetic_files[index].name}"
                ),
                use_container_width=True,
            )


# =========================================================
# PIXEL ANALYSIS
# =========================================================

def calculate_image_statistics(files, limit=200):

    means = []
    stds = []
    brightness = []
    sharpness = []

    for path in files[:limit]:

        img = load_image(path)

        if img is None:
            continue

        arr = np.asarray(
            img,
            dtype=np.float32,
        ) / 255.0

        means.append(
            float(arr.mean())
        )

        stds.append(
            float(arr.std())
        )

        brightness.append(
            float(arr.mean())
        )

        # Simple gradient-based sharpness proxy.
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
        color_discrete_map={
            "Real": PRIMARY,
            "Synthetic": SOFT_GOLD,
        },
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

    # Detect class column.
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
            color_discrete_sequence=[
                SOFT_BLUE,
                SOFT_TEAL,
                SOFT_GOLD,
                SOFT_PURPLE,
                SOFT_GREEN,
            ],
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
                "Accuracy": accuracy * 100
                if accuracy <= 1
                else accuracy,
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
        results["Experiment"].astype(str).str.len() > 0
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
            color_discrete_sequence=[
                SOFT_BLUE,
                SOFT_TEAL,
                SOFT_GOLD,
                SOFT_PURPLE,
            ],
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

    df = detailed if detailed is not None else summary

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
        ] + [
            str(x)
            for x in working[class_col]
            .dropna()
            .unique()
        ],
        key="mri_quality_class_selector",
    )

    if selected_class != "All Classes":
        selected_df = working[
            working[class_col].astype(str)
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
    )

    fig.update_traces(
        marker_color=SOFT_BLUE,
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
# MAIN DASHBOARD
# =========================================================

def show_mri_dashboard():

    st.title(
        "🧠 MRI Intelligence"
    )

    st.caption(
        "Brain MRI synthetic-data generation, "
        "visual inspection, quality analysis and ML utility."
    )

    # -----------------------------------------------------
    # HEADER KPIs
    # -----------------------------------------------------

    (
        train_counts,
        preprocessed_counts,
        test_counts,
        synthetic_counts,
    ) = get_dataset_counts()

    train_total = sum(
        train_counts.values()
    )

    test_total = sum(
        test_counts.values()
    )

    preprocessed_total = sum(
        preprocessed_counts.values()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "MRI Classes",
            "4",
            "Glioma • Meningioma • No Tumor • Pituitary",
        )

    with c2:
        metric_card(
            "Real Training",
            f"{train_total:,}",
            "GAN training data",
        )

    with c3:
        metric_card(
            "Real Testing",
            f"{test_total:,}",
            "Evaluation only",
        )

    with c4:
        metric_card(
            "128×128",
            "Grayscale",
            "Preprocessed MRI format",
        )

    st.divider()

    # -----------------------------------------------------
    # NAVIGATION
    # -----------------------------------------------------

    tabs = st.tabs([
        "📊 Overview",
        "🖼️ Real MRI",
        "✨ Synthetic MRI",
        "🔄 Real vs Synthetic",
        "🔬 Image Analysis",
        "📈 GAN Quality",
        "🤖 ML Utility",
        "🧩 Class Quality",
        "⬇️ Outputs",
    ])

    with tabs[0]:
        show_dataset_overview()

    with tabs[1]:
        show_real_gallery()

    with tabs[2]:
        root = show_version_selector("mri_synthetic_version_gallery")

        if root is not None:
            show_synthetic_gallery(
                root
            )

    with tabs[3]:
        root = show_version_selector("mri_synthetic_version_comparison")

        if root is not None:
            show_image_comparison(
                root
            )

    with tabs[4]:
        root = show_version_selector("mri_synthetic_version_analysis")

        if root is not None:
            show_pixel_analysis(
                root
            )

    with tabs[5]:
        show_v4_quality()

    with tabs[6]:
        show_ml_validation()

    with tabs[7]:
        show_class_quality()

    with tabs[8]:
        show_downloads()

    st.divider()

    st.info(
        "⚠️ MRI quality metrics shown here are engineering and ML "
        "evaluation metrics. They do not establish medical realism, "
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

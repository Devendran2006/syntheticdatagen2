"""
MRI TESTING V2 - GENERATION RECOVERY

PURPOSE
-------
Generate synthetic MRI images from an ALREADY TRAINED
VQ-VAE + Conditional Latent Diffusion V2 model.

IMPORTANT
---------
This script DOES NOT train anything.

It loads:

    outputs/mri_testing_vq_diffusion_v2/
        run_20260918_062424/
            models/
                vqvae_final.pt
                diffusion_final.pt
                latent_statistics.pt

CURRENT CHECKPOINT
------------------
The checkpoint was trained in PILOT_MODE on GLIOMA only.

Therefore this recovery script generates GLIOMA only.

TARGET
------
800 synthetic Glioma MRI images.

OUTPUT
------
A NEW recovery folder is created.

No old checkpoint is modified.
No Testing images are modified.
No Training images are read.
No old synthetic outputs are modified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is:
# E:\A1593 DA python live\synthetic data\src\mri\
#
# parents[2] =
# E:\A1593 DA python live\synthetic data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# EXACT TRAINED RUN
# ------------------------------------------------------------

TRAINED_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_vq_diffusion_v2"
    / "run_20260918_062424"
)

MODEL_ROOT = TRAINED_RUN / "models"

VQVAE_CHECKPOINT = MODEL_ROOT / "vqvae_final.pt"
DIFFUSION_CHECKPOINT = MODEL_ROOT / "diffusion_final.pt"
LATENT_STATS = MODEL_ROOT / "latent_statistics.pt"


# ============================================================
# NEW OUTPUT DIRECTORY
# ============================================================

RECOVERY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mri_testing_vq_diffusion_v2"
    / time.strftime(
        "generation_recovery_%Y%m%d_%H%M%S"
    )
)

SYNTH_ROOT = RECOVERY_ROOT / "synthetic"
PREVIEW_ROOT = RECOVERY_ROOT / "previews"
REPORT_ROOT = RECOVERY_ROOT / "reports"


# ============================================================
# GENERATION CONFIG
# ============================================================

CLASS_NAME = "glioma"

# The checkpoint was trained only on Glioma.
CLASS_ID = 0

SYNTHETIC_PER_CLASS = 800

GEN_BATCH_SIZE = 32

IMAGE_SIZE = 128

LATENT_CHANNELS = 8
LATENT_SIZE = 16

DIFFUSION_STEPS = 1000
DDIM_STEPS = 50

SEED = 42


# ============================================================
# IMPORT THE EXACT V2 ARCHITECTURE
# ============================================================

# We import the model definitions from the original V2 file.
#
# IMPORTANT:
# We do NOT call its main().
#
# Therefore:
# - VQ-VAE is NOT trained
# - Diffusion is NOT trained
# - No generation from the original script occurs
#
# We only reuse the exact architecture/classes.

SOURCE_DIR = Path(__file__).resolve().parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


try:
    import mri_testing_vq_diffusion_v2 as v2

except Exception as exc:
    raise RuntimeError(
        "\nCould not import the original V2 architecture.\n"
        f"Expected source file:\n{SOURCE_DIR / 'mri_testing_vq_diffusion_v2.py'}\n\n"
        f"Original error:\n{exc}"
    ) from exc


# Exact architecture classes
VQVAE = v2.VQVAE
LatentDiffusionUNet = v2.LatentDiffusionUNet
EMA = v2.EMA


# Exact diffusion schedule
ALPHA_BAR = v2.ALPHA_BAR


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = SEED):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

SYNTH_DIR = (
    SYNTH_ROOT
    / CLASS_NAME
)

PREVIEW_DIR = (
    PREVIEW_ROOT
    / CLASS_NAME
)

SYNTH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print("\n" + "=" * 78)
    print("CHECKING TRAINED V2 CHECKPOINT")
    print("=" * 78)

    print(
        f"Trained run : {TRAINED_RUN}"
    )

    print(
        f"VQ-VAE      : {VQVAE_CHECKPOINT}"
    )

    print(
        f"Diffusion   : {DIFFUSION_CHECKPOINT}"
    )

    print(
        f"Latent stats: {LATENT_STATS}"
    )

    required = [
        VQVAE_CHECKPOINT,
        DIFFUSION_CHECKPOINT,
        LATENT_STATS,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:

        print("\nMISSING FILES:")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "\nThe trained V2 checkpoint is incomplete."
        )

    print("\nAll required checkpoint files found.")


# ============================================================
# LOAD VQ-VAE
# ============================================================

def load_vqvae():

    print("\n" + "=" * 78)
    print("LOADING VQ-VAE")
    print("=" * 78)

    checkpoint = torch.load(
        VQVAE_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = VQVAE().to(DEVICE)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    print("VQ-VAE loaded successfully.")

    print(
        f"Device: {DEVICE}"
    )

    return model


# ============================================================
# LOAD DIFFUSION + EMA
# ============================================================

def load_diffusion():

    print("\n" + "=" * 78)
    print("LOADING LATENT DIFFUSION")
    print("=" * 78)

    checkpoint = torch.load(
        DIFFUSION_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    diffusion = (
        LatentDiffusionUNet()
        .to(DEVICE)
    )

    diffusion.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    # --------------------------------------------------------
    # Load EMA weights
    # --------------------------------------------------------

    ema = EMA(diffusion)

    if "ema" not in checkpoint:

        raise RuntimeError(
            "EMA weights are missing from "
            "diffusion_final.pt"
        )

    ema.load_state_dict(
        checkpoint["ema"]
    )

    # Use EMA weights for generation.
    ema.apply(diffusion)

    diffusion.eval()

    print(
        "Diffusion model loaded successfully."
    )

    print(
        "EMA weights applied."
    )

    return diffusion


# ============================================================
# LOAD LATENT STATISTICS
# ============================================================

def load_latent_statistics():

    print("\n" + "=" * 78)
    print("LOADING LATENT STATISTICS")
    print("=" * 78)

    stats = torch.load(
        LATENT_STATS,
        map_location=DEVICE,
        weights_only=False,
    )

    latent_mean = stats["mean"].to(
        DEVICE
    )

    latent_std = stats["std"].to(
        DEVICE
    )

    # Safety protection against zero std.
    latent_std = torch.clamp(
        latent_std,
        min=1e-6,
    )

    stored_shape = stats.get(
        "shape",
        None,
    )

    print(
        f"Stored latent shape: {stored_shape}"
    )

    print(
        f"Mean shape: {tuple(latent_mean.shape)}"
    )

    print(
        f"Std shape : {tuple(latent_std.shape)}"
    )

    return (
        latent_mean,
        latent_std,
    )


# ============================================================
# DDIM SAMPLER
# ============================================================

@torch.no_grad()
def ddim_sample_recovery(
    model,
    labels,
    latent_mean,
    latent_std,
):

    model.eval()

    batch = labels.shape[0]

    # --------------------------------------------------------
    # Start from Gaussian noise
    # --------------------------------------------------------

    x = torch.randn(
        batch,
        LATENT_CHANNELS,
        LATENT_SIZE,
        LATENT_SIZE,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Same 1000 -> 0 schedule used by V2
    # --------------------------------------------------------

    times = torch.linspace(
        DIFFUSION_STEPS - 1,
        0,
        DDIM_STEPS,
        device=DEVICE,
    ).long()

    # --------------------------------------------------------
    # DDIM reverse process
    # --------------------------------------------------------

    for i, tv in enumerate(times):

        t = torch.full(
            (batch,),
            int(tv.item()),
            device=DEVICE,
            dtype=torch.long,
        )

        eps = model(
            x,
            t,
            labels,
        )

        a_t = ALPHA_BAR[t].view(
            -1,
            1,
            1,
            1,
        )

        if i + 1 < len(times):

            prev_t = torch.full(
                (batch,),
                int(
                    times[i + 1].item()
                ),
                device=DEVICE,
                dtype=torch.long,
            )

            a_prev = ALPHA_BAR[
                prev_t
            ].view(
                -1,
                1,
                1,
                1,
            )

        else:

            a_prev = torch.ones_like(
                a_t
            )

        # ----------------------------------------------------
        # Estimate clean latent
        # ----------------------------------------------------

        x0 = (
            x
            - torch.sqrt(1 - a_t) * eps
        ) / torch.sqrt(a_t)

        # ----------------------------------------------------
        # Deterministic DDIM update
        # ----------------------------------------------------

        x = (
            torch.sqrt(a_prev) * x0
            + torch.sqrt(1 - a_prev) * eps
        )

    # --------------------------------------------------------
    # Undo latent normalization
    # --------------------------------------------------------

    x = (
        x * latent_std
        + latent_mean
    )

    return x


# ============================================================
# SAFE TENSOR -> PIL IMAGE
# ============================================================

def tensor_to_image(tensor):

    """
    Converts:

        [1, 1, 128, 128]

    or:

        [1, 128, 128]

    into:

        PIL grayscale 128x128

    IMPORTANT
    ---------
    The original V2 code had:

        numpy()[0]

    which produced:

        [1, 128, 128]

    for a [1,1,128,128] tensor.

    This recovery function explicitly selects:

        [0, 0]

    so the result is:

        [128,128]
    """

    x = tensor.detach()

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    if x.ndim == 4:

        # [B,C,H,W]
        x = x[0, 0]

    elif x.ndim == 3:

        # [C,H,W]
        x = x[0]

    elif x.ndim == 2:

        # [H,W]
        pass

    else:

        raise ValueError(
            f"Unexpected image tensor shape: {tuple(x.shape)}"
        )

    # --------------------------------------------------------
    # Model output is [-1, 1]
    # Convert to [0, 1]
    # --------------------------------------------------------

    x = (
        (x + 1.0) / 2.0
    ).clamp(
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------

    arr = (
        x.cpu()
        .numpy()
        * 255.0
    ).round().astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if arr.ndim != 2:

        raise ValueError(
            "Image conversion failed. "
            f"Final array shape: {arr.shape}"
        )

    return Image.fromarray(
        arr,
        mode="L",
    )


# ============================================================
# PREVIEW IMAGE
# ============================================================

def create_preview(image):

    """
    Creates a lightly sharpened preview.

    RAW synthetic images remain unchanged.
    """

    preview = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=60,
            threshold=3,
        )
    )

    return preview


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    image_paths,
    output_path,
    title="Synthetic Glioma Preview",
):

    if not image_paths:
        return

    # Use at most 64 images for the contact sheet.
    selected = image_paths[:64]

    thumb_size = 128

    columns = 8

    rows = int(
        np.ceil(
            len(selected) / columns
        )
    )

    title_height = 50

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_size,
            rows * thumb_size + title_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    draw.text(
        (10, 15),
        title,
        fill="black",
    )

    for index, path in enumerate(
        selected
    ):

        image = Image.open(
            path
        ).convert("L")

        image = ImageOps.fit(
            image,
            (
                thumb_size,
                thumb_size,
            ),
        )

        image_rgb = image.convert(
            "RGB"
        )

        x = (
            index % columns
        ) * thumb_size

        y = (
            index // columns
        ) * thumb_size + title_height

        sheet.paste(
            image_rgb,
            (x, y),
        )

    sheet.save(
        output_path
    )

    print(
        f"Contact sheet saved: {output_path}"
    )


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_glioma(
    vqvae,
    diffusion,
    latent_mean,
    latent_std,
):

    print("\n" + "=" * 78)
    print("GENERATING SYNTHETIC GLIOMA MRI")
    print("=" * 78)

    print(
        f"Class             : {CLASS_NAME}"
    )

    print(
        f"Class ID          : {CLASS_ID}"
    )

    print(
        f"Target images     : {SYNTHETIC_PER_CLASS}"
    )

    print(
        f"Generation batch  : {GEN_BATCH_SIZE}"
    )

    print(
        f"Image size        : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Latent            : "
        f"{LATENT_CHANNELS}x"
        f"{LATENT_SIZE}x"
        f"{LATENT_SIZE}"
    )

    print(
        f"DDIM steps        : {DDIM_STEPS}"
    )

    print(
        f"Output            : {SYNTH_DIR}"
    )

    print("=" * 78)

    start_time = time.time()

    generated = 0

    saved_paths = []

    while generated < SYNTHETIC_PER_CLASS:

        current_batch = min(
            GEN_BATCH_SIZE,
            SYNTHETIC_PER_CLASS - generated,
        )

        # ----------------------------------------------------
        # Class labels
        # ----------------------------------------------------

        labels = torch.full(
            (current_batch,),
            CLASS_ID,
            device=DEVICE,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Sample latent
        # ----------------------------------------------------

        z = ddim_sample_recovery(
            diffusion,
            labels,
            latent_mean,
            latent_std,
        )

        # ----------------------------------------------------
        # Decode latent -> image
        # ----------------------------------------------------

        images = vqvae.decode(
            z
        )

        # ----------------------------------------------------
        # Save every image
        # ----------------------------------------------------

        for j in range(
            current_batch
        ):

            image = tensor_to_image(
                images[j:j + 1]
            )

            index = (
                generated
                + j
                + 1
            )

            output_path = (
                SYNTH_DIR
                / f"synthetic_{index:04d}.png"
            )

            image.save(
                output_path
            )

            saved_paths.append(
                output_path
            )

        generated += current_batch

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            generated / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            SYNTHETIC_PER_CLASS
            - generated
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"Generated "
            f"{generated:4d}/"
            f"{SYNTHETIC_PER_CLASS} "
            f"| {rate:.2f} img/s "
            f"| ETA {eta:.1f}s"
        )

    total_time = (
        time.time()
        - start_time
    )

    print("\nGeneration complete.")

    print(
        f"Images generated : {generated}"
    )

    print(
        f"Time             : {total_time:.2f}s"
    )

    print(
        f"Output folder    : {SYNTH_DIR}"
    )

    return saved_paths, total_time


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    generated_count,
    generation_seconds,
):

    report = {

        "experiment": (
            "MRI Testing V2 "
            "Generation Recovery"
        ),

        "trained_run": str(
            TRAINED_RUN
        ),

        "recovery_output": str(
            RECOVERY_ROOT
        ),

        "class": CLASS_NAME,

        "class_id": CLASS_ID,

        "synthetic_count": (
            generated_count
        ),

        "image_size": [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],

        "latent_channels": (
            LATENT_CHANNELS
        ),

        "latent_size": (
            LATENT_SIZE
        ),

        "diffusion_steps": (
            DIFFUSION_STEPS
        ),

        "ddim_steps": (
            DDIM_STEPS
        ),

        "generation_batch_size": (
            GEN_BATCH_SIZE
        ),

        "generation_seconds": (
            generation_seconds
        ),

        "device": str(
            DEVICE
        ),

        "training_performed": False,

        "vqvae_retrained": False,

        "diffusion_retrained": False,

        "testing_data_modified": False,

        "training_data_modified": False,

        "previous_outputs_modified": False,

        "ema_used": True,

        "tensor_conversion_fix": (
            "[0,0] grayscale extraction"
        ),
    }

    report_path = (
        REPORT_ROOT
        / "generation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"Report saved: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 78)
    print("MRI TESTING V2 - GENERATION RECOVERY")
    print("RESIDUAL VQ-VAE + CONDITIONAL LATENT DIFFUSION")
    print("=" * 78)

    print(
        "\nTHIS SCRIPT DOES NOT TRAIN."
    )

    print(
        "It loads the completed V2 checkpoint "
        "and generates synthetic Glioma images."
    )

    # --------------------------------------------------------
    # Checkpoint validation
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Load VQ-VAE
    # --------------------------------------------------------

    vqvae = load_vqvae()

    # --------------------------------------------------------
    # Load diffusion
    # --------------------------------------------------------

    diffusion = load_diffusion()

    # --------------------------------------------------------
    # Load latent statistics
    # --------------------------------------------------------

    (
        latent_mean,
        latent_std,
    ) = load_latent_statistics()

    # --------------------------------------------------------
    # Generate 800 Glioma images
    # --------------------------------------------------------

    (
        saved_paths,
        generation_seconds,
    ) = generate_glioma(
        vqvae,
        diffusion,
        latent_mean,
        latent_std,
    )

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    contact_sheet_path = (
        PREVIEW_ROOT
        / "glioma_contact_sheet.png"
    )

    create_contact_sheet(
        saved_paths,
        contact_sheet_path,
        title=(
            "V2 Synthetic Glioma "
            "(800 generated)"
        ),
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        len(saved_paths),
        generation_seconds,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("GENERATION RECOVERY COMPLETE")
    print("=" * 78)

    print(
        f"\nClass       : {CLASS_NAME}"
    )

    print(
        f"Generated   : {len(saved_paths)}"
    )

    print(
        f"Synthetic   : {SYNTH_DIR}"
    )

    print(
        f"Preview     : {contact_sheet_path}"
    )

    print(
        f"Report      : "
        f"{REPORT_ROOT / 'generation_report.json'}"
    )

    print("\nPROTECTION STATUS")
    print("-" * 78)
    print("Training data       : NOT MODIFIED")
    print("Testing data        : NOT MODIFIED")
    print("Old V2 checkpoint   : NOT MODIFIED")
    print("Old V2 outputs      : NOT MODIFIED")
    print("Training performed  : NO")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
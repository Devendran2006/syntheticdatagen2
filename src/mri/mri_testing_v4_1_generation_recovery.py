"""
MRI TESTING V4.1 — GENERATION-ONLY RECOVERY
Uses the already-trained V4.1 checkpoint. NO training.
Generates only 16 images/class for visual validation.
"""

from pathlib import Path
import sys, json, math, time
import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINED_RUN = PROJECT_ROOT / "outputs/mri_testing_v4_1_quality_diffusion/pilot_20261001_061420"
CHECKPOINT = TRAINED_RUN / "models/v4_1_best_ema_diffusion.pt"

CACHE_RUN = PROJECT_ROOT / "outputs/mri_testing_v4_latents/run_20261001_044203"
LATENT_FILE = CACHE_RUN / "latents.pt"
STATS_FILE = CACHE_RUN / "latent_statistics.pt"

V3_CHECKPOINT = (
    PROJECT_ROOT / "outputs/mri_testing_v3_reconstruction/"
    "run_20260922_044321/models/best_autoencoder.pt"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs/mri_testing_v4_1_quality_diffusion"
RUN = OUTPUT_ROOT / f"generation_recovery_{time.strftime('%Y%m%d_%H%M%S')}"
SYNTH_DIR = RUN / "synthetic"
PREVIEW_DIR = RUN / "previews"
REPORT_DIR = RUN / "reports"

for d in (SYNTH_DIR, PREVIEW_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
PILOT_PER_CLASS = 16
DDIM_STEPS = 75
CFG_SCALE = 2.5
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_MRI = PROJECT_ROOT / "src" / "mri"
sys.path.insert(0, str(SRC_MRI))

try:
    import mri_testing_v4_1_quality_diffusion as v41
except Exception as exc:
    raise RuntimeError(
        "Could not import mri_testing_v4_1_quality_diffusion.py. "
        f"Original error: {exc}"
    )


def check_required_files():
    print("\nChecking required files...")
    for path in [CHECKPOINT, LATENT_FILE, STATS_FILE, V3_CHECKPOINT]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")
        print(f"  OK  {path}")


def tensor_to_pil(tensor):
    array = (((tensor + 1.0) / 2.0).clamp(0, 1)
             .detach().cpu().numpy())
    if array.ndim == 4:
        array = array[0, 0]
    elif array.ndim == 3:
        array = array[0]
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_contact_sheet(images, class_name):
    columns = 4
    rows = math.ceil(len(images) / columns)
    label_height = 24
    sheet = Image.new(
        "L",
        (columns * IMAGE_SIZE, rows * (IMAGE_SIZE + label_height)),
        255,
    )
    draw = ImageDraw.Draw(sheet)

    for index, image in enumerate(images):
        x = (index % columns) * IMAGE_SIZE
        y = (index // columns) * (IMAGE_SIZE + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + IMAGE_SIZE + 4),
                  f"{class_name} {index + 1:02d}", fill=0)

    path = PREVIEW_DIR / f"{class_name}_v4_1_recovery_contact_sheet.png"
    sheet.save(path)
    return path


def main():
    print("\n" + "#" * 80)
    print("# MRI TESTING V4.1 — GENERATION-ONLY RECOVERY")
    print("#" * 80)
    print(f"\nDevice       : {DEVICE}")
    print(f"Training run : {TRAINED_RUN}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Output run   : {RUN}")
    print(f"Pilot        : {PILOT_PER_CLASS} images/class "
          f"({PILOT_PER_CLASS * len(CLASSES)} total)")

    check_required_files()

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    print("\nCheckpoint loaded.")
    print(f"  Best validation loss : {checkpoint.get('best_val_loss')}")
    print(f"  Training epochs      : {checkpoint.get('epochs')}")
    print(f"  Checkpoint CFG       : {checkpoint.get('cfg_scale')}")
    print(f"  Checkpoint DDIM      : {checkpoint.get('ddim_steps')}")

    latent_shape = checkpoint.get("latent_shape", [16, 32, 32])

    model = v41.QualityConditionalUNet(
        latent_channels=latent_shape[0],
        num_classes=len(CLASSES),
        base=checkpoint.get("base_channels", 80),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters     : {parameter_count / 1e6:.2f}M")
    print("  Model weights        : loaded successfully")

    stats = torch.load(STATS_FILE, map_location="cpu")
    mean = stats["mean"].float()
    std = stats["std"].float().clamp_min(1e-5)

    payload = torch.load(LATENT_FILE, map_location="cpu")
    real_latents = payload["latents"].float()
    real_labels = payload["labels"].long()

    normalized_real = ((real_latents - mean) / std).clamp(-5.0, 5.0)

    _, _, alpha_bars = v41.cosine_alpha_bars(
        v41.DIFFUSION_STEPS, DEVICE
    )

    decoder = v41.load_v3_decoder()

    class_results = {}
    all_generated = []
    generation_start = time.time()

    print("\n" + "=" * 80)
    print("GENERATING V4.1 PILOT")
    print("=" * 80)

    for class_name in CLASSES:
        class_id = CLASS_TO_ID[class_name]
        labels = torch.full(
            (PILOT_PER_CLASS,),
            class_id,
            device=DEVICE,
            dtype=torch.long,
        )

        class_start = time.time()

        generated_normalized = v41.ddim_sample(
            model=model,
            labels=labels,
            latent_shape=tuple(latent_shape),
            alpha_bars=alpha_bars,
            steps=DDIM_STEPS,
            guidance_scale=CFG_SCALE,
        )

        generated_cpu = generated_normalized.detach().cpu()
        real_class = normalized_real[real_labels == class_id]

        latent_metrics = v41.latent_global_metrics(
            generated_cpu, real_class
        )
        nearest_metrics = v41.nearest_latent_distance(
            generated_cpu, real_class
        )

        generated_latents = (
            generated_normalized * std.to(DEVICE) + mean.to(DEVICE)
        )

        with torch.no_grad():
            decoded = decoder.decode(generated_latents)

        class_dir = SYNTH_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for index in range(decoded.shape[0]):
            image = tensor_to_pil(decoded[index:index + 1])
            images.append(image)
            image.save(
                class_dir / (
                    f"{class_name}_synthetic_{index + 1:04d}.png"
                )
            )

        contact_sheet = save_contact_sheet(images, class_name)
        elapsed = time.time() - class_start

        class_results[class_name] = {
            "count": PILOT_PER_CLASS,
            "seconds": elapsed,
            "contact_sheet": str(contact_sheet),
            "latent_metrics": latent_metrics,
            "nearest_metrics": nearest_metrics,
        }
        all_generated.append(generated_cpu)

        print(f"\n{class_name:12s} | {PILOT_PER_CLASS:02d} images | "
              f"{elapsed:.1f}s")
        print(f"  Mean : {latent_metrics['generated_mean']:.4f} "
              f"vs real {latent_metrics['real_mean']:.4f}")
        print(f"  Std  : {latent_metrics['generated_std']:.4f} "
              f"vs real {latent_metrics['real_std']:.4f}")
        print(f"  Nearest latent distance: "
              f"{nearest_metrics['mean_nearest_distance']:.4f}")
        print(f"  Contact sheet: {contact_sheet}")

    generated_all = torch.cat(all_generated, dim=0)
    global_metrics = v41.latent_global_metrics(
        generated_all, normalized_real
    )
    global_nearest = v41.nearest_latent_distance(
        generated_all, normalized_real
    )

    report = {
        "mode": "generation_only_recovery",
        "source_training_run": str(TRAINED_RUN),
        "checkpoint": str(CHECKPOINT),
        "cache_run": str(CACHE_RUN),
        "v3_checkpoint": str(V3_CHECKPOINT),
        "device": str(DEVICE),
        "classes": CLASSES,
        "images_per_class": PILOT_PER_CLASS,
        "total_images": PILOT_PER_CLASS * len(CLASSES),
        "ddim_steps": DDIM_STEPS,
        "cfg_scale": CFG_SCALE,
        "latent_shape": latent_shape,
        "global_latent_metrics": global_metrics,
        "global_nearest_metrics": global_nearest,
        "class_results": class_results,
        "training_performed": False,
        "testing_data_modified": False,
        "v3_checkpoint_modified": False,
        "v4_cache_modified": False,
        "v4_1_checkpoint_modified": False,
        "production_800_per_class": False,
        "total_generation_seconds": time.time() - generation_start,
    }

    report_path = REPORT_DIR / "v4_1_generation_recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "#" * 80)
    print("# V4.1 GENERATION RECOVERY COMPLETE")
    print("#" * 80)
    print(f"\nSynthetic output : {SYNTH_DIR}")
    print(f"Contact sheets   : {PREVIEW_DIR}")
    print(f"Report           : {report_path}")
    print("\nTraining performed : NO")
    print("800/class generated: NO")
    print("\nInspect all four contact sheets before production generation.")


if __name__ == "__main__":
    main()

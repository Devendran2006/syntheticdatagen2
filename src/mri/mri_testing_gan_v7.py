import os
import time
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================================================================
# MRI TESTING GAN V7
# ================================================================
# Purpose:
#   Train a GAN using ONLY the MRI Testing dataset.
#   Existing Training data and previous GAN outputs are untouched.
#
# Main V7 idea:
#   128x128 MRI -> quality-preserving preprocessing -> autoencoder
#   -> latent-space LSGAN -> decoder -> synthetic 128x128 MRI.
#
# V7 is deliberately conservative on CPU. Start with one class:
#     PILOT_CLASSES = ["glioma"]
# After checking quality, set PILOT_CLASSES = None for all classes.
# ================================================================

# ----------------------------- PATHS -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTING_ROOT = PROJECT_ROOT / "data" / "imaging" / "MRI" / "Testing"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mri_testing_gan_v7"

# --------------------------- CONFIG ------------------------------
SEED = 42
IMAGE_SIZE = 128
CHANNELS = 1
BATCH_SIZE = 16
NUM_WORKERS = 0

# CPU-friendly defaults. Increase only after visual quality is confirmed.
AE_EPOCHS = 8
GAN_EPOCHS = 12
LATENT_CHANNELS = 128
NOISE_DIM = 128
LEARNING_RATE_AE = 2e-4
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
BETAS = (0.5, 0.999)
LAMBDA_EDGE = 0.15
LAMBDA_LATENT = 0.05

SYNTHETIC_PER_CLASS = 100
PREVIEW_EVERY = 2
CHECKPOINT_EVERY = 4

# IMPORTANT: run Glioma first. Set to None only after the pilot looks good.
PILOT_CLASSES = ["glioma"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- REPRODUCIBILITY -----------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything()

# ----------------------------- DIRS ------------------------------
DIRS = {
    "models": OUTPUT_ROOT / "models",
    "ae": OUTPUT_ROOT / "autoencoder",
    "previews": OUTPUT_ROOT / "previews",
    "checkpoints": OUTPUT_ROOT / "checkpoints",
    "synthetic": OUTPUT_ROOT / "synthetic_samples",
    "history": OUTPUT_ROOT / "training_history",
    "reports": OUTPUT_ROOT / "reports",
}

for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

# --------------------------- UTILITIES ---------------------------
def banner(text, char="="):
    print("\n" + char * 72)
    print(text)
    print(char * 72)


def normalize_mri(arr):
    """Robust MRI normalization using percentile clipping."""
    arr = np.asarray(arr).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)

    if hi <= lo + 1e-6:
        lo = float(arr.min())
        hi = float(arr.max())

    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)

    # Keep background close to black while preserving tissue contrast.
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def pil_to_tensor(path):
    with Image.open(path) as im:
        im = im.convert("L")
        im = ImageOps.fit(
            im,
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        arr = normalize_mri(np.asarray(im))
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_image(t):
    t = t.detach().cpu().float().clamp(0, 1)
    arr = (t.squeeze().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def enhance_for_preview(img):
    """Preview-only contrast enhancement; does NOT alter saved model output."""
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img


def list_images(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_classes():
    if not TESTING_ROOT.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n{TESTING_ROOT}\n\n"
            "Expected: data/imaging/MRI/Testing/<class>/images"
        )

    classes = []
    for p in sorted(TESTING_ROOT.iterdir()):
        if p.is_dir() and len(list_images(p)) > 0:
            classes.append(p.name)

    if not classes:
        raise FileNotFoundError(f"No MRI classes/images found in:\n{TESTING_ROOT}")

    return classes


def get_class_images(class_name):
    folder = TESTING_ROOT / class_name
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images found for class: {class_name}\n{folder}")
    return paths


def save_grid(images, path, title=None, cols=4, cell=256):
    if not images:
        return

    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("L", (cols * cell, rows * cell), color=0)

    for i, img in enumerate(images):
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize((cell, cell), Image.Resampling.LANCZOS)
        x = (i % cols) * cell
        y = (i // cols) * cell
        canvas.paste(img, (x, y))

    if title:
        # Keep file generation dependency-free; title is printed in metadata.
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_side_by_side(real_images, recon_images, fake_images, path):
    rows = min(len(real_images), len(recon_images), len(fake_images))
    if rows == 0:
        return

    cell = 256
    canvas = Image.new("L", (3 * cell, rows * cell), color=0)

    for i in range(rows):
        trio = [real_images[i], recon_images[i], fake_images[i]]
        for j, img in enumerate(trio):
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(img, (j * cell, i * cell))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def sobel_edges(x):
    """Differentiable normalized Sobel magnitude."""
    gx = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0)
    gy = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0)
    px = F.conv2d(x, gx, padding=1)
    py = F.conv2d(x, gy, padding=1)
    return torch.sqrt(px * px + py * py + 1e-8)


def edge_loss(a, b):
    return F.l1_loss(sobel_edges(a), sobel_edges(b))


def source_statistics(paths, max_items=200):
    sample = paths[:max_items]
    means, stds, edges = [], [], []
    for p in sample:
        try:
            x = pil_to_tensor(p).unsqueeze(0)
            means.append(float(x.mean()))
            stds.append(float(x.std()))
            edges.append(float(sobel_edges(x).mean()))
        except Exception:
            continue

    if not means:
        return {"mean": 0.5, "std": 0.2, "edge": 0.1}

    return {
        "mean": float(np.mean(means)),
        "std": float(np.mean(stds)),
        "edge": float(np.mean(edges)),
    }

# ----------------------------- DATASET ---------------------------
class MRIDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            x = pil_to_tensor(path)
        except Exception:
            # Deterministic black fallback is safer than crashing a long CPU run.
            x = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        return x, str(path)

# -------------------------- AUTOENCODER --------------------------
class Encoder(nn.Module):
    def __init__(self, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),      # 64
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),     # 32
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 96, 4, 2, 1),     # 16
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),

            nn.Conv2d(96, latent_channels, 4, 2, 1),  # 8
            nn.GroupNorm(16, latent_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels=LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 96, 4, 2, 1),  # 16
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(96, 64, 4, 2, 1),                # 32
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),                # 64
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),                # 128
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),

            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# --------------------------- LATENT GAN --------------------------
class LatentGenerator(nn.Module):
    """Noise -> 128x8x8 latent feature map."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM, 512 * 4 * 4),
            nn.SiLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, LATENT_CHANNELS, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        x = self.fc(noise).view(-1, 512, 4, 4)
        return self.net(x)


class LatentDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(LATENT_CHANNELS, 128, 3, 2, 1)),  # 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)),                # 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 2, 2, 0)),                # 1
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, z):
        x = self.net(z).flatten(1)
        return self.fc(x)

# ------------------------------ EMA ------------------------------
@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name in ema_params:
        ema_params[name].mul_(decay).add_(model_params[name], alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name in ema_buffers:
        ema_buffers[name].copy_(model_buffers[name])

# ------------------------ AUTOENCODER TRAIN ----------------------
def train_autoencoder(class_name, paths):
    banner(f"AUTOENCODER PRETRAINING: {class_name}")
    print(f"Images      : {len(paths)}")
    print(f"Resolution  : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Epochs      : {AE_EPOCHS}")
    print(f"Device      : {DEVICE}")

    dataset = MRIDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    ae = AutoEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(
        ae.parameters(), lr=LEARNING_RATE_AE, betas=BETAS, weight_decay=1e-4
    )

    history = []
    preview_dir = DIRS["ae"] / class_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, AE_EPOCHS + 1):
        start = time.time()
        ae.train()
        losses = []

        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            recon, _ = ae(x)

            # L1 preserves anatomical intensity transitions better than MSE alone.
            loss_recon = F.l1_loss(recon, x)
            loss_edge = edge_loss(recon, x)
            loss = loss_recon + LAMBDA_EDGE * loss_edge

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        avg = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - start
        history.append({
            "epoch": epoch,
            "loss": avg,
            "time_sec": elapsed,
        })

        print(
            f"AE Epoch [{epoch:02d}/{AE_EPOCHS}] | "
            f"Loss: {avg:.5f} | Time: {elapsed:.1f}s"
        )

        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}

        if epoch == 1 or epoch == AE_EPOCHS:
            ae.eval()
            with torch.no_grad():
                sample = torch.stack([pil_to_tensor(p) for p in paths[:8]]).to(DEVICE)
                recon, _ = ae(sample)
            images = [enhance_for_preview(tensor_to_image(v)) for v in recon]
            out = preview_dir / f"epoch_{epoch:04d}.png"
            save_grid(images, out, cols=4)
            print(f"AE preview saved: {out}")

    if best_state is not None:
        ae.load_state_dict(best_state)

    model_path = DIRS["models"] / f"{class_name}_autoencoder_v7.pth"
    torch.save({
        "model_state_dict": ae.state_dict(),
        "image_size": IMAGE_SIZE,
        "latent_channels": LATENT_CHANNELS,
        "best_loss": best_loss,
    }, model_path)

    pd.DataFrame(history).to_csv(
        DIRS["history"] / f"{class_name}_autoencoder_history.csv", index=False
    )

    print(f"Best AE loss   : {best_loss:.5f}")
    print(f"Autoencoder    : {model_path}")
    return ae, history

# -------------------------- GAN TRAIN ----------------------------
def latent_stats(real_latents):
    with torch.no_grad():
        mean = real_latents.mean(dim=(0, 2, 3), keepdim=True)
        std = real_latents.std(dim=(0, 2, 3), keepdim=True).clamp_min(0.05)
    return mean, std


def train_latent_gan(class_name, paths, ae, source_stats):
    banner(f"LATENT GAN TRAINING: {class_name}")
    print(f"Real Testing images : {len(paths)}")
    print(f"Resolution          : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"GAN Epochs          : {GAN_EPOCHS}")
    print(f"Batch size          : {BATCH_SIZE}")
    print("Objective            : LSGAN in learned latent space")
    print("Generator            : latent-space generator")
    print("Decoder              : pretrained autoencoder decoder")
    print("EMA                  : Enabled")
    print(f"Device               : {DEVICE}")

    dataset = MRIDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Freeze encoder/decoder during GAN training. This prevents the decoder from
    # forgetting anatomical detail learned from the Testing dataset.
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    encoder = ae.encoder
    decoder = ae.decoder

    generator = LatentGenerator().to(DEVICE)
    discriminator = LatentDiscriminator().to(DEVICE)
    ema_generator = LatentGenerator().to(DEVICE)
    ema_generator.load_state_dict(generator.state_dict())
    ema_generator.eval()

    opt_g = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE_G, betas=BETAS
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE_D, betas=BETAS
    )

    # Calculate target latent statistics from real Testing images.
    real_latents_all = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            z = encoder(x)
            real_latents_all.append(z.cpu())

    real_latents_all = torch.cat(real_latents_all, dim=0)
    target_mean = float(real_latents_all.mean())
    target_std = float(real_latents_all.std().clamp_min(0.05))
    del real_latents_all

    history = []
    best_g = float("inf")
    best_state = None
    best_ema_state = None

    preview_dir = DIRS["previews"] / class_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, GAN_EPOCHS + 1):
        start = time.time()
        g_losses, d_losses, recon_losses = [], [], []

        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                z_real = encoder(x)

            # ------------------------------------------------------
            # D step: LSGAN
            # ------------------------------------------------------
            noise = torch.randn(x.size(0), NOISE_DIM, device=DEVICE)
            z_fake = generator(noise)

            d_real = discriminator(z_real.detach())
            d_fake = discriminator(z_fake.detach())

            d_loss = 0.5 * (
                F.mse_loss(d_real, torch.ones_like(d_real))
                + F.mse_loss(d_fake, torch.zeros_like(d_fake))
            )

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            opt_d.step()

            # ------------------------------------------------------
            # G step
            # ------------------------------------------------------
            noise = torch.randn(x.size(0), NOISE_DIM, device=DEVICE)
            z_fake = generator(noise)
            d_fake_for_g = discriminator(z_fake)

            g_adv = F.mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))

            # Keep the generated latent distribution near the real latent
            # distribution. This helps avoid degenerate nearly-black outputs.
            fake_mean = z_fake.mean()
            fake_std = z_fake.std().clamp_min(1e-4)
            latent_stat_loss = (
                (fake_mean - target_mean) ** 2
                + (fake_std - target_std) ** 2
            )

            g_loss = g_adv + LAMBDA_LATENT * latent_stat_loss

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            opt_g.step()

            update_ema(ema_generator, generator)

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))
            recon_losses.append(float(latent_stat_loss.item()))

        g_avg = float(np.mean(g_losses)) if g_losses else 0.0
        d_avg = float(np.mean(d_losses)) if d_losses else 0.0
        latent_avg = float(np.mean(recon_losses)) if recon_losses else 0.0
        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "generator_loss": g_avg,
            "discriminator_loss": d_avg,
            "latent_stat_loss": latent_avg,
            "time_sec": elapsed,
        })

        print(
            f"Epoch [{epoch:02d}/{GAN_EPOCHS}] | "
            f"G: {g_avg:.5f} | D: {d_avg:.5f} | "
            f"Latent: {latent_avg:.5f} | Time: {elapsed:.1f}s"
        )

        # Use a conservative criterion based on generator loss, but keep EMA.
        if g_avg < best_g:
            best_g = g_avg
            best_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}
            best_ema_state = {k: v.detach().cpu().clone() for k, v in ema_generator.state_dict().items()}

        if epoch == 1 or epoch % PREVIEW_EVERY == 0 or epoch == GAN_EPOCHS:
            save_gan_preview(
                class_name,
                epoch,
                paths,
                encoder,
                decoder,
                generator,
                ema_generator,
            )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == GAN_EPOCHS:
            checkpoint = DIRS["checkpoints"] / f"{class_name}_v7_latest.pth"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "best_generator_loss": best_g,
                "target_latent_mean": target_mean,
                "target_latent_std": target_std,
            }, checkpoint)
            print(f"Checkpoint saved: {checkpoint}")

    if best_state is not None:
        generator.load_state_dict(best_state)
    if best_ema_state is not None:
        ema_generator.load_state_dict(best_ema_state)

    g_path = DIRS["models"] / f"{class_name}_generator_v7.pth"
    ema_path = DIRS["models"] / f"{class_name}_generator_ema_v7.pth"
    d_path = DIRS["models"] / f"{class_name}_discriminator_v7.pth"

    torch.save(generator.state_dict(), g_path)
    torch.save(ema_generator.state_dict(), ema_path)
    torch.save(discriminator.state_dict(), d_path)

    pd.DataFrame(history).to_csv(
        DIRS["history"] / f"{class_name}_gan_history.csv", index=False
    )

    print(f"Best G loss : {best_g:.5f}")
    print(f"Generator   : {g_path}")
    print(f"EMA         : {ema_path}")
    print(f"Discriminator: {d_path}")

    return generator, ema_generator, discriminator, history

# -------------------------- PREVIEW ------------------------------
@torch.no_grad()
def save_gan_preview(class_name, epoch, paths, encoder, decoder, generator, ema_generator):
    generator.eval()
    ema_generator.eval()
    encoder.eval()
    decoder.eval()

    sample_paths = paths[:8]
    real = torch.stack([pil_to_tensor(p) for p in sample_paths]).to(DEVICE)
    z_real = encoder(real)

    # Reconstruction is an important diagnostic: if reconstruction is sharp,
    # the decoder can preserve useful structure; if it is blurry, GAN training
    # cannot fix the decoder itself.
    recon = decoder(z_real)

    noise = torch.randn(len(sample_paths), NOISE_DIM, device=DEVICE)
    z_fake = ema_generator(noise)
    fake = decoder(z_fake).clamp(0, 1)

    # Avoid arbitrary post-processing in the saved model output. The preview
    # uses mild contrast only so dark structures are easier to inspect.
    real_imgs = [enhance_for_preview(tensor_to_image(v)) for v in real]
    recon_imgs = [enhance_for_preview(tensor_to_image(v)) for v in recon]
    fake_imgs = [enhance_for_preview(tensor_to_image(v)) for v in fake]

    out = DIRS["previews"] / class_name / f"epoch_{epoch:04d}_real_recon_fake.png"
    save_side_by_side(real_imgs, recon_imgs, fake_imgs, out)
    print(f"Preview saved: {out}")

# ---------------------- QUALITY FILTER ---------------------------
def image_quality_score(img_tensor, source):
    """Simple non-medical image sanity score; not a diagnostic metric."""
    mean = float(img_tensor.mean())
    std = float(img_tensor.std())
    edge = float(sobel_edges(img_tensor).mean())

    mean_err = abs(mean - source["mean"])
    std_err = abs(std - source["std"])
    edge_err = abs(edge - source["edge"])

    score = 1.0 / (1.0 + 2.0 * mean_err + 2.0 * std_err + 1.5 * edge_err)
    return float(score), mean, std, edge


@torch.no_grad()
def generate_synthetic(class_name, paths, ae, ema_generator, count=SYNTHETIC_PER_CLASS):
    banner(f"GENERATING SYNTHETIC MRI: {class_name}", "-")
    print(f"Requested synthetic images : {count}")
    print("Using EMA generator         : YES")
    print("Output resolution           : 128 x 128")

    ae.eval()
    ema_generator.eval()

    source = source_statistics(paths)
    out_dir = DIRS["synthetic"] / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Do not delete previous V7 files. Continue numbering from existing files.
    existing = list_images(out_dir)
    start_index = len(existing) + 1

    accepted = []
    attempts = 0
    max_attempts = max(count * 5, count + 20)
    quality_rows = []

    while len(accepted) < count and attempts < max_attempts:
        batch = min(BATCH_SIZE, count - len(accepted) + 4)
        noise = torch.randn(batch, NOISE_DIM, device=DEVICE)
        z_fake = ema_generator(noise)
        fake = ae.decoder(z_fake).clamp(0, 1)

        for i in range(fake.size(0)):
            img = fake[i:i + 1]
            score, mean, std, edge = image_quality_score(img, source)

            # Sanity filters are intentionally broad. We don't claim they
            # establish medical validity.
            acceptable = (
                0.01 < std < 0.45
                and mean > 0.015
                and score > 0.55
            )

            quality_rows.append({
                "attempt": attempts + 1,
                "accepted": int(acceptable),
                "quality_score": score,
                "mean": mean,
                "std": std,
                "edge": edge,
            })

            attempts += 1

            if acceptable and len(accepted) < count:
                accepted.append(img.squeeze(0).cpu())

            if len(accepted) >= count or attempts >= max_attempts:
                break

    if len(accepted) < count:
        print(
            f"WARNING: only {len(accepted)}/{count} images passed the broad "
            "sanity filter after {attempts} attempts."
        )

    for i, tensor in enumerate(accepted, start=start_index):
        img = tensor_to_image(tensor)
        img.save(out_dir / f"synthetic_{i:04d}.png")

    pd.DataFrame(quality_rows).to_csv(
        DIRS["reports"] / f"{class_name}_quality_filter.csv", index=False
    )

    # Final preview of accepted images.
    preview = [tensor_to_image(x) for x in accepted[:16]]
    preview_path = DIRS["previews"] / class_name / "final_synthetic_preview.png"
    save_grid(preview, preview_path, cols=4)

    print(f"Accepted synthetic images : {len(accepted)}")
    print(f"Synthetic folder          : {out_dir}")
    print(f"Final preview              : {preview_path}")
    return len(accepted)

# ------------------------- REPORTING -----------------------------
def save_summary(rows):
    df = pd.DataFrame(rows)
    path = DIRS["reports"] / "v7_summary.csv"
    df.to_csv(path, index=False)

    json_path = DIRS["reports"] / "v7_summary.json"
    json_path.write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return df

# -------------------------- MAIN PIPELINE ------------------------
def train_class(class_name):
    paths = get_class_images(class_name)
    source = source_statistics(paths)

    banner(f"STARTING V7: {class_name}", "#")
    print(f"Testing images : {len(paths)}")
    print(f"Source mean    : {source['mean']:.5f}")
    print(f"Source std     : {source['std']:.5f}")
    print(f"Source edge    : {source['edge']:.5f}")

    ae, ae_history = train_autoencoder(class_name, paths)
    generator, ema_generator, discriminator, gan_history = train_latent_gan(
        class_name, paths, ae, source
    )
    synthetic_count = generate_synthetic(
        class_name, paths, ae, ema_generator, SYNTHETIC_PER_CLASS
    )

    row = {
        "class": class_name,
        "real_images": len(paths),
        "synthetic_images": synthetic_count,
        "resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "ae_epochs": AE_EPOCHS,
        "gan_epochs": GAN_EPOCHS,
        "latent_channels": LATENT_CHANNELS,
        "noise_dim": NOISE_DIM,
        "device": str(DEVICE),
    }

    banner(f"{class_name.upper()} V7 COMPLETED", "-")
    for k, v in row.items():
        print(f"{k:18}: {v}")

    return row


def train_all_classes():
    banner("MRI TESTING GAN V7")
    print("Testing dataset ONLY")
    print("Quality-focused latent-space GAN")
    print("Native 128x128 output")
    print("Existing training data protected")
    print("Existing GAN outputs protected")

    banner("PROTECTED DATA", "-")
    print("Training                  : NOT MODIFIED")
    print("Training_Preprocessed     : NOT MODIFIED")
    print("outputs/mri               : NOT MODIFIED")
    print("mri_testing_gan           : NOT MODIFIED")
    print("mri_testing_gan_v3        : NOT MODIFIED")
    print("mri_testing_gan_v4        : NOT MODIFIED")
    print("mri_testing_gan_v5        : NOT MODIFIED")
    print("mri_testing_gan_v6        : NOT MODIFIED")

    banner("V7 CONFIGURATION", "-")
    print(f"Input     : {TESTING_ROOT}")
    print(f"Output    : {OUTPUT_ROOT}")
    print(f"Device    : {DEVICE}")
    print(f"Resolution: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"AE epochs : {AE_EPOCHS}")
    print(f"GAN epochs: {GAN_EPOCHS}")

    classes = discover_classes()
    print("\nDetected classes:")
    for c in classes:
        print(f"  {c:<14} {len(get_class_images(c))} images")

    if PILOT_CLASSES is not None:
        classes = [c for c in classes if c.lower() in {x.lower() for x in PILOT_CLASSES}]
        if not classes:
            raise ValueError(f"None of PILOT_CLASSES found. Available: {discover_classes()}")
        print("\nPILOT MODE:")
        print("  " + ", ".join(classes))

    rows = []
    for class_name in classes:
        rows.append(train_class(class_name))

    df = save_summary(rows)

    banner("MRI TESTING GAN V7 FINISHED")
    print(df.to_string(index=False))
    print(f"\nV7 output       : {OUTPUT_ROOT}")
    print(f"Synthetic MRI   : {DIRS['synthetic']}")
    print(f"Models          : {DIRS['models']}")
    print(f"Previews        : {DIRS['previews']}")
    print(f"Reports         : {DIRS['reports']}")


if __name__ == "__main__":
    train_all_classes()

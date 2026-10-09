"""
MRI TESTING V3 — RECONSTRUCTION-FIRST AUTOENCODER

Reads ONLY:
    data/imaging/MRI/Testing

Does NOT:
    read Training
    modify old outputs
    train diffusion
    generate synthetic images

Goal:
    Train a stronger residual autoencoder on all 4 Testing classes.
    Validate REAL -> RECONSTRUCTION quality before diffusion.
"""

from __future__ import annotations
import json, math, random, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "data" / "imaging" / "MRI" / "Testing"
RUN = ROOT / "outputs" / "mri_testing_v3_reconstruction" / time.strftime("run_%Y%m%d_%H%M%S")
MODEL_DIR = RUN / "models"
RECON_DIR = RUN / "reconstruction"
PREVIEW_DIR = RUN / "previews"
REPORT_DIR = RUN / "reports"

for p in [MODEL_DIR, RECON_DIR, PREVIEW_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

BATCH_SIZE = 16
EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.10

# V3 keeps substantially more information than V2:
# 128 -> 64 -> 32, giving 16 x 32 x 32 latent.
LATENT_CHANNELS = 16
LATENT_SIZE = 32

L1_W = 1.0
SSIM_W = 0.25
EDGE_W = 0.10

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# DATA
# ============================================================

def paths_for_class(c):
    d = TEST_ROOT / c
    if not d.exists():
        return []
    return sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def preprocess(path):
    im = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    a = np.asarray(im, dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, [1, 99])
    if hi > lo + 1e-6:
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    return a.astype(np.float32)


class MRIDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    @classmethod
    def all_testing(cls, augment=False):
        items = []
        for c in CLASSES:
            for p in paths_for_class(c):
                items.append((p, CLASS_ID[c], c))
        if not items:
            raise RuntimeError(f"No images found: {TEST_ROOT}")
        return cls(items, augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, c = self.items[i]
        a = preprocess(p)
        if self.augment and random.random() < 0.5:
            a = np.fliplr(a).copy()
        x = torch.from_numpy(a).float().unsqueeze(0)
        return x * 2 - 1, torch.tensor(y), str(p), c


# ============================================================
# MODEL
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.SiLU(),
            ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 96, 4, 2, 1),
            nn.SiLU(),
            ResBlock(96), ResBlock(96),
            nn.Conv2d(96, LATENT_CHANNELS, 4, 2, 1),
            nn.SiLU(),
            ResBlock(LATENT_CHANNELS),
            ResBlock(LATENT_CHANNELS),
            nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 96, 3, padding=1),
            nn.SiLU(),
            ResBlock(96), ResBlock(96),
            nn.ConvTranspose2d(96, 96, 4, 2, 1),
            nn.SiLU(),
            ResBlock(96), ResBlock(96),
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.SiLU(),
            ResBlock(64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class MRIReconstructionAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# ============================================================
# LOSSES
# ============================================================

def sobel(x):
    kx = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        dtype=x.dtype, device=x.device
    ).view(1,1,3,3)
    ky = torch.tensor(
        [[-1,-2,-1],[0,0,0],[1,2,1]],
        dtype=x.dtype, device=x.device
    ).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def ssim_loss(x, y):
    mx = F.avg_pool2d(x, 7, 1, 3)
    my = F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x*x, 7, 1, 3) - mx*mx
    vy = F.avg_pool2d(y*y, 7, 1, 3) - my*my
    cov = F.avg_pool2d(x*y, 7, 1, 3) - mx*my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2*mx*my+c1)*(2*cov+c2)) / (
        (mx*mx+my*my+c1)*(vx+vy+c2) + 1e-6
    )
    return 1 - s.clamp(-1, 1).mean()


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-10))


# ============================================================
# TRAINING
# ============================================================

def loaders():
    full = MRIDataset.all_testing(False)
    val_n = max(len(CLASSES), int(len(full) * VAL_RATIO))
    train_n = len(full) - val_n

    g = torch.Generator().manual_seed(SEED)
    train_base, val_base = random_split(full, [train_n, val_n], generator=g)

    train = MRIDataset(
        [full.items[i] for i in train_base.indices], True
    )
    val = MRIDataset(
        [full.items[i] for i in val_base.indices], False
    )

    kw = dict(batch_size=BATCH_SIZE, num_workers=0,
              pin_memory=(DEVICE.type == "cuda"))
    return (
        DataLoader(train, shuffle=True, drop_last=True, **kw),
        DataLoader(val, shuffle=False, **kw),
    )


def train():
    tr, va = loaders()
    model = MRIReconstructionAE().to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.99),
        weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=2e-5
    )

    amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = float("inf")
    history = []

    print("\n" + "="*80)
    print("MRI TESTING V3 — RECONSTRUCTION-FIRST")
    print("="*80)
    print(f"Testing root : {TEST_ROOT}")
    print(f"Run folder   : {RUN}")
    print(f"Device       : {DEVICE}")
    print(f"Latent       : {LATENT_CHANNELS}x{LATENT_SIZE}x{LATENT_SIZE}")
    print(f"Train        : {len(tr.dataset)}")
    print(f"Validation   : {len(va.dataset)}")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        ts = [0, 0, 0, 0]

        for x, _, _, _ in tr:
            x = x.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                r, _ = model(x)
                l1 = F.l1_loss(r, x)
                ss = ssim_loss(r, x)
                ed = F.l1_loss(sobel(r), sobel(x))
                loss = L1_W*l1 + SSIM_W*ss + EDGE_W*ed

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ts[0] += loss.item()
            ts[1] += l1.item()
            ts[2] += ss.item()
            ts[3] += ed.item()

        model.eval()
        vs = [0, 0, 0, 0, 0]

        with torch.no_grad():
            for x, _, _, _ in va:
                x = x.to(DEVICE, non_blocking=True)
                r, _ = model(x)
                l1 = F.l1_loss(r, x)
                ss = ssim_loss(r, x)
                ed = F.l1_loss(sobel(r), sobel(x))
                mse = F.mse_loss(r, x)
                loss = L1_W*l1 + SSIM_W*ss + EDGE_W*ed
                vs[0] += loss.item()
                vs[1] += l1.item()
                vs[2] += ss.item()
                vs[3] += ed.item()
                vs[4] += mse.item()

        tn, vn = max(len(tr),1), max(len(va),1)
        sched.step()

        row = {
            "epoch": ep,
            "train_loss": ts[0]/tn,
            "train_l1": ts[1]/tn,
            "train_ssim_loss": ts[2]/tn,
            "train_edge": ts[3]/tn,
            "val_loss": vs[0]/vn,
            "val_l1": vs[1]/vn,
            "val_ssim_loss": vs[2]/vn,
            "val_edge": vs[3]/vn,
            "val_mse": vs[4]/vn,
            "val_psnr": psnr(vs[4]/vn),
            "seconds": time.time()-t0,
        }
        history.append(row)

        print(
            f"Epoch {ep:02d}/{EPOCHS} | "
            f"Train {row['train_loss']:.4f} | "
            f"Val {row['val_loss']:.4f} | "
            f"PSNR {row['val_psnr']:.2f} dB | "
            f"SSIM-loss {row['val_ssim_loss']:.4f} | "
            f"{row['seconds']:.1f}s"
        )

        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "image_size": IMAGE_SIZE,
                    "latent_channels": LATENT_CHANNELS,
                    "latent_size": LATENT_SIZE,
                    "classes": CLASSES,
                },
                "epoch": ep,
                "val_loss": row["val_loss"],
                "val_psnr": row["val_psnr"],
            }, MODEL_DIR / "best_autoencoder.pt")

    torch.save({
        "model": model.state_dict(),
        "config": {
            "image_size": IMAGE_SIZE,
            "latent_channels": LATENT_CHANNELS,
            "latent_size": LATENT_SIZE,
            "classes": CLASSES,
        },
        "history": history,
    }, MODEL_DIR / "final_autoencoder.pt")

    (REPORT_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    return model


# ============================================================
# PREVIEW
# ============================================================

def to_pil(x):
    x = ((x + 1) / 2).clamp(0, 1)
    a = x.detach().cpu().numpy()
    if a.ndim == 4:
        a = a[0, 0]
    elif a.ndim == 3:
        a = a[0]
    return Image.fromarray((a*255).round().astype(np.uint8), "L")


@torch.no_grad()
def make_previews(model):
    model.eval()
    metrics = {}

    for c in CLASSES:
        paths = paths_for_class(c)[:12]
        if not paths:
            continue

        out = RECON_DIR / c
        out.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, p in enumerate(paths):
            a = preprocess(p)
            x = torch.from_numpy(a).float()[None,None].to(DEVICE)
            x = x * 2 - 1
            r, z = model(x)

            real = ((x+1)/2).clamp(0,1)
            rec = ((r+1)/2).clamp(0,1)
            mse = F.mse_loss(rec, real).item()

            rows.append({
                "file": p.name,
                "mae": F.l1_loss(rec, real).item(),
                "psnr": psnr(mse),
                "ssim_loss": ssim_loss(rec, real).item(),
            })

            pair = Image.new("L", (IMAGE_SIZE*2, IMAGE_SIZE))
            pair.paste(to_pil(x), (0,0))
            pair.paste(to_pil(r), (IMAGE_SIZE,0))
            pair.save(out / f"{i+1:03d}_real_vs_recon.png")

        metrics[c] = rows
        print(
            f"{c:15s} | "
            f"MAE {np.mean([r['mae'] for r in rows]):.4f} | "
            f"PSNR {np.mean([r['psnr'] for r in rows]):.2f} dB | "
            f"SSIM-loss {np.mean([r['ssim_loss'] for r in rows]):.4f}"
        )

    (REPORT_DIR / "preview_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return metrics


def contact_sheet(model, c):
    model.eval()
    paths = paths_for_class(c)[:12]
    if not paths:
        return

    cell_w, cell_h = IMAGE_SIZE*2, IMAGE_SIZE+22
    cols = 2
    rows = math.ceil(len(paths)/cols)
    sheet = Image.new("L", (cell_w*cols, cell_h*rows), 255)
    draw = ImageDraw.Draw(sheet)

    for i, p in enumerate(paths):
        a = preprocess(p)
        x = torch.from_numpy(a).float()[None,None].to(DEVICE)
        x = x*2-1
        with torch.no_grad():
            r, _ = model(x)

        pair = Image.new("L", (IMAGE_SIZE*2, IMAGE_SIZE))
        pair.paste(to_pil(x), (0,0))
        pair.paste(to_pil(r), (IMAGE_SIZE,0))

        xx = (i % cols) * cell_w
        yy = (i // cols) * cell_h
        sheet.paste(pair, (xx,yy))
        draw.text((xx+4, yy+IMAGE_SIZE+3), f"{i+1:02d}  {c}", fill=0)

    path = PREVIEW_DIR / f"{c}_reconstruction_contact_sheet.png"
    sheet.save(path)
    print(f"Contact sheet : {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()

    print("\n" + "#"*80)
    print("# MRI TESTING V3")
    print("# RECONSTRUCTION FIRST — NO DIFFUSION YET")
    print("#"*80)

    total = 0
    for c in CLASSES:
        n = len(paths_for_class(c))
        total += n
        print(f"{c:15s}: {n}")
    print(f"{'TOTAL':15s}: {total}")

    model = train()

    ckpt = torch.load(
        MODEL_DIR / "best_autoencoder.pt",
        map_location=DEVICE
    )
    model.load_state_dict(ckpt["model"])

    print("\nBest checkpoint:")
    print(f"Epoch    : {ckpt['epoch']}")
    print(f"Val loss : {ckpt['val_loss']:.6f}")
    print(f"Val PSNR : {ckpt['val_psnr']:.2f} dB")

    metrics = make_previews(model)

    for c in CLASSES:
        contact_sheet(model, c)

    report = {
        "version": "V3 reconstruction-first",
        "testing_root": str(TEST_ROOT),
        "run": str(RUN),
        "classes": CLASSES,
        "latent_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
        "best_epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "best_val_psnr": ckpt["val_psnr"],
        "preview_metrics": metrics,
        "training_folder_read": False,
        "old_outputs_modified": False,
        "diffusion_trained": False,
        "synthetic_generated": False,
        "seconds": time.time()-start,
    }

    (REPORT_DIR / "v3_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n" + "="*80)
    print("V3 COMPLETE")
    print("="*80)
    print(f"Run folder: {RUN}")
    print("\nOpen the *_reconstruction_contact_sheet.png files.")
    print("Each pair is REAL on the LEFT and RECONSTRUCTION on the RIGHT.")
    print("\nDO NOT train diffusion yet.")
    print("If reconstructions are good, the next step is V4 conditional latent diffusion.")
    print("If reconstructions are poor, we fix the autoencoder before diffusion.")


if __name__ == "__main__":
    main()

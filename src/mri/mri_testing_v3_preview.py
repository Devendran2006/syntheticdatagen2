from pathlib import Path
import json, math, sys
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
TRAINED_RUN=ROOT/"outputs"/"mri_testing_v3_reconstruction"/"run_20260922_044321"
CHECKPOINT=TRAINED_RUN/"models"/"best_autoencoder.pt"
TEST_ROOT=ROOT/"data"/"imaging"/"MRI"/"Testing"
OUTPUT_ROOT=TRAINED_RUN/"preview_epoch7"
PREVIEW_ROOT=OUTPUT_ROOT/"contact_sheets"
PAIR_ROOT=OUTPUT_ROOT/"real_vs_reconstruction"
REPORT_ROOT=OUTPUT_ROOT/"reports"
for d in [PREVIEW_ROOT,PAIR_ROOT,REPORT_ROOT]: d.mkdir(parents=True,exist_ok=True)

IMAGE_SIZE=128
CLASSES=["glioma","meningioma","notumor","pituitary"]
EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_DIR=ROOT/"src"/"mri"
sys.path.insert(0,str(SRC_DIR))
from mri_testing_v3_reconstruction import MRIReconstructionAE

def image_paths(c):
    d=TEST_ROOT/c
    return sorted([x for x in d.rglob("*") if x.is_file() and x.suffix.lower() in EXTS]) if d.exists() else []

def preprocess(path):
    im=Image.open(path).convert("L").resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.float32)/255.0
    lo,hi=np.percentile(a,[1,99])
    if hi>lo+1e-6: a=np.clip((a-lo)/(hi-lo),0,1)
    return a.astype(np.float32)

def to_pil(x):
    a=(((x+1)/2).clamp(0,1)).detach().cpu().numpy()
    if a.ndim==4: a=a[0,0]
    elif a.ndim==3: a=a[0]
    return Image.fromarray((a*255).round().astype(np.uint8),"L")

def psnr(mse): return -10*math.log10(max(float(mse),1e-10))

def ssim_loss(x,y):
    mx=F.avg_pool2d(x,7,1,3); my=F.avg_pool2d(y,7,1,3)
    vx=F.avg_pool2d(x*x,7,1,3)-mx*mx
    vy=F.avg_pool2d(y*y,7,1,3)-my*my
    cov=F.avg_pool2d(x*y,7,1,3)-mx*my
    c1,c2=.01**2,.03**2
    s=((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-6)
    return float((1-s.clamp(-1,1).mean()).item())

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    ckpt=torch.load(CHECKPOINT,map_location=DEVICE)
    model=MRIReconstructionAE().to(DEVICE)
    model.load_state_dict(ckpt["model"]); model.eval()
    print("="*80)
    print("MRI TESTING V3 — PREVIEW ONLY")
    print("="*80)
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Epoch      : {ckpt.get('epoch')}")
    print(f"Val loss   : {ckpt.get('val_loss')}")
    print(f"Val PSNR   : {ckpt.get('val_psnr')}")
    print(f"Device     : {DEVICE}")

    all_metrics={}
    with torch.no_grad():
        for c in CLASSES:
            paths=image_paths(c)[:12]
            pair_dir=PAIR_ROOT/c; pair_dir.mkdir(parents=True,exist_ok=True)
            rows=[]
            for i,path in enumerate(paths,1):
                a=preprocess(path)
                x=torch.from_numpy(a).float()[None,None].to(DEVICE)*2-1
                r,z=model(x)
                real=((x+1)/2).clamp(0,1); rec=((r+1)/2).clamp(0,1)
                mse=F.mse_loss(rec,real).item()
                rows.append({"image":path.name,"mae":float(F.l1_loss(rec,real)),"psnr_db":psnr(mse),"ssim_loss":ssim_loss(rec,real)})
                pair=Image.new("L",(IMAGE_SIZE*2,IMAGE_SIZE),255)
                pair.paste(to_pil(x),(0,0)); pair.paste(to_pil(r),(IMAGE_SIZE,0))
                pair.save(pair_dir/f"{i:03d}_real_vs_reconstruction.png")
            all_metrics[c]=rows
            cols=2; cw=IMAGE_SIZE*2; ch=IMAGE_SIZE+24
            sheet=Image.new("L",(cw*cols,ch*math.ceil(len(paths)/cols)),255)
            draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(paths):
                a=preprocess(path); x=torch.from_numpy(a).float()[None,None].to(DEVICE)*2-1
                r,_=model(x)
                pair=Image.new("L",(IMAGE_SIZE*2,IMAGE_SIZE),255)
                pair.paste(to_pil(x),(0,0)); pair.paste(to_pil(r),(IMAGE_SIZE,0))
                px=(i%cols)*cw; py=(i//cols)*ch
                sheet.paste(pair,(px,py)); draw.text((px+4,py+IMAGE_SIZE+4),f"{i+1:02d} {c}",fill=0)
            out=PREVIEW_ROOT/f"{c}_reconstruction_contact_sheet.png"
            sheet.save(out)
            print(f"{c:12s} | PSNR {np.mean([r['psnr_db'] for r in rows]):.2f} dB | MAE {np.mean([r['mae'] for r in rows]):.4f} | {out}")

    (REPORT_ROOT/"preview_report.json").write_text(json.dumps({"epoch":ckpt.get("epoch"),"val_psnr":ckpt.get("val_psnr"),"metrics":all_metrics},indent=2),encoding="utf-8")
    print("\nPREVIEW COMPLETE — inspect the four contact sheets before diffusion.")

if __name__=="__main__": main()

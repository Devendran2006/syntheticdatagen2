from __future__ import annotations
import json, sys, time, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / 'data' / 'imaging' / 'MRI' / 'Testing'
V3_RUN = ROOT / 'outputs' / 'mri_testing_v3_reconstruction' / 'run_20260922_044321'
V3_CHECKPOINT = V3_RUN / 'models' / 'best_autoencoder.pt'
RUN = ROOT / 'outputs' / 'mri_testing_v4_latents' / time.strftime('run_%Y%m%d_%H%M%S')
PREVIEW_DIR = RUN / 'previews'; REPORT_DIR = RUN / 'reports'
for d in (RUN, PREVIEW_DIR, REPORT_DIR): d.mkdir(parents=True, exist_ok=True)
LATENT_FILE = RUN / 'latents.pt'; STATS_FILE = RUN / 'latent_statistics.pt'; METADATA_FILE = RUN / 'metadata.json'; REPORT_FILE = REPORT_DIR / 'cache_report.json'
IMAGE_SIZE=128; BATCH_SIZE=16; SEED=42
CLASSES=['glioma','meningioma','notumor','pituitary']; CLASS_TO_ID={c:i for i,c in enumerate(CLASSES)}
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_DIR=ROOT/'src'/'mri'; sys.path.insert(0,str(SRC_DIR))
from mri_testing_v3_reconstruction import MRIReconstructionAE

def image_paths(c):
    d=TEST_ROOT/c
    return sorted(p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in EXTS) if d.exists() else []

def preprocess(p):
    im=Image.open(p).convert('L').resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.float32)/255.0
    lo,hi=np.percentile(a,[1,99])
    if hi>lo+1e-6: a=np.clip((a-lo)/(hi-lo),0,1)
    return torch.from_numpy(a).float().unsqueeze(0)*2-1

class MRIDataset(Dataset):
    def __init__(self, records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        p,c,y=self.records[i]
        return preprocess(p), torch.tensor(y,dtype=torch.long), str(p), c

def tensor_to_pil(x):
    a=(((x+1)/2).clamp(0,1)).detach().cpu().numpy()
    if a.ndim==4: a=a[0,0]
    elif a.ndim==3: a=a[0]
    return Image.fromarray((a*255).round().astype(np.uint8),'L')

def collect():
    records=[]
    for c in CLASSES:
        ps=image_paths(c); print(f'{c:12s}: {len(ps)}')
        records += [(p,c,CLASS_TO_ID[c]) for p in ps]
    if not records: raise RuntimeError(f'No images found: {TEST_ROOT}')
    print(f'TOTAL       : {len(records)}')
    return records

def load_model():
    if not V3_CHECKPOINT.exists(): raise FileNotFoundError(f'V3 checkpoint not found: {V3_CHECKPOINT}')
    ck=torch.load(V3_CHECKPOINT,map_location=DEVICE)
    m=MRIReconstructionAE().to(DEVICE); m.load_state_dict(ck['model']); m.eval()
    print(f'V3 epoch: {ck.get("epoch")} | Val PSNR: {ck.get("val_psnr")} dB | Device: {DEVICE}')
    return m,ck

@torch.no_grad()
def cache(model,records):
    dl=DataLoader(MRIDataset(records),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=DEVICE.type=='cuda')
    zs=[]; ys=[]; paths=[]; names=[]; total=len(records); start=time.time()
    for bi,(x,y,p,n) in enumerate(dl,1):
        z=model.encode(x.to(DEVICE,non_blocking=True)).cpu().float()
        zs.append(z); ys.append(y.cpu()); paths.extend(p); names.extend(n)
        done=min(bi*BATCH_SIZE,total); print(f'Encoded {done:4d}/{total} | {done/max(time.time()-start,1e-6):.2f} img/s')
    return torch.cat(zs),torch.cat(ys),paths,names

@torch.no_grad()
def preview(model,records,z):
    pairs=[]
    for i,(p,c,_) in enumerate(records[:8]):
        real=preprocess(p).unsqueeze(0).to(DEVICE); rec=model.decode(z[i:i+1].to(DEVICE))
        pair=Image.new('L',(IMAGE_SIZE*2,IMAGE_SIZE),255); pair.paste(tensor_to_pil(real),(0,0)); pair.paste(tensor_to_pil(rec),(IMAGE_SIZE,0)); pairs.append((pair,c))
    sheet=Image.new('L',(IMAGE_SIZE*4,IMAGE_SIZE*4+96),255); draw=ImageDraw.Draw(sheet)
    for i,(pair,c) in enumerate(pairs):
        x=(i%2)*IMAGE_SIZE*2; y=(i//2)*(IMAGE_SIZE+24); sheet.paste(pair,(x,y)); draw.text((x+4,y+IMAGE_SIZE+4),f'{i+1:02d} {c}',fill=0)
    out=PREVIEW_DIR/'cached_latent_decode_preview.png'; sheet.save(out); print(f'Preview: {out}')

def main():
    print('\n'+'='*80); print('MRI TESTING V4 — STEP 1: CACHE V3 LATENTS'); print('='*80)
    print(f'Testing: {TEST_ROOT}\nV3 checkpoint: {V3_CHECKPOINT}\nNew run: {RUN}')
    records=collect(); model,ck=load_model(); z,y,paths,names=cache(model,records)
    expected=(16,32,32)
    if tuple(z.shape[1:])!=expected: raise RuntimeError(f'Unexpected latent shape {tuple(z.shape[1:])}; expected {expected}')
    mean=z.mean(dim=(0,2,3),keepdim=True); std=z.std(dim=(0,2,3),keepdim=True).clamp_min(1e-6)
    stats={'mean':mean,'std':std,'global_mean':float(z.mean()),'global_std':float(z.std()),'global_min':float(z.min()),'global_max':float(z.max()),'latent_shape':list(z.shape[1:])}
    torch.save({'latents':z,'labels':y,'classes':CLASSES,'class_to_id':CLASS_TO_ID,'image_size':IMAGE_SIZE,'latent_shape':list(z.shape[1:])},LATENT_FILE)
    torch.save(stats,STATS_FILE)
    metadata={'version':'V4 latent cache','testing_root':str(TEST_ROOT),'v3_checkpoint':str(V3_CHECKPOINT),'v3_epoch':ck.get('epoch'),'v3_val_psnr':ck.get('val_psnr'),'classes':CLASSES,'image_count':len(paths),'latent_shape':list(z.shape[1:]),'source_images':paths,'source_class_names':names,'training_performed':False,'diffusion_performed':False,'synthetic_generated':False,'testing_modified':False,'v3_checkpoint_modified':False}
    METADATA_FILE.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report={'run':str(RUN),'image_count':len(paths),'class_counts':{c:int((y==CLASS_TO_ID[c]).sum()) for c in CLASSES},'latent_shape':list(z.shape[1:]),'global_mean':stats['global_mean'],'global_std':stats['global_std'],'global_min':stats['global_min'],'global_max':stats['global_max'],'source_testing_modified':False,'v3_checkpoint_modified':False,'diffusion_trained':False,'synthetic_generated':False}
    REPORT_FILE.write_text(json.dumps(report,indent=2),encoding='utf-8')
    preview(model,records,z)
    print('\n'+'='*80); print('V4 LATENT CACHE COMPLETE'); print('='*80)
    print(f'Latents: {LATENT_FILE}\nStats:   {STATS_FILE}\nReport:  {REPORT_FILE}')
    print(f'Latent shape: {tuple(z.shape)}'); print(f'Global mean/std: {stats["global_mean"]:.6f} / {stats["global_std"]:.6f}')
    print('\nNO synthetic MRI generated yet. Next step: V4 conditional latent diffusion pilot.')

if __name__=='__main__': main()

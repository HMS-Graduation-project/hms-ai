"""
Train ResNet50 for pneumonia detection on the CLEANED dataset.

Usage:
    cd hms-ai
    python -m ml.training.train_resnet50 --epochs 11
"""

from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ml.training.config import (
    BATCH_SIZE, CHECKPOINT_DIR, CLASS_NAMES, DATA_DIR, DEVICE,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LEARNING_RATE,
    METRICS_DIR, NUM_CLASSES, NUM_WORKERS, PATIENCE, RANDOM_SEED,
    TRAIN_DIR, VAL_DIR, WEIGHT_DECAY,
)

MODEL_VERSION = "pneumonia-resnet50-v1"
BEST_CKPT = CHECKPOINT_DIR / "pneumonia_resnet50_best.pt"
OUT_DIR = METRICS_DIR / "resnet50"

def _seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def verify():
    if "chest_xray_cleaned" not in str(DATA_DIR): print("ERROR: Use cleaned dataset"); sys.exit(1)
    for n, d in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        for c in CLASS_NAMES:
            cd = d / c
            cnt = sum(1 for f in cd.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}) if cd.exists() else 0
            print(f"  {n}/{c}: {cnt}")

def get_transforms(train):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def compute_weights(d):
    n = sum(1 for f in (d/"NORMAL").iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"})
    p = sum(1 for f in (d/"PNEUMONIA").iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"})
    t = n + p; w0 = t/(2*max(n,1)); w1 = t/(2*max(p,1))
    print(f"  Class weights: NORMAL={w0:.4f}  PNEUMONIA={w1:.4f}")
    return torch.tensor([w0, w1], dtype=torch.float32)

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

def train_epoch(model, loader, crit, opt, scaler):
    model.train(); lsum=0; cor=0; tot=0
    for imgs, lbls in tqdm(loader, desc="  Train", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE); opt.zero_grad()
        if scaler and DEVICE.type=="cuda":
            with torch.amp.autocast("cuda"): out=model(imgs); loss=crit(out,lbls)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: out=model(imgs); loss=crit(out,lbls); loss.backward(); opt.step()
        lsum+=loss.item()*imgs.size(0); cor+=out.max(1)[1].eq(lbls).sum().item(); tot+=lbls.size(0)
    return lsum/tot, cor/tot

@torch.no_grad()
def validate(model, loader, crit):
    model.eval(); lsum=0; tot=0; yl=[]; yp=[]; ypr=[]
    for imgs, lbls in tqdm(loader, desc="  Val  ", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        if DEVICE.type=="cuda":
            with torch.amp.autocast("cuda"): out=model(imgs); loss=crit(out,lbls)
        else: out=model(imgs); loss=crit(out,lbls)
        lsum+=loss.item()*imgs.size(0); probs=torch.softmax(out,1)
        yl.extend(lbls.cpu().numpy()); yp.extend(out.max(1)[1].cpu().numpy())
        ypr.extend(probs[:,1].cpu().numpy()); tot+=lbls.size(0)
    yt=np.array(yl); ypp=np.array(yp); yprr=np.array(ypr)
    tp=int(((ypp==1)&(yt==1)).sum()); tn=int(((ypp==0)&(yt==0)).sum())
    fp=int(((ypp==1)&(yt==0)).sum()); fn=int(((ypp==0)&(yt==1)).sum())
    acc=(tp+tn)/max(tot,1); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    spec=tn/max(tn+fp,1); f1=2*prec*rec/max(prec+rec,1e-8)
    try: auc=roc_auc_score(yt,yprr)
    except: auc=0.0
    return {"loss":lsum/tot,"accuracy":acc,"precision":prec,"recall":rec,"specificity":spec,"f1":f1,"auc":auc}

def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 pneumonia classifier")
    parser.add_argument("--epochs",type=int,default=11)
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    parser.add_argument("--lr",type=float,default=LEARNING_RATE)
    parser.add_argument("--weight-decay",type=float,default=WEIGHT_DECAY)
    parser.add_argument("--patience",type=int,default=PATIENCE)
    parser.add_argument("--resume",type=str,default=None)
    args = parser.parse_args(); _seed(RANDOM_SEED)

    print("="*64); print("  ResNet50 Pneumonia Training (Cleaned Dataset)"); print("="*64)
    print(f"  Device: {DEVICE}");
    if DEVICE.type=="cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dataset: {DATA_DIR}\n  Epochs: {args.epochs}\n  Batch: {args.batch_size}")
    print(f"  LR: {args.lr}\n  Model: {MODEL_VERSION}\n"); verify()

    tr_ds=datasets.ImageFolder(str(TRAIN_DIR),transform=get_transforms(True))
    va_ds=datasets.ImageFolder(str(VAL_DIR),transform=get_transforms(False))
    assert list(tr_ds.class_to_idx.keys())==CLASS_NAMES
    tr_ld=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    va_ld=DataLoader(va_ds,batch_size=args.batch_size,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
    print(f"\n  Train: {len(tr_ds)} imgs, {len(tr_ld)} batches\n  Val: {len(va_ds)} imgs, {len(va_ld)} batches\n")

    model=build_model(); print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    cw=compute_weights(TRAIN_DIR).to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=cw)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2)
    scaler=torch.amp.GradScaler("cuda") if DEVICE.type=="cuda" else None

    start_ep=0; best_f1=0.0; no_imp=0
    if args.resume:
        ck=Path(args.resume)
        if ck.exists():
            ckpt=torch.load(ck,map_location=DEVICE,weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"]); opt.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt: sched.load_state_dict(ckpt["scheduler_state_dict"])
            start_ep=ckpt.get("epoch",0)+1; best_f1=ckpt.get("metrics",{}).get("f1",0.0)
            print(f"  Resumed at epoch {start_ep}, best F1: {best_f1:.4f}\n")

    CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True); OUT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]; fields=["epoch","train_loss","val_loss","val_accuracy","val_precision","val_recall","val_specificity","val_f1","val_auc","lr"]
    print("Starting training...\n"); t0=time.time()

    for ep in range(start_ep, args.epochs):
        print(f"Epoch {ep+1}/{args.epochs}")
        tl,ta=train_epoch(model,tr_ld,crit,opt,scaler); vm=validate(model,va_ld,crit)
        lr=opt.param_groups[0]["lr"]; sched.step(vm["f1"])
        print(f"  Train Loss: {tl:.4f}  Acc: {ta:.4f}")
        print(f"  Val   Loss: {vm['loss']:.4f}  Acc: {vm['accuracy']:.4f}  F1: {vm['f1']:.4f}  Recall: {vm['recall']:.4f}  AUC: {vm['auc']:.4f}  LR: {lr:.2e}")
        rows.append({"epoch":ep+1,"train_loss":round(tl,4),"val_loss":round(vm["loss"],4),
            "val_accuracy":round(vm["accuracy"],4),"val_precision":round(vm["precision"],4),
            "val_recall":round(vm["recall"],4),"val_specificity":round(vm["specificity"],4),
            "val_f1":round(vm["f1"],4),"val_auc":round(vm["auc"],4),"lr":lr})
        if vm["f1"]>best_f1:
            best_f1=vm["f1"]; no_imp=0
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sched.state_dict(),"class_to_idx":tr_ds.class_to_idx,"metrics":vm,
                "preprocessing":{"image_size":IMAGE_SIZE,"normalize_mean":IMAGENET_MEAN,"normalize_std":IMAGENET_STD},
                "model_version":MODEL_VERSION,"architecture":"resnet50"},BEST_CKPT)
            print(f"  >> Saved best (F1={best_f1:.4f})")
        else: no_imp+=1; print(f"  No improvement ({no_imp}/{args.patience})")
        if no_imp>=args.patience: print("\n  Early stopping."); break
        print()

    with open(OUT_DIR/"training_metrics.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: training_metrics.csv")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ep=[r["epoch"] for r in rows]; fig,axes=plt.subplots(1,3,figsize=(18,5))
        axes[0].plot(ep,[r["train_loss"] for r in rows],"b-o",label="Train"); axes[0].plot(ep,[r["val_loss"] for r in rows],"r-o",label="Val")
        axes[0].set_title("Loss",fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep,[r["val_f1"] for r in rows],"b-o",label="F1"); axes[1].plot(ep,[r["val_recall"] for r in rows],"r-o",label="Recall")
        axes[1].plot(ep,[r["val_accuracy"] for r in rows],"g-o",label="Acc"); axes[1].set_title("Metrics",fontweight="bold"); axes[1].legend(); axes[1].set_ylim(0,1); axes[1].grid(alpha=0.3)
        axes[2].plot(ep,[r["val_auc"] for r in rows],"m-o",label="AUC"); axes[2].set_title("AUC",fontweight="bold"); axes[2].legend(); axes[2].set_ylim(0,1); axes[2].grid(alpha=0.3)
        plt.suptitle("Training Curves -- ResNet50",fontsize=14,fontweight="bold"); plt.tight_layout()
        plt.savefig(OUT_DIR/"training_curves.png",dpi=130); plt.close(); print(f"  Saved: training_curves.png")
    except ImportError: pass

    print(f"\n{'='*64}\n  Training complete in {(time.time()-t0)/60:.1f} min\n  Best F1: {best_f1:.4f}\n  Checkpoint: {BEST_CKPT}\n{'='*64}")

if __name__=="__main__": main()

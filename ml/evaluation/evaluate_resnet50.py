"""
Evaluate ResNet50 + threshold optimization.

Usage:
    cd hms-ai
    python -m ml.evaluation.evaluate_resnet50
"""
from __future__ import annotations
import csv, sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from ml.training.config import (BATCH_SIZE, CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR, NUM_CLASSES, NUM_WORKERS, TEST_DIR)

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "pneumonia_resnet50_best.pt"
OUT = METRICS_DIR / "resnet50"
VER = "pneumonia-resnet50-v1"

def load_model():
    if not CKPT.exists(): print(f"ERROR: {CKPT}"); sys.exit(1)
    m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False); m.load_state_dict(ck["model_state_dict"])
    m.to(DEVICE).eval(); print(f"  Loaded: epoch={ck.get('epoch','?')}, val_f1={ck.get('metrics',{}).get('f1','?')}")
    return m

@torch.no_grad()
def get_preds(model, loader):
    yl, ypr = [], []
    for imgs, lbls in tqdm(loader, desc="  Eval"):
        imgs = imgs.to(DEVICE)
        if DEVICE.type=="cuda":
            with torch.amp.autocast("cuda"): logits=model(imgs)
        else: logits=model(imgs)
        probs=F.softmax(logits,1).cpu().numpy(); yl.extend(lbls.numpy()); ypr.append(probs)
    return np.array(yl), np.vstack(ypr)

def met_at(yt, yp, t):
    pred=(yp>=t).astype(int); cm=confusion_matrix(yt,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    n=len(yt); acc=(tp+tn)/n; prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    f1=2*prec*rec/max(prec+rec,1e-8)
    try: auc=roc_auc_score(yt,yp)
    except: auc=0.0
    return {"threshold":round(t,4),"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
            "specificity":round(spec,4),"f1":round(f1,4),"auc":round(auc,4),"TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)}

def main():
    print("="*64); print("  ResNet50 Evaluation + Threshold Optimization"); print("="*64)
    print(f"  Device: {DEVICE}\n  Test: {TEST_DIR}\n"); OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    ds = datasets.ImageFolder(str(TEST_DIR), transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]))
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Test: {len(ds)} images\n")
    labels, probs = get_preds(model, ld); yp = probs[:,1]

    print("\n--- Classification Report (0.50) ---\n")
    print(classification_report(labels, (yp>=0.5).astype(int), target_names=CLASS_NAMES, digits=4))

    # Threshold sweep
    fine = [met_at(labels, yp, t) for t in np.arange(0.01, 1.0, 0.01)]
    with open(OUT/"threshold_optimization.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(fine[0].keys())); w.writeheader(); w.writerows(fine)
    print("  Saved: threshold_optimization.csv")

    bal = max(fine, key=lambda r: r["f1"])
    scr = max([r for r in fine if r["recall"]>=0.95], key=lambda r: r["specificity"], default=fine[0])
    dft = met_at(labels, yp, 0.50)

    print(f"\n  Default (0.50):   F1={dft['f1']:.4f}  Recall={dft['recall']:.4f}  Spec={dft['specificity']:.4f}  FP={dft['FP']}  FN={dft['FN']}")
    print(f"  Balanced ({bal['threshold']:.2f}):  F1={bal['f1']:.4f}  Recall={bal['recall']:.4f}  Spec={bal['specificity']:.4f}  FP={bal['FP']}  FN={bal['FN']}")
    print(f"  Screening ({scr['threshold']:.2f}): F1={scr['f1']:.4f}  Recall={scr['recall']:.4f}  Spec={scr['specificity']:.4f}  FP={scr['FP']}  FN={scr['FN']}")

    # Save metrics JSON
    json.dump({"model":VER,"optimal_threshold":bal["threshold"],"balanced":bal,"screening":scr,"default":dft},
              open(OUT/"metrics.json","w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import seaborn as sns
        # Threshold plot
        fig,axes=plt.subplots(1,2,figsize=(16,6)); ts=[r["threshold"] for r in fine]
        axes[0].plot(ts,[r["recall"] for r in fine],"r-",lw=2,label="Recall")
        axes[0].plot(ts,[r["specificity"] for r in fine],"b-",lw=2,label="Specificity")
        axes[0].plot(ts,[r["f1"] for r in fine],"g-",lw=2,label="F1")
        axes[0].axvline(bal["threshold"],color="green",ls="--",alpha=0.5,label=f"Best F1={bal['threshold']:.2f}")
        axes[0].set_title("Metrics vs Threshold",fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0,1.05)
        axes[1].plot(ts,[r["FP"] for r in fine],"b-",lw=2,label="FP"); axes[1].plot(ts,[r["FN"] for r in fine],"r-",lw=2,label="FN")
        axes[1].set_title("Errors vs Threshold",fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.suptitle("ResNet50 Threshold Optimization",fontsize=14,fontweight="bold"); plt.tight_layout()
        plt.savefig(OUT/"threshold_optimization.png",dpi=130); plt.close(); print("  Saved: threshold_optimization.png")
        # Confusion matrix
        cm=confusion_matrix(labels,(yp>=bal["threshold"]).astype(int)); fig,ax=plt.subplots(figsize=(7,6))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=ax,annot_kws={"size":16})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"ResNet50 Confusion (t={bal['threshold']:.2f}, F1={bal['f1']:.4f})",fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=150); plt.close(); print("  Saved: confusion_matrix.png")
        # ROC
        fpr,tpr,_=roc_curve(labels,yp); fig,ax=plt.subplots(figsize=(7,6))
        ax.plot(fpr,tpr,"b-",lw=2.5,label=f"AUC={bal['auc']:.4f}"); ax.plot([0,1],[0,1],"--",color="gray")
        ax.set_title("ROC -- ResNet50",fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT/"roc_curve.png",dpi=150); plt.close(); print("  Saved: roc_curve.png")
    except ImportError: pass

    report = f"""# ResNet50 Evaluation Report\n\n## Model: {VER}\n- Checkpoint: {CKPT.name}\n- Test: {len(ds)} images\n- Optimal threshold: {bal['threshold']}\n
| Metric | Default (0.50) | Balanced ({bal['threshold']:.2f}) | Screening ({scr['threshold']:.2f}) |
|--------|---------------|------|------|
| Accuracy | {dft['accuracy']:.4f} | {bal['accuracy']:.4f} | {scr['accuracy']:.4f} |
| Precision | {dft['precision']:.4f} | {bal['precision']:.4f} | {scr['precision']:.4f} |
| Recall | {dft['recall']:.4f} | {bal['recall']:.4f} | {scr['recall']:.4f} |
| Specificity | {dft['specificity']:.4f} | {bal['specificity']:.4f} | {scr['specificity']:.4f} |
| F1 | {dft['f1']:.4f} | {bal['f1']:.4f} | {scr['f1']:.4f} |
| AUC | {dft['auc']:.4f} | {bal['auc']:.4f} | {scr['auc']:.4f} |
| FP | {dft['FP']} | {bal['FP']} | {scr['FP']} |
| FN | {dft['FN']} | {bal['FN']} | {scr['FN']} |
"""
    with open(OUT/"evaluation_report.md","w",encoding="utf-8") as f: f.write(report)
    print("  Saved: evaluation_report.md")
    print(f"\n{'='*64}\n  Evaluation complete.\n{'='*64}")

if __name__=="__main__": main()

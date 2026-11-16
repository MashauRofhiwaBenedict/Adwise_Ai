"""
05_solution_techniques.py  -  RUBRIC: SOLUTION TECHNIQUES (5 marks)
Shows, with measured results, how the AI model's accuracy is improved: class weights, random under/over-sampling
(book ch.5 'Class Imbalance'), feature engineering, hyper-parameter tuning (GridSearchCV) and threshold tuning.
Run:  python 05_solution_techniques.py      (takes about 1-3 minutes)
"""
import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, recall_score, precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0])
const = [c for c in df.columns if df[c].nunique() == 1]; df = df.drop(columns=const)
y = df.Conversion; X = pd.get_dummies(df.drop(columns=["CustomerID", "Conversion"]), drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   # SAME split as script 03
def score(name, proba, rows, thr=0.5):
    flag = proba < thr; non = (yte == 0).values
    rows.append((name, round(roc_auc_score(yte, proba), 3), round((flag & non).sum() / non.sum(), 3), round((flag & non).sum() / max(flag.sum(), 1), 3)))
    print(f"{name:52s} ROC-AUC {rows[-1][1]:.3f} | recall(non-conv) {rows[-1][2]:.3f} | precision {rows[-1][3]:.3f}")
rows = []; print("Recall/precision are for the NON-converting class at the stated threshold.\n")

gb = lambda **k: GradientBoostingClassifier(random_state=42, **k)
m0 = gb().fit(Xtr, ytr); p0 = m0.predict_proba(Xte)[:, 1]; score("1. Baseline Gradient Boosting (threshold 0.5)", p0, rows)
score("2. Baseline + threshold tuning (0.80)", p0, rows, 0.8)

w = compute_sample_weight("balanced", ytr); m1 = gb().fit(Xtr, ytr, sample_weight=w); p1 = m1.predict_proba(Xte)[:, 1]; score("3. Class weights (balanced), threshold 0.5", p1, rows)

tr = pd.concat([Xtr, ytr], axis=1); maj, mino = tr[tr.Conversion == 1], tr[tr.Conversion == 0]
under = pd.concat([resample(maj, n_samples=len(mino), replace=False, random_state=1), mino]); mu = gb().fit(under.drop(columns="Conversion"), under.Conversion)
score("4. Random UNDER-sampling (majority cut to minority size)", mu.predict_proba(Xte)[:, 1], rows)
over = pd.concat([maj, resample(mino, n_samples=len(maj), replace=True, random_state=1)]); mo = gb().fit(over.drop(columns="Conversion"), over.Conversion)
score("5. Random OVER-sampling (minority replicated)", mo.predict_proba(Xte)[:, 1], rows)

def feats(d):
    d = d.copy(); d["SpendPerVisit"] = d.AdSpend / (d.WebsiteVisits + 1); d["EmailClickRate"] = d.EmailClicks / (d.EmailOpens + 1)
    d["Engagement"] = d.TimeOnSite * d.PagesPerVisit; d["LoyaltyPerPurchase"] = d.LoyaltyPoints / (d.PreviousPurchases + 1); return d
mf = gb().fit(feats(Xtr), ytr); pf = mf.predict_proba(feats(Xte))[:, 1]; score("6. Feature engineering (4 new features)", pf, rows)

print("\nGridSearchCV (3-fold, scoring = ROC-AUC) ...")
grid = GridSearchCV(GradientBoostingClassifier(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4], "subsample": [0.8]},
                    cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
print("Best parameters:", grid.best_params_, "| CV ROC-AUC %.3f" % grid.best_score_)
pt = grid.best_estimator_.predict_proba(Xte)[:, 1]; score("7. Hyper-parameter tuning, threshold 0.5", pt, rows)
score("8. Tuned model + threshold 0.80", pt, rows, 0.8)
def f1_non(th):
    flag = pt < th; non = (yte == 0).values; tp = (flag & non).sum(); pr = tp / max(flag.sum(), 1); rc = tp / non.sum(); return 2 * pr * rc / (pr + rc) if pr + rc else 0
best_t = max(np.arange(0.5, 0.95, 0.05), key=f1_non)
score("9. Tuned model + F1-optimal threshold %.2f" % best_t, pt, rows, best_t)

res = pd.DataFrame(rows, columns=["Technique", "ROC_AUC", "Recall_nonconv", "Precision_nonconv"]); res.to_csv(os.path.join(OUT, "05_solution_techniques.csv"), index=False)
ax = res.set_index("Technique")[["ROC_AUC", "Recall_nonconv", "Precision_nonconv"]].plot.barh(figsize=(10, 6)); plt.title("Effect of each improvement technique"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "05_solution_techniques.png"), dpi=150); show_chart(os.path.join(OUT, "05_solution_techniques.png"))
print("\nSaved outputs/05_solution_techniques.csv and .png")

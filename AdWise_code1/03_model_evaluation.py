"""
03_model_evaluation.py  -  RUBRIC: MODEL - how accuracy is evaluated (part of the 10 marks)
Trains the chosen model (Gradient Boosting) and evaluates it with: confusion matrix, accuracy, precision, recall,
F1, ROC-AUC, ROC curve, threshold analysis, business savings, leakage check and group fairness check (book ch.5).
Run:  python 03_model_evaluation.py
Outputs: models/adwise_model.joblib, outputs/oof_scores.csv (honest out-of-fold scores used by the chatbot), charts.
"""
import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); MOD = os.path.join(HERE, "models")
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0])
const = [c for c in df.columns if df[c].nunique() == 1]
y = df.Conversion; X = pd.get_dummies(df.drop(columns=const + ["CustomerID", "Conversion"]), drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y); spend_te = df.loc[Xte.index, "AdSpend"]

model = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr); p = model.predict_proba(Xte)[:, 1]
print("=== 1. HEADLINE METRICS (unseen test set, %d customers) ===" % len(yte))
print("Majority-class baseline accuracy : %.3f" % max(yte.mean(), 1 - yte.mean()))
print("Accuracy at threshold 0.5        : %.3f  (looks good but is barely above baseline)" % accuracy_score(yte, p >= .5))
print("ROC-AUC                          : %.3f  (0.5 = guessing, 1.0 = perfect)" % roc_auc_score(yte, p))
cvp = cross_val_predict(GradientBoostingClassifier(random_state=42), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
print("ROC-AUC, 5-fold out-of-fold      : %.3f" % roc_auc_score(y, cvp))

print("\n=== 2. THRESHOLD ANALYSIS (flag customer as 'unlikely to convert' when probability < threshold) ===")
rows = []
for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
    flag = p < t; non = (yte == 0).values
    tp = (flag & non).sum(); fp = (flag & ~non).sum()
    waste_caught = spend_te[flag & non].sum() / spend_te[non].sum()
    saved = spend_te[flag & non].sum(); breakeven = saved / fp if fp else np.nan
    rows.append((t, flag.sum(), tp / non.sum(), tp / max(flag.sum(), 1), waste_caught, fp / (~non).sum(), breakeven))
th = pd.DataFrame(rows, columns=["Threshold", "Flagged", "Recall_nonconv", "Precision_nonconv", "Waste_caught", "Converters_lost", "Breakeven_profit_per_sale"])
print(th.round(3).to_string(index=False)); th.to_csv(os.path.join(OUT, "03_threshold_analysis.csv"), index=False)
T = 0.80; pred = (p >= T).astype(int)
print("\n=== 3. CHOSEN THRESHOLD = %.2f ===" % T); cm = confusion_matrix(yte, pred); print("Confusion matrix [rows actual 0/1, cols predicted 0/1]:\n", cm)
print(classification_report(yte, pred, target_names=["No conversion", "Conversion"]))
print("Break-even: the saving only beats the lost sales if profit per conversion is below ~%.0f monetary units." % th.loc[th.Threshold == T, "Breakeven_profit_per_sale"].iloc[0])

print("\n=== 4. LEAKAGE CHECK (risk: rate columns may be calculated from the outcome) ===")
for label, drop in {"all features": [], "without ConversionRate": ["ConversionRate"], "without ConversionRate + ClickThroughRate": ["ConversionRate", "ClickThroughRate"]}.items():
    m = GradientBoostingClassifier(random_state=42).fit(Xtr.drop(columns=drop), ytr)
    print(f"  {label:45s} ROC-AUC {roc_auc_score(yte, m.predict_proba(Xte.drop(columns=drop))[:, 1]):.3f}")

print("\n=== 5. FAIRNESS CHECK (ROC-AUC by group on the test set) ===")
g = df.loc[Xte.index].assign(p=p, y=yte); g["AgeBand"] = pd.cut(g.Age, [17, 30, 45, 60, 70], labels=["18-30", "31-45", "46-60", "61-69"])
for col in ("Gender", "AgeBand", "CampaignChannel"):
    print(" ", col, {str(k): round(roc_auc_score(s.y, s.p), 3) for k, s in g.groupby(col, observed=True) if s.y.nunique() == 2})

print("\n=== 6. CHARTS AND SAVED FILES ===")
fpr, tpr, _ = roc_curve(yte, p); fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(fpr, tpr, color="#1F4E79", label="AUC %.3f" % roc_auc_score(yte, p)); ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC curve"); ax[0].legend(); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[1].imshow(cm, cmap="Blues"); ax[1].set_title("Confusion matrix (threshold 0.80)"); ax[1].set_xticks([0, 1], ["flagged", "ok"]); ax[1].set_yticks([0, 1], ["no conv.", "converted"])
[ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14) for i in range(2) for j in range(2)]
imp = pd.Series(model.feature_importances_, X.columns).sort_values().tail(10); imp.plot.barh(ax=ax[2], color="#1F4E79"); ax[2].set_title("Top 10 feature importances")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "03_evaluation.png"), dpi=150); show_chart(os.path.join(OUT, "03_evaluation.png"))
pd.DataFrame({"CustomerID": df.CustomerID, "p_convert": cvp, "AdSpend": df.AdSpend, "Conversion": y, "CampaignChannel": df.CampaignChannel, "CampaignType": df.CampaignType}).to_csv(os.path.join(OUT, "oof_scores.csv"), index=False)
final = GradientBoostingClassifier(random_state=42).fit(X, y)
joblib.dump({"model": final, "columns": list(X.columns), "threshold": T, "medians": X.median().to_dict()}, os.path.join(MOD, "adwise_model.joblib"))
print("Saved models/adwise_model.joblib, outputs/oof_scores.csv, outputs/03_evaluation.png, 03_threshold_analysis.csv")

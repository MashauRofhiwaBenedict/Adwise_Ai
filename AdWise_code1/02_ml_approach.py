"""
02_ml_approach.py  -  RUBRIC: MACHINE LEARNING APPROACH (5 marks)
Compares the algorithms from the book (ch.3, 5, 8, 15) on the same data with 5-fold cross-validation,
then adds unsupervised learning (K-means clustering + silhouette score, ch.8) for customer segments.
Run:  python 02_ml_approach.py
"""
import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0])
const = [c for c in df.columns if df[c].nunique() == 1]
y = df.Conversion; X = pd.get_dummies(df.drop(columns=const + ["CustomerID", "Conversion"]), drop_first=True).astype(float)

print("TASK 1: SUPERVISED CLASSIFICATION  (predict Conversion = 1/0)")
print("Majority-class baseline accuracy: %.3f\n" % max(y.mean(), 1 - y.mean()))
sc = lambda m: make_pipeline(StandardScaler(), m)
models = {
 "Logistic Regression": sc(LogisticRegression(max_iter=1000, class_weight="balanced")),
 "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
 "Naive Bayes": GaussianNB(),
 "K-Nearest Neighbours": sc(KNeighborsClassifier(15)),
 "SVM (RBF kernel)": sc(SVC(class_weight="balanced")),
 "Random Forest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
 "Gradient Boosting": GradientBoostingClassifier(random_state=42),
 "Neural Network (MLP)": sc(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=1)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=1); rows = []
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "accuracy", "recall_macro"])
    rows.append((name, r["test_roc_auc"].mean(), r["test_accuracy"].mean(), r["test_recall_macro"].mean()))
    print(f"{name:22s} ROC-AUC {rows[-1][1]:.3f} | accuracy {rows[-1][2]:.3f} | macro recall {rows[-1][3]:.3f}")
res = pd.DataFrame(rows, columns=["Model", "ROC_AUC", "Accuracy", "Macro_recall"]).sort_values("ROC_AUC", ascending=False)
res.to_csv(os.path.join(OUT, "02_model_comparison.csv"), index=False)
print("\nBest model by ROC-AUC:", res.iloc[0, 0])
res.set_index("Model").ROC_AUC.sort_values().plot.barh(color="#1F4E79", figsize=(7, 4)); plt.xlim(0.5, 0.9)
plt.title("5-fold cross-validated ROC-AUC"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "02_model_comparison.png"), dpi=150); show_chart(os.path.join(OUT, "02_model_comparison.png"))

print("\nTASK 2: UNSUPERVISED LEARNING - K-MEANS CLUSTERING  (customer segments)")
cols = ["Age", "Income", "AdSpend", "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "EmailOpens", "PreviousPurchases", "LoyaltyPoints"]
Z = StandardScaler().fit_transform(df[cols])
for k in range(2, 8):
    print("  k =", k, " silhouette =", round(silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)), 3))
print("Silhouette near 0 means clusters overlap strongly: segments are descriptive, not sharp groups.")
df["Segment"] = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
prof = df.groupby("Segment")[["Age", "Income", "AdSpend", "WebsiteVisits", "LoyaltyPoints", "PreviousPurchases", "Conversion"]].mean().round(2)
prof["Customers"] = df.Segment.value_counts().sort_index(); print(prof.to_string()); prof.to_csv(os.path.join(OUT, "02_segments.csv"))
P = PCA(2).fit_transform(Z); plt.figure(figsize=(6, 5)); plt.scatter(P[:, 0], P[:, 1], c=df.Segment, cmap="tab10", s=6)
plt.title("Customer segments (PCA view)"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "02_segments.png"), dpi=150); show_chart(os.path.join(OUT, "02_segments.png"))
print("\nSaved outputs/02_model_comparison.csv/.png, 02_segments.csv/.png")

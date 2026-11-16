"""
01_data_preparation.py  -  RUBRIC: DATA (5 marks)
Loads the campaign dataset, checks its quality, explains every form of data,
and prepares it for machine learning (book ch.4: encoding, scaling, train/test split).
Run:  python 01_data_preparation.py
Outputs: outputs/clean_data.csv + charts (PNG)
"""
import os, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))
if not files: raise SystemExit("Put digital_marketing_campaign_dataset.csv in the same folder as this script.")
df = pd.read_csv(files[0])

print("=== 1. SHAPE AND TYPES ===")
print("Rows, columns:", df.shape); print(df.dtypes.to_string())
print("\n=== 2. DATA QUALITY ===")
print("Missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()),
      "| duplicate CustomerIDs:", int(df.CustomerID.duplicated().sum()))
const = [c for c in df.columns if df[c].nunique() == 1]; print("Constant columns (no information):", const)
print("\nNumeric summary:\n", df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())

print("\n=== 3. TARGET (Conversion) AND CLASS IMBALANCE ===")
vc = df.Conversion.value_counts(); print(vc.to_string()); print("Converted share: %.2f%%" % (100 * df.Conversion.mean()))
print("A model that always says 'converted' is already %.2f%% accurate -> accuracy alone is misleading." % (100 * df.Conversion.mean()))

print("\n=== 3b. ROW-ORDER CHECK (is the file in time order?) ===")
blk = df.sort_values("CustomerID").reset_index(drop=True).groupby(lambda i: i // 1000).Conversion.mean().round(3)
print("Conversion rate per block of 1,000 rows:", blk.tolist())
print("Correlation of CustomerID with Conversion: %.3f -> the LAST ~1,000 customers convert far less (%.0f%% vs %.0f%%)." % (df.CustomerID.corr(df.Conversion), 100 * blk.iloc[-1], 100 * blk.iloc[:-1].mean()))
print("So row order is NOT chronological. Always use a random split, never use CustomerID as a feature, and confirm with the data owner.")

print("\n=== 4. FORMS OF DATA IN THE DATASET ===")
groups = {"Demographic": ["Age", "Gender", "Income"], "Campaign": ["CampaignChannel", "CampaignType", "AdSpend"],
          "Engagement": ["ClickThroughRate", "ConversionRate", "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks"],
          "Loyalty": ["PreviousPurchases", "LoyaltyPoints"], "Target": ["Conversion"]}
for g, cols in groups.items(): print(f"{g:12s}", cols)
print("Categorical:", list(df.select_dtypes(exclude="number").columns)); print("No date/time column -> time series needs assumed dates (see script 04).")

print("\n=== 5. CORRELATION WITH CONVERSION ===")
corr = df.drop(columns=["CustomerID"]).corr(numeric_only=True)["Conversion"].drop("Conversion").sort_values()
print(corr.round(3).to_string())

print("\n=== 6. PREPARATION ===")
df = df.drop(columns=const + ["CustomerID"]); print("Dropped constant columns and CustomerID. Remaining:", df.shape[1], "columns")
# (a) label encoding as in the book (ch.4) - shown for Gender
le = preprocessing.LabelEncoder().fit(df.Gender); print("LabelEncoder on Gender:", dict(zip(le.classes_, le.transform(le.classes_))))
# (b) one-hot encoding used for modelling
X = pd.get_dummies(df.drop(columns="Conversion"), drop_first=True).astype(float); y = df.Conversion
print("After one-hot encoding:", X.shape[1], "input columns")
# (c) split first, then scale using TRAINING data only (avoids leakage)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", Xtr.shape, "Test:", Xte.shape, "| converted share train %.3f test %.3f" % (ytr.mean(), yte.mean()))
sc = preprocessing.StandardScaler().fit(Xtr); Z = sc.transform(Xtr)
print("Standard scaling -> train mean ~", round(float(Z.mean()), 6), "std ~", round(float(Z.std()), 3))
mm = preprocessing.MinMaxScaler().fit(Xtr[["AdSpend"]]); print("Min-max scaling of AdSpend: 0 to 1 range, e.g.", mm.transform(Xtr[["AdSpend"]])[:3].ravel().round(3))
pd.concat([X, y], axis=1).to_csv(os.path.join(OUT, "clean_data.csv"), index=False)

print("\n=== 7. CHARTS ===")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df.Conversion.value_counts().sort_index().plot.bar(ax=ax[0], color=["#C0392B", "#1F4E79"], rot=0); ax[0].set_title("Class balance (0 = no, 1 = converted)")
df.groupby("CampaignChannel").Conversion.mean().plot.bar(ax=ax[1], color="#1F4E79", rot=30); ax[1].set_title("Conversion rate by channel"); ax[1].set_ylim(0.8, 0.92)
corr.plot.barh(ax=ax[2], color="#1F4E79"); ax[2].set_title("Correlation with Conversion")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "01_data_overview.png"), dpi=150); show_chart(os.path.join(OUT, "01_data_overview.png"))
num = df.select_dtypes("number"); plt.figure(figsize=(8, 6.5)); plt.imshow(num.corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(num.shape[1]), num.columns, rotation=90, fontsize=7); plt.yticks(range(num.shape[1]), num.columns, fontsize=7)
plt.colorbar(); plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "01_correlation_matrix.png"), dpi=150); show_chart(os.path.join(OUT, "01_correlation_matrix.png"))
print("Saved outputs/clean_data.csv, 01_data_overview.png, 01_correlation_matrix.png")

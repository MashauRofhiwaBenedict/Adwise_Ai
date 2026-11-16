"""
04_time_series_analysis.py  -  RUBRIC: TIME SERIES ANALYSIS ON DATA (part of the 10 marks)
Follows book ch.11: pandas date_range, indexing/slicing, resampling, rolling mean, statistics, and a
Hidden Markov Model (hmmlearn).  Adds a forecast that is compared with a naive baseline.
IMPORTANT: the dataset has NO date column, and the ORDER OF ROWS IS NOT CHRONOLOGICAL (the last 1,000 rows convert
at ~62% versus ~91% for the rest, so using row order as time would create a FAKE trend). We therefore give every
50 randomly shuffled (fixed seed) customers one calendar day from 2026-01-01 (160 days). The numbers below show the
METHOD working; they are not real business trends. With real timestamps, only the block marked 'ASSUMED DATES' changes.
Run:  python 04_time_series_analysis.py             (shuffled, recommended)
      python 04_time_series_analysis.py --row-order (shows the fake trend that row order would create)
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0]).sort_values("CustomerID").reset_index(drop=True)
print("Conversion rate by block of 1,000 rows (CustomerID order):", df.groupby(df.index // 1000).Conversion.mean().round(2).tolist())
print("-> last block is much lower: row order is NOT time. Using seeded shuffle unless --row-order is given.\n")

# ---- ASSUMED DATES ---------------------------------------------------------------------------
per_day = 50
if "--row-order" not in sys.argv: df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df["Date"] = pd.date_range("2026-01-01", periods=int(np.ceil(len(df) / per_day)), freq="D").repeat(per_day)[:len(df)]
# -----------------------------------------------------------------------------------------------

print("=== 1. BUILD THE TIME SERIES (pandas Series indexed by date) ===")
daily = df.groupby("Date").agg(conv=("Conversion", "mean"), spend=("AdSpend", "sum"), customers=("Conversion", "size"))
daily["wasted"] = df.assign(w=df.AdSpend * (1 - df.Conversion)).groupby("Date").w.sum()
print(daily.head().round(3).to_string()); print("Days:", len(daily), "| from", daily.index.min().date(), "to", daily.index.max().date())

print("\n=== 2. SLICING (book: timeseries['1980':'1990']) ===")
print("March 2026 mean conversion rate:", round(daily.loc["2026-03", "conv"].mean(), 4))
print("Mean wasted spend per day, Jan:", round(daily.loc["2026-01", "wasted"].mean(), 0), "| Jun:", round(daily.loc["2026-06", "wasted"].mean(), 0))

print("\n=== 3. STATISTICS (mean, max, min, describe) ===")
print(daily.conv.describe().round(4).to_string())

print("\n=== 4. RE-SAMPLING (daily -> weekly -> monthly) ===")
weekly = daily.resample("W").agg({"conv": "mean", "spend": "sum", "wasted": "sum"}); monthly = daily.resample("MS").agg({"conv": "mean", "spend": "sum", "wasted": "sum"})
print("Weekly conversion (first 5):\n", weekly.conv.head().round(4).to_string()); print("Monthly:\n", monthly.round(3).to_string())
weekly.to_csv(os.path.join(OUT, "weekly_series.csv"))

print("\n=== 5. ROLLING MEAN AND CHANGE ===")
daily["roll7"] = daily.conv.rolling(window=7).mean(); daily["roll28"] = daily.conv.rolling(28).mean(); daily["pct_change"] = daily.conv.pct_change()
print(daily[["conv", "roll7", "roll28"]].dropna().tail(3).round(4).to_string())
print("Lag-1 autocorrelation of daily conversion:", round(daily.conv.autocorr(1), 3), "| lag-7:", round(daily.conv.autocorr(7), 3),
      "-> values near 0 mean no strong trend or weekly seasonality in this (synthetic) data")
trend = np.polyfit(np.arange(len(daily)), daily.conv.values, 1)[0]; print("Linear trend per day: %+.6f (conversion rate points)" % trend)

print("\n=== 6. FORECAST vs NAIVE BASELINE (hold out the last 14 days) ===")
train, test = daily.conv.iloc[:-14], daily.conv.iloc[-14:]
f_naive = np.repeat(train.iloc[-1], 14); f_mean = np.repeat(train.tail(28).mean(), 14); f_all = np.repeat(train.mean(), 14)
mae = lambda f: float(np.mean(np.abs(test.values - f)))
print("MAE last-value naive: %.4f | 28-day moving average: %.4f | overall mean: %.4f" % (mae(f_naive), mae(f_mean), mae(f_all)))
print("If a smoothed forecast is not clearly better than the naive one, the series has no usable trend/seasonality (expected for assumed dates).")
print("\n=== 6b. EARLY-WARNING CONTROL CHART (business use: alert when conversion drops unusually) ===")
mu, sd = daily.conv.mean(), daily.conv.std(); alerts = daily[daily.conv < mu - 3 * sd]
print("Centre line %.3f, lower control limit (mean - 3 SD) %.3f -> %d alert day(s)" % (mu, mu - 3 * sd, len(alerts))); print(alerts[["conv"]].round(3).to_string() if len(alerts) else "No alerts.")

print("\n=== 7. HIDDEN MARKOV MODEL (book ch.11, GaussianHMM) ===")
try:
    from hmmlearn.hmm import GaussianHMM
    Xh = daily[["conv", "spend"]].values; Xh = (Xh - Xh.mean(0)) / Xh.std(0)
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=500, random_state=7).fit(Xh); daily["state"] = hmm.predict(Xh)
    print("Days in each hidden state:", daily.state.value_counts().sort_index().to_dict()); print(daily.groupby("state")[["conv", "spend"]].mean().round(4).to_string())
    print("Transition matrix:\n", hmm.transmat_.round(3))
except ImportError: print("hmmlearn not installed (pip install hmmlearn). Skipping."); daily["state"] = 0

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
daily[["conv", "roll7", "roll28"]].plot(ax=ax[0], alpha=.9); ax[0].set_title("Daily conversion rate, 7-day and 28-day rolling means (assumed dates, shuffled rows)")
weekly.conv.plot(ax=ax[1], marker="o", color="#1F4E79"); ax[1].set_title("Weekly resampled conversion rate")
ax[0].axhline(mu - 3 * sd, color="red", ls="--", lw=1); ax[2].scatter(daily.index, daily.spend / 1e3, c=daily.state, cmap="coolwarm", s=12); ax[2].set_title("Daily ad spend (thousands) coloured by HMM hidden state")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "04_time_series.png"), dpi=150); show_chart(os.path.join(OUT, "04_time_series.png"))
print("\nSaved outputs/04_time_series.png and outputs/weekly_series.csv")

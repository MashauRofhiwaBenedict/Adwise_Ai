"""
09_deep_learning_lstm.py  -  RUBRIC: DEEP LEARNING (2 of 3)  - recurrent network (LSTM) for TIME SERIES forecasting
Predicts tomorrow's conversion rate from the previous 7 days and compares with simple baselines.
IMPORTANT: the dataset has no dates, so the daily series uses assumed dates on shuffled rows (see script 04).
The result shows the METHOD; real time-stamped campaign data is needed for real forecasts.
If TensorFlow is not installed (not available for Python 3.13/3.14) a scikit-learn neural-network forecaster is used instead.
Run:  python 09_deep_learning_lstm.py
"""
import os, glob, warnings; warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart
from sklearn.neural_network import MLPRegressor
try:
    import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers; HAVE_TF = True; tf.keras.utils.set_random_seed(42)
except ImportError: HAVE_TF = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0]).sample(frac=1, random_state=7).reset_index(drop=True)   # shuffled: row order is not time
df["Date"] = pd.date_range("2026-01-01", periods=160).repeat(50)[:len(df)]; s = df.groupby("Date").Conversion.mean().values.astype("float32")
W = 7; Xs = np.array([s[i:i + W] for i in range(len(s) - W)])[..., None]; ys = s[W:]; n_test = 21
Xtr, ytr, Xte, yte = Xs[:-n_test], ys[:-n_test], Xs[-n_test:], ys[-n_test:]; mu, sd = Xtr.mean(), Xtr.std()
if HAVE_TF:
    name = "LSTM forecast"
    net = keras.Sequential([layers.Input(shape=(W, 1)), layers.LSTM(16), layers.Dense(1)]); net.compile(optimizer="adam", loss="mae"); net.summary()
    net.fit((Xtr - mu) / sd, (ytr - mu) / sd, epochs=150, batch_size=16, verbose=0, validation_split=0.15, callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    pred = net.predict((Xte - mu) / sd, verbose=0).ravel() * sd + mu
else:
    name = "Neural-network forecast (no TensorFlow)"
    print("TensorFlow is not installed for this Python version: using a scikit-learn neural network on the 7-day windows instead of the LSTM.")
    print("For the real LSTM use Python 3.10-3.12 and run: pip install tensorflow\n")
    m = MLPRegressor(hidden_layer_sizes=(16,), max_iter=2000, early_stopping=True, random_state=42).fit(((Xtr - mu) / sd).reshape(len(Xtr), W), (ytr - mu) / sd)
    pred = m.predict(((Xte - mu) / sd).reshape(len(Xte), W)) * sd + mu
mae = lambda f: float(np.mean(np.abs(yte - f)))
print("\nMAE on the last %d days (lower is better):" % n_test); print("  %-38s: %.4f" % (name, mae(pred))); print("  Naive (yesterday's value)             : %.4f" % mae(Xte[:, -1, 0]))
print("  Mean of previous 7 days               : %.4f" % mae(Xte.mean(axis=(1, 2)))); print("  Training-set mean                     : %.4f" % mae(np.repeat(ytr.mean(), n_test)))
print("If the network is not clearly better than the mean, the series has no learnable pattern (expected here). With real time-stamped data the same code is used.")
plt.figure(figsize=(9, 3.5)); plt.plot(yte, "o-", label="actual"); plt.plot(pred, "s--", label=name); plt.legend(); plt.title("One-day-ahead forecast (assumed dates)"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_lstm_forecast.png"), dpi=150); show_chart(os.path.join(OUT, "09_lstm_forecast.png")); print("Saved outputs/09_lstm_forecast.png")

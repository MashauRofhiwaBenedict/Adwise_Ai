"""
08_deep_learning_ann.py  -  RUBRIC: DEEP LEARNING (1 of 3 scripts: 08 ANN, 09 LSTM, 10 CNN)
Keras artificial neural network (book ch.15 + ch.19) that predicts conversion: Sequential model, Dense layers with ReLU,
Dropout against over-fitting, sigmoid output, Adam optimiser, binary cross-entropy, early stopping, class weights.
Compared fairly with Gradient Boosting on the SAME test set as scripts 03 and 05.
If TensorFlow is not installed (it does not support Python 3.13/3.14 yet) the script automatically uses the
scikit-learn neural network instead and says so. For Keras use Python 3.10 - 3.12:  pip install tensorflow
Run:  python 08_deep_learning_ann.py
"""
import os, glob, warnings; warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from show_utils import show_chart
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
try:
    import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers; HAVE_TF = True; tf.keras.utils.set_random_seed(42)
except ImportError: HAVE_TF = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0]); df = df.drop(columns=[c for c in df.columns if df[c].nunique() == 1])
y = df.Conversion; X = pd.get_dummies(df.drop(columns=["CustomerID", "Conversion"]), drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)            # neural networks need scaled inputs

if HAVE_TF:
    name = "Keras neural network"
    net = keras.Sequential([layers.Input(shape=(A.shape[1],)), layers.Dense(64, activation="relu"), layers.Dropout(0.3),
                            layers.Dense(32, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
    net.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")]); net.summary()
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}; print("Class weights:", {k: round(v, 2) for k, v in cw.items()})
    hist = net.fit(A, ytr, validation_split=0.2, epochs=100, batch_size=64, class_weight=cw, verbose=0,
                   callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True)])
    print("Training stopped after", len(hist.history["loss"]), "epochs (early stopping)")
    p = net.predict(B, verbose=0).ravel(); curves = (hist.history["loss"], hist.history["val_loss"], hist.history["auc"], hist.history["val_auc"], "Loss", "ROC-AUC")
else:
    name = "scikit-learn neural network"
    print("TensorFlow is not installed for this Python version, so the scikit-learn neural network (same idea: layers of neurons) is used instead.")
    print("For the Keras version use Python 3.10-3.12 and run: pip install tensorflow\n")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, max_iter=300, random_state=42).fit(A, ytr)
    p = mlp.predict_proba(B)[:, 1]; print("Training stopped after", mlp.n_iter_, "epochs (early stopping)")
    curves = (mlp.loss_curve_, None, None, mlp.validation_scores_, "Training loss", "Validation accuracy")
gb = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr); pg = gb.predict_proba(Xte)[:, 1]
print("\nTEST SET RESULTS (unseen 1,600 customers)")
for nm, pr, thr in [(name, p, 0.5), ("Gradient Boosting", pg, 0.8)]:
    flag = pr < thr; non = (yte == 0).values; tp = (flag & non).sum()
    print(f"  {nm:28s} ROC-AUC {roc_auc_score(yte, pr):.3f} | non-converter recall {tp / non.sum():.3f} | precision {tp / max(flag.sum(), 1):.3f} (threshold {thr})")
print("Conclusion: on a small table of 8,000 rows tree ensembles usually beat neural networks; deep learning is kept for text, images and sequences.")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); l, vl, a, va, t1, t2 = curves
ax[0].plot(l, label="train"); (ax[0].plot(vl, label="validation") if vl is not None else None); ax[0].set_title(t1); ax[0].legend()
(ax[1].plot(a, label="train") if a is not None else None); ax[1].plot(va, label="validation"); ax[1].set_title(t2); ax[1].legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_ann_training.png"), dpi=150); show_chart(os.path.join(OUT, "08_ann_training.png")); print("Saved outputs/08_ann_training.png")

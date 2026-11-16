"""
11_marketbot_chatbot.py  -  RUBRIC: OTHER FEATURES (Chatbot / Softbot) and PRACTICAL SOLUTION
MarketBot: a softbot that lets a non-technical manager ask questions about campaign waste, channels, customers.
Uses: NLP intent classifier (nlp_utils.py, TF-IDF + Naive Bayes), the Gradient Boosting model saved by 03,
      honest out-of-fold scores from 03, customer segments from 02, optional speech input/output (ch.12).
First run  03_model_evaluation.py  and  02_ml_approach.py  (or run_all.py), then:
Run:  python 11_marketbot_chatbot.py               opens the MarketBot CHAT WINDOW (pop-up)
      python 11_marketbot_chatbot.py --text        chat in the terminal instead
      python 11_marketbot_chatbot.py --voice       terminal chat with spoken questions (microphone + internet)
      python 11_marketbot_chatbot.py --speak       bot reads answers aloud
      python 11_marketbot_chatbot.py --demo        scripted demo conversation in the terminal (good for screenshots)
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import nlp_utils as nu
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); MOD = os.path.join(HERE, "models")
try:
    DF = pd.read_csv(sorted(glob.glob(os.path.join(HERE, "digital_marketing_campaign_dataset*.csv")))[0]); PKG = joblib.load(os.path.join(MOD, "adwise_model.joblib"))
    OOF = pd.read_csv(os.path.join(OUT, "oof_scores.csv")); SEG = pd.read_csv(os.path.join(OUT, "02_segments.csv"))
except Exception as e: raise SystemExit("Missing file (%s).\nRun  python run_all.py  (or scripts 02 and 03) first, with the CSV in this folder." % e)
MODEL, COLS, THR, MED = PKG["model"], PKG["columns"], PKG["threshold"], PKG["medians"]; DF["Wasted"] = DF.AdSpend * (1 - DF.Conversion); INTENT = nu.build_model(); MIN_CONF = 0.15
HELP = ("I can: (1) predict a customer's conversion chance, e.g. 'predict a customer with 25 visits, time on site 12 and 5 purchases'; (2) compare channels or campaign types; "
        "(3) show wasted ad spend; (4) show what drives conversion; (5) describe customer segments; (6) recommend where to cut spend. Type 'bye' to exit.")

def do_predict(text):
    e = {k: v for k, v in nu.extract_entities(text).items() if k in MED}
    if not e: return "Give me some details, e.g. 'predict a customer aged 40 with 25 visits, time on site 12 and 5 purchases'."
    row = dict(MED); row.update(e); p = MODEL.predict_proba(pd.DataFrame([row])[COLS])[0, 1]
    v = "LOW conversion chance - consider NOT spending on this customer." if p < THR else "Good conversion chance - worth targeting."
    return "Estimated conversion probability: %.0f%%. %s\n(Used: %s; all other details set to typical values, so treat this as a rough estimate.)" % (100 * p, v, ", ".join("%s=%g" % kv for kv in e.items()))
def do_channel(_):
    g = DF.groupby("CampaignChannel").agg(rate=("Conversion", "mean"), spend=("AdSpend", "sum")).sort_values("rate", ascending=False)
    return "Conversion rate by channel:\n" + "\n".join("  %s: %.1f%% (spend %s)" % (i, 100 * r.rate, format(r.spend, ",.0f")) for i, r in g.iterrows()) + "\nThe differences are small (under 2 percentage points), so channel alone does not explain waste."
def do_type(_):
    g = DF.groupby("CampaignType").Conversion.mean().sort_values(ascending=False); return "Conversion rate by campaign type:\n" + "\n".join("  %s: %.1f%%" % (i, 100 * v) for i, v in g.items())
def do_waste(_):
    g = DF.groupby("CampaignChannel").Wasted.sum().sort_values(ascending=False); t = DF.Wasted.sum()
    return "Total wasted spend (spend on customers who did not convert): %s = %.1f%% of all spend.\n%s" % (format(t, ",.0f"), 100 * t / DF.AdSpend.sum(), "\n".join("  %s: %s" % (i, format(v, ",.0f")) for i, v in g.items()))
def do_drivers(_):
    imp = pd.Series(MODEL.feature_importances_, COLS).sort_values(ascending=False).head(5); return "Top 5 drivers of conversion (model importance):\n" + "\n".join("  %s: %.1f%%" % (i, 100 * v) for i, v in imp.items())
def do_segment(_):
    return "Customer segments (K-means, 4 groups; they overlap, so treat as descriptive):\n" + "\n".join(
        "  Segment %d: %d customers, avg income %s, avg loyalty points %s, conversion %.0f%%" % (r.Segment, r.Customers, format(r.Income, ",.0f"), format(r.LoyaltyPoints, ",.0f"), 100 * r.Conversion) for r in SEG.itertuples())
def do_recommend(_):
    o = OOF; non = o.Conversion == 0; fl = o.p_convert < THR; saved = o.AdSpend[fl & non].sum(); lost = int((fl & ~non).sum())
    top = o[fl].groupby("CampaignChannel").AdSpend.sum().sort_values(ascending=False)
    return ("At threshold %.2f the model flags %d customers (%.0f%% of spend) as unlikely to convert. %.0f%% of them truly would not convert.\n"
            "Not advertising to them would avoid %s of spend (%.0f%% of all wasted spend) but would also lose %d customers who would have converted.\n"
            "Most flagged spend: %s (%s). Act on this only if your profit per sale is below about %s, otherwise the lost sales cost more than the saving."
            % (THR, fl.sum(), 100 * o.AdSpend[fl].sum() / o.AdSpend.sum(), 100 * (fl & non).sum() / fl.sum(), format(saved, ",.0f"), 100 * saved / o.AdSpend[non].sum(), lost, top.index[0], format(top.iloc[0], ",.0f"), format(saved / max(lost, 1), ",.0f")))
ACT = {"predict": do_predict, "channel": do_channel, "type": do_type, "waste": do_waste, "drivers": do_drivers, "segment": do_segment, "recommend": do_recommend}
def reply(text):
    lab, conf = nu.classify(INTENT, text, MIN_CONF)
    if lab == "unknown": return "Sorry, I did not understand that. " + HELP
    return {"greet": "Hello! I am MarketBot. " + HELP, "help": HELP, "bye": "BYE"}.get(lab) or ACT[lab](text)

def listen():
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as src: r.adjust_for_ambient_noise(src); print("Listening ..."); return r.recognize_google(r.listen(src, timeout=8))
def say(t):
    try: import pyttsx3; e = pyttsx3.init(); e.say(t); e.runAndWait()
    except Exception: pass

def gui(speak=False):
    """Pop-up chat window (tkinter comes with Python). Enter or the Send button asks; the Mic button listens."""
    import tkinter as tk; from tkinter import scrolledtext
    root = tk.Tk(); root.title("MarketBot - AdWise AI"); root.geometry("760x620")
    box = scrolledtext.ScrolledText(root, wrap="word", state="disabled", font=("Arial", 11), padx=8, pady=8); box.pack(fill="both", expand=True, padx=8, pady=(8, 4))
    box.tag_config("you", foreground="#12304F", font=("Arial", 11, "bold")); box.tag_config("bot", foreground="#1B6B67")
    def add(who, text):
        box.config(state="normal"); box.insert("end", ("You: " if who == "you" else "MarketBot: ") + text + "\n\n", who); box.config(state="disabled"); box.see("end")
    row = tk.Frame(root); row.pack(fill="x", padx=8, pady=(0, 8)); entry = tk.Entry(row, font=("Arial", 12)); entry.pack(side="left", fill="x", expand=True, ipady=5)
    def ask(q):
        q = q.strip()
        if not q: return
        add("you", q); a = reply(q)
        if a == "BYE": add("bot", "Goodbye!"); root.after(900, root.destroy); return
        add("bot", a)
        if speak: root.update(); say(a)
    def send(_=None): q = entry.get(); entry.delete(0, "end"); ask(q)
    def mic():
        try: add("bot", "Listening ... speak now."); root.update(); ask(listen())
        except Exception as ex: add("bot", "Could not use the microphone (needs PyAudio and internet): %s" % ex)
    entry.bind("<Return>", send); tk.Button(row, text="Send", command=send, width=8).pack(side="left", padx=(6, 0)); tk.Button(row, text="Mic", command=mic, width=6).pack(side="left", padx=(6, 0))
    add("bot", "Hello! I am MarketBot. " + HELP); entry.focus(); root.mainloop()

if __name__ == "__main__":
    a = sys.argv[1:]; voice, speak = "--voice" in a, "--speak" in a
    if "--demo" in a:
        qs = ["hi", "how much money are we wasting", "which channel performs best", "what drives conversion", "describe our customer segments",
              "predict a customer with click through rate 0.02, conversion rate 0.01, 3 visits, time on site 1, 1 pages, 0 purchases, 1 email opens",
              "predict a customer aged 40 with 25 visits, time on site 12 and 5 purchases", "where should I cut spend", "pizza recipe", "bye"]
        for q in qs: print("\nYou: " + q); print("MarketBot: " + reply(q).replace("BYE", "Goodbye!"))
        raise SystemExit
    if not (voice or "--text" in a):
        try: gui(speak); raise SystemExit
        except ImportError: print("tkinter not available - using the terminal chat instead.")
    print("MarketBot ready.", HELP)
    while True:
        try: q = listen() if voice else input("\nYou: ")
        except EOFError: break
        except Exception as ex: print("Could not hear you:", ex); continue
        if voice: print("You said:", q)
        ans = reply(q)
        if ans == "BYE": print("MarketBot: Goodbye!"); break
        print("MarketBot:", ans); say(ans) if speak else None

"""
06_nlp_intent_classifier.py  -  RUBRIC: NATURAL LANGUAGE PROCESSING (part of the 5 marks for NLP / Speech)
Shows each NLP step from the book (ch.9-10) on real MarketBot questions and MEASURES how well the chatbot
understands them: tokenisation, stop words, stemming, bag of words, TF-IDF, Multinomial Naive Bayes, entity extraction.
Run:  python 06_nlp_intent_classifier.py
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nlp_utils as nu

print("=== 1. TOKENISATION -> STOP WORDS -> STEMMING ===")
for s in ["How much money are we wasting on ads?", "Predict a customer aged 40 with 25 visits", "Which channels convert most?"]:
    print(f"  {s!r}\n    tokens (stop words removed, stemmed): {nu.preprocess(s)}")

print("\n=== 2. BAG-OF-WORDS (book: document-term matrix) ===")
demo = ["how much money are we wasting", "which channel performs best", "what drives conversion"]
cv = CountVectorizer(analyzer=nu.preprocess).fit(demo); print("Vocabulary:", sorted(cv.vocabulary_)); print(pd.DataFrame(cv.transform(demo).toarray(), columns=cv.get_feature_names_out(), index=["waste", "channel", "drivers"]).to_string())
print("\n=== 3. TF-IDF WEIGHTS (rare words weigh more than common ones) ===")
tf = TfidfTransformer().fit(cv.transform(demo)); print(pd.DataFrame(tf.transform(cv.transform(demo)).toarray().round(2), columns=cv.get_feature_names_out(), index=["waste", "channel", "drivers"]).to_string())

print("\n=== 4. TRAINING DATA ===")
n = {k: len(v) for k, v in nu.TRAIN.items()}; print(n, "| total", sum(n.values()), "team-written example questions")
X = [s for v in nu.TRAIN.values() for s in v]; y = [k for k, v in nu.TRAIN.items() for _ in v]

print("\n=== 5. EVALUATION A: 5-fold cross-validation on the training questions ===")
cv5 = StratifiedKFold(5, shuffle=True, random_state=1)
for name, clf in {"TF-IDF + Multinomial Naive Bayes (chosen, as in book)": MultinomialNB(alpha=0.3), "TF-IDF + Logistic Regression": LogisticRegression(max_iter=2000, C=10)}.items():
    pipe = make_pipeline(CountVectorizer(analyzer=nu.preprocess), TfidfTransformer(), clf); print(f"  {name:55s} accuracy {cross_val_score(pipe, X, y, cv=cv5).mean():.3f}")

print("\n=== 6. EVALUATION B: unseen questions written separately (hold-out test) ===")
TEST = {
 "greet": ["hello there", "hi bot", "good day", "hey marketbot"], "help": ["what are you able to do", "help me please", "which questions can i ask", "give me instructions"],
 "bye": ["see you", "goodbye marketbot", "i am finished", "exit now"], "predict": ["will a customer with 30 visits and 4 purchases convert", "predict conversion for a customer aged 25 with 2 email opens", "check if this customer is worth targeting", "what is the probability a customer with ctr 0.2 buys"],
 "channel": ["which marketing channel is best", "how is social media performing", "compare seo and email", "what channel converts the most customers"], "type": ["which campaign type is most effective", "compare retention and awareness campaigns", "how do consideration campaigns perform", "best type of campaign"],
 "waste": ["how much of our spend is wasted", "show me the wasted budget", "which channel wastes the most money", "what are we losing on non converting customers"], "drivers": ["which factors drive conversion", "what influences whether customers buy", "show me the main drivers", "what are the most important features"],
 "segment": ["describe our customer segments", "what kind of customers do we have", "show me the clusters", "profile the customer groups"], "recommend": ["how can we cut our ad costs", "where can i save budget", "what do you recommend", "advise me on reducing waste"]}
model = nu.build_model(); Xt = [s for v in TEST.values() for s in v]; yt = [k for k, v in TEST.items() for _ in v]
pred = [nu.classify(model, s)[0] for s in Xt]; print("Hold-out accuracy: %.3f  (%d questions)" % (accuracy_score(yt, pred), len(yt)))
wrong = [(s, t, p) for s, t, p in zip(Xt, yt, pred) if t != p]; print("Misunderstood questions:", wrong if wrong else "none")
print(classification_report(yt, pred, zero_division=0))

print("=== 7. OUT-OF-SCOPE QUESTIONS (the bot should say it did not understand) ===")
OOS = ["pizza recipe", "what is the weather today", "who won the soccer match", "tell me a joke", "book me a flight", "how old is the president", "play some music", "translate this to french"]
for thr in (0.0, 0.30):
    rej = sum(nu.classify(model, s, thr)[0] == "unknown" for s in OOS); print(f"  min confidence {thr:.2f}: rejected {rej}/{len(OOS)} out-of-scope questions")
for s in OOS[:4]: print("   ", s, "->", nu.classify(model, s, 0.30))

print("\n=== 8. ENTITY EXTRACTION (pattern matching) ===")
for s in ["predict a customer aged 40 with 25 visits, time on site 12 and 5 purchases", "will a customer with ctr 0.02, 3 visits and 1 email opens convert",
          "score a male customer with income 90000 and 8 purchases"]: print(f"  {s}\n    -> {nu.extract_entities(s)}")
print("\nLimitations: fixed set of 10 intents; short training set written by the team; mis-spellings are not corrected. Next step: collect real user questions.")

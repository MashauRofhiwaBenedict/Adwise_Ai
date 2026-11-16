"""
nlp_utils.py - shared NLP helpers used by 06_nlp_intent_classifier.py and 11_marketbot_chatbot.py
(tokenisation, stop-word removal, stemming, TF-IDF + Multinomial Naive Bayes intent model, entity extraction).
Keep this file in the same folder as the other scripts.
"""
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

try:                                    # NLTK stemmer needs no extra download
    from nltk.stem.porter import PorterStemmer; _stem = PorterStemmer().stem
except Exception:                       # works without NLTK too
    _stem = lambda w: w

STOP = set("a an the is are was were be do does did i me my we our you your it its this that of to for in on at by with what which who how can will should would and or please tell about any there am".split())

def preprocess(text):
    """Lower-case -> tokenise (letters only) -> remove stop words -> stem."""
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]

TRAIN = {
 "greet": ["hi", "hello", "hey", "hello marketbot", "good morning", "good afternoon", "hey there bot", "howzit", "hi there", "greetings", "hello bot", "good evening", "hi marketbot", "morning"],
 "help": ["help", "what can you do", "how do i use you", "show me options", "what commands are available", "how does this work", "what questions can i ask", "list your features", "i need help", "guide me", "what do you know", "options please", "what can i ask you"],
 "bye": ["bye", "goodbye", "exit", "quit", "see you later", "bye bye", "thanks bye", "i am done", "stop", "close", "that is all", "end chat", "cheers bye", "talk later"],
 "predict": ["will this customer convert", "predict conversion for a customer", "what is the chance a customer buys", "check this customer aged 30 with income 60000", "likelihood of conversion for this customer",
             "should i target a customer with 10 visits and time on site 8", "score this customer", "predict a customer with 25 visits and 5 purchases", "will a customer with 3 email opens and 2 purchases buy",
             "estimate conversion probability", "is this customer worth targeting", "how likely is a customer to convert with click through rate 0.1", "probability that a customer converts", "customer aged 40 with 12 visits will they respond"],
 "channel": ["which channel performs best", "compare campaign channels", "best marketing channel", "how is email doing", "ppc versus seo results", "channel performance", "which channel converts most",
             "how does social media compare to referral", "rank the channels", "what is the worst channel", "compare email ppc seo referral and social media", "which platform gives the best conversion", "show channel results", "how is ppc performing"],
 "type": ["which campaign type works best", "awareness vs retention", "campaign type results", "does conversion campaign work", "compare campaign types", "retention or consideration campaigns",
          "which type of campaign converts most", "how do awareness campaigns perform", "best campaign objective", "campaign type performance", "are retention campaigns better", "compare awareness consideration conversion retention"],
 "waste": ["how much money are we wasting", "wasted ad spend", "how much budget is lost", "where is spend wasted", "show me wasted spend by channel", "cost of non converting ads", "money wasted on ads",
           "wasting budget", "ad waste total", "lost spend", "how much did we waste", "which channel wastes the most", "spend on customers who did not convert", "what is our wasted budget"],
 "drivers": ["what drives conversion", "why do customers convert", "important factors for buying", "top features", "what influences customer behaviour", "main drivers", "key factors", "what makes people buy",
             "what matters most for conversion", "which variables are important", "explain what affects conversion", "most important predictors", "what factors increase conversion", "feature importance"],
 "segment": ["show customer segments", "describe the customer groups", "who are our customers", "customer segmentation", "what groups of customers do we have", "profile the segments", "tell me about customer clusters",
             "segment profiles", "cluster summary", "group customers by behaviour", "what types of customers exist"],
 "recommend": ["where should i cut spend", "how can i save money", "recommend budget changes", "which customers should i stop targeting", "advice to reduce waste", "how do i reduce wasted spend",
               "what should i do to save budget", "suggest savings", "where can we save", "give me a recommendation", "which spend can be cut", "cost saving advice"],
}

def build_model(train=TRAIN):
    """TF-IDF + Multinomial Naive Bayes (book ch.9-10, 'Category Prediction')."""
    X = [s for v in train.values() for s in v]; y = [k for k, v in train.items() for _ in v]
    pipe = make_pipeline(CountVectorizer(analyzer=preprocess), TfidfTransformer(), MultinomialNB(alpha=0.3)); return pipe.fit(X, y)

def classify(model, text, min_conf=0.0):
    """Return (intent, confidence). 'unknown' when no known words are present or confidence is too low."""
    toks = [t for t in preprocess(text) if t in model[0].vocabulary_]
    if not toks: return "unknown", 0.0
    p = model.predict_proba([text])[0]; i = p.argmax()
    return (str(model.classes_[i]), float(p[i])) if p[i] >= min_conf else ("unknown", float(p[i]))

FIELDS = {"Age": r"(?:aged?|age)\s*(\d+)", "Income": r"income\s*(?:of\s*)?(\d+)", "AdSpend": r"(?:ad\s*spend|spend)\s*(?:of\s*)?(\d+)",
          "WebsiteVisits": r"(\d+)\s*(?:website\s*)?visits", "TimeOnSite": r"time on site\s*(?:of\s*)?(\d+(?:\.\d+)?)", "PagesPerVisit": r"(\d+(?:\.\d+)?)\s*pages",
          "PreviousPurchases": r"(\d+)\s*(?:previous\s*)?purchases", "EmailOpens": r"(\d+)\s*email opens", "EmailClicks": r"(\d+)\s*email clicks",
          "ClickThroughRate": r"(?:click[- ]?through rate|ctr)\s*(?:of\s*)?(\d*\.?\d+)", "ConversionRate": r"conversion rate\s*(?:of\s*)?(\d*\.?\d+)", "LoyaltyPoints": r"(\d+)\s*loyalty"}
def extract_entities(text):
    """Entity extraction by pattern matching: finds numbers attached to customer fields."""
    t = text.lower(); out = {}
    for f, rx in FIELDS.items():
        m = re.search(rx, t)
        if m: out[f] = float(m.group(1))
    for ch in ["Email", "PPC", "SEO", "Referral", "Social Media"]:
        if re.search(r"\b" + ch.lower() + r"\b(?! opens| clicks)", t) and "predict" in t: out["CampaignChannel_" + ch] = 1.0
    if re.search(r"\bmale\b", t) and "female" not in t: out["Gender_Male"] = 1.0
    return out

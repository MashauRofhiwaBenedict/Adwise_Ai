"""run_all.py - runs every analysis script in order (01 to 09), then tells you how to start the chatbot.
Run:  python run_all.py              (chart windows pop up: close each one to continue)
      python run_all.py --nowindows   (no pop-ups, charts only saved)
      (puts all charts in outputs/ and models in models/; about 3-6 minutes)"""
import subprocess, sys, os, time
ENV = dict(os.environ)
if "--nowindows" in sys.argv: ENV["MPLBACKEND"] = "Agg"; ENV["ADWISE_NO_POPUP"] = "1"   # no pop-up windows; charts are only saved to outputs/
HERE = os.path.dirname(os.path.abspath(__file__))
for s in ["01_data_preparation.py", "02_ml_approach.py", "03_model_evaluation.py", "04_time_series_analysis.py", "05_solution_techniques.py",
          "06_nlp_intent_classifier.py", "07_speech_recognition.py", "08_deep_learning_ann.py", "09_deep_learning_lstm.py", "10_deep_learning_cnn_template.py"]:
    print("\n" + "=" * 90 + "\nRUNNING", s, "\n" + "=" * 90); t = time.time(); r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE, env=ENV)
    print("-> %s %s (%.0fs)" % (s, "finished" if r.returncode == 0 else "FAILED (see message above; e.g. pip install tensorflow)", time.time() - t))
print("\nAll done. Now chat with the bot:   python 11_marketbot_chatbot.py   (add --demo for a scripted demo)")

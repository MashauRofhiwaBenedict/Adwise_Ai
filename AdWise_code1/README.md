# AdWise AI - code folder (BA 3.2 project)
Put `digital_marketing_campaign_dataset.csv` (any name starting with `digital_marketing_campaign_dataset`) in this folder.
Install packages: `pip install -r requirements.txt`   Run everything: `python run_all.py` (charts pop up; if your Python has no chart-window support they open in your image viewer; `--nowindows` skips them). Then `python 11_marketbot_chatbot.py` opens the MarketBot chat window.
TensorFlow (scripts 08-10) needs Python 3.10-3.12; on newer Python, 08 and 09 automatically use scikit-learn neural networks instead.

| Script | Rubric item |
|---|---|
| 01_data_preparation.py | Data |
| 02_ml_approach.py | Machine Learning Approach |
| 03_model_evaluation.py | Model (how accuracy is evaluated) |
| 04_time_series_analysis.py | Time Series Analysis |
| 05_solution_techniques.py | Solution Techniques |
| 06_nlp_intent_classifier.py (+ nlp_utils.py) | Natural Language Processing |
| 07_speech_recognition.py | Speech Recognition / Synthesis |
| 08_deep_learning_ann.py, 09_deep_learning_lstm.py, 10_deep_learning_cnn_template.py | Deep Learning |
| 11_marketbot_chatbot.py | Chatbot / Softbot + Practical |
Each script also runs alone, except 11, which needs the files made by 02 and 03.

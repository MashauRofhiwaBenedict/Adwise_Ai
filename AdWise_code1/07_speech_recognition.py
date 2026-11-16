"""
07_speech_recognition.py  -  RUBRIC: SPEECH RECOGNITION / SYNTHESIS (part of the 5 marks for NLP / Speech)
Follows book ch.12.  Three parts:
  A) Signal analysis that needs NO microphone: create an audio file, read it, FFT (frequency domain), MFCC and filter-bank features.
  B) Speech-to-text (Google Speech API through the SpeechRecognition package) from the MICROPHONE or a WAV FILE, then
     passes the text to the NLP intent classifier (nlp_utils.py) so a spoken question reaches MarketBot.
  C) Speech synthesis (text-to-speech) with pyttsx3, so the bot can read answers aloud.
Run:  python 07_speech_recognition.py                      (Part A)
      python 07_speech_recognition.py --mic                (Part B, needs microphone + PyAudio + internet)
      python 07_speech_recognition.py --file question.wav  (Part B from a recorded file)
      python 07_speech_recognition.py --speak "Total wasted spend is ten percent"   (Part C)
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from show_utils import show_chart   # charts open as pop-up windows (or in your image viewer)
from scipy.io import wavfile
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)

def part_a():
    print("=== PART A: AUDIO SIGNAL ANALYSIS ===")
    fs, dur = 16000, 3.0; t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    # a synthetic 'voice-like' signal: three tones (words) separated by silence, plus a little background noise
    sig = np.zeros_like(t)
    for f0, a, b in [(220, 0.2, 0.9), (330, 1.1, 1.8), (262, 2.0, 2.8)]:
        m = (t >= a) & (t < b); sig[m] = 0.5 * np.sin(2 * np.pi * f0 * t[m]) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t[m])
    sig += 0.01 * np.random.default_rng(0).standard_normal(len(t)); wav = os.path.join(OUT, "07_test_signal.wav")
    wavfile.write(wav, fs, (sig * 32767).astype(np.int16))
    fs2, x = wavfile.read(wav); print("Sampling frequency:", fs2, "Hz | samples:", x.shape[0], "| dtype:", x.dtype, "| duration:", round(x.shape[0] / fs2, 2), "s")
    x = x / np.power(2, 15)                                              # normalise (book)
    X = np.fft.rfft(x); power = 10 * np.log10((np.abs(X) / len(x)) ** 2 + 1e-12); freqs = np.fft.rfftfreq(len(x), 1 / fs2)
    from scipy.signal import find_peaks
    idx = find_peaks(power, distance=40)[0]; top = idx[np.argsort(power[idx])[-6:]]
    print("Strongest frequency peaks (Hz):", sorted(int(round(freqs[i])) for i in top), "-> tones at 220, 262, 330 Hz and their harmonics")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6)); ax[0].plot(t[:2000] * 1000 + 1100, x[1100 * fs // 1000: 1100 * fs // 1000 + 2000]); ax[0].set_title("Time domain (zoom)"); ax[0].set_xlabel("ms")
    ax[1].plot(freqs / 1000, power, color="k", lw=.6); ax[1].set_xlim(0, 2); ax[1].set_title("Frequency domain (Fourier transform)"); ax[1].set_xlabel("kHz"); ax[1].set_ylabel("dB")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "07_signal_fft.png"), dpi=150); show_chart(os.path.join(OUT, "07_signal_fft.png"))
    try:
        from python_speech_features import mfcc, logfbank
        m = mfcc(x[:15000], fs2); fb = logfbank(x[:15000], fs2); print("MFCC features: windows =", m.shape[0], "| coefficients per window =", m.shape[1]); print("Filter-bank features: windows =", fb.shape[0], "| filters =", fb.shape[1])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5)); ax[0].matshow(m.T, aspect="auto"); ax[0].set_title("MFCC"); ax[1].matshow(fb.T, aspect="auto"); ax[1].set_title("Filter bank"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "07_mfcc.png"), dpi=150); show_chart(os.path.join(OUT, "07_mfcc.png"))
    except ImportError: print("python_speech_features not installed (pip install python_speech_features): skipping MFCC.")
    print("Saved outputs/07_test_signal.wav, 07_signal_fft.png, 07_mfcc.png")

def intent_of(text):
    try:
        import nlp_utils as nu; return nu.classify(nu.build_model(), text, 0.0)
    except Exception as e: return ("(NLP not available: %s)" % e, 0)

def part_b(wav=None):
    print("=== PART B: SPEECH-TO-TEXT ===")
    try: import speech_recognition as sr
    except ImportError: print("Install first:  pip install SpeechRecognition pyaudio"); return
    r = sr.Recognizer()
    try:
        if wav:
            with sr.AudioFile(wav) as src: audio = r.record(src)
        else:
            with sr.Microphone() as src: r.adjust_for_ambient_noise(src); print("Speak now ..."); audio = r.listen(src, timeout=8)
        text = r.recognize_google(audio); print("You said:", text); print("MarketBot intent:", intent_of(text))
    except sr.UnknownValueError: print("Could not understand the audio (no speech found or too noisy).")
    except sr.RequestError as e: print("Google Speech API unreachable (internet needed):", e)
    except Exception as e: print("Audio input problem (is a microphone / PyAudio available?):", e)

def part_c(text):
    print("=== PART C: SPEECH SYNTHESIS ===")
    try: import pyttsx3; e = pyttsx3.init(); e.say(text); e.runAndWait(); print("Spoke:", text)
    except Exception as ex: print("Text-to-speech not available (pip install pyttsx3):", ex)

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--mic" in a: part_b()
    elif "--file" in a: part_b(a[a.index("--file") + 1])
    elif "--speak" in a: part_c(" ".join(a[a.index("--speak") + 1:]))
    else: part_a(); print("\nNext: --mic or --file <wav> for speech-to-text, --speak \"text\" for speech synthesis.")

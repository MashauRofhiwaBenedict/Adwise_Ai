"""
show_utils.py - makes charts appear on screen no matter what.
 * If matplotlib has a window backend, the chart opens in a normal pop-up window (close it to continue).
 * If not (common on brand-new Python versions / Macs without tkinter), the chart is opened as a picture
   in your computer's image viewer (Preview on Mac, Photos on Windows) instead.
Keep this file in the same folder as the other scripts.
"""
import os, sys, subprocess, matplotlib
import matplotlib.pyplot as plt

def _interactive():
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "pgf", "cairo", "template")

def open_file(path):
    try:
        if sys.platform == "darwin": subprocess.Popen(["open", path])
        elif sys.platform.startswith("win"): os.startfile(path)
        else: subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("  Chart saved here (open it yourself):", path)

def show_chart(path=None):
    if _interactive(): plt.show(); return
    plt.close()
    if path and not os.environ.get("ADWISE_NO_POPUP"):
        print("  [no chart-window backend found, opening the saved picture instead: %s]" % os.path.basename(path)); open_file(path)

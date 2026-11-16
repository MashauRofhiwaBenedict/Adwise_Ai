"""
10_deep_learning_cnn_template.py  -  RUBRIC: DEEP LEARNING (3 of 3)  - Convolutional Neural Network for ADVERT IMAGES
Book ch.19 image classifier (Conv2D -> MaxPooling2D -> Flatten -> Dense -> sigmoid). Purpose in AdWise AI: learn which
advert creatives (images) lead to conversions, e.g. 'high_converting' vs 'low_converting'.
The campaign CSV has NO images, so this is a READY-TO-RUN TEMPLATE:
  * If you create the folders below with real advert images it TRAINS and evaluates the CNN.
  * If not, it only builds the model and prints its architecture (no fake data is used).
Folder layout:   ad_images/train/high_converting/*.jpg   ad_images/train/low_converting/*.jpg
                 ad_images/test/high_converting/*.jpg    ad_images/test/low_converting/*.jpg
Run:  python 10_deep_learning_cnn_template.py      (needs: pip install tensorflow)
"""
import os, warnings; warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers
except ImportError:
    print("TensorFlow is not installed for this Python version (it needs Python 3.10-3.12), so the CNN cannot be built here.")
    print("Architecture (book ch.19): Input 64x64x3 -> Rescaling -> Conv2D(32) -> MaxPooling2D -> Conv2D(32) -> MaxPooling2D -> Flatten -> Dense(128, relu) -> Dropout(0.3) -> Dense(1, sigmoid)")
    print("Install TensorFlow on Python 3.10-3.12 and add advert images in ad_images/train and ad_images/test to train it.")
    raise SystemExit(0)
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "ad_images"); tf.keras.utils.set_random_seed(42)
cnn = keras.Sequential([layers.Input(shape=(64, 64, 3)), layers.Rescaling(1. / 255), layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")])
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]); cnn.summary()
if os.path.isdir(os.path.join(D, "train")) and os.path.isdir(os.path.join(D, "test")):
    aug = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomZoom(0.2)])      # image augmentation (book)
    tr = keras.utils.image_dataset_from_directory(os.path.join(D, "train"), image_size=(64, 64), batch_size=32, label_mode="binary")
    te = keras.utils.image_dataset_from_directory(os.path.join(D, "test"), image_size=(64, 64), batch_size=32, label_mode="binary")
    cnn.fit(tr.map(lambda x, y: (aug(x, training=True), y)), validation_data=te, epochs=15); print("Test loss/accuracy:", cnn.evaluate(te, verbose=0))
else: print("\nNo ad_images folder found: architecture built only. Add labelled advert images (see the folder layout above) to train it.")

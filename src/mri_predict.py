import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "models/mri_classifier.h5"
)

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

def predict_mri(img):

    img = img.resize((128,128))
    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    print("IMAGE SHAPE =", img.shape)
    print("MODEL INPUT =", model.input_shape)
    pred = model.predict(img, verbose=0)

    idx = np.argmax(pred)

    confidence = np.max(pred)

    finding = classes[idx]

    if finding == "glioma":
        risk = "High"
        region = "Temporal Lobe"

    elif finding == "meningioma":
        risk = "Medium"
        region = "Frontal Lobe"

    elif finding == "pituitary":
        risk = "Medium"
        region = "Pituitary Gland"

    else:
        risk = "Low"
        region = "Normal Brain"

    return {
        "finding": finding,
        "confidence": round(float(confidence) * 100, 2),
        "risk": risk,
        "region": region
    }
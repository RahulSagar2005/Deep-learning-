from flask import Flask, request, render_template_string
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- CONFIG ----------------
MODEL_PATH = "blood_group.h5"
UPLOAD_FOLDER = "uploads"
IMG_SIZE = 224   # 🔴 MUST MATCH TRAINING

CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
# ---------------------------------------

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# -------- HTML (INLINE) --------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Blood Group Detection</title>
</head>
<body style="font-family: Arial; background:#eef2f7;">

<div style="width:420px; margin:60px auto; padding:20px; background:white;
            border-radius:8px; text-align:center; box-shadow:0 0 10px #ccc;">
    <h2>🩸 Blood Group Detection</h2>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <hr>
        <p><b>Predicted Blood Group:</b> {{ prediction }}</p>
        <p><b>Confidence:</b> {{ confidence }}%</p>
        <img src="{{ img_path }}" width="200">
    {% endif %}
</div>

</body>
</html>
"""

# -------- PREPROCESSING (MATCHES TRAINING) --------
def preprocess_image(img_path):
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb"   # 🔴 IMPORTANT
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == 'POST':
        file = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        img_array = preprocess_image(img_path)

        preds = model.predict(img_array)
        print("RAW PREDICTIONS:", preds)  # debug (optional)

        prediction = CLASS_NAMES[np.argmax(preds)]
        confidence = round(float(np.max(preds) * 100), 2)

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, Response
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
# -------------------------
# LOAD MODEL
# -------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "final_plant_model.keras"))

with open(os.path.join(BASE_DIR, "class_names.json")) as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

IMG_SIZE = (224, 224)

# -------------------------
# FERTILIZER DB (same)
# -------------------------
fertilizer_db = {
    "Corn_Cercospora_leaf_spot Gray_leaf_spot": {
        "fertilizer": "Propiconazole (Curative Fungicide)",
        "image": "fertilizers/Propiconazole (Curative Fungicide).webp",
        "usage": "1.Mix 1 ml per liter water\n2.Spray uniformly on infected plants",
        "benefits": "1.Controls existing infection\n2.Systemic action (works inside plant)\n3.Improves plant health"
    },
    "Corn_Common_rust": {
        "fertilizer": "Propiconazole (Best Curative Fungicide)",
        "image": "fertilizers/Propiconazole (Curative Fungicide).webp",
        "usage": "1.Mix 1 ml per liter water\n2.Spray uniformly on infected plants",
        "benefits": "1.Controls existing infection\n2.Systemic action (works inside plant)\n3.Improves plant health"
    },
    "Corn_Northern_Leaf_Blight": {
          "fertilizer": "Propiconazole (Best Curative Fungicide)",
        "image": "fertilizers/Propiconazole (Curative Fungicide).webp",
        "usage": "1.Mix 1 ml per liter water\n2.Spray uniformly on infected plants",
        "benefits": "1.Controls existing infection\n2.Systemic action (works inside plant)\n3.Improves plant health"
    },
     "Corn_healthy": {
        "fertilizer": "Potash (MOP – Muriate of Potash)",
        "image": "fertilizers/Potash MOP – Muriate of Potash.webp",
        "usage": "Apply in soil during growth stage",
        "benefits": "1.Improves disease resistance\n2.Strengthens stems\n3.Increases grain quality"
    },
     "Pepper,_bell___Bacterial_spot": {
        "fertilizer": "Liquid Soap or Horticultural Oil",
        "image": "fertilizers/Propiconazole (Curative Fungicide).webp",
        "usage": "1.Mix 1 tablespoon liquid soap in 1 gallon of water\n2.Spray leaves once per week\n3.Helps suffocate bacterial spores",
        "benefits": "1.Natural treatment\n2.Works as a barrier for spores\n3.Safe for environment"
    },
     "Pepper,_bell___healthy": {
        "fertilizer": "Calcium Nitrate (For Strong Fruits)",
        "image": "fertilizers/Calcium Nitrate (For Strong Fruits).webp",
        "usage": "1.Mix 2–3 grams per liter water\n2.Spray during flowering and fruit stage",
        "benefits": "1.Prevents blossom end rot\n2.Strengthens fruit quality\n3.Improves shelf life"
    },
     "Potato___Early_blight": {
        "fertilizer": "Chlorothalonil (Best Preventive)",
        "image": "fertilizers/Chlorothalonil (Best Preventive).webp",
        "usage": "1.Mix 2 grams per liter water\n2.Spray every 7–10 days",
        "benefits": "1.Prevents disease spread\n2.Protects healthy leaves\n3.Good for early stage"
    },
     "Potato___Late_blight": {
        "fertilizer": "Cymoxanil + Mancozeb (Fast Curative Action)",
        "image": "fertilizers/Cymoxanil + Mancozeb (Fast Curative Action).avif",
        "usage": "1.Mix 2–3 grams per liter water\n2.Spray before disease outbreak",
        "benefits": "1.Prevents infection\n2.Affordable option\n3.Suitable for early protection"
    },
     "Potato___healthy": {
        "fertilizer": "Potash (MOP – Potassium Fertilizer)",
        "image": "fertilizers/Potash (MOP – Potassium Fertilizer).webp",
        "usage": "Apply during tuber formation stage",
        "benefits": "1.Improves tuber size and quality\n2.Increases disease resistance\n3.Enhances storage life"
    }
}
# -------------------------
# IMAGE PREDICTION
# -------------------------
def predict_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return class_names[class_id], confidence


# -------------------------
# LIVE CAMERA FUNCTION
# -------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize & preprocess
        img = cv2.resize(frame, IMG_SIZE)
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred))
        label = class_names[class_id]

        # Draw result
        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")

    result, confidence = predict_image(img)
    data = fertilizer_db.get(result, None)

    return render_template(
        "predict.html",
        result=result,
        confidence=round(confidence*100, 2),
        data=data,
        alert="healthy" not in result.lower()
    )


# 🔥 LIVE PAGE
@app.route("/live")
def live():
    return render_template("live.html")


# 🔥 VIDEO STREAM
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
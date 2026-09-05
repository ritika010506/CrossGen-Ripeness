from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
from keras.applications import mobilenet_v2, efficientnet, resnet50
import numpy as np
from PIL import Image
import io, json
from huggingface_hub import hf_hub_download

app = FastAPI()

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your Vercel URL after deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────────────────
print("Downloading/loading models from Hugging Face...")

HF_REPO_ID = "ritika010506/ripeness-models"

mobilenet_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="mobilenet_model.h5"
)

efficientnet_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="efficientnet_model.h5"
)

resnet_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="resnet_model.h5"
)

weights_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="ensemble_weights.json"
)

model1 = keras.models.load_model(mobilenet_path)
model2 = keras.models.load_model(efficientnet_path)
model3 = keras.models.load_model(resnet_path)

with open(weights_path) as f:
    ew = json.load(f)

w1, w2, w3 = ew["w1"], ew["w2"], ew["w3"]

print("Models ready.")

CLASS_NAMES = ["overripe", "ripe", "unripe"]  # must match your training folder order
print("Models ready.")

# ── Preprocessing helper ──────────────────────────────────────────────────
def prepare_image(image_bytes, preprocess_fn):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_fn(arr)
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# ── Prediction endpoint ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Run all 3 models with their correct preprocessing
    p1 = model1.predict(prepare_image(image_bytes, mobilenet_v2.preprocess_input),  verbose=0)
    p2 = model2.predict(prepare_image(image_bytes, efficientnet.preprocess_input),  verbose=0)
    p3 = model3.predict(prepare_image(image_bytes, resnet50.preprocess_input),       verbose=0)

    # Weighted ensemble
    ensemble = (w1 * p1) + (w2 * p2) + (w3 * p3)
    ensemble = ensemble[0]   # shape: (3,)

    predicted_index = int(np.argmax(ensemble))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(ensemble[predicted_index])

    return {
        "prediction":  predicted_class,
        "confidence":  round(confidence * 100, 2),
        "all_scores": {
            CLASS_NAMES[i]: round(float(ensemble[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }

@app.get("/")
def health():
    return {"status": "ok"}

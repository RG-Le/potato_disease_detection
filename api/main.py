import uvicorn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(r"C:\Users\rajat\OneDrive\Desktop\PotatoDisease\models\1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "HELLO WORLD :)!!!!"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    prediction = MODEL.predict(tf.expand_dims(image, axis=0))
    predicted_class = CLASS_NAMES[tf.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ ADD THIS AFTER app init
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev: allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Keras model
model = tf.keras.models.load_model("toxicity.h5")

# Dummy tokenizer (retrain or load actual tokenizer if needed)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(["dummy example text for tokenizer setup"])

# Define request body
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"msg": "Toxicity Detection API Running ✅"}

@app.post("/predict")
def predict_text(input: TextInput):
    text = input.text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
    prediction = model.predict(padded)
    label = "toxic" if prediction[0][0] >= 0.5 else "non-toxic"
    
    return {
        "input": text,
        "prediction": label,
        "confidence": float(prediction[0][0])
    }


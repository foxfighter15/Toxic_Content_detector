from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# CORS enable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("toxicity_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.get("/")
def root():
    return {"message": "API is working!"}

@app.get("/predict")
def predict_toxicity(text: str):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    score = model.predict_proba(text_vector)[0][1]
    return {
        "text": text,
        "is_toxic": bool(prediction),
        "toxicity_score": round(float(score), 4)
    }


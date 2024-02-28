from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = "gyesibiney/miniLm-emotions-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an emotion analysis pipeline
emotion_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a request body model
class EmotionRequest(BaseModel):
    text: str

# Define a response model
class EmotionResponse(BaseModel):
    emotion: str
    score: float

# Create an endpoint for emotion analysis with a query parameter
@app.post("/analyze_emotion/")
async def analyze_emotion(request: EmotionRequest):
    result = emotion_analyzer(request.text)
    emotion_label = result[0]["label"]
    emotion_score = result[0]["score"]

    # Map emotion label to emotion string (you might need to adjust this based on your label mapping)
    emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    emotion_value = emotion_mapping.get(emotion_label, 'unknown')

    return EmotionResponse(emotion=emotion_value, score=emotion_score)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

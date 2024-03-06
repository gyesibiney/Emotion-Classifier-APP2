from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Load the pre-trained model and tokenizer for emotion analysis
model_name = "gyesibiney/miniLm-emotions-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an emotion analysis pipeline
emotion = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Mapping from label to emotion
label2id = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# Define a request body model for emotion analysis
class EmotionRequest(BaseModel):
    text: str

# Define a response model for emotion analysis
class EmotionResponse(BaseModel):
    emotion: str
    score: float

# Create an endpoint for emotion analysis with a query parameter
@app.get("/emotion/")
async def analyze_emotion(text: str = Query(..., description="Input text for emotion analysis")):
    result = emotion(text)
    emotion_label = result[0]["label"]
    emotion_score = result[0]["score"]

    # Correctly map the model's label to the desired emotion
    emotion_value = label2id.get(emotion_label,-1)  # Extracting the numeric part and mapping

    return EmotionResponse(emotion=emotion_value, score=emotion_score)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

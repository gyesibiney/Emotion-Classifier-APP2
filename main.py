from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load label2id from the saved JSON file
with open('label2id.json', 'r') as json_file:
    label2id = json.load(json_file)


app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = "gyesibiney/miniLm-emotions-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an emotion analysis pipeline
emotion = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Mapping from label to emotion
label2id = {idx: feature['label'].int2str(idx) for idx in range(6)}

# Define a request body model
class EmotionRequest(BaseModel):
    text: str

# Define a response model
class EmotionResponse(BaseModel):
    emotion: str  # Specify the possible emotions based on your model
    score: float

# Create an endpoint for emotion analysis with a query parameter
@app.get("/emotion/")
async def analyze_emotion(text: str = Query(..., description="Input text for emotion analysis")):
    result = emotion(text)
    emotion_label = result[0]["label"]
    emotion_score = result[0]["score"]

    emotion_value = label2id.get(emotion_label, 'unknown')  # Default to "unknown" for unknown labels

    return EmotionResponse(emotion=emotion_value, score=emotion_score)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

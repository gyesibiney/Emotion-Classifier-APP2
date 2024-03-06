from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import TextClassificationPipeline

app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = "gyesibiney/miniLm-emotions-finetuned"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text classification pipeline
emotion_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, task="sentiment-analysis")

# Define a request body model
class EmotionRequest(BaseModel):
    text: str

# Define a response model
class EmotionResponse(BaseModel):
    emotion: str
    score: float

# Create an endpoint for emotion analysis with a query parameter
@app.get("/emotion/")
async def analyze_emotion(text: str = Query(..., description="Input text for emotion analysis")):
    result = emotion_pipeline(text)[0]

    return EmotionResponse(emotion=result['label'], score=result['score'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

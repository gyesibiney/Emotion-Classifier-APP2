from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = "gyesibiney/Sentiment-review-analysis-roberta-3"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Create a dictionary to map sentiment labels to positive and negative strings
sentiment_label_mapping = {
    "LABEL_1": "positive",
    "LABEL_0": "negative",
}

# Define a request body model
class SentimentRequest(BaseModel):
    text: str

# Define a response model
class SentimentResponse(BaseModel):
    sentiment: str  # 1 for positive, 0 for negative
    score: float

# Create an endpoint for sentiment analysis with query parameter
@app.get("/sentiment/")
async def analyze_sentiment(text: str = Query(..., description="Input text for sentiment analysis")):
    result = sentiment(text)
    sentiment_label = result[0]["label"]
    sentiment_score = result[0]["score"]
    
    sentiment_value = sentiment_label_mapping.get(sentiment_label, -1)  # Default to -1 for unknown labels
    
    return SentimentResponse(sentiment=sentiment_value, score=sentiment_score)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
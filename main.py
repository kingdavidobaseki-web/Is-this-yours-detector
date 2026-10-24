import os
from typing import List, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"
SPACE_SECRET = os.environ.get("SPACE_SECRET")

app = FastAPI()

# Load once at startup, then shrink it with dynamic quantization —
# roughly halves memory use and speeds up CPU inference, both of which
# matter on a free tier's limited RAM.
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
_model = torch.quantization.quantize_dynamic(
    _model, {torch.nn.Linear}, dtype=torch.qint8
)
classifier = pipeline("text-classification", model=_model, tokenizer=_tokenizer)


class PredictRequest(BaseModel):
    inputs: List[str]


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict")
def predict(req: PredictRequest, x_space_key: Optional[str] = Header(None)):
    if SPACE_SECRET and x_space_key != SPACE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing key.")

    if not req.inputs:
        raise HTTPException(status_code=400, detail="inputs must be a non-empty list.")

    results = classifier(req.inputs, truncation=True)
    return results

import re
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="AITA Post Classifier")

# Transformers will instantly load the model from the local build cache
tokenizer = AutoTokenizer.from_pretrained("marvinwong12/roberta-aita")
model = AutoModelForSequenceClassification.from_pretrained("marvinwong12/roberta-aita")
model.eval()

def clean_and_format(title: str, story: str) -> str:
    """Preprocesses input text identically to training logic."""
    t_clean = re.sub(r'\s+', ' ', str(title or '')).strip()
    b_clean = re.sub(r'\s+', ' ', str(story or '')).strip()
    
    if t_clean and b_clean:
        return f"TITLE: {t_clean}\nSTORY: {b_clean}"
    elif t_clean:
        return f"TITLE: {t_clean}"
    return f"STORY: {b_clean}"

class PostRequest(BaseModel):
    title: str = ""
    story: str = ""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves a single-page web app interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AITA Classifier</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; }
            .card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            textarea, input { width: 100%; padding: 10px; margin-top: 8px; margin-bottom: 16px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { background: #ff4500; color: white; border: none; padding: 12px 20px; font-weight: bold; border-radius: 4px; cursor: pointer; width: 100%; }
            button:hover { background: #e03d00; }
            #result { margin-top: 20px; padding: 15px; border-radius: 4px; display: none; font-weight: bold; }
            .YTA { background-color: #ffdddd; color: #a00000; }
            .NTA { background-color: #ddffdd; color: #006000; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Am I The Asshole? Classifier</h2>
            <label>Post Title</label>
            <input type="text" id="title" placeholder="AITA for eating my roommate's cake?">
            <label>Story Body</label>
            <textarea id="story" rows="6" placeholder="I came home late and saw a cake on the counter..."></textarea>
            <button onclick="classifyPost()">Classify Post</button>
            <div id="result"></div>
        </div>

        <script>
            async function classifyPost() {
                const title = document.getElementById('title').value;
                const story = document.getElementById('story').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.style.display = 'block';
                resultDiv.className = '';
                resultDiv.innerText = 'Analyzing post...';

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, story })
                });

                const data = await response.json();
                const isYTA = data.prediction === "Asshole (YTA)";
                
                resultDiv.className = isYTA ? 'YTA' : 'NTA';
                resultDiv.innerHTML = `<strong>Verdict:</strong> ${data.prediction}<br>
                                       <strong>Asshole Probability:</strong> ${(data.asshole_probability * 100).toFixed(1)}%`;
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: PostRequest):
    formatted_text = clean_and_format(data.title, data.story)
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        
    pred_idx = torch.argmax(probs).item()
    labels = {0: "Not Asshole (NTA)", 1: "Asshole (YTA)"}
    
    return {
        "prediction": labels[pred_idx],
        "asshole_probability": round(float(probs[1]), 4),
        "not_asshole_probability": round(float(probs[0]), 4)
    }

@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and container orchestrators."""
    return {"status": "healthy", "model_loaded": model is not None}
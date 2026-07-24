import re
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="r/AITA Classifier v2")

# Optimized Decision Threshold (use the best value found during training, e.g., 0.40)
# We handle this on the backend to ensure consistent decision making.
OPTIMAL_THRESHOLD = 0.50

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
    """Serves an overhauled, highly Reddit-like interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>r/AITA Classifier - Reddit Style</title>
        <style>
            /* Reddit Modern Light Mode Palette */
            :root {
                --body-bg: #DAE0E6; /* Modern light gray background */
                --card-bg: #FFFFFF; /* White cards */
                --text-main: #1A1A1B; /* Dark gray text */
                --text-secondary: #7c7c7c; /* Gray secondary text */
                --reddit-orange: #FF4500; /* Submission Button */
                --nta-green: #008000;
                --yta-red: #B00020;
                --border-color: #EDEFF1; /* Thin gray border */
                --focus-blue: #0079D3;
            }

            body { 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
                max-width: 760px; 
                margin: 40px auto; 
                padding: 0 20px; 
                background-color: var(--body-bg); 
                color: var(--text-main); 
                line-height: 1.5;
            }

            /* Main Container Card */
            .card { 
                background-color: var(--card-bg); 
                padding: 24px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
                border: 1px solid var(--border-color);
            }

            /* Reddit-style Header */
            .header-bar {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 16px;
            }

            h2 { 
                margin: 0; 
                font-size: 20px; 
                font-weight: 600;
            }

            label { 
                display: block;
                font-size: 14px;
                font-weight: 500;
                color: var(--text-secondary);
                margin-top: 16px;
                margin-bottom: 4px;
            }

            /* Reddit-style Inputs: Pill shape, lighter background */
            textarea, input { 
                width: 100%; 
                padding: 12px 16px; 
                margin-bottom: 12px; 
                border: 1px solid var(--border-color); 
                border-radius: 4px; 
                box-sizing: border-box; 
                font-family: inherit;
                background-color: #F6F7F8;
            }

            textarea:focus, input:focus {
                outline: none;
                border-color: var(--focus-blue);
                background-color: var(--card-bg);
            }

            textarea {
                resize: vertical;
                min-height: 120px;
            }

            /* Full Pill-shaped Reddit Button */
            button { 
                background-color: var(--reddit-orange); 
                color: white; 
                border: none; 
                padding: 12px 28px; 
                font-weight: 700; 
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-radius: 999px; 
                cursor: pointer; 
                display: block;
                margin: 20px auto 0;
                width: auto;
                min-width: 180px;
                transition: background-color 0.2s, box-shadow 0.2s;
            }
            
            button:hover { 
                background-color: #E03D00; 
                box-shadow: 0 0 8px rgba(255, 69, 0, 0.4);
            }

            /* Result Display Overhaul: Looks like a stickied comment verdict */
            #result { 
                margin-top: 24px; 
                padding: 20px; 
                border-radius: 8px; 
                display: none; 
                background-color: #F6F7F8;
                border: 1px solid var(--border-color);
            }

            .verdict-header {
                display: flex;
                align-items: center;
                gap: 12px;
                font-weight: 800;
                font-size: 20px;
                margin-bottom: 12px;
            }
            
            .reddit-icon { font-size: 24px; }

            .stats-row {
                font-size: 14px;
                color: var(--text-secondary);
                margin-bottom: 8px;
            }

            /* The fun judgment message box */
            .judgment-msg {
                margin-top: 18px;
                font-weight: 500;
                font-size: 16px;
                border-top: 1px solid var(--border-color);
                padding-top: 18px;
                line-height: 1.6;
            }

            /* Verdict styling */
            .YTA .verdict-header { color: var(--yta-red); }
            .NTA .verdict-header { color: var(--nta-green); }

        </style>
    </head>
    <body>
        <div class="card">
            <div class="header-bar">
                <h2>🤖 r/AITA Classifier v2</h2>
            </div>

            <label>Title (AITA for...)</label>
            <input type="text" id="title" placeholder="AITA for eating my roommate's cake?">
            
            <label>Story Body (The juicy details)</label>
            <textarea id="story" placeholder="So, I came home late, and there was this beautiful cake on the counter..."></textarea>
            
            <button onclick="classifyPost()">Classify Post</button>
            
            <div id="result"></div>
        </div>

        <script>
            // Custom fun messages based on Asshole probability buckets (10%)
            // Bucket definitions are inclusive: [0, 10), [10, 20), ... [90, 100]
            const getJudgmentMessage = (prob) => {
                const p = prob * 100;
                if (p < 10)  return "<strong>The Verdict: Pure NTA.</strong> The bot detected essentially zero assholery. You are an angel, or perhaps you just manipulated the story perfectly. Acceptance is confirmed.";
                if (p < 20)  return "<strong>The Verdict: Definitively NTA.</strong> Just a normal interaction where someone else was probably being difficult. Walk away with your head high.";
                if (p < 30)  return "<strong>The Verdict: Probably fine, but...</strong> You are mostly not the asshole, but maybe you made *one* situation slightly awkward? The bot is mostly on your side.";
                if (p < 40)  return "<strong>The Verdict: The Gray Area.</strong> We are bordering on ESH (Everyone Sucks Here). You contributed to this mess, even if they started it.";
                if (p < 50)  return "<strong>The Verdict: Soft YTA.</strong> Yeah, you did it. It wasn't malicious, but you made the wrong call. Time for a polite apology.";
                if (p < 60)  return "<strong>The Verdict: Firmly YTA.</strong> Your actions were the primary cause of this issue. You need to own this mistake, big time.";
                if (p < 70)  return "<strong>The Verdict: Asshole (Confirmed).</strong> This was concentrated assholery. The bot is displeased with your behavior.";
                if (p < 80)  return "<strong>The Verdict: YIKES.</strong> That was dense. Do you often prioritize your own minor needs over basic social rules?";
                if (p < 90)  return "<strong>The Verdict: Absolute Legend...</strong> of terrible decisions. You seemingly made an active effort to be the worst part of someone's day.";
                return "<strong>The Verdict: The final boss of AITA.</strong> 100% pure assholery, aged to perfection. The model is essentially screaming YTA at the screen. You are the champion, in the worst possible way.";
            };

            async function classifyPost() {
                const title = document.getElementById('title').value;
                const story = document.getElementById('story').value;
                const resultDiv = document.getElementById('result');
                
                // Visual reset while thinking
                resultDiv.style.display = 'block';
                resultDiv.className = '';
                resultDiv.innerHTML = '<div class="verdict-header">Analyzing Post...</div>';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title, story })
                    });

                    if (!response.ok) throw new Error("Classifier took a nap.");

                    const data = await response.json();
                    
                    // The threshold check (0.40) is handled entirely by the backend now!
                    const isYTA = data.prediction === "Asshole (YTA)";
                    
                    resultDiv.className = isYTA ? 'YTA' : 'NTA';

                    // Build the detailed Reddit-style sticks response
                    resultDiv.innerHTML = `
                        <div class="verdict-header">
                            <span class="reddit-icon">🤖</span> Verdict: ${data.prediction}
                        </div>
                        <div class="stats-row"><strong>Asshole Probability:</strong> ${(data.asshole_probability * 100).toFixed(1)}%</div>
                        <div class="stats-row">Not Asshole Probability: ${(data.not_asshole_probability * 100).toFixed(1)}%</div>
                        <div class="judgment-msg">${getJudgmentMessage(data.asshole_probability)}</div>
                    `;

                } catch (error) {
                    resultDiv.className = 'YTA';
                    resultDiv.innerHTML = '<div class="verdict-header">Error</div>The bot experienced an existential crisis. Did you submit a blank post?';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: PostRequest):
    """
    Predict endpoint implementing optimal decision thresholding (0.40 YTA).
    If probability >= 0.40, the verdict is YTA.
    """
    formatted_text = clean_and_format(data.title, data.story)
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        
    asshole_prob = float(probs[1])
    
    # Apply optimal threshold (e.g., 0.40) instead of default argmax (0.50)
    # We enforce this decision here on the backend.
    is_asshole = asshole_prob >= OPTIMAL_THRESHOLD
    prediction = "Asshole (YTA)" if is_asshole else "Not Asshole (NTA)"
    
    return {
        "prediction": prediction,
        "asshole_probability": round(asshole_prob, 4),
        "not_asshole_probability": round(float(probs[0]), 4),
        "threshold_used": OPTIMAL_THRESHOLD
    }

@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and container orchestrators."""
    return {"status": "healthy", "model_loaded": model is not None}